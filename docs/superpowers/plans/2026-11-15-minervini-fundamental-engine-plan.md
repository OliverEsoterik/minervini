# Minervini Fundamental Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-metric `FundamentalFilter` with a 5-dimension `MinerviniFundamentalEngine` that checks EPS growth, EPS acceleration, revenue growth, revenue acceleration, and profit margin improvement using tiered Minervini thresholds (15% YoY / 25% QoQ).

**Architecture:** New dataclasses `MetricResult` and `MinerviniFundamentalResult` replace `FundamentalFilterResult`. New class `MinerviniFundamentalEngine` extracts all 5 dimensions from a single `yfinance.Ticker.quarterly_financials` fetch. The Pipeline calls it instead of the old filter; approval logic uses a configurable `min_score` (default 3/5).

**Tech Stack:** Python 3.10+, yfinance, numpy, pandas

## Global Constraints

- All data from `yfinance` — no mock data, no hardcoded prices
- Data comes from `Ticker.quarterly_financials` — single call per ticker
- Each dimension with insufficient data marks `passed=False` with a clear detail string, never crashes
- `Basic EPS` preferred; fallback to `Diluted EPS` if Basic is missing
- All thresholds: YoY ≥ 15%, QoQ ≥ 25%, margin improvement ≥ 1pp or strictly upward trend
- All changes on `feature/minervini-fundamental-engine` branch

---

## File Changes

| File | Change |
|------|--------|
| `vcp_momentum_engine.py` | Add 2 dataclasses + 1 class. Replace `FundamentalFilter` reference in `Pipeline`. Update `ScreeningResult`. |
| `batch_screen.py` | Update near-miss output to show dimensional breakdown |
| `check_trend.py` | Show fundamental breakdown alongside trend template |
| `README.md` | Document new 5-dimension fundamental filter |

---

### Task 1: Add MetricResult and MinerviniFundamentalResult dataclasses

**Files:**
- Modify: `vcp_momentum_engine.py` — insert after existing `FundamentalFilterResult` dataclass (or replace it)

**Interfaces:**
- Consumes: nothing
- Produces: `MetricResult(label, passed, value, threshold, quarters, detail)` and `MinerviniFundamentalResult(ticker, eps_growth, eps_acceleration, revenue_growth, revenue_acceleration, profit_margin, score, total_dims, overall_passed, details)`

- [ ] **Step 1: Locate the insertion point**

Read the current `FundamentalFilterResult` dataclass in `vcp_momentum_engine.py` to find where to insert the new dataclasses (replacing the old one).

- [ ] **Step 2: Replace `FundamentalFilterResult` with the new dataclasses**

Replace the existing `FundamentalFilterResult` dataclass (and any related imports/dataclass) with:

```python
@dataclass
class MetricResult:
    """Result for one fundamental dimension (EPS, revenue, margin, etc.)."""
    label: str                       # e.g. "EPS growth (YoY)"
    passed: bool                     # meets threshold?
    value: Optional[float] = None    # growth rate / margin % (as decimal, e.g. 0.22)
    threshold: Optional[float] = None
    quarters: List[float] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "passed": self._to_native(self.passed),
            "value": round(self._to_native(self.value), 4) if self.value is not None else None,
            "threshold": self.threshold,
            "detail": self.detail,
        }

    @staticmethod
    def _to_native(v: Any) -> Any:
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v


@dataclass
class MinerviniFundamentalResult:
    """Aggregate result of the 5-dimension Minervini fundamental check."""
    ticker: str
    eps_growth: MetricResult
    eps_acceleration: MetricResult
    revenue_growth: MetricResult
    revenue_acceleration: MetricResult
    profit_margin: MetricResult
    score: int = 0
    total_dims: int = 5
    overall_passed: bool = False
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "score": self.score,
            "total_dims": self.total_dims,
            "eps_growth": self.eps_growth.to_dict(),
            "eps_acceleration": self.eps_acceleration.to_dict(),
            "revenue_growth": self.revenue_growth.to_dict(),
            "revenue_acceleration": self.revenue_acceleration.to_dict(),
            "profit_margin": self.profit_margin.to_dict(),
        }
```

- [ ] **Step 3: Run syntax check**

```bash
python3 -c "import ast; ast.parse(open('vcp_momentum_engine.py').read()); print('OK')"
```

Expected: OK

- [ ] **Step 4: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "feat: add MetricResult and MinerviniFundamentalResult dataclasses"
```

---

### Task 2: Implement MinerviniFundamentalEngine class

**Files:**
- Modify: `vcp_momentum_engine.py` — add `MinerviniFundamentalEngine` class after the dataclasses, replacing the old `FundamentalFilter` class

**Interfaces:**
- Consumes: `MetricResult`, `MinerviniFundamentalResult` (from Task 1)
- Consumes: `DataIngestion.fetch_quarterly_financials()` (existing)
- Produces: `MinerviniFundamentalEngine.screen(ticker: str) -> MinerviniFundamentalResult`

- [ ] **Step 1: Write `MinerviniFundamentalEngine` class**

Add the following class, replacing the entire old `FundamentalFilter` class and `FundamentalFilterResult`:

```python
class MinerviniFundamentalEngine:
    """5-dimension fundamental engine based on Mark Minervini's criteria.

    Evaluates EPS growth, EPS acceleration, revenue growth, revenue
    acceleration, and profit margin improvement from quarterly financials.
    """

    YOY_THRESHOLD: float = 0.15    # 15 %
    QOQ_THRESHOLD: float = 0.25    # 25 %
    MIN_MARGIN_IMPROVEMENT_PP: float = 0.01  # 1 percentage point

    def __init__(self, min_score: int = 3) -> None:
        self.min_score = min_score

    # ------------------------------------------------------------------
    def screen(self, ticker: str) -> MinerviniFundamentalResult:
        """Fetch quarterly financials and evaluate all 5 dimensions.

        Returns a ``MinerviniFundamentalResult`` with per-dimension breakdown.
        Never raises — missing data produces ``passed=False`` with a clear
        detail string.
        """
        revenues, earnings = DataIngestion.fetch_quarterly_financials(ticker)
        rev_series = revenues     # pd.Series, oldest-first
        ear_series = earnings

        # Fetch EPS and Gross Profit from a fresh call
        eps_series, gp_series, oi_series = self._fetch_extra_metrics(ticker)

        # Compute each dimension
        eps_growth = self._eval_eps_growth(eps_series)
        eps_accel = self._eval_eps_acceleration(eps_series)
        rev_growth = self._eval_revenue_growth(rev_series)
        rev_accel = self._eval_revenue_acceleration(rev_series)
        margin = self._eval_profit_margin(gp_series, ear_series, rev_series)

        dims = [eps_growth, eps_accel, rev_growth, rev_accel, margin]
        score = sum(1 for d in dims if d.passed)
        total = sum(1 for d in dims if d.value is not None or "insufficient" not in d.detail.lower())

        detail_lines = [
            f"  EPS growth:       {'✅' if eps_growth.passed else '❌'} {eps_growth.detail}",
            f"  EPS acceleration: {'✅' if eps_accel.passed else '❌'} {eps_accel.detail}",
            f"  Rev growth:       {'✅' if rev_growth.passed else '❌'} {rev_growth.detail}",
            f"  Rev acceleration: {'✅' if rev_accel.passed else '❌'} {rev_accel.detail}",
            f"  Profit margin:    {'✅' if margin.passed else '❌'} {margin.detail}",
            f"  Score: {score}/{total} passed",
        ]
        overall = score >= self.min_score

        logger.info(
            "MinerviniFundamental [%s]: %s (%d/%d dims, min_score=%d)",
            ticker, "PASS" if overall else "FAIL", score, total, self.min_score,
        )

        return MinerviniFundamentalResult(
            ticker=ticker,
            eps_growth=eps_growth,
            eps_acceleration=eps_accel,
            revenue_growth=rev_growth,
            revenue_acceleration=rev_accel,
            profit_margin=margin,
            score=score,
            total_dims=total,
            overall_passed=overall,
            details="\n".join(detail_lines),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_extra_metrics(
        ticker: str,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Fetch EPS, Gross Profit, and Operating Income from quarterly financials."""
        eps = pd.Series()
        gp = pd.Series()
        oi = pd.Series()
        try:
            t = yf.Ticker(ticker)
            qf = t.quarterly_financials
            if qf is None or qf.empty:
                return eps, gp, oi

            if "Basic EPS" in qf.index:
                eps = qf.loc["Basic EPS"].dropna().astype(float).sort_index()
            elif "Diluted EPS" in qf.index:
                eps = qf.loc["Diluted EPS"].dropna().astype(float).sort_index()

            if "Gross Profit" in qf.index:
                gp = qf.loc["Gross Profit"].dropna().astype(float).sort_index()

            if "Operating Income" in qf.index:
                oi = qf.loc["Operating Income"].dropna().astype(float).sort_index()

        except Exception as exc:
            logger.warning("_fetch_extra_metrics(%s) failed: %s", ticker, exc)

        return eps, gp, oi

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_yoy_growth(series: pd.Series) -> List[float]:
        """Compute YoY growth rates: (val[t] - val[t-4]) / val[t-4]."""
        vals = series.values
        rates: List[float] = []
        for i in range(4, len(vals)):
            prev = vals[i - 4]
            curr = vals[i]
            if prev > 0:
                rates.append(float((curr - prev) / prev))
            else:
                rates.append(0.0)
        return rates

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_eps_growth(eps: pd.Series) -> MetricResult:
        """Check EPS growth against YoY and QoQ thresholds."""
        if len(eps) < 2:
            return MetricResult(
                label="EPS growth", passed=False,
                detail="Insufficient EPS data.",
            )

        vals = eps.values
        # Prefer YoY (≥ 5 quarters)
        if len(vals) >= 5:
            latest = (vals[-1] - vals[-5]) / vals[-5] if vals[-5] > 0 else 0.0
            threshold = 0.15
            mode = "YoY"
        else:
            latest = (vals[-1] - vals[-2]) / vals[-2] if vals[-2] > 0 else 0.0
            threshold = 0.25
            mode = "QoQ"

        passed = latest >= threshold
        return MetricResult(
            label=f"EPS growth ({mode})",
            passed=passed,
            value=latest,
            threshold=threshold,
            quarters=[float(v) for v in vals],
            detail=f"{latest*100:+.2f}% {'≥' if passed else '<'} {threshold*100:.0f}% ({mode})",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_eps_acceleration(eps: pd.Series) -> MetricResult:
        """Check if EPS YoY growth rate is accelerating."""
        if len(eps) < 6:
            return MetricResult(
                label="EPS acceleration",
                passed=False,
                detail="Insufficient data (need ≥ 6 quarters).",
            )

        rates = MinerviniFundamentalEngine._compute_yoy_growth(eps)
        if len(rates) < 2:
            return MetricResult(
                label="EPS acceleration",
                passed=False,
                detail="Insufficient YoY comparisons.",
            )

        latest = rates[-1]
        prior = rates[-2]
        accelerating = latest > prior
        return MetricResult(
            label="EPS acceleration",
            passed=accelerating,
            value=latest,
            threshold=prior,
            quarters=[float(r) for r in rates],
            detail=(
                f"{latest*100:+.2f}% → {prior*100:+.2f}% "
                f"{'accelerating' if accelerating else 'decelerating'}"
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_revenue_growth(rev: pd.Series) -> MetricResult:
        """Check revenue growth against YoY and QoQ thresholds."""
        if len(rev) < 2:
            return MetricResult(
                label="Revenue growth", passed=False,
                detail="Insufficient revenue data.",
            )

        vals = rev.values
        if len(vals) >= 5:
            latest = (vals[-1] - vals[-5]) / vals[-5] if vals[-5] > 0 else 0.0
            threshold = 0.15
            mode = "YoY"
        else:
            latest = (vals[-1] - vals[-2]) / vals[-2] if vals[-2] > 0 else 0.0
            threshold = 0.25
            mode = "QoQ"

        passed = latest >= threshold
        return MetricResult(
            label=f"Revenue growth ({mode})",
            passed=passed,
            value=latest,
            threshold=threshold,
            quarters=[float(v) for v in vals],
            detail=f"{latest*100:+.2f}% {'≥' if passed else '<'} {threshold*100:.0f}% ({mode})",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_revenue_acceleration(rev: pd.Series) -> MetricResult:
        """Check if revenue YoY growth rate is accelerating."""
        if len(rev) < 6:
            return MetricResult(
                label="Revenue acceleration",
                passed=False,
                detail="Insufficient data (need ≥ 6 quarters).",
            )

        rates = MinerviniFundamentalEngine._compute_yoy_growth(rev)
        if len(rates) < 2:
            return MetricResult(
                label="Revenue acceleration",
                passed=False,
                detail="Insufficient YoY comparisons.",
            )

        latest = rates[-1]
        prior = rates[-2]
        accelerating = latest > prior
        return MetricResult(
            label="Revenue acceleration",
            passed=accelerating,
            value=latest,
            threshold=prior,
            quarters=[float(r) for r in rates],
            detail=(
                f"{latest*100:+.2f}% → {prior*100:+.2f}% "
                f"{'accelerating' if accelerating else 'decelerating'}"
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_profit_margin(
        gp: pd.Series, ear: pd.Series, rev: pd.Series,
    ) -> MetricResult:
        """Check if gross and net profit margins are improving.

        Uses Gross Profit / Total Revenue for gross margin.
        Uses Net Income / Total Revenue for net margin.
        Requires ≥ 3 quarters to detect a trend.
        """
        if rev.empty or len(rev) < 3:
            return MetricResult(
                label="Profit margin",
                passed=False,
                detail="Insufficient data (need ≥ 3 quarters of revenue).",
            )

        rev_vals = rev.values
        gross_margins: List[float] = []
        net_margins: List[float] = []

        if not gp.empty:
            gp_vals = gp.values
            for i in range(min(len(gp_vals), len(rev_vals))):
                if rev_vals[i] > 0:
                    gross_margins.append(float(gp_vals[i] / rev_vals[i]))

        if not ear.empty:
            ear_vals = ear.values
            for i in range(min(len(ear_vals), len(rev_vals))):
                if rev_vals[i] > 0:
                    net_margins.append(float(ear_vals[i] / rev_vals[i]))

        if not gross_margins and not net_margins:
            return MetricResult(
                label="Profit margin",
                passed=False,
                detail="No margin data available.",
            )

        # Check if margins are improving over last 3 periods
        improving = False
        detail_parts: List[str] = []

        if len(gross_margins) >= 3:
            gm = gross_margins[-3:]
            gm_improving = gm[-1] > gm[-2] > gm[-3]
            improving = improving or gm_improving
            detail_parts.append(
                f"gross {'↑' if gm_improving else '→'} "
                f"{gm[0]*100:.1f}%→{gm[1]*100:.1f}%→{gm[2]*100:.1f}%"
            )
        elif gross_margins:
            detail_parts.append(f"gross {gross_margins[-1]*100:.1f}%")

        if len(net_margins) >= 3:
            nm = net_margins[-3:]
            nm_improving = nm[-1] > nm[-2] > nm[-3]
            improving = improving or nm_improving
            detail_parts.append(
                f"net {'↑' if nm_improving else '→'} "
                f"{nm[0]*100:.1f}%→{nm[1]*100:.1f}%→{nm[2]*100:.1f}%"
            )
        elif net_margins:
            detail_parts.append(f"net {net_margins[-1]*100:.1f}%")

        # Build a value for MetricResult — margin improvement in pp
        if gross_margins:
            margin_improvement_pp = (gross_margins[-1] - gross_margins[0]) * 100
        elif net_margins:
            margin_improvement_pp = (net_margins[-1] - net_margins[0]) * 100
        else:
            margin_improvement_pp = 0.0

        # At least 1pp improvement required for pass
        passed = improving and margin_improvement_pp >= 1.0

        return MetricResult(
            label="Profit margin",
            passed=passed,
            value=round(margin_improvement_pp, 2),
            threshold=1.0,
            quarters=gross_margins or net_margins,
            detail=" | ".join(detail_parts),
        )
```

- [ ] **Step 2: Remove the old `FundamentalFilterResult` and `FundamentalFilter` classes**

Delete the entire `FundamentalFilterResult` dataclass and `FundamentalFilter` class (they're replaced by the new classes above). The old `FundamentalFilterResult` starts around line 385 and the `FundamentalFilter` class around line 415.

Make sure to also remove the old `FundamentalFilterResult` reference — the dataclass definition. The class `FundamentalFilter` is about 60 lines. Remove both cleanly.

- [ ] **Step 3: Run syntax check**

```bash
python3 -c "import ast; ast.parse(open('vcp_momentum_engine.py').read()); print('OK')"
```

Expected: OK

- [ ] **Step 4: Quick import check**

```bash
python3 -c "from vcp_momentum_engine import MinerviniFundamentalEngine, MetricResult, MinerviniFundamentalResult; print('Import OK')"
```

Expected: Import OK

- [ ] **Step 5: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "feat: implement MinerviniFundamentalEngine with 5 dimensions"
```

---

### Task 3: Integrate into Pipeline

**Files:**
- Modify: `vcp_momentum_engine.py` — update `Pipeline.__init__`, `Pipeline.screen()`, and `ScreeningResult`

**Interfaces:**
- Consumes: `MinerviniFundamentalEngine`, `MinerviniFundamentalResult` (Task 2)
- Produces: updated `ScreeningResult` with `minervini_fundamental` field

- [ ] **Step 1: Update `ScreeningResult` dataclass**

Replace `fundamental_filter: Optional[FundamentalFilterResult] = None` with:

```python
minervini_fundamental: Optional[MinerviniFundamentalResult] = None
```

Also remove the old `fundamental_filter` from the `to_dict()` method and add the new one.

- [ ] **Step 2: Update `Pipeline.__init__`**

Replace:
```python
self.fundamental = FundamentalFilter()
```
with:
```python
self.fundamental = MinerviniFundamentalEngine(min_score=3)
```

- [ ] **Step 3: Update `Pipeline.screen()`**

Replace the old fundamental filter call:
```python
fundamental_result = self.fundamental.screen(ticker)
result.fundamental_filter = fundamental_result
```

With:
```python
fundamental_result = self.fundamental.screen(ticker)
result.minervini_fundamental = fundamental_result
```

And update the approval logic to use the new field:
```python
ff_pass = fundamental_result.overall_passed
```

Update the approval summary log line to use `overall_passed` instead of `.passed`.

- [ ] **Step 4: Update `ScreeningResult.to_dict()`**

Remove the old `fundamental_filter` section and replace with:

```python
if self.minervini_fundamental:
    d["minervini_fundamental"] = self.minervini_fundamental.to_dict()
```

Also update the `canslim` section to reference the new field if needed.

- [ ] **Step 5: Run a test**

```bash
python3 vcp_momentum_engine.py ADI COP 2>&1 | grep -E "(APPROVED|Minervini|Score)"
```

Expected: both ADI and COP should show their fundamental score breakdown.

- [ ] **Step 6: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "feat: integrate MinerviniFundamentalEngine into Pipeline"
```

---

### Task 4: Update batch_screen.py output

**Files:**
- Modify: `batch_screen.py`

- [ ] **Step 1: Update the near-miss sections**

Find the "NEAR-MISSES (TT+VCP PASS, FUND FAIL)" section. Replace the simple `ff_note` that only shows YoY growth with the dimensional breakdown:

```python
if near_miss_ff:
    print(f"\n{'─'*72}")
    print(f"  ⚠  NEAR-MISSES (TT+VCP PASS, FUND FAIL)")
    print(f"{'─'*72}")
    for r in sorted(near_miss_ff, key=lambda x: x.ticker):
        mf = r.minervi_fundamental
        if mf:
            dims = [
                f"EPS:{'✅' if mf.eps_growth.passed else '❌'}"
                    + (f"{mf.eps_growth.value*100:.0f}%" if mf.eps_growth.value is not None else ""),
                f"Rev:{'✅' if mf.revenue_growth.passed else '❌'}"
                    + (f"{mf.revenue_growth.value*100:.0f}%" if mf.revenue_growth.value is not None else ""),
                f"Marg:{'✅' if mf.profit_margin.passed else '❌'}",
                f"Accel:{mf.score}/5",
            ]
            print(f"  {r.ticker:8s} → {' | '.join(dims)}")
        else:
            print(f"  {r.ticker:8s} → no fundamental data")
```

- [ ] **Step 2: Commit**

```bash
git add batch_screen.py
git commit -m "feat: update batch_screen.py with dimensional fundamental breakdown"
```

---

### Task 5: Update check_trend.py output

**Files:**
- Modify: `check_trend.py`

- [ ] **Step 1: Add fundamental breakdown to the output**

After the trend template verdict section in `check_ticker()`, add:

```python
# --- fundamental check --------------------------------------------
from vcp_momentum_engine import MinerviniFundamentalEngine
mf_engine = MinerviniFundamentalEngine()
mf_result = mf_engine.screen(ticker)
print(f"  Fundamental (Minervini 5-dim):")
print(f"    {'✅' if mf_result.eps_growth.passed else '❌'}  EPS growth:       {mf_result.eps_growth.detail}")
print(f"    {'✅' if mf_result.eps_acceleration.passed else '❌'}  EPS acceleration: {mf_result.eps_acceleration.detail}")
print(f"    {'✅' if mf_result.revenue_growth.passed else '❌'}  Rev growth:       {mf_result.revenue_growth.detail}")
print(f"    {'✅' if mf_result.revenue_acceleration.passed else '❌'}  Rev acceleration: {mf_result.revenue_acceleration.detail}")
print(f"    {'✅' if mf_result.profit_margin.passed else '❌'}  Profit margin:    {mf_result.profit_margin.detail}")
print(f"    Score: {mf_result.score}/{mf_result.total_dims}")
print()
```

- [ ] **Step 2: Commit**

```bash
git add check_trend.py
git commit -m "feat: show Minervini fundamental breakdown in check_trend.py"
```

---

### Task 6: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the "Fundamental Acceleration Filter" section**

Replace the current Fundamental Filter section with:

```markdown
### Minervini Fundamental Engine (5-Dimension)

Based on Mark Minervini's fundamental analysis criteria from *Think and Trade Like a Champion*.
All data is extracted from `yfinance` quarterly financials in a single call.

| # | Dimension | Source | Threshold |
|---|-----------|--------|-----------|
| 1 | **EPS growth** | Basic EPS (or Diluted EPS) | YoY ≥ 15% / QoQ ≥ 25% |
| 2 | **EPS acceleration** | Same EPS series | Growth rate increasing across consecutive periods |
| 3 | **Revenue growth** | Total Revenue | YoY ≥ 15% / QoQ ≥ 25% |
| 4 | **Revenue acceleration** | Same revenue series | Growth rate increasing across consecutive periods |
| 5 | **Profit margin improvement** | Gross Profit / Total Revenue + Net Income / Total Revenue | Upward trend across 3+ quarters, ≥ 1pp improvement |

**Scoring:** Each dimension is independently evaluated. The default approval threshold is
**3 out of 5** dimensions passing. Configure via `MinerviniFundamentalEngine(min_score=N)`.

**Edge cases:** Missing quarterly data produces `passed=False` with a clear reason. The
engine never crashes — it reports what data is available and scores accordingly.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with 5-dimension fundamental filter"
```

---

### Task 7: Run full test on watchlist

**Files:**
- Run: `batch_screen.py` on a small subset

- [ ] **Step 1: Test on the original 4-ticker watchlist**

```bash
python3 vcp_momentum_engine.py AAPL MSFT NVDA PLTR 2>&1 | grep -E "(MinerviniFundamental|\d/\d)"
```

Expected: each ticker shows its 5-dimension breakdown with score.

- [ ] **Step 2: Verify old FundamentalFilter is fully removed**

```bash
grep -n "FundamentalFilter\|fundamental_filter" vcp_momentum_engine.py | grep -v "MinerviniFundamental\|minervini_fundamental\|#.*deprecated"
```

Expected: no references to the old class remain (except comments).

- [ ] **Step 3: Run batch_screen.py on a 10-ticker sample**

```bash
python3 vcp_momentum_engine.py ADI COP EXPD HUM REGN XOM AAPL KO MSFT AMZN 2>&1 | grep -E "(APPROVED|Score|EPS|Rev|Marg)"
```

Expected: each ticker shows dimensional breakdown. At least some show passing EPS/Rev with score info.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "fix: final adjustments after full test"
```

---

### Task 8: Push branch

- [ ] **Step 1: Push to origin**

```bash
git push -u origin feature/minervini-fundamental-engine
```