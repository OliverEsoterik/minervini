# Fix Minervini Engine Bugs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 systemic bugs in the Minervini screening engine that caused false-positive approval of REGN (and potentially other stocks).

**Architecture:** All fixes are in `vcp_momentum_engine.py`:
1. Add 150-day SMA to `TrendTemplateValidator` (was missing entirely — the root cause of REGN passing Trend Template)
2. Fix dimension counting in `MinerviniFundamentalEngine` (floating denominator let missing data make passing easier)
3. Add ROE check to `MinerviniFundamentalEngine` (was a missing dimension)

**Tech Stack:** Python 3, yfinance, pandas, numpy

**File:** `vcp_momentum_engine.py` (1556 lines, single-module engine)

## Global Constraints

- No changes to any file except `vcp_momentum_engine.py`
- No new dependencies
- All existing tests/pipelines must continue to work (COP must still pass)
- Follow existing code style (dataclass results, MetricResult pattern, logger pattern)
- Each fix is independently testable — verify before/after behavior for both COP and REGN

---

## File Structure Changes

| Change | File | Lines |
|--------|------|-------|
| Add SMA-150 to TrendTemplate | `vcp_momentum_engine.py` | Module 2 (~L130-270) |
| Fix dimension counting | `vcp_momentum_engine.py` | Module 3, `screen()` method |
| Add ROE check | `vcp_momentum_engine.py` | Module 3, new method + dataclass field |

---

### Task 1: Add 150-day SMA to TrendTemplateValidator

**Files:**
- Modify: `vcp_momentum_engine.py` — Module 2 section

**Interfaces:**
- Consumes: `TrendTemplateResult` dataclass, `TrendTemplateValidator.validate()` method
- Produces: Updated `TrendTemplateResult` with `sma_150` field; updated validator that checks 6 conditions

**Context:**
The current validator checks only Close > SMA-50, SMA-50 > SMA-200, and slope of both. Minervini's full Trend Template requires the three-step MA chain: Close > SMA-50 > SMA-150 > SMA-200. The script was using SMA-50 > SMA-200 as a proxy for the two-step chain, but this fails to catch stocks where SMA-150 hasn't crossed SMA-200 yet (REGN: SMA-150=$742 < SMA-200=$748, while SMA-50=$821 > SMA-200=$748).

- [ ] **Step 1: Add `sma_150` and `sma_150_slope` fields to `TrendTemplateResult` dataclass**

Add two new optional float fields to the dataclass at approximately line 130:

```python
@dataclass
class TrendTemplateResult:
    """Result of running the Trend Template on one ticker."""
    ticker: str
    passed: bool
    current_close: float
    sma_50: float
    sma_150: Optional[float] = None     # NEW
    sma_200: float
    sma_50_slope: float
    sma_150_slope: Optional[float] = None  # NEW
    sma_200_slope: float
    details: str = ""
```

- [ ] **Step 2: Compute SMA-150 and add 3 new criteria checks in `TrendTemplateValidator.validate()`**

In the `validate()` method, after the existing SMA-50 and SMA-200 computation, add:

```python
# --- compute SMA-150 (min 150 periods) ---
sma_150 = close.rolling(window=150, min_periods=150).mean()
latest_sma_150 = float(sma_150.iloc[-1]) if len(close) >= 150 else float('nan')
slope_150 = self._slope(sma_150) if len(close) >= 150 else 0.0
```

Then expand the checks list from 4 to 7 items (the existing 4 plus 3 new ones). The new checks are:

```python
# --- new checks ------------------------------------------------
(
    latest_close > latest_sma_150 if not math.isnan(latest_sma_150) else False,
    f"Close {latest_close:.2f} > SMA-150 {latest_sma_150:.2f}",
),
(
    latest_sma_50 > latest_sma_150 if not math.isnan(latest_sma_150) else False,
    f"SMA-50 {latest_sma_50:.2f} > SMA-150 {latest_sma_150:.2f}",
),
(
    latest_sma_150 > latest_sma_200 if not math.isnan(latest_sma_150) else False,
    f"SMA-150 {latest_sma_150:.2f} > SMA-200 {latest_sma_200:.2f}",
),
```

Also update the `TrendTemplateResult` construction to pass `sma_150=latest_sma_150` and `sma_150_slope=slope_150`.

Add `import math` at the top of the file if not already present.

- [ ] **Step 3: Verify REGN now FAILS Trend Template**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import DataIngestion, TrendTemplateValidator
data = DataIngestion().fetch('REGN')
result = TrendTemplateValidator().validate(data)
print(f'Passed: {result.passed}')
print(f'Details: {result.details}')
print(f'SMA-150: {result.sma_150:.2f}')
assert not result.passed, 'REGN should FAIL trend template now'
print('✅ REGN correctly fails')
"
```

Expected: `Passed: False` with the check `Close > SMA-50 > SMA-150 > SMA-200` showing SMA-150 < SMA-200.

- [ ] **Step 4: Verify COP still PASSES Trend Template**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import DataIngestion, TrendTemplateValidator
data = DataIngestion().fetch('COP')
result = TrendTemplateValidator().validate(data)
print(f'Passed: {result.passed}')
print(f'Details: {result.details}')
print(f'SMA-150: {result.sma_150:.2f}')
assert result.passed, 'COP should still pass trend template'
print('✅ COP correctly passes')
"
```

Expected: `Passed: True` with all 6 checks passing.

- [ ] **Step 5: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "fix: add 150-day SMA to TrendTemplateValidator

Adds SMA-150 computation and 3 missing Minervini criteria checks:
- Close > SMA-150
- SMA-50 > SMA-150
- SMA-150 > SMA-200

Previously the validator used SMA-50 > SMA-200 as a proxy for the
two-step chain, missing stocks where SMA-150 hasn't crossed SMA-200
yet (e.g. REGN)."
```

---

### Task 2: Fix fundamental engine dimension counting

**Files:**
- Modify: `vcp_momentum_engine.py` — Module 3, `MinerviniFundamentalEngine.screen()` method

**Interfaces:**
- Consumes: `MinerviniFundamentalResult.overall_passed` and `total_dims` fields
- Produces: Honest dimension counting where missing data doesn't shrink the denominator

**Context:**
The engine calculates `total = sum(1 for d in dims if d.value is not None)`. When EPS acceleration and Revenue acceleration can't be evaluated (need ≥6 quarters, REGN has 5), they return `value=None` and are excluded from the denominator. With only 3 active dimensions and `min_score=3`, passing is trivially 3/3 = 100%. The fix makes the denominator always equal the number of dimensions (currently 5, will be 6 after Task 3).

- [ ] **Step 1: Fix the `total` computation in `screen()`**

Find the lines in `MinerviniFundamentalEngine.screen()` (approximately line 540):

```python
# Count dimensions where data was available (value set)
total = sum(1 for d in dims if d.value is not None)
```

Replace with:

```python
# Total dimensions in the engine (fixed denominator — missing data counts as fail)
total = len(dims)
```

- [ ] **Step 2: Also update the log message to show actual vs available**

Find the line:

```python
logger.info(
    "MinerviniFundamental [%s]: %s (%d/%d dims, min_score=%d)",
    ticker, "PASS" if overall else "FAIL", score, total, self.min_score,
)
```

And update it to include data-available count for transparency:

```python
available_dims = sum(1 for d in dims if d.value is not None)
logger.info(
    "MinerviniFundamental [%s]: %s (%d/%d dims, %d available, min_score=%d)",
    ticker, "PASS" if overall else "FAIL", score, total, available_dims, self.min_score,
)
```

- [ ] **Step 3: Verify REGN's fundamental now shows honest 3/5 (or 3/6 after Task 3)**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
result = MinerviniFundamentalEngine().screen('REGN')
print(f'Score: {result.score}/{result.total_dims}')
print(f'Overall passed: {result.overall_passed}')
print(result.details)
assert result.score < result.total_dims, 'Score should show not all dims passed'
"
```

After Task 3 (ROE) is also done, the expected output is that REGN's `total_dims` is 6, not 3.

- [ ] **Step 4: Verify COP still passes fundamental**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
result = MinerviniFundamentalEngine().screen('COP')
print(f'Score: {result.score}/{result.total_dims}')
print(f'Overall passed: {result.overall_passed}')
print(result.details)
"
```

- [ ] **Step 5: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "fix: use fixed denominator in fundamental dimension counting

Changes total_dims from a floating count (excluding dims where value
is None) to a fixed count of all dimensions. Missing data now counts
as a fail rather than shrinking the denominator, preventing degenerate
passes like REGN scoring 3/3 when 2 of 5 dimensions were excluded."
```

---

### Task 3: Add ROE check to MinerviniFundamentalEngine

**Files:**
- Modify: `vcp_momentum_engine.py` — Module 3

**Interfaces:**
- Consumes: `MinerviniFundamentalResult` dataclass — add `roe` field
- Produces: ROE check as 6th fundamental dimension

**Context:**
Minervini's fundamental screening includes Return on Equity > 17%. The engine was missing this entirely. REGN fails ROE at ~14%. The ROE is computed as: trailing 12 months Net Income (sum of latest 4 quarters) divided by latest Stockholders Equity from the quarterly balance sheet.

- [ ] **Step 1: Add `roe` field to `MinerviniFundamentalResult` dataclass**

```python
@dataclass
class MinerviniFundamentalResult:
    """Aggregate result of the 5-dimension Minervini fundamental check."""
    ticker: str
    eps_growth: MetricResult
    eps_acceleration: MetricResult
    revenue_growth: MetricResult
    revenue_acceleration: MetricResult
    profit_margin: MetricResult
    roe: MetricResult          # NEW
    score: int = 0
    total_dims: int = 6        # CHANGED: 5 → 6
    overall_passed: bool = False
    details: str = ""
```

Also update the `to_dict()` method to include `roe`.

- [ ] **Step 2: Add `_eval_roe()` static method**

Add this method to `MinerviniFundamentalEngine`:

```python
@staticmethod
def _eval_roe(ticker: str) -> MetricResult:
    """Compute Return on Equity and check against 17% threshold.

    ROE = Trailing 12-month Net Income / Latest Stockholders Equity

    Uses quarterly balance sheet for equity and quarterly financials
    for net income, both from yfinance.
    """
    ROE_THRESHOLD = 0.17  # 17%

    try:
        t = yf.Ticker(ticker)

        # Get net income from quarterly financials
        qf = t.quarterly_financials
        if qf is None or qf.empty or "Net Income" not in qf.index:
            return MetricResult(
                label="Return on Equity (ROE)",
                passed=False,
                detail="No net income data available.",
            )

        ni = qf.loc["Net Income"].dropna().astype(float)
        if len(ni) < 4:
            return MetricResult(
                label="Return on Equity (ROE)",
                passed=False,
                detail=f"Need ≥4 quarters of net income, got {len(ni)}.",
            )

        trailing_ni = float(ni.tail(4).sum())

        # Get stockholders' equity from quarterly balance sheet
        qbs = t.quarterly_balance_sheet
        if qbs is None or qbs.empty or "Stockholders Equity" not in qbs.index:
            # Fall back to annual balance sheet
            bs = t.balance_sheet
            if bs is None or bs.empty or "Stockholders Equity" not in bs.index:
                return MetricResult(
                    label="Return on Equity (ROE)",
                    passed=False,
                    detail="No stockholders' equity data available.",
                )
            equity = float(bs.loc["Stockholders Equity"].dropna().iloc[-1])
        else:
            equity_series = qbs.loc["Stockholders Equity"].dropna()
            if equity_series.empty:
                return MetricResult(
                    label="Return on Equity (ROE)",
                    passed=False,
                    detail="Stockholders equity series is empty.",
                )
            equity = float(equity_series.iloc[-1])

        if equity <= 0:
            return MetricResult(
                label="Return on Equity (ROE)",
                passed=False,
                detail=f"Non-positive equity: {equity}.",
            )

        roe = trailing_ni / equity
        passed = roe >= ROE_THRESHOLD

        return MetricResult(
            label="Return on Equity (ROE)",
            passed=passed,
            value=roe,
            threshold=ROE_THRESHOLD,
            detail=(
                f"{roe*100:.2f}% {'≥' if passed else '<'} "
                f"{ROE_THRESHOLD*100:.0f}% "
                f"(NI={trailing_ni:.0f}, EQ={equity:.0f})"
            ),
        )

    except Exception as exc:
        logger.warning("ROE check failed for %s: %s", ticker, exc)
        return MetricResult(
            label="Return on Equity (ROE)",
            passed=False,
            detail=f"Error: {exc}",
        )
```

Add `import yfinance as yf` at the top of the module if it's not already there (it should be — check).

- [ ] **Step 3: Integrate ROE into `screen()` method**

In the `screen()` method, add the ROE evaluation call and include it in the dims list:

```python
# Compute each dimension
eps_growth = self._eval_eps_growth(eps_series)
eps_accel = self._eval_eps_acceleration(eps_series)
rev_growth = self._eval_revenue_growth(rev_series)
rev_accel = self._eval_revenue_acceleration(rev_series)
margin = self._eval_profit_margin(gp_series, ear_series, rev_series)
roe = self._eval_roe(ticker)      # NEW

dims = [eps_growth, eps_accel, rev_growth, rev_accel, margin, roe]  # CHANGED: +roe
```

Update the detail_lines and the `MinerviniFundamentalResult` constructor to include the `roe` field.

Also update the error-return path (the one that fires when quarterly financials fetch fails) to include a placeholder `roe` field.

- [ ] **Step 4: Write the initial `screen()` call to also check `roe` dimension**

Also in `screen()`, update the detail lines:

```python
detail_lines = [
    f"  EPS growth:       {'✅' if eps_growth.passed else '❌'} {eps_growth.detail}",
    f"  EPS acceleration: {'✅' if eps_accel.passed else '❌'} {eps_accel.detail}",
    f"  Rev growth:       {'✅' if rev_growth.passed else '❌'} {rev_growth.detail}",
    f"  Rev acceleration: {'✅' if rev_accel.passed else '❌'} {rev_accel.detail}",
    f"  Profit margin:    {'✅' if margin.passed else '❌'} {margin.detail}",
    f"  ROE:              {'✅' if roe.passed else '❌'} {roe.detail}",   # NEW
    f"  Score: {score}/{total} passed",
]
```

- [ ] **Step 5: Verify REGN now shows ROE failure in fundamental**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
result = MinerviniFundamentalEngine().screen('REGN')
print(f'Score: {result.score}/{result.total_dims}')
print(f'Overall passed: {result.overall_passed}')
print(result.details)
# ROE should be present and failing
assert result.roe is not None
print(f'ROE passed: {result.roe.passed}, ROE value: {result.roe.value*100 if result.roe.value else None:.2f}%')
"
```

Expected: Score 3/6, overall_passed=False (3 < min_score=3? Wait — min_score is still 3). Actually with min_score=3 and 6 dims, 3 >= 3 so it still passes. Let me address this.

**Decision point:** With ROE as a 6th dimension, should `min_score` stay at 3 or go to 4?

Current behavior with all fixes applied:
- REGN: Score=3/6 (EPS growth ✅, Revenue growth ✅, Profit margin ✅; EPS accel ❌, Rev accel ❌, ROE ❌)
  - With min_score=3: 3 >= 3 → PASS
  - With min_score=4: 3 < 4 → FAIL
- COP: Score=5/6 or 6/6 (all dims pass except possibly ROE or acceleration)
  - Either way, passes both thresholds

The analysis shows REGN should fail fundamental on ROE (14.95% < 17%) — but ROE alone only contributes 1 fail. With min_score=3, REGN still passes fundamental via the other 3 dims. With min_score=4, REGN fails.

However, REGN is already blocked by the Trend Template (Task 1). The ROE fix is for correctness and for stocks that might pass trend template but fail on fundamentals.

**Recommendation:** Keep `min_score=3`. The Trend Template fix is the primary gate for REGN. ROE is an additional check that will catch stocks passing trend template but with weak returns. If in the future too many marginal stocks pass, bump min_score to 4.

- [ ] **Step 6: Full pipeline test — verify REGN is rejected and COP is approved**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import Pipeline
p = Pipeline()

r = p.screen('REGN')
print('=== REGN ===')
print(f'Approved: {r.approved}')
print(f'TT passed: {r.trend_template.passed}')
print(f'Fund passed: {r.minervini_fundamental.overall_passed}')
print(f'VCP detected: {r.vcp.vcp_detected}')
assert not r.approved, 'REGN should NOT be approved'
print('✅ REGN correctly rejected')

print()

r = p.screen('COP')
print('=== COP ===')
print(f'Approved: {r.approved}')
print(f'TT passed: {r.trend_template.passed}')
print(f'Fund passed: {r.minervini_fundamental.overall_passed}')
print(f'VCP detected: {r.vcp.vcp_detected}')
# COP may or may not be approved depending on VCP — but TT and Fund should pass
assert r.trend_template.passed, 'COP should pass trend template'
print('✅ COP correctly passes trend template')
"
```

- [ ] **Step 7: Commit**

```bash
git add vcp_momentum_engine.py
git commit -m "fix: add ROE check to Minervini fundamental engine

Adds Return on Equity as a 6th fundamental dimension, computed as
trailing 12-month Net Income / latest Stockholders Equity, with
a 17% threshold per Minervini's criteria.

Also bumps total_dims default from 5 to 6 to reflect the new dimension."
```

---

### Task 4: Run full batch to verify no regressions

- [ ] **Step 1: Run the batch screener on the full watchlist**

```bash
cd /home/oliver/minervi && python3 batch_screen.py 2>&1 | tail -40
```

Verify:
- COP is still approved (if VCP also detects it)
- REGN is NOT approved (rejected on Trend Template)
- No new errors introduced
- The near-misses section shows REGN in TT+FUND pass, VCP fail (or if TT now fails, REGN won't appear there either)

- [ ] **Step 2: Run check_trend.py on REGN to verify clear sell signal**

```bash
cd /home/oliver/minervi && python3 check_trend.py REGN
```

Expected output: Trend Template should fail with the SMA-150 < SMA-200 criteria showing.

- [ ] **Step 3: Commit the final batch verification**

```bash
git add vcp_momentum_engine.py
git commit -m "chore: verify fix — REGN rejected, COP still passes"
```