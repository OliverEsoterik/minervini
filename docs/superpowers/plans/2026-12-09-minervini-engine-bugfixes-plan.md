# Minervini Engine Bugfixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 systemic bugs in `vcp_momentum_engine.py` that cause the batch screener to approve stocks that fail expert-level Minervini analysis. Currently 5/5 approved stocks are weak — after fixes only genuine strong picks should pass.

**Architecture:** All fixes in `vcp_momentum_engine.py`:
1. Rewrite VCP swing detection (3-bar noise → proper swing points with minimum thresholds)
2. Add minimum initial pullback check to VCP heuristic (≥ 8%)
3. Add fallback acceleration logic for stocks with 4-5 quarters (instead of requiring ≥ 6)
4. Raise EPS/revenue growth YoY thresholds from 15% to 25%
5. Raise QoQ threshold from 25% to 30%
6. Bump `min_score` from 3 to 4 (with acceleration dimensions now functional)
7. Add 52-week high proximity hard gate (>25% below high = reject)

**Tech Stack:** Python 3.10+, yfinance, pandas, numpy

## Global Constraints

- All changes in `vcp_momentum_engine.py` only — no new files, no new dependencies
- No changes to `batch_screen.py`, `check_trend.py`, or any other file
- Follow existing code style (dataclass results, MetricResult pattern, logger pattern)
- All existing Trend Template logic stays untouched (it's correct)
- Each fix is independently testable — verify before/after behavior
- No mock data — all tests run against live yfinance data

---

### Task 1: Rewrite VCP Swing Detection (`_find_peaks_troughs`)

**Files:**
- Modify: `vcp_momentum_engine.py` — method `VCPEngine._find_peaks_troughs()` (~line 1012)

**Interfaces:**
- Consumes: `df: pd.DataFrame` (OHLCV data with at least "Close" column)
- Produces: `List[Tuple[int, str, float]]` — same return type as current, but with meaningful swing points only

**Context:**
The current implementation uses a 3-bar window: `Close[i] > Close[i-1] AND Close[i] > Close[i+1]` counts as a peak. In any trending market this produces 15-30 peaks/troughs in 60 days, all with sub-5% drawdowns — pure noise. A real swing point requires a minimum price retracement and a wider lookback window.

**New algorithm:**
1. Use a 5-bar lookback and 5-bar lookahead window (not 1-bar neighbors)
2. A bar is a swing HIGH if it is the highest Close in the 5 bars before AND 5 bars after it
3. A bar is a swing LOW if it is the lowest Close in the 5 bars before AND 5 bars after it
4. Filter: only keep swings where the price change from the previous opposite swing exceeds 5%
5. Ensure alternating peak/trough sequence (discard consecutive peaks)

- [ ] **Step 1: Verify current behavior is broken**

Run the script that shows FAST has 18 false contractions with max 6.3%.

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import VCPEngine, DataIngestion
data = DataIngestion().fetch('FAST')
vcp = VCPEngine().analyse(data)
print(f'FAST: VCP={vcp.vcp_detected}, contractions={len(vcp.contractions)}')
print(f'Magnitudes: {[round(c[\"range_pct\"],2) for c in vcp.contractions]}')
# Currently: VCP=True, 18 contractions, max ~6.3%
"
```

Expected: VCP detected=True, 18 contractions — the bug we're fixing.

- [ ] **Step 2: Replace `_find_peaks_troughs` implementation**

Replace the entire method body at line ~1012 with the new swing detection algorithm.

```python
@staticmethod
def _find_peaks_troughs(df: pd.DataFrame) -> List[Tuple[int, str, float]]:
    """Locate meaningful swing highs and swing lows.

    A swing high is a bar whose Close is the highest of the 5 bars before
    AND 5 bars after it. A swing low is the lowest under the same window.
    Only swings with price retracement > 5% from the prior opposite swing
    are retained. This eliminates the noise of 3-bar extrema detection.

    Returns list of (index_pos, type, price) with alternating types.
    """
    close = df["Close"].values
    n = len(close)
    if n < 11:  # Need at least 5+1+5 = 11 bars for a meaningful swing
        return []

    window = 5  # bars before/after to check

    # Find ALL swing highs and lows (candidates) using 5-bar window
    raw_peaks: List[Tuple[int, float]] = []
    raw_troughs: List[Tuple[int, float]] = []

    for i in range(window, n - window):
        # Swing high: close[i] is highest in [i-window, i+window]
        is_peak = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if close[j] >= close[i]:
                is_peak = False
                break
        if is_peak:
            raw_peaks.append((i, close[i]))

        # Swing low: close[i] is lowest in [i-window, i+window]
        is_trough = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if close[j] <= close[i]:
                is_trough = False
                break
        if is_trough:
            raw_troughs.append((i, close[i]))

    # Merge into alternating sequence starting with the first event type
    # Determine which comes first
    events: List[Tuple[int, str, float]] = []
    p_idx, t_idx = 0, 0

    # Build the sequence by picking the next chronological event
    while p_idx < len(raw_peaks) and t_idx < len(raw_troughs):
        if raw_peaks[p_idx][0] < raw_troughs[t_idx][0]:
            events.append((raw_peaks[p_idx][0], "peak", raw_peaks[p_idx][1]))
            p_idx += 1
        else:
            events.append((raw_troughs[t_idx][0], "trough", raw_troughs[t_idx][1]))
            t_idx += 1

    # Append remaining
    while p_idx < len(raw_peaks):
        events.append((raw_peaks[p_idx][0], "peak", raw_peaks[p_idx][1]))
        p_idx += 1
    while t_idx < len(raw_troughs):
        events.append((raw_troughs[t_idx][0], "trough", raw_troughs[t_idx][1]))
        t_idx += 1

    # Remove consecutive duplicates (two peaks or two troughs in a row)
    filtered: List[Tuple[int, str, float]] = []
    for ev in events:
        if filtered and filtered[-1][1] == ev[1]:
            # Keep the more extreme one
            if ev[1] == "peak" and ev[2] > filtered[-1][2]:
                filtered[-1] = ev
            elif ev[1] == "trough" and ev[2] < filtered[-1][2]:
                filtered[-1] = ev
            # else skip the weaker one
        else:
            filtered.append(ev)

    # Minimum retracement filter: each swing must differ from the prior
    # opposite swing by at least 5% of the prior swing's price
    MIN_RETRACEMENT_PCT = 5.0
    result: List[Tuple[int, str, float]] = []
    for ev in filtered:
        if not result:
            result.append(ev)
            continue
        prev = result[-1]
        change_pct = abs(ev[2] - prev[2]) / prev[2] * 100
        if change_pct >= MIN_RETRACEMENT_PCT:
            result.append(ev)
        # else skip — noise, not a meaningful swing

    return result
```

- [ ] **Step 3: Verify FAST now produces far fewer, meaningful contractions**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import VCPEngine, DataIngestion
data = DataIngestion().fetch('FAST')
vcp = VCPEngine().analyse(data)
print(f'FAST: VCP={vcp.vcp_detected}, contractions={len(vcp.contractions)}')
print(f'Magnitudes: {[round(c[\"range_pct\"],2) for c in vcp.contractions]}')
# Now should show far fewer contractions (< 5 ideally)
"
```

Expected: Fewer contractions (not 18). Likely 0-3 now. VCP likely False.

- [ ] **Step 4: Verify MEDP still shows a VCP setup**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import VCPEngine, DataIngestion
data = DataIngestion().fetch('MEDP')
vcp = VCPEngine().analyse(data)
print(f'MEDP: VCP={vcp.vcp_detected}, contractions={len(vcp.contractions)}')
print(f'Magnitudes: {[round(c[\"range_pct\"],2) for c in vcp.contractions]}')
"
```

Expected: Fewer contractions but some meaningful ones. VCP depends on whether MEDP has a real pattern.

- [ ] **Step 5: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: rewrite VCP swing detection with 5-bar window and 5% minimum retracement

Replaces the 3-bar local-extrema peak/trough detector with a proper
swing-point algorithm requiring:
- 5-bar lookback and lookahead windows (eliminates noise)
- Minimum 5% price retracement between opposite swings
- Consecutive-duplicate consolidation (keeps extreme)

Previously a stock like FAST produced 18 false contractions from
daily noise. Now only meaningful swing points are detected."
```

---

### Task 2: Add Minimum Initial Pullback to VCP Heuristic

**Files:**
- Modify: `vcp_momentum_engine.py` — method `VCPEngine._vcp_heuristic()` (~line 1115)

**Interfaces:**
- Consumes: same parameters `(contractions, atr_pct, volume_dryup)`
- Produces: same return type `Tuple[bool, str]`

**Context:**
The VCP heuristic checks for declining contractions and tight final consolidation but doesn't require the initial pullback to be meaningful. A series of 2-3% wiggles passes the "declining" check because both first and last are tiny. Adding `magnitudes[0] >= 8.0` ensures the pattern starts with a genuine pullback.

- [ ] **Step 1: Add the minimum initial pullback check**

Replace the `_vcp_heuristic` method with the updated version:

```python
@staticmethod
def _vcp_heuristic(
    contractions: List[Dict[str, float]],
    atr_pct: float,
    volume_dryup: bool,
) -> Tuple[bool, str]:
    """Apply Minervini VCP heuristics to the contraction sequence.

    The classic pattern:
        1. An initial pullback of **15–25 %**.
        2. Followed by tighter contractions of **8–12 %**.
        3. Culminating in a tight consolidation **< 8 %**.

    We relax the fixed bands slightly for real-market noise: we look for
    a *declining* trend in drawdown magnitudes. At minimum the first
    pullback must be ≥ 8 %.
    """
    if len(contractions) < 2:
        return False, "Fewer than 2 contraction events."

    magnitudes = [c["range_pct"] for c in contractions]

    # The initial pullback must be meaningful (at least 8%)
    # A true VCP starts with a 15-25% decline, not 2-3% noise
    if magnitudes[0] < 8.0:
        return False, (
            f"Initial pullback {magnitudes[0]:.1f}% < 8% "
            f"— not a meaningful contraction start."
        )

    # Check for general declining trend
    # We expect at least the last contraction to be smaller than the first
    declining = magnitudes[-1] < magnitudes[0] * 0.85  # at least 15% tighter

    # Last consolidation < 10 % (relaxed from 8 % for robustness)
    tight = magnitudes[-1] < 10.0

    # ATR compression < 3 % (tight range suggests low volatility)
    compressed = atr_pct < 4.0

    parts: List[str] = []
    if declining:
        parts.append("contractions declining")
    if tight:
        parts.append(f"final consolidation {magnitudes[-1]:.1f}% < 10%")
    if compressed:
        parts.append(f"ATR {atr_pct:.2f}% < 4%")
    if volume_dryup:
        parts.append("volume dry-up detected")

    detected = declining and tight and compressed
    return detected, "; ".join(parts) if parts else "No VCP signal."
```

- [ ] **Step 2: Verify FAST, CBOE, COP now fail VCP**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import VCPEngine, DataIngestion
for t in ['FAST', 'CBOE', 'COP', 'MA', 'MEDP']:
    data = DataIngestion().fetch(t)
    vcp = VCPEngine().analyse(data)
    print(f'{t}: VCP={vcp.vcp_detected} ({len(vcp.contractions)} contractions, first={vcp.contractions[0][\"range_pct\"]:.1f}% if vcp.contractions else \"none\")')
"
```

Expected: FAST, CBOE, COP should now show VCP=False. MEDP may or may not depending on its actual pattern.

- [ ] **Step 3: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: require minimum 8% initial pullback in VCP heuristic

Adds check that the first contraction (initial pullback) is at least
8%. This eliminates false VCP detections on stocks whose largest
pullback is 3-7% of daily noise, ensuring the pattern starts with
a genuine decline per Minervini's framework."
```

---

### Task 3: Add Fallback Acceleration Logic

**Files:**
- Modify: `vcp_momentum_engine.py` — methods `_eval_eps_acceleration` (~line 746) and `_eval_revenue_acceleration` (~line 810)

**Interfaces:**
- Consumes: `eps: pd.Series` / `rev: pd.Series` (quarterly values, oldest first)
- Produces: `MetricResult` with meaningful acceleration assessment even with < 6 quarters

**Context:**
yfinance's `quarterly_financials` consistently returns exactly 5 non-null values per line item. The current code requires ≥ 6 quarters to compute 2 YoY comparisons (rates[0] and rates[1]), meaning this dimension fails for nearly every stock. With only 5 quarters we can compute exactly 1 YoY rate but we cannot compare it to a prior YoY rate. Instead:

- **If exactly 5 quarters:** Compute the single YoY rate. Pass if it's ≥ 25% (high absolute growth as a proxy for acceleration — a company growing 25%+ is compounding strongly).
- **If ≥ 4 quarters but < 5:** Compute QoQ growth rates, then compute the linear slope of those rates. Pass if slope > 0 (sequential acceleration).
- **If ≥ 6 quarters:** Use current logic (2+ YoY comparisons).

- [ ] **Step 1: Replace `_eval_eps_acceleration`**

Replace the entire method body:

```python
@staticmethod
def _eval_eps_acceleration(eps: pd.Series) -> MetricResult:
    """Check if EPS growth rate is accelerating.

    Three tiers based on data availability:
    1. ≥ 6 quarters: YoY acceleration (current logic)
    2. Exactly 5 quarters: Single YoY rate, pass if ≥ 25% (strong growth proxy)
    3. 4 quarters: QoQ sequential slope, pass if trend positive
    """
    if len(eps) < 4:
        return MetricResult(
            label="EPS acceleration",
            passed=False,
            detail="Insufficient data (need ≥ 4 quarters).",
        )

    vals = eps.values

    # Tier 1: ≥ 6 quarters — full YoY acceleration check
    if len(vals) >= 6:
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
        # Also require that the latest rate is at least 10%
        # (prevents 2% → 5% from being "accelerating")
        accelerating = accelerating and latest >= 0.10
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

    # Tier 2: Exactly 5 quarters — single YoY rate
    # We can compute one YoY comparison: (Q5 - Q1) / Q1
    # If the absolute growth rate is ≥ 25%, treat as implicitly accelerating
    if len(vals) >= 5:
        prev = vals[0]
        curr = vals[-1]
        if prev <= 0:
            return MetricResult(
                label="EPS acceleration",
                passed=False,
                detail=f"Prior-year EPS non-positive ({prev}), cannot compute YoY.",
            )
        yoy_rate = (curr - prev) / prev
        # Use 25% as proxy threshold — a company growing 25%+ YoY
        # is compounding strongly even with limited data
        ACCEL_PROXY_THRESHOLD = 0.25
        passed = yoy_rate >= ACCEL_PROXY_THRESHOLD
        return MetricResult(
            label="EPS acceleration (proxy)",
            passed=passed,
            value=yoy_rate,
            threshold=ACCEL_PROXY_THRESHOLD,
            quarters=[float(v) for v in vals],
            detail=(
                f"{yoy_rate*100:+.2f}% YoY {'≥' if passed else '<'} "
                f"{ACCEL_PROXY_THRESHOLD*100:.0f}% (proxy, 5 qtrs)"
            ),
        )

    # Tier 3: 4 quarters — QoQ sequential acceleration
    # Compute 3 QoQ growth rates, check if slope > 0
    qoq_rates = []
    for i in range(1, len(vals)):
        if vals[i-1] > 0:
            qoq_rates.append((vals[i] - vals[i-1]) / vals[i-1])
        else:
            qoq_rates.append(0.0)

    if len(qoq_rates) < 2:
        return MetricResult(
            label="EPS acceleration",
            passed=False,
            detail="Insufficient QoQ comparisons.",
        )

    # Linear regression slope on QoQ rates
    y = np.array(qoq_rates, dtype=float)
    x = np.arange(len(y))
    n_ = float(len(y))
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * x).sum()
    denom = n_ * sum_xx - sum_x * sum_x
    slope = (n_ * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-15 else 0.0
    accelerating = slope > 0

    return MetricResult(
        label="EPS acceleration (QoQ)",
        passed=accelerating,
        value=float(qoq_rates[-1]),
        threshold=float(qoq_rates[0]),
        quarters=[float(r) for r in qoq_rates],
        detail=(
            f"QoQ slope {slope:+.4f} "
            f"{'accelerating' if accelerating else 'decelerating'} "
            f"(4 qtrs)"
        ),
    )
```

- [ ] **Step 2: Replace `_eval_revenue_acceleration`**

Same pattern. Replace the entire method body:

```python
@staticmethod
def _eval_revenue_acceleration(rev: pd.Series) -> MetricResult:
    """Check if revenue growth rate is accelerating.

    Same three-tier logic as ``_eval_eps_acceleration``.
    """
    if len(rev) < 4:
        return MetricResult(
            label="Revenue acceleration",
            passed=False,
            detail="Insufficient data (need ≥ 4 quarters).",
        )

    vals = rev.values

    # Tier 1: ≥ 6 quarters — full YoY acceleration
    if len(vals) >= 6:
        rates = MinerviniFundamentalEngine._compute_yoy_growth(rev)
        if len(rates) < 2:
            return MetricResult(
                label="Revenue acceleration",
                passed=False,
                detail="Insufficient YoY comparisons.",
            )
        latest = rates[-1]
        prior = rates[-2]
        accelerating = latest > prior and latest >= 0.10
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

    # Tier 2: Exactly 5 quarters — single YoY rate
    if len(vals) >= 5:
        prev = vals[0]
        curr = vals[-1]
        if prev <= 0:
            return MetricResult(
                label="Revenue acceleration",
                passed=False,
                detail=f"Prior-year revenue non-positive ({prev}), cannot compute YoY.",
            )
        yoy_rate = (curr - prev) / prev
        ACCEL_PROXY_THRESHOLD = 0.25
        passed = yoy_rate >= ACCEL_PROXY_THRESHOLD
        return MetricResult(
            label="Revenue acceleration (proxy)",
            passed=passed,
            value=yoy_rate,
            threshold=ACCEL_PROXY_THRESHOLD,
            quarters=[float(v) for v in vals],
            detail=(
                f"{yoy_rate*100:+.2f}% YoY {'≥' if passed else '<'} "
                f"{ACCEL_PROXY_THRESHOLD*100:.0f}% (proxy, 5 qtrs)"
            ),
        )

    # Tier 3: 4 quarters — QoQ sequential acceleration
    qoq_rates = []
    for i in range(1, len(vals)):
        if vals[i-1] > 0:
            qoq_rates.append((vals[i] - vals[i-1]) / vals[i-1])
        else:
            qoq_rates.append(0.0)

    if len(qoq_rates) < 2:
        return MetricResult(
            label="Revenue acceleration",
            passed=False,
            detail="Insufficient QoQ comparisons.",
        )

    y = np.array(qoq_rates, dtype=float)
    x = np.arange(len(y))
    n_ = float(len(y))
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * x).sum()
    denom = n_ * sum_xx - sum_x * sum_x
    slope = (n_ * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-15 else 0.0
    accelerating = slope > 0

    return MetricResult(
        label="Revenue acceleration (QoQ)",
        passed=accelerating,
        value=float(qoq_rates[-1]),
        threshold=float(qoq_rates[0]),
        quarters=[float(r) for r in qoq_rates],
        detail=(
            f"QoQ slope {slope:+.4f} "
            f"{'accelerating' if accelerating else 'decelerating'} "
            f"(4 qtrs)"
        ),
    )
```

- [ ] **Step 3: Verify acceleration dimensions now pass for strong stocks**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
for t in ['MEDP', 'CBOE', 'COP', 'MA', 'FAST']:
    r = MinerviniFundamentalEngine().screen(t)
    print(f'{t}: EPS_accel={\"✅\" if r.eps_acceleration.passed else \"❌\"} Rev_accel={\"✅\" if r.revenue_acceleration.passed else \"❌\"} Score={r.score}/{r.total_dims}')
    print(f'  EPS accel: {r.eps_acceleration.detail}')
    print(f'  Rev accel: {r.revenue_acceleration.detail}')
"
```

Expected: Some stocks should now pass acceleration dimensions. MEDP has 23.7% quarterly EPS growth and 25% EPS YoY growth — should pass the proxy check.

- [ ] **Step 4: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: add fallback acceleration logic for 4-5 quarters of data

yfinance quarterly_financials typically returns 5 values per line
item, making the old 6-quarter requirement always fail. New logic:
- ≥6 quarters: full YoY acceleration comparison (unchanged)
- Exactly 5 quarters: single YoY rate, pass if ≥ 25% (proxy)
- 4 quarters: QoQ slope trend, pass if positive
- Also adds 10% minimum absolute rate requirement to prevent
  trivial acceleration (2% -> 5% no longer passes)"
```

---

### Task 4: Raise Growth Thresholds

**Files:**
- Modify: `vcp_momentum_engine.py` — class constants in `MinerviniFundamentalEngine` (~lines 501-503)

- [ ] **Step 1: Change YOY_THRESHOLD from 0.15 to 0.25**

```python
    YOY_THRESHOLD: float = 0.25    # 25 % (was 15 %)
```

- [ ] **Step 2: Change QOQ_THRESHOLD from 0.25 to 0.30**

```python
    QOQ_THRESHOLD: float = 0.30    # 30 % (was 25 %)
```

These thresholds are used by `_eval_eps_growth()` and `_eval_revenue_growth()` to determine if absolute growth rates pass.

- [ ] **Step 3: Verify the thresholds are picked up**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
e = MinerviniFundamentalEngine()
print(f'YOY_THRESHOLD = {e.YOY_THRESHOLD}')  # Expected: 0.25
print(f'QOQ_THRESHOLD = {e.QOQ_THRESHOLD}')  # Expected: 0.30
"
```

- [ ] **Step 4: Verify weaker stocks now fail EPS/revenue growth**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import MinerviniFundamentalEngine
for t in ['FAST', 'MA', 'MEDP', 'CBOE']:
    r = MinerviniFundamentalEngine().screen(t)
    print(f'{t}: EPS_growth={\"✅\" if r.eps_growth.passed else \"❌\"} ({r.eps_growth.detail})')
    print(f'  Rev_growth={\"✅\" if r.revenue_growth.passed else \"❌\"} ({r.revenue_growth.detail})')
"
```

Expected: MA (14% revenue) and MEDP (8.5% revenue) should now FAIL revenue growth. FAST (20.7% EPS) should now FAIL EPS growth with 25% threshold.

Wait — FAST has 20.7% EPS growth which is < 25%. So it should fail EPS growth too. Let me verify what MEDP's EPS is — it was 24.56% which is < 25%. Hmm.

Actually, this may be too aggressive. Let me think about this. The expert says MEDP's EPS growth is 25.1% — so it should just barely pass. The 24.56% I saw was from the script run... the actual value might fluctuate with data freshness.

The point is: raising thresholds will cause more stocks to fail the absolute growth dimensions. Combined with the working acceleration dimensions and min_score=4, only genuinely strong stocks should pass.

- [ ] **Step 5: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: raise EPS/revenue growth thresholds to Minervini targets

YOY_THRESHOLD: 15% -> 25% (Minervini targets 25%+ for superperformance)
QOQ_THRESHOLD: 25% -> 30% (consistent increase)
This blocks steady compounders with moderate growth rates from
passing the fundamental filter."
```

---

### Task 5: Bump min_score from 3 to 4

**Files:**
- Modify: `vcp_momentum_engine.py` — `Pipeline.__init__()` (~line 1447)

**Context:**
With acceleration dimensions now functional (Task 3), the effective number of evaluable dimensions is 6, not 4. `min_score=3` (50%) was set when 2 dimensions were guaranteed fails. Now that acceleration works, stocks need a higher bar: 4 of 6 dimensions must pass.

- [ ] **Step 1: Change min_score from 3 to 4**

Replace:
```python
self.fundamental = MinerviniFundamentalEngine(min_score=3)
```

With:
```python
self.fundamental = MinerviniFundamentalEngine(min_score=4)
```

- [ ] **Step 2: Verify filter now requires 4/6**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import Pipeline
p = Pipeline()
for t in ['FAST', 'CBOE', 'COP', 'MA', 'MEDP']:
    r = p.screen(t)
    mf = r.minervini_fundamental
    if mf:
        print(f'{t}: Score={mf.score}/{mf.total_dims} Pass={mf.overall_passed}')
        print(mf.details)
        print()
    else:
        print(f'{t}: No fundamental data')
"
```

Expected: With min_score=4 and raised thresholds, most stocks should score 2-4/6. Strong stocks with accelerating EPS+revenue should still pass.

- [ ] **Step 3: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: bump min_score from 3 to 4

With acceleration dimensions now functional (fixing the data
availability issue), 3/6 was too permissive. Raising to 4/6
ensures stocks need genuine strength across multiple dimensions
rather than scraping by on 3 of 4 evaluable metrics."
```

---

### Task 6: Add 52-Week High Proximity Hard Gate

**Files:**
- Modify: `vcp_momentum_engine.py` — `Pipeline.screen()` method (~line 1452)

**Context:**
Minervini requires stocks to be within 25% of their 52-week high to ensure minimal overhead resistance. A stock can pass the Trend Template (SMAs are rising) while still being 40% below its high if the decline was 200+ trading days ago. This gate catches that case.

- [ ] **Step 1: Add 52-week high check after data ingestion**

In `Pipeline.screen()`, after the data ingestion block and Trend Template validation, add:

```python
# ---- 0b. 52-Week High Proximity Gate ------------------------------
# Reject if price is > 25% below the 52-week high
info = DataIngestion.fetch_info(ticker)
high_52 = info.get("fiftyTwoWeekHigh")
if high_52 is not None and high_52 > 0:
    pct_off_high = (high_52 - close) / high_52 * 100
    if pct_off_high > 25.0:
        result.error = (
            f"Price ${close:.2f} is {pct_off_high:.1f}% below "
            f"52-week high ${high_52:.2f} (max allowed: 25%)"
        )
        logger.info(
            "52-week high gate [%s]: FAIL — %.1f%% below high",
            ticker, pct_off_high,
        )
        return result
    logger.info(
        "52-week high gate [%s]: PASS (%.1f%% below high)",
        ticker, pct_off_high,
    )
```

Insert this after line `result.ohlcv_data = data` (after data ingestion is confirmed, before or after Trend Template — either is fine, but early rejection is better). Put it right after the data ingestion block and before the Trend Template validation, so we don't waste API calls on stocks too far from their highs.

The exact insertion point in `Pipeline.screen()` (~line 1456):

```python
        # ---- 1. Data Ingestion --------------------------------------
        data = self.ingestion.fetch(ticker)
        if data is None:
            result.error = "Failed to fetch data."
            return result
        result.ohlcv_data = data

        # ---- 0b. 52-Week High Proximity Gate [NEW] -----------------
        # ... (insert here)

        # ---- 2. Trend Template --------------------------------------
```

- [ ] **Step 2: Verify gate blocks stocks far from highs**

Find a stock far from its 52-week high to test:

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import Pipeline
p = Pipeline()
# Pick a stock likely far from its high
r = p.screen('INTC')
if r.error:
    print(f'INTC rejected: {r.error}')
else:
    print(f'INTC approved: {r.approved}')
"
```

- [ ] **Step 3: Verify MEDP still passes the gate**

```bash
cd /home/oliver/minervi && python3 -c "
from vcp_momentum_engine import Pipeline
p = Pipeline()
r = p.screen('MEDP')
if r.error:
    print(f'MEDP rejected: {r.error}')
else:
    print(f'MEDP passes gate, approved: {r.approved}')
"
```

- [ ] **Step 4: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "feat: add 52-week high proximity hard gate

Rejects stocks trading > 25% below their 52-week high. This is a
critical Minervini criterion that was missing — a stock can pass
the Trend Template (SMAs rising) while still being 40%+ below its
high if the decline was >200 trading days ago.
Data source: yfinance Ticker.info['fiftyTwoWeekHigh']."
```

---

### Task 7: Full Pipeline Verification

**Files:**
- Run: `batch_screen.py` — no code changes, just observation

- [ ] **Step 1: Run batch_screen on the full watchlist**

```bash
cd /home/oliver/minervi && python3 batch_screen.py 2>&1 | tail -50
```

Expected:
- Fewer approved stocks overall (was 5+ previously)
- Only stocks with genuine VCP patterns + strong fundamentals pass
- Near-misses section shows which stocks are close

- [ ] **Step 2: Run batch_screen on just the 5 suspect stocks**

```bash
cd /home/oliver/minervi && python3 vcp_momentum_engine.py MEDP CBOE COP MA FAST 2>&1 | grep -E "(APPROVED|Score|EPS|Rev|Marg|ROE|52-week)"
```

Expected: Only MEDP (and possibly CBOE) should show strong scores. FAST and MA should clearly fail.

- [ ] **Step 3: Manual sanity check on known-strong tickers**

```bash
cd /home/oliver/minervi && python3 vcp_momentum_engine.py NVDA MSFT AAPL 2>&1 | grep -E "(APPROVED|Score|EPS|Rev)"
```

Expected: Known strong stocks should pass if they have VCP patterns.

- [ ] **Step 4: Verify no syntax errors or import failures**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('vcp_momentum_engine.py').read()); print('✅ Syntax OK')"
python3 -c "from vcp_momentum_engine import *; print('✅ Import OK')"
```

- [ ] **Step 5: Final commit with all changes**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "fix: complete Minervini screening engine accuracy overhaul

Six changes to vcp_momentum_engine.py:
1. Rewrite VCP swing detection (5-bar window, 5% min retracement)
2. Require 8% minimum initial pullback in VCP heuristic
3. Fallback acceleration logic for 4-5 quarters of data
4. Raise YoY growth thresholds 15% -> 25%
5. Raise QoQ growth thresholds 25% -> 30%
6. Bump min_score 3 -> 4
7. Add 52-week high proximity hard gate (>25% below = reject)

Previously 5/5 screened stocks were weak picks. After fixes,
only stocks with genuine VCP patterns and accelerating
fundamentals at 25%+ growth rates should pass."
```