# Minervini Screening Engine — Bugfix Design

**Date:** 2026-12-09
**Status:** Approved for implementation

## Problem

The `batch_screen.py` pipeline in `vcp_momentum_engine.py` approves stocks that don't pass expert-level Minervini scrutiny. Diagnostics showed:

1. VCP engine detects false patterns in any low-volatility uptrend (3-bar extrema → noise)
2. Fundamental acceleration dimensions require 6 quarters of data — yfinance provides 5 → always fail
3. Growth thresholds (15% YoY) are below Minervini's targets (25%+)
4. `min_score=3` of 6 dimensions (50%) drifted from original 3/5 (60%) when ROE was added
5. No 52-week high proximity check — a stock could pass Trend Template while 40% below its high

## Proposed Changes

All changes in `vcp_momentum_engine.py`. No new files, no new dependencies, no API changes.

### Change 1 — Rewrite VCP Swing Detection

**File:** `VCPEngine._find_peaks_troughs()` (~line 900)

Replace the 3-bar local extrema with a proper swing-point detector:
- Minimum 5-bar lookback and lookahead windows (not 1-bar)
- Price retracement must exceed a minimum threshold (5%) to register as a peak/trough
- This eliminates the noise that produced 15-30 false contractions per stock

### Change 2 — Minimum Initial Pullback in VCP Heuristic

**File:** `VCPEngine._vcp_heuristic()` (~line 960)

Add `magnitudes[0] >= 8.0` check. A true VCP starts with a meaningful pullback (15-25%), not 2-3% noise.

### Change 3 — Fallback Acceleration Logic

**File:** `MinerviniFundamentalEngine._eval_eps_acceleration()` and `_eval_revenue_acceleration()`

When < 6 quarters of data available:
- **If exactly 5 quarters:** Compute single YoY rate. Pass if ≥ 25% (high absolute rate as proxy for acceleration — a company growing 25%+ YoY is by definition accelerating from prior levels).
- **If ≥ 4 quarters:** Compute QoQ growth rates and check if the linear slope of rates is positive (sequential acceleration).
- If < 4 quarters: fail with "insufficient data" (current behavior).

### Change 4 — Raise Growth Thresholds

**File:** `MinerviniFundamentalEngine` class constants

- `YOY_THRESHOLD`: 0.15 → 0.25
- `QOQ_THRESHOLD`: 0.25 → 0.30
- `MIN_MARGIN_IMPROVEMENT_PP`: stays 0.01

### Change 5 — Bump min_score

**File:** `Pipeline.__init__()`

- `MinerviniFundamentalEngine(min_score=3)` → `min_score=4`
- `total_dims` stays 6 (acceleration dimensions now functional)

### Change 6 — 52-Week High Proximity Gate

**File:** `Pipeline.screen()` — new check after `DataIngestion.fetch()`

```python
# Fetch 52-week high from yfinance info
info = DataIngestion.fetch_info(ticker)
high_52 = info.get("fiftyTwoWeekHigh")
if high_52 and close < 0.75 * high_52:
    result.error = f"Price ${close:.2f} >25% below 52-week high ${high_52:.2f}"
    return result
```

This is a hard gate: if the stock is >25% below its 52-week high, it's rejected immediately with a clear error, regardless of other scores.

### What's NOT Changing

- Earnings surprise dimension: yfinance quarterly earnings data is deprecated/None for most tickers. Not a reliable source. Skip.
- No new files or modules.
- No changes to `batch_screen.py` or `check_trend.py`.
- No changes to the CAN SLIM or Position Sizing modules.
- No changes to the Trend Template validator (it's already correct with 150-day SMA).

## Verification Plan

After all fixes:

| Stock | Expert Rating | Should Pass? | Key Reason |
|-------|--------------|-------------|------------|
| MEDP | ★★★★★ | ✅ | Strong VCP + 25% EPS growth + accelerating |
| CBOE | ★★★★½ | ❌ (for now) | No VCP, needs tightening |
| COP | ★★★★ | ❌ | Cyclical recovery, not sustained growth |
| MA | ★★★½ | ❌ | Revenue < 25% threshold, decelerating |
| FAST | ★★★ | ❌ | No VCP, no acceleration, no catalyst |

Expected: only MEDP (and possibly CBOE, COP depending on VCP) pass the full pipeline.

## Implementation Order

1. Fix VCP swing detection (Change 1)
2. Add minimum initial pullback (Change 2)
3. Add fallback acceleration (Change 3)
4. Raise thresholds (Change 4)
5. Bump min_score (Change 5)
6. Add 52-week high gate (Change 6)

Each fix is independently testable. Order chosen so each step improves accuracy incrementally.