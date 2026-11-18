# Minervini Relative Strength (RS) Rating Module

**Date:** 2026-11-18
**Status:** Design spec — approved for implementation

---

## Motivation

Mark Minervini's SEPA framework requires a Relative Strength (RS) Rating of **≥ 70** (preferably ≥ 80) as the 8th criterion of his Trend Template. The current `vcp_momentum_engine.py` has **zero RS implementation** — every other Trend Template criterion is checked except RS.

Without RS ranking, the system cannot distinguish market leaders from mere uptrenders. A stock can pass all SMA/slope checks while being a market laggard — ranking in the bottom 50th percentile versus peers. Minervini's research (and O'Neil's before him) shows that nearly every massive winner ranked in the top 10-30% of RS before its advance.

## What Is RS Rating (Minervini Method)

The RS Rating is **NOT** the RSI oscillator. It is a **cross-sectional percentile rank** comparing each stock's price performance against all peers in the universe.

**Calculation (weighted multi-period return):**

```
Weighted Return = P6(126d) × 0.50 + P3(63d) × 0.30 + P1(21d) × 0.20
```

Where:
- P6 = (current close - close 126 trading days ago) / close 126 trading days ago
- P3 = (current close - close 63 trading days ago) / close 63 trading days ago
- P1 = (current close - close 21 trading days ago) / close 21 trading days ago

**Ranking:** All weighted returns are sorted descending. Each ticker gets a percentile:
```
RS Rank = (number_of_stocks - position + 1) / number_of_stocks × 99
```
Top performer = 99, median ≈ 50, worst ≈ 0.

**Gate:** RS ≥ 70 (hard reject). RS ≥ 80 (preferred — informational).

## Universe

Ranking is computed **within the watchlist** (~500 large-cap US stocks). The watchlist already represents the largest stocks by market cap, so within-universe ranking is meaningful. A pre-screened list of market-cap leaders means low-ranked stocks here are genuinely lagging the large-cap space.

## Architecture

### New File: `rs_rating.py`

A standalone module with one public class `RSModule`. No dependencies on `vcp_momentum_engine.py` or `batch_screen.py` — it can be tested and used independently.

```
rs_rating.py
├── RSModule
│   ├── __init__(cache_dir: str = "rs_cache")
│   ├── fetch_returns(tickers: List[str]) -> Dict[str, Optional[float]]
│   │   • Batch yfinance download for 6 months (~126 trading days) of daily closes
│   │   • Compute weighted score per ticker
│   │   • Returns None for IPOs / failed fetches
│   │
│   ├── compute_percentile_ranks(returns: Dict[str, Optional[float]]) -> Dict[str, float]
│   │   • Rank non-None scores → percentile 0-99
│   │   • Top performer = 99, worst of ranked = 0
│   │
│   ├── get_rs_ratings(tickers: List[str], force_refresh: bool = False) -> Dict[str, float]
│   │   • Orchestrator: check cache → fetch → compute → save cache → return
│   │
│   └── _cache_path, _load_cache(), _save_cache()
│       • JSON cache with timestamp — 1-day TTL
```

### Data Structures

```python
@dataclass
class RSResult:
    """Result for one ticker's RS rating."""
    ticker: str
    rs_rank: float           # 0-99, None if unavailable
    weighted_return: Optional[float]  # raw weighted score before ranking
    detail: str               # human-readable

@dataclass 
class RSCache:
    """Persistent cache of RS ratings."""
    fetched_at: str           # ISO timestamp
    ratings: Dict[str, float] # {ticker: rs_rank}
    scores: Dict[str, float]  # {ticker: weighted_return} (for debugging)
```

### Data Flow

```
batch_screen.py
    │
    ├── RSModule.get_rs_ratings(watchlist)  ← PRE-PHASE
    │   └── returns {ticker: rs_rank}
    │
    ├── for each ticker in parallel:
    │   └── Pipeline.screen(ticker, rs_rank=rs_rank)
    │       └── RS gate: reject if rs_rank < 70
    │
    └── output with RS rank in results
```

### Caching

Caching avoids re-downloading ~500 tickers on every run. The cache is a single JSON file:

```json
{
  "fetched_at": "2026-11-18T16:30:00",
  "ratings": {"AAPL": 85.3, "MSFT": 92.1, ...},
  "scores": {"AAPL": 0.321, "MSFT": 0.445, ...}
}
```

- **TTL:** 1 trading day (cache is valid if < 24 hours old)
- **Format:** JSON — human-readable, diff-friendly in git
- **Location:** `rs_cache/rs_ratings.json` (configurable via `cache_dir`)
- **Force refresh:** `get_rs_ratings(force_refresh=True)` bypasses cache

## Changes to Existing Files

### `vcp_momentum_engine.py` — Pipeline.screen()

Add an `rs_rank` parameter to `Pipeline.screen()`:

```python
def screen(self, ticker, entry_price=None, stop_loss_pct=0.04, rs_rank=None):
```

Add an RS gate after the 52-week high proximity gate:

```python
# ---- 0c. RS Rating Gate ----------------------------------------
if rs_rank is not None and rs_rank < 70:
    result.error = (
        f"RS Rating {rs_rank:.0f} < 70 (Minervini minimum)"
    )
    return result
```

The `rs_rank=None` default preserves backward compatibility for standalone CLI use.

### `ScreeningResult` — New Field

Add an `rs_rank: Optional[float] = None` field to `ScreeningResult`. Populate it in `Pipeline.screen()` before returning.

Update `ScreeningResult.to_dict()` to include RS rank.

### `batch_screen.py` — Pre-Phase + RS Output

1. **Pre-phase** before the ThreadPoolExecutor:

```python
from rs_rating import RSModule

print("  Pre-computing RS ratings for watchlist...")
rs_module = RSModule()
rs_ranks = rs_module.get_rs_ratings(WATCHLIST)
valid_ranks = {t: r for t, r in rs_ranks.items() if r is not None}
print(f"  RS computed: {len(valid_ranks)}/{len(WATCHLIST)} tickers")
```

2. **Modified screen_one** to accept and pass rs_rank:

```python
def screen_one(ticker, pipeline, rs_ranks):
    rs_rank = rs_ranks.get(ticker)
    return pipeline.screen(ticker, rs_rank=rs_rank)
```

3. **Output enhancement**: Show RS rank in approved setups and near-misses.

### `check_trend.py` — No changes (standalone HOLD/SELL, no RS needed)

### `README.md` — Documentation update

Document the new RS module, gate threshold, and how it affects screening.

## Backward Compatibility

| Usage Scenario | RS Available? | Behavior |
|---------------|---------------|----------|
| `batch_screen.py` | ✅ Yes | Full RS gate applied |
| `vcp_momentum_engine.py AAPL MSFT` | ❌ No (rs_rank=None) | RS gate skipped |
| `vcp_momentum_engine.py` (embedded watchlist) | ❌ No | RS gate skipped |
| Direct Pipeline.screen() calls from tests | ❌ No | RS gate skipped |

CLI usage doesn't get RS ranking because there's no pre-computation phase. This is acceptable — the main screening workflow is through `batch_screen.py`. If CLI RS is desired later, the RSModule can be added there too.

## Edge Cases

| Case | Handling |
|------|----------|
| IPO < 6 months old (< 126 trading days) | Return None — RS gate skipped (not rejected) |
| Ticker delisted / no data | Return None — RS gate skipped |
| All tickers return None | RSModule logs warning, returns empty dict |
| Fewer than 20 valid tickers | RS ranking is unreliable — RSModule logs warning, ranking falls back to no gate |
| yfinance batch download timeout | RSModule logs error and retries once |
| Cache file corrupted (invalid JSON) | RSModule treats as cache miss, re-fetches |
| IPO < 126 trading days, but ≥ 63 | Compute available windows (P3, P1) with proportional weight redistribution: P3×0.60 + P1×0.40. Skip P6. |
| IPO < 63 trading days, but ≥ 21 | Compute P1 only. Weight = 1.0. Single-window RS is less reliable but still directional. |
| IPO < 21 trading days | Return None — insufficient price history for any window. RS gate skipped. |

## Testing Strategy

| Test | What It Verifies |
|------|-----------------|
| `test_weighted_return()` | Correct multi-period weighting |
| `test_percentile_rank()` | Top stock gets 99, bottom gets 0, median ≈ 50 |
| `test_identical_returns()` | Ties get same percentile rank |
| `test_single_ticker()` | Single entry → RS = 99 |
| `test_all_none()` | No valid data → empty dict |
| `test_rs_gate_below_70()` | Pipeline rejects RS < 70 |
| `test_rs_gate_above_70()` | Pipeline passes RS ≥ 70 |
| `test_rs_gate_none()` | Pipeline passes when rs_rank is None |
| `test_cache_ttl()` | Fresh cache used; expired cache triggers re-fetch |
| `test_cache_corrupted()` | Invalid JSON treated as cache miss |

## Files Changed

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `rs_rating.py` | **New file** | ~150 lines |
| `vcp_momentum_engine.py` | Modify | ~20 lines (Pipeline.screen + ScreeningResult) |
| `batch_screen.py` | Modify | ~25 lines (pre-phase + screen_one) |
| `README.md` | Modify | ~15 lines (document RS) |

## Implementation Order

1. Create `rs_rating.py` with RSModule class
2. Add rs_rank parameter and RS gate to Pipeline.screen()
3. Add rs_rank field to ScreeningResult + to_dict()
4. Add pre-phase + modified screen_one to batch_screen.py
5. Update README with RS documentation
6. Test: batch_screen.py on full watchlist, verify RS filtering