# Minervini 5-Dimension Fundamental Engine

**Date:** 2026-11-15
**Branch:** `feature/minervini-fundamental-engine`
**Status:** Design spec — approved for implementation

---

## Motivation

The current `FundamentalFilter` checks a single metric (revenue YoY growth ≥ 20%). Mark Minervini's actual fundamental analysis is broader: he evaluates EPS growth, EPS acceleration, revenue growth, revenue acceleration, and profit margin trends across multiple quarters with tiered thresholds.

## Data Source

All data comes from a single `yfinance.Ticker(ticker).quarterly_financials` call — a DataFrame with fiscal-quarter columns and financial-statement line items as rows.

### Extracted Line Items

| yfinance Row | Purpose |
|-------------|---------|
| `Total Revenue` | Revenue growth + acceleration |
| `Net Income` | Net profit margin |
| `Basic EPS` (fallback: `Diluted EPS`) | EPS growth + acceleration |
| `Gross Profit` | Gross profit margin |
| `Operating Income` | Operating margin (secondary) |

### Availability

Confirmed across 5 diverse tickers (AAPL, COP, ADI, KO, MSFT): all line items present with 5–7 quarters of history. `Cost of Revenue` is the only common missing row and is not needed.

---

## The 5 Dimensions

### 1. EPS Growth

**Source:** `Basic EPS` row, quarterly values sorted oldest-first.

**Yearly** (preferred, more data-stable):
```
growth = (eps[t] - eps[t-4]) / eps[t-4]
threshold: ≥ 15%
```

**Quarterly** (fallback when < 5 quarters):
```
growth = (eps[t] - eps[t-1]) / eps[t-1]
threshold: ≥ 25%
```

### 2. EPS Acceleration

**Source:** Same EPS series.

Compare consecutive YoY growth rates:
```
acceleration = latest_YoY_growth > prior_YoY_growth
```

Requires ≥ 6 quarters (≥ 2 YoY comparisons). With exactly 5 quarters, flag as `accelerating = None` with detail "insufficient data for acceleration check".

### 3. Revenue Growth

**Source:** `Total Revenue` row.

Same logic as EPS growth with same thresholds (15% YoY, 25% QoQ). Replaces the current single 20% threshold.

### 4. Revenue Acceleration

**Source:** Same revenue series.

Same acceleration logic as EPS acceleration. Requires ≥ 6 quarters.

### 5. Profit Margin Improvement

**Source:** `Gross Profit` and `Net Income` divided by `Total Revenue`.

**Gross margin:** `Gross Profit / Total Revenue`
**Net margin:** `Net Income / Total Revenue`

Check if margin improves across 3 consecutive quarters:
```
improving = margin[t] > margin[t-1] > margin[t-2]
```

Threshold: ≥ 1 percentage point improvement over the period, or strictly upward trend.

Reports both gross and net margin trends.

---

## Data Structures

### MetricResult (per dimension)

```python
@dataclass
class MetricResult:
    label: str                     # e.g. "EPS growth (YoY)"
    passed: bool                   # meets threshold?
    value: Optional[float]         # growth rate / margin % (decimal)
    threshold: Optional[float]     # threshold it was tested against
    quarters: List[float]          # raw quarterly values
    detail: str                    # human-readable explanation
```

### MinerviniFundamentalResult (aggregate)

```python
@dataclass
class MinerviniFundamentalResult:
    ticker: str
    eps_growth: MetricResult
    eps_acceleration: MetricResult
    revenue_growth: MetricResult
    revenue_acceleration: MetricResult
    profit_margin: MetricResult
    score: int                     # how many dimensions passed
    total_dims: int                # how many had sufficient data (max 5)
    overall_passed: bool           # score >= threshold
    details: str                   # multi-line breakdown
```

---

## Integration

### Pipeline Changes

- `Pipeline.__init__` gains `fundamental_min_score: int = 3` parameter.
- `Pipeline.screen()` calls `MinerviniFundamentalEngine.screen(ticker)` instead of `FundamentalFilter.screen(ticker)`.
- Approval logic: `TrendTemplate.passed AND vcp.vcp_detected AND fundamental.overall_passed`.
- `ScreeningResult.fundamental_filter` is replaced by `ScreeningResult.minervini_fundamental` (old field kept as deprecated alias).

### Output Changes

**Terminal output** shows the full breakdown per dimension instead of a single line.

**batch_screen.py** — near-miss sections become richer, showing which dimensions failed.

**check_trend.py** — shows fundamental breakdown alongside the trend template.

---

## Edge Cases

| Situation | Handling |
|-----------|----------|
| < 5 quarters | Growth dimensions use QoQ threshold; acceleration flagged insufficient |
| EPS row missing | Fallback to Diluted EPS; if both missing, mark with `passed=False` |
| Only 5 quarters | Acceleration check returns `None` for accelerating flag; still reports growth |
| Gross Profit missing | Margin dimension reports net margin only |
| Negative growth | Reported as-is (negative growth = fail) |
| Division by zero | Guarded: if denominator ≤ 0, mark as failed with detail |

---

## Implementation Order

1. Create `MetricResult` and `MinerviniFundamentalResult` dataclasses
2. Implement `MinerviniFundamentalEngine` class with all 5 dimensions
3. Integration into `Pipeline` (replace old filter, wire up approval)
4. Update `ScreeningResult.to_dict()` for the new structure
5. Update `batch_screen.py` output to show dimensional breakdown
6. Update `check_trend.py` to show fundamental breakdown
7. Update README with new fundamental filter documentation
8. Test on current watchlist and compare results