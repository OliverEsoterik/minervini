# VCP Momentum Engine

**Mark Minervini's Volatility Contraction Pattern + Trend Template + William O'Neil's CAN SLIM** — implemented as a production-ready Python screening engine using live Yahoo Finance data.

All data is fetched live via `yfinance`. No mock data, no hardcoded prices. Every decision is based on the current market structure and quarterly financials.

---

## What it does

The engine runs **three independent filters** against every ticker and only approves trades when all three agree:

| Filter | What it checks | Method |
|--------|----------------|--------|
| **Trend Template** | Is the stock in a structural bull market? | Close > SMA-50 > SMA-200, both SMAs sloping up |
| **Fundamental Filter** | Is revenue accelerating YoY? | Quarterly revenue growth > 20% baseline |
| **VCP Engine** | Is there a volatility contraction pattern? | Tightening drawdowns, ATR compression, volume dry-up |

This produces a ~1–3% approval rate on a broad watchlist — intentionally strict.

---

## Files

| File | Purpose |
|------|---------|
| `vcp_momentum_engine.py` | Core module — all 5 modules + pipeline. CLI: `python3 vcp_momentum_engine.py [TICKERS...]` |
| `batch_screen.py` | Parallel batch screening (8 workers) for the full watchlist |
| `check_trend.py` | Quick HOLD/SELL check for open positions |
| `INSTRUCTIONS.md` | Original specification |

---

## Setup

```bash
pip install yfinance numpy pandas
```

Python 3.10+ required.

---

## Usage

### 1. Screen the full watchlist (fastest)

```bash
python3 batch_screen.py
```

Processes ~500 tickers in parallel across 8 workers (~5 minutes). Output: approved setups with entry prices, stop-losses, and share counts.

### 2. Screen tickers via command line

```bash
# Full embedded watchlist (~500 tickers)
python3 vcp_momentum_engine.py

# One or more specific tickers
python3 vcp_momentum_engine.py AAPL MSFT COP ADI

# Single ticker
python3 vcp_momentum_engine.py SEZL

# Usage info
python3 vcp_momentum_engine.py --help
```

Runs the full pipeline (Trend Template + Fundamental + VCP + Position Sizing) sequentially.
With no arguments, screens the embedded watchlist.  With tickers on the command line,
screens only those.

### 3. Check if open positions are still valid

```bash
# Quick check — just the trend template
python3 check_trend.py ADI COP

# With entry price and trailing high watermark
ENTRY=142.28 HIGH=175.00 python3 check_trend.py COP
```

Output:

- **✅ HOLD** — all trend template checks pass. Position is valid.
- **🛑 SELL** — trend template broken. Reasons listed (Close < SMA-50, SMA-50 slope negative, etc.).
- **⚠ Trailing stop warning** — if within 15–25% of the high watermark.

---

## How the three filters work

### Trend Template (Minervini)

Four rules validated on daily OHLCV data:

1. **Close > SMA-50** — the stock isn't in a deep correction
2. **SMA-50 > SMA-200** — the intermediate trend is above the long-term trend
3. **SMA-50 sloping up** (20-day window) — the intermediate trend is strengthening
4. **SMA-200 sloping up** — the long-term trend is strengthening (informational)

All SMAs are simple moving averages (not exponential), matching Minervini's published logic. The slope is calculated via ordinary least squares over the last 20 trading days.

### Fundamental Acceleration Filter (O'Neil — CAN SLIM "A")

Retrieves quarterly Total Revenue from `yfinance.Ticker.quarterly_financials` and computes year-over-year growth by comparing each quarter to the same quarter four periods back:

- **≥ 6 quarters (≥ 2 YoY comparisons):** latest YoY growth must exceed prior YoY growth *and* both must exceed 20%.
- **5 quarters (1 YoY comparison):** latest YoY growth must exceed 20%.

Missing quarterly data produces a `passed=False` with `revenue_available=False` — a clean fallback flag rather than a crash.

### VCP Engine (Minervini)

Scans the most recent ~75 calendar days (≈60 trading days) for:

- **Progressive drawdown tightening:** peak-to-trough declines shrinking from 15–25% → 8–12% → < 10%.
- **ATR compression:** Average True Range < 4% of the mean close price.
- **Volume dry-up:** latest session volume < 75% of the 20-day volume SMA (a precursor signal).

The engine identifies local peaks and troughs via a simple extremum scan, then checks if the contraction sequence is declining and the final consolidation is tight.

---

## CAN SLIM Position Sizing

Based on O'Neil's institutional risk framework (*How to Make Money in Stocks*):

```
Dollar Risk Allowed = Account Equity × Risk Percentage
Risk Per Share      = Entry Price - Stop-Loss Price
Shares to Buy       = floor(Dollar Risk Allowed / Risk Per Share)
```

Defaults: $100,000 equity, 0.3% risk per trade (=$300), 4% stop-loss below entry.

---

## Exit rules

Three-stage discipline:

| Trigger | Action |
|---------|--------|
| Price hits stop-loss (-4%) | Out. Small defined loss. |
| Price falls 25%+ from high | Out via trailing stop with gains intact. |
| Trend template breaks | Out the full position — the structural thesis is gone. |

The trend template is your primary sell signal. Run `check_trend.py` weekly on open positions. If any of Close > SMA-50, SMA-50 > SMA-200, or SMA-50 slope > 0 fails, exit.

---

## Default watchlist

The embedded watchlist covers ~500 tickers across sectors: FAANG, semiconductors, financials, healthcare, energy, consumer staples, industrials, and more. Edit the `watchlist` list in `vcp_momentum_engine.py` or `batch_screen.py` to customize.

Pass tickers directly on the command line to screen a subset without editing files:

```bash
python3 vcp_momentum_engine.py ADI COP EXPD
```

---

## Methodology

This engine is based on two complementary frameworks:

- **Mark Minervini's SEPA / VCP / Trend Template** — covered in *Trade Like a Stock Market Wizard*, *The Mind of the Market Wizard*, and his public seminars. Minervini's core insight: stocks in strong uptrends routinely pause to consolidate, and those consolidations produce a detectable pattern of tightening volatility (the VCP). Enter at the point of tightest consolidation for the lowest-risk, highest-probability breakout trade.

- **William O'Neil's CAN SLIM** — from *How to Make Money in Stocks*. The seven criteria (Current price > 200-day MA, Accelerating earnings, New products, Supply/demand imbalance, Leadership, Institutional sponsorship, Merger potential) provide the fundamental thesis for why a stock can go up. This engine automates the C and A criteria; the remaining five are populated from yfinance metadata for manual review.