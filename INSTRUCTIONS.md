You are an expert quantitative software engineer and algorithmic trading system architect. 

Your task is to build a robust, production-ready Python module named `vcp_momentum_engine.py` that implements Mark Minervini’s SEPA (Specific Entry Point Analysis) framework, Volatility Contraction Pattern (VCP) detection, Trend Template filters, and institutional risk-management position sizing using `yfinance` for data ingestion.

### Core Architectural Requirements

1. **Strict Production Standards:**
   - Write fully typed Python code (using Python 3.10+ type hints: `dict`, `list`, `Optional`, `Tuple`, etc.).
   - Include comprehensive Google-style docstrings for all classes and methods.
   - Implement structured logging (`logging` module) instead of `print` statements.
   - Use `dataclasses` or `pydantic` models for structured data outputs (e.g., trade setups, screening results).
   - **NO MOCK DATA:** All logic must process real data retrieved dynamically via `yfinance`.

2. **Module 1: Data Ingestion & Normalization**
   - Fetch daily OHLCV historical data for a given ticker (default: 400 calendar days to guarantee at least 200 trading days for moving averages).
   - Automatically handle and normalize `yfinance` MultiIndex column anomalies (which occur in newer library versions).
   - Handle network timeouts, empty dataframes, and missing symbols gracefully with custom try/except blocks and logging.

3. **Module 2: Minervini Trend Template Engine**
   Programmatically validate the following structural rules on the daily series:
   - Current closing price > 50-day Simple Moving Average (SMA).
   - 50-day SMA > 200-day SMA.
   - Both 50-day and 200-day SMAs must be sloping upward over a lookback window of the last 20 trading days (~1 month).

4. **Module 3: Fundamental Acceleration Filter**
   - Query `ticker.quarterly_financials` to extract sequential quarterly revenue and earnings data.
   - Verify year-over-year (YoY) acceleration in quarterly revenue growth (e.g., latest quarter YoY growth > prior quarter YoY growth, with a minimum baseline growth threshold of 20%).
   - If fundamental quarterly data is missing or sparse for a given asset, log a warning and provide a clean fallback flag rather than crashing the pipeline.

5. **Module 4: VCP & Volatility Contraction Engine**
   Analyze the most recent 60 trading days of price and volume action:
   - Identify progressive contractions in peak-to-trough drawdowns (e.g., initial pullback of 15–25%, followed by tighter contractions of 8–12%, culminating in a tight consolidation range < 8%).
   - Compute Average True Range (ATR) percentage to measure price compression.
   - Detect volume dry-ups: check if the latest session volume is significantly below the 20-day average volume baseline (e.g., < 75% of 20-day volume SMA).

6. **Module 5: Risk Management & Position Sizing Calculator**
   Implement deterministic position sizing based on institutional risk parameters:
   - Inputs: Account equity (default: $100,000), maximum risk percentage per trade (default: 0.3% / `0.003`), entry price, and stop-loss price (3% to 5% below entry).
   - Formulas:
     $$\text{Dollar Risk Allowed} = \text{Account Equity} \times \text{Risk Percentage}$$
     $$\text{Risk Per Share} = \text{Entry Price} - \text{Stop-Loss Price}$$
     $$\text{Shares to Buy} = \left\lfloor \frac{\text{Dollar Risk Allowed}}{\text{Risk Per Share}} \right\rfloor$$
   - Validate that `Risk Per Share > 0` and raise a ValueError if invalid.

### Output Deliverable
Provide the complete, self-contained Python script containing all classes (`DataIngestion`, `TrendTemplateValidator`, `FundamentalFilter`, `VCPEngine`, `RiskManager`) along with an execution block that screens a sample watchlist (e.g., `["NVDA", "PLTR", "AAPL", "MSFT"]`) and outputs structured JSON/dictionary summaries of approved setups.
