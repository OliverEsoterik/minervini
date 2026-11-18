#!/usr/bin/env python3
"""
VCP Momentum Engine — Mark Minervini's VCP Detection & Trend Template
with William O'Neil's CAN SLIM risk-framework integration.

Production-grade module implementing:

  • Module 1 — Data Ingestion & Normalization (yfinance OHLCV)
  • Module 2 — Minervini Trend Template (price > 50 SMA > 200 SMA, slope)
  • Module 3 — Minervini Fundamental Engine (5-dimension: EPS/revenue/margins)
  • Module 4 — VCP & Volatility Contraction Engine (drawdown, ATR, volume)
  • Module 5 — CAN SLIM Risk Management & Position Sizing

All data is fetched live via yfinance — no mock data, no hardcoded prices.
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# numpy compatibility: np.RankWarning was removed in numpy 2.x
NUMPY_MAJOR = int(np.__version__.split(".")[0])
if NUMPY_MAJOR >= 2:
    RankWarning = np.exceptions.RankWarning  # type: ignore[attr-defined]
else:
    RankWarning = np.RankWarning  # type: ignore[attr-defined]

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("vcp_momentum_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_DAYS: int = 400  # guarantee >= 200 trading days
VCP_LOOKBACK_DAYS: int = 75       # conservative buffer around 60 trading days
TREND_SLOPE_WINDOW: int = 20      # trading days for SMA slope check
DEFAULT_EQUITY: float = 100_000.0
DEFAULT_RISK_PCT: float = 0.003   # 0.3 % max risk per trade
VOLUME_DRYUP_THRESHOLD: float = 0.75     # < 75 % of 20-d volume SMA


# ===================================================================
# Module 1 — Data Ingestion & Normalization
# ===================================================================

@dataclass
class OHLCVData:
    """Normalised OHLCV daily price/volume series with metadata."""
    ticker: str
    dataframe: pd.DataFrame
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=None))
    trading_days: int = 0
    interval: str = "1d"

    @property
    def close(self) -> pd.Series:
        return self.dataframe["Close"]

    @property
    def volume(self) -> pd.Series:
        return self.dataframe["Volume"]

    @property
    def high(self) -> pd.Series:
        return self.dataframe["High"]

    @property
    def low(self) -> pd.Series:
        return self.dataframe["Low"]

    @property
    def open_(self) -> pd.Series:
        return self.dataframe["Open"]


class DataIngestion:
    """Fetch and normalise daily OHLCV data from Yahoo Finance.

    Handles yfinance MultiIndex column quirks, network errors, empty
    responses, and missing symbols transparently.
    """

    def __init__(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        self.lookback_days = lookback_days

    # ------------------------------------------------------------------
    def fetch(self, ticker: str) -> Optional[OHLCVData]:
        """Download OHLCV history for *ticker* and return a normalised
        :class:`OHLCVData` container, or ``None`` on failure.

        Steps:
          1. Call ``yfinance.download(…)`` with a ``period`` long enough to
             guarantee at least 200 trading days.
          2. Flatten the MultiIndex columns that newer yfinance versions
             return (a tuple → single level with a space separator).
          3. Drop rows with null Close prices (non-trading days) and sort.
          4. Wrap all network / parse exceptions in a try/except with
             structured logging.
        """
        logger.info("Fetching OHLCV data for %s …", ticker)
        try:
            raw = yf.download(
                tickers=ticker,
                period=f"{self.lookback_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error(
                "yfinance.download failed for %s: %s", ticker, exc
            )
            return None

        if raw is None or raw.empty:
            logger.warning("Empty DataFrame returned for %s", ticker)
            return None

        # ---- flatten MultiIndex columns ----------------------------
        df = self._flatten_columns(raw)

        # ---- keep only OHLCV columns -------------------------------
        expected = {"Open", "High", "Low", "Close", "Volume"}
        available = set(df.columns)
        if not expected.issubset(available):
            missing = expected - available
            logger.warning(
                "Missing columns %s for %s — got %s", missing, ticker, available
            )
            return None

        df = df[list(expected)].copy()

        # ---- drop non-trading days (NaN Close) ---------------------
        before = len(df)
        df = df.dropna(subset=["Close"])
        after = len(df)

        # ---- sort index ---------------------------------------------
        df = df.sort_index()
        trading_days = len(df)

        logger.info(
            "%s: %d trading days (%d rows dropped for NaN Close)",
            ticker, trading_days, before - after,
        )

        # ---- sanity ------------------------------------------------
        if trading_days < 200:
            logger.warning(
                "%s: only %d trading days (< 200). "
                "Consider increasing lookback_days.",
                ticker, trading_days,
            )

        return OHLCVData(
            ticker=ticker,
            dataframe=df,
            trading_days=trading_days,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten a yfinance MultiIndex column frame into single-level.

        yfinance 0.5+ returns columns like ``('Close', 'AAPL')``.  We drop
        the second level (ticker name) and keep the first as a plain string.
        If columns are already single-level this is a no-op.
        """
        if isinstance(df.columns, pd.MultiIndex):
            # Keep only the first level (Open, High, Low, Close, Volume)
            df.columns = df.columns.get_level_values(0)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def fetch_info(ticker: str) -> Dict[str, Any]:
        """Return the raw ``yfinance.Ticker.info`` dict, or ``{}`` on error.

        Useful for CAN SLIM metadata (sector, industry, market-cap).
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            return {k: v for k, v in info.items() if v is not None}
        except Exception as exc:
            logger.warning("fetch_info(%s) failed: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------
    @staticmethod
    def fetch_quarterly_financials(
        ticker: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """Return (revenue_series, earnings_series) from quarterly financials.

        Each Series is indexed by fiscal-quarter timestamp and contains
        the sequential quarterly values for that line item, sorted oldest
        first.  Returns empty Series if data is unavailable.
        """
        try:
            t = yf.Ticker(ticker)
            qf = t.quarterly_financials
            if qf is None or qf.empty:
                logger.warning("No quarterly financials for %s", ticker)
                return pd.Series(), pd.Series()

            # qf.columns = fiscal-quarter timestamps (most recent last)
            revenue = pd.Series()
            earnings = pd.Series()

            if "Total Revenue" in qf.index:
                revenue = qf.loc["Total Revenue"].dropna().astype(float)
                revenue = revenue.sort_index()  # oldest first
            else:
                logger.warning(
                    "%s quarterly — 'Total Revenue' row not found; "
                    "available rows: %s",
                    ticker, list(qf.index),
                )

            if "Net Income" in qf.index:
                earnings = qf.loc["Net Income"].dropna().astype(float)
            elif "Net Earnings" in qf.index:
                earnings = qf.loc["Net Earnings"].dropna().astype(float)
            else:
                logger.warning(
                    "%s quarterly — 'Net Income' / 'Net Earnings' row not found; "
                    "available rows: %s",
                    ticker, list(qf.index),
                )

            if not earnings.empty:
                earnings = earnings.sort_index()

            return revenue, earnings

        except Exception as exc:
            logger.error(
                "fetch_quarterly_financials(%s) failed: %s", ticker, exc
            )
            return pd.Series(), pd.Series()


# ===================================================================
# Module 2 — Minervini Trend Template Validator
# ===================================================================

@dataclass
class TrendTemplateResult:
    """Result of running the Trend Template on one ticker."""
    ticker: str
    passed: bool
    current_close: float
    sma_50: float
    sma_200: float
    sma_50_slope: float   # average daily change over last 20 days
    sma_200_slope: float
    sma_150: Optional[float] = None     # 150-day SMA (current value)
    sma_150_slope: Optional[float] = None  # slope of 150-day SMA
    details: str = ""


class TrendTemplateValidator:
    """Minervini Trend Template — structural bull-market pre-screen.

    Rules
    -----
    1. Current closing price > 50-day SMA.
    2. Current closing price > 150-day SMA.
    3. 50-day SMA > 150-day SMA.
    4. 150-day SMA > 200-day SMA.
    5. Both the 50-day and 200-day SMAs are sloping *upward* over a
       lookback window of **TREND_SLOPE_WINDOW** trading days (~1 month).

    This implements the full 3-step MA chain (50 > 150 > 200) per
    Minervini's Trend Template, closing the gap where SMA-150 hasn't
    crossed SMA-200 yet.
    """

    def __init__(self, slope_window: int = TREND_SLOPE_WINDOW) -> None:
        self.slope_window = slope_window

    # ------------------------------------------------------------------
    def validate(self, data: OHLCVData) -> TrendTemplateResult:
        """Run the Trend Template against *data* and return a verdict.

        Uses simple moving averages (``.rolling(…).mean()``) — not
        exponential — matching Minervini's published logic.
        """
        ticker = data.ticker
        close = data.close
        n = len(close)

        # --- compute SMAs -------------------------------------------
        sma_50 = close.rolling(window=50, min_periods=50).mean()
        sma_150 = close.rolling(window=150, min_periods=150).mean()
        sma_200 = close.rolling(window=200, min_periods=200).mean()

        latest_close = float(close.iloc[-1])
        latest_sma_50 = float(sma_50.iloc[-1])
        latest_sma_150 = float(sma_150.iloc[-1]) if len(close) >= 150 else float('nan')
        latest_sma_200 = float(sma_200.iloc[-1])

        # --- slope over the last TREND_SLOPE_WINDOW days ------------
        slope_50 = self._slope(sma_50)
        slope_150 = self._slope(sma_150) if len(close) >= 150 else 0.0
        slope_200 = self._slope(sma_200)

        # --- evaluate -----------------------------------------------
        checks: List[Tuple[bool, str]] = [
            (
                latest_close > latest_sma_50,
                f"Close {latest_close:.2f} > SMA-50 {latest_sma_50:.2f}",
            ),
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
            (
                latest_sma_50 > latest_sma_200,
                f"SMA-50 {latest_sma_50:.2f} > SMA-200 {latest_sma_200:.2f}",
            ),
            (
                slope_50 > 0,
                f"SMA-50 slope {slope_50:.4f} > 0",
            ),
            (
                slope_200 > 0,
                f"SMA-200 slope {slope_200:.4f} > 0",
            ),
        ]

        passed = all(c[0] for c in checks)
        detail = " | ".join(
            ("PASS" if c[0] else "FAIL") + " — " + c[1] for c in checks
        )

        logger.info(
            "TrendTemplate [%s]: %s — %s",
            ticker, "PASS" if passed else "FAIL", detail,
        )

        return TrendTemplateResult(
            ticker=ticker,
            passed=passed,
            current_close=latest_close,
            sma_50=latest_sma_50,
            sma_150=latest_sma_150,
            sma_200=latest_sma_200,
            sma_50_slope=slope_50,
            sma_150_slope=slope_150,
            sma_200_slope=slope_200,
            details=detail,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _slope(series: pd.Series, window: int = TREND_SLOPE_WINDOW) -> float:
        """Linear slope of *series* over the last *window* non-NaN values.

        Uses ordinary least-squares (closed-form) to avoid numpy version
        headaches with ``np.polyfit`` / ``RankWarning``.
        Returns 0.0 if insufficient data.
        """
        valid = series.dropna().tail(window)
        if len(valid) < 2:
            return 0.0
        y = valid.values.astype(float)
        x = np.arange(len(y))
        n = float(len(y))
        # Simple linear regression: y = mx + c
        # m = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_xx = (x * x).sum()
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-15:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return float(slope)


# ===================================================================
# Module 3 — Minervini Fundamental Engine (5-Dimension)
# ===================================================================


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



def _extract_series(df: pd.DataFrame | None, label: str) -> pd.Series:
    """Extract a single line-item row from a quarterly-financials DataFrame.

    Returns the Series sorted oldest-first, or empty Series if missing.
    """
    if df is None or df.empty or label not in df.index:
        return pd.Series()
    return df.loc[label].dropna().astype(float).sort_index()


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
        # Single yfinance call for all quarterly line items
        try:
            qf = yf.Ticker(ticker).quarterly_financials
        except Exception as exc:
            logger.warning(
                "screen(%s) — quarterly financials fetch failed: %s", ticker, exc
            )
            return MinerviniFundamentalResult(
                ticker=ticker,
                eps_growth=MetricResult(label="EPS growth", passed=False, detail="yfinance error"),
                eps_acceleration=MetricResult(label="EPS acceleration", passed=False, detail="yfinance error"),
                revenue_growth=MetricResult(label="Revenue growth", passed=False, detail="yfinance error"),
                revenue_acceleration=MetricResult(label="Revenue acceleration", passed=False, detail="yfinance error"),
                profit_margin=MetricResult(label="Profit margin", passed=False, detail="yfinance error"),
                score=0,
                total_dims=0,
                overall_passed=False,
                details=f"Quarterly financials fetch failed for {ticker}: {exc}",
           )
        rev_series = _extract_series(qf, "Total Revenue")
        ear_series = _extract_series(qf, "Net Income")
        if ear_series.empty:
            ear_series = _extract_series(qf, "Net Earnings")
        eps_series, gp_series = self._fetch_extra_metrics(qf)

        # Compute each dimension
        eps_growth = self._eval_eps_growth(eps_series)
        eps_accel = self._eval_eps_acceleration(eps_series)
        rev_growth = self._eval_revenue_growth(rev_series)
        rev_accel = self._eval_revenue_acceleration(rev_series)
        margin = self._eval_profit_margin(gp_series, ear_series, rev_series)

        dims = [eps_growth, eps_accel, rev_growth, rev_accel, margin]
        score = sum(1 for d in dims if d.passed)
        # Count dimensions where data was available (value set)
        total = sum(1 for d in dims if d.value is not None)

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
        qf: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """Extract EPS and Gross Profit from a quarterly-financials DataFrame."""
        eps = pd.Series()
        gp = pd.Series()
        try:
            if qf is None or qf.empty:
                return eps, gp

            if "Basic EPS" in qf.index:
                eps = qf.loc["Basic EPS"].dropna().astype(float).sort_index()
            elif "Diluted EPS" in qf.index:
                eps = qf.loc["Diluted EPS"].dropna().astype(float).sort_index()

            if "Gross Profit" in qf.index:
                gp = qf.loc["Gross Profit"].dropna().astype(float).sort_index()

        except Exception as exc:
            logger.warning("_fetch_extra_metrics failed: %s", exc)

        return eps, gp

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


# ===================================================================
# Module 4 — VCP & Volatility Contraction Engine
# ===================================================================

@dataclass
class VCPResult:
    """Output of the VCP engine for one ticker."""
    ticker: str
    vcp_detected: bool
    contractions: List[Dict[str, float]] = field(default_factory=list)
    latest_consolidation_pct: Optional[float] = None
    atr_pct: Optional[float] = None
    volume_dryup: bool = False
    volume_ratio: Optional[float] = None
    details: str = ""


class VCPEngine:
    """Detect Minervini Volatility Contraction Patterns.

    Analysed over the most recent ~60 trading days:

    * **Drawdown contractions**: progressive tightening from initial
      15–25 % pullback → 8–12 % → < 8 % consolidation.
    * **ATR percent**: Average True Range as a percent of price (compression
      metric).
    * **Volume dry-up**: latest session volume < 75 % of 20-day volume SMA.
    """

    def __init__(self, lookback: int = VCP_LOOKBACK_DAYS) -> None:
        self.lookback = lookback

    # ------------------------------------------------------------------
    def analyse(self, data: OHLCVData) -> VCPResult:
        """Run the VCP detection algorithm on *data*.

        Returns a :class:`VCPResult` with the contraction series, ATR
        compression metric, and volume dry-up flag.
        """
        ticker = data.ticker
        df = data.dataframe.tail(self.lookback).copy()
        if len(df) < 30:
            return VCPResult(
                ticker=ticker,
                vcp_detected=False,
                details=f"Only {len(df)} days in lookback window (< 30).",
            )

        # --- 1. Identify drawdown events by local peaks/troughs -----
        peaks_troughs = self._find_peaks_troughs(df)
        contractions = self._compute_contractions(peaks_troughs, df)

        # --- 2. ATR compression --------------------------------------
        atr_pct = self._atr_percent(df)

        # --- 3. Volume dry-up ----------------------------------------
        volume_ratio, dryup = self._volume_dryup(df)

        # --- 4. VCP heuristic ----------------------------------------
        vcp_detected, detail = self._vcp_heuristic(contractions, atr_pct, dryup)

        latest_con = contractions[-1]["range_pct"] if contractions else None

        logger.info(
            "VCP [%s]: %s — %d contractions, ATR %.2f%%, volume-dryup=%s",
            ticker, "DETECTED" if vcp_detected else "NONE",
            len(contractions), atr_pct, dryup,
        )

        return VCPResult(
            ticker=ticker,
            vcp_detected=vcp_detected,
            contractions=contractions,
            latest_consolidation_pct=latest_con,
            atr_pct=atr_pct,
            volume_dryup=dryup,
            volume_ratio=volume_ratio,
            details=detail,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _find_peaks_troughs(df: pd.DataFrame) -> List[Tuple[int, str, float]]:
        """Locate alternating swing highs and swing lows.

        Return list of (index_pos, type, price) where type is "peak" or
        "trough".  Uses a simple local-extrema scan over the Close series.
        """
        close = df["Close"].values
        n = len(close)
        if n < 3:
            return []

        events: List[Tuple[int, str, float]] = []
        # Always start with the first data point as a reference
        for i in range(1, n - 1):
            if close[i] > close[i - 1] and close[i] > close[i + 1]:
                events.append((i, "peak", close[i]))
            elif close[i] < close[i - 1] and close[i] < close[i + 1]:
                events.append((i, "trough", close[i]))

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_contractions(
        events: List[Tuple[int, str, float]],
        df: pd.DataFrame,
    ) -> List[Dict[str, float]]:
        """Convert peak/trough events into drawdown ranges (%).

        A *contraction* is defined as the percent decline from a peak to
        the next trough (peak → trough).  Returns a list of dicts:

            {"peak_idx": …, "trough_idx": …, "peak": …, "trough": …,
             "range_pct": …}
        """
        contractions: List[Dict[str, float]] = []
        for i in range(len(events) - 1):
            if events[i][1] == "peak" and events[i + 1][1] == "trough":
                p_idx, _, peak = events[i]
                t_idx, _, trough = events[i + 1]
                rng_pct = (peak - trough) / peak * 100
                contractions.append(
                    {
                        "peak_idx": p_idx,
                        "trough_idx": t_idx,
                        "peak": round(peak, 2),
                        "trough": round(trough, 2),
                        "range_pct": round(rng_pct, 2),
                    }
                )
        return contractions

    # ------------------------------------------------------------------
    @staticmethod
    def _atr_percent(df: pd.DataFrame) -> float:
        """Compute Average True Range as a percentage of the mean Close.

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        """
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        tr: List[float] = []
        for i in range(1, len(close)):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr.append(max(h_l, h_pc, l_pc))

        if not tr:
            return 0.0

        mean_tr = float(np.mean(tr))
        mean_close = float(np.mean(close))
        if mean_close == 0:
            return 0.0
        return mean_tr / mean_close * 100

    # ------------------------------------------------------------------
    @staticmethod
    def _volume_dryup(df: pd.DataFrame) -> Tuple[Optional[float], bool]:
        """Check if latest volume is significantly below 20-day SMA.

        Returns (volume_ratio, is_dry).  *volume_ratio* = latest_volume /
        volume_sma_20, clipped at 0.
        """
        volume = df["Volume"].dropna()
        if len(volume) < 20:
            return None, False

        sma_20 = volume.rolling(window=20, min_periods=20).mean()
        latest_vol = float(volume.iloc[-1])
        latest_sma = float(sma_20.iloc[-1])

        if latest_sma == 0:
            return None, False

        ratio = latest_vol / latest_sma
        return ratio, ratio < VOLUME_DRYUP_THRESHOLD

    # ------------------------------------------------------------------
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
        a *declining* trend in drawdown magnitudes.
        """
        if len(contractions) < 2:
            return False, "Fewer than 2 contraction events."

        magnitudes = [c["range_pct"] for c in contractions]

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


# ===================================================================
# Module 5 — CAN SLIM Risk Management & Position Sizing
# ===================================================================

@dataclass
class PositionSizingResult:
    """Position-sizing output based on institutional risk parameters.

    Implements O'Neil's risk-asymmetric position sizing from
    *The Great Investors* and CAN SLIM framework:
    bounded downside → let the upside run.
    """
    ticker: str
    account_equity: float
    risk_per_trade_pct: float
    entry_price: float
    stop_loss_price: float
    dollar_risk_allowed: float
    risk_per_share: float
    shares_to_buy: int
    total_risk_deployed: float
    risk_pct_of_equity: float


class CANSLIMProfile:
    """William O'Neil CAN SLIM qualitative assessment dataclass.

    Fields C, A are derived from quantitative checks (Trend Template,
    MinerviniFundamentalEngine).  N, S, L, I, M are populated from available
    metadata or left as ``None`` for the user to fill in.
    """

    C_price_above_200ma: Optional[bool] = None
    A_earnings_acceleration: Optional[bool] = None
    N_new_products: Optional[str] = None      # qualitative
    S_supply_demand: Optional[str] = None     # qualitative
    L_leadership: Optional[str] = None        # qualitative
    I_institutional: Optional[str] = None     # qualitative
    M_merger_takeover: Optional[str] = None   # qualitative
    overall_score: int = 0  # 0–7 criteria met


class RiskManager:
    """Position sizing and CAN SLIM risk assessment.

    Position Sizing (Module 5)
    -------------------------
    Deterministic formula based on institutional risk parameters
    (see Section 7.1 of O'Neil's *How to Make Money in Stocks*):

        Dollar Risk Allowed = Account Equity × Risk Percentage

        Risk Per Share = Entry Price - Stop-Loss Price

        Shares to Buy = floor(Dollar Risk Allowed / Risk Per Share)

    CAN SLIM Profile
    ----------------
    Populates the 7 CAN SLIM criteria from available data sources and
    returns a structured profile.
    """

    def __init__(
        self,
        equity: float = DEFAULT_EQUITY,
        risk_pct: float = DEFAULT_RISK_PCT,
    ) -> None:
        if equity <= 0:
            raise ValueError(f"equity must be > 0, got {equity}")
        if not 0 < risk_pct < 1:
            raise ValueError(
                f"risk_pct must be in (0, 1), got {risk_pct}"
            )
        self.equity = equity
        self.risk_pct = risk_pct

    # ------------------------------------------------------------------
    def size_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss_price: float,
    ) -> PositionSizingResult:
        """Compute the number of shares to buy given entry and stop-loss.

        Raises
        ------
        ValueError
            If *stop_loss_price* >= *entry_price* (no risk per share) or if
            prices are non-positive.
        """
        if entry_price <= 0:
            raise ValueError(
                f"entry_price must be > 0, got {entry_price}"
            )
        if stop_loss_price <= 0:
            raise ValueError(
                f"stop_loss_price must be > 0, got {stop_loss_price}"
            )

        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            raise ValueError(
                f"Risk per share must be > 0. "
                f"Entry {entry_price} <= Stop-loss {stop_loss_price}"
            )

        dollar_risk_allowed = self.equity * self.risk_pct
        shares = int(math.floor(dollar_risk_allowed / risk_per_share))
        total_risk_deployed = shares * risk_per_share
        risk_pct_of_equity = total_risk_deployed / self.equity

        logger.info(
            "Position [%s]: entry=%.2f stop=%.2f "
            "risk/share=%.4f shares=%d deployed=%.2f (%.4f%% of equity)",
            ticker,
            entry_price,
            stop_loss_price,
            risk_per_share,
            shares,
            total_risk_deployed,
            risk_pct_of_equity * 100,
        )

        return PositionSizingResult(
            ticker=ticker,
            account_equity=self.equity,
            risk_per_trade_pct=self.risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            dollar_risk_allowed=round(dollar_risk_allowed, 2),
            risk_per_share=round(risk_per_share, 4),
            shares_to_buy=shares,
            total_risk_deployed=round(total_risk_deployed, 2),
            risk_pct_of_equity=round(risk_pct_of_equity, 6),
        )

    # ------------------------------------------------------------------
    def build_canslim_profile(
        self,
        ticker: str,
        trend_result: Optional[TrendTemplateResult] = None,
        fundamental_result: Optional[MinerviniFundamentalResult] = None,
    ) -> CANSLIMProfile:
        """Assemble a CAN SLIM profile from available analysis results.

        *C* and *A* are set automatically from the Trend Template and
        Fundamental Filter.  The remaining five fields (N, S, L, I, M)
        are seeded with industry metadata from yfinance ``info`` where
        available, or left as ``None`` for user review.

        *overall_score* counts how many of the 7 criteria are either
        ``True`` (for C, A) or non-empty (for N, S, L, I, M).
        """
        profile = CANSLIMProfile()

        # --- C: Price above 200-day SMA -----------------------------
        if trend_result is not None:
            profile.C_price_above_200ma = (
                trend_result.current_close > trend_result.sma_200
            )

        # --- A: Earnings acceleration -------------------------------
        if fundamental_result is not None:
            profile.A_earnings_acceleration = fundamental_result.eps_acceleration.passed

        # --- N, S, L, I, M from yfinance info -----------------------
        info = DataIngestion.fetch_info(ticker)
        profile.L_leadership = info.get("industry", info.get("sector"))
        profile.I_institutional = info.get("marketCap")

        # --- Score (C and A counted as booleans; N/S/L/I/M count if
        #     they have a non-None value).
        bool_checks = [
            profile.C_price_above_200ma or False,
            profile.A_earnings_acceleration or False,
        ]
        str_checks = [
            profile.N_new_products,
            profile.S_supply_demand,
            profile.L_leadership,
            profile.I_institutional is not None,
            profile.M_merger_takeover,
        ]
        profile.overall_score = sum(
            1 for c in bool_checks if c
        ) + sum(
            1 for c in str_checks if c
        )

        return profile


# ===================================================================
# Pipeline — Screen a watchlist
# ===================================================================

@dataclass
class ScreeningResult:
    """Aggregate result for one ticker through the full pipeline."""
    ticker: str
    error: Optional[str] = None
    ohlcv_data: Optional[OHLCVData] = None
    trend_template: Optional[TrendTemplateResult] = None
    minervini_fundamental: Optional[MinerviniFundamentalResult] = None
    vcp: Optional[VCPResult] = None
    position_sizing: Optional[PositionSizingResult] = None
    canslim: Optional[CANSLIMProfile] = None
    approved: bool = False

    @property
    def fundamental_filter(self) -> Optional[MinerviniFundamentalResult]:
        """Deprecated alias for minervini_fundamental. Use .minervini_fundamental instead."""
        return self.minervini_fundamental

    @staticmethod
    def _to_native(v: Any) -> Any:
        """Convert numpy scalars to Python native types for JSON."""
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        d: Dict[str, Any] = {"ticker": self.ticker, "approved": self.approved}
        if self.error:
            d["error"] = self.error
        if self.trend_template:
            d["trend_template"] = {
                "passed": self.trend_template.passed,
                "close_sma50_sma200": (
                    f"{self.trend_template.current_close:.2f} / "
                    f"{self.trend_template.sma_50:.2f} / "
                    f"{self.trend_template.sma_200:.2f}"
                ),
            }
        if self.minervini_fundamental:
            d["minervini_fundamental"] = self.minervini_fundamental.to_dict()
        if self.vcp:
            d["vcp"] = {
                "detected": self._to_native(self.vcp.vcp_detected),
                "contractions": len(self.vcp.contractions),
                "atr_pct": round(self._to_native(self.vcp.atr_pct), 2),
                "volume_dryup": self._to_native(self.vcp.volume_dryup),
                "latest_consolidation_pct": (
                    round(self._to_native(self.vcp.latest_consolidation_pct), 2)
                    if self.vcp.latest_consolidation_pct is not None
                    else None
                ),
            }
        if self.position_sizing:
            d["position_sizing"] = {
                "entry": round(self.position_sizing.entry_price, 2),
                "stop_loss": round(self.position_sizing.stop_loss_price, 2),
                "shares": self.position_sizing.shares_to_buy,
                "dollar_risk": round(self.position_sizing.dollar_risk_allowed, 2),
            }
        if self.canslim is not None:
            d["canslim"] = {
                "score": self.canslim.overall_score,
                "C_price_above_200ma": self._to_native(self.canslim.C_price_above_200ma),
                "A_acceleration": self._to_native(self.canslim.A_earnings_acceleration),
            }
        return d


class Pipeline:
    """End-to-end screening pipeline.

    Runs each ticker through all five modules and emits structured
    ``ScreeningResult`` entries.  A ticker is **approved** when:

    1. Trend Template passes (structural bull market).
    2. VCP pattern is detected (volatility contraction).
    3. The Fundamental Filter passes or gracefully falls back (no crash).
    """

    def __init__(
        self,
        equity: float = DEFAULT_EQUITY,
        risk_pct: float = DEFAULT_RISK_PCT,
    ) -> None:
        self.ingestion = DataIngestion()
        self.trend = TrendTemplateValidator()
        self.fundamental = MinerviniFundamentalEngine(min_score=3)
        self.vcp = VCPEngine()
        self.risk = RiskManager(equity=equity, risk_pct=risk_pct)

    # ------------------------------------------------------------------
    def screen(
        self,
        ticker: str,
        entry_price: Optional[float] = None,
        stop_loss_pct: float = 0.04,   # 4 % below entry
    ) -> ScreeningResult:
        """Run the full pipeline for a single *ticker*.

        Parameters
        ----------
        ticker :
            Yahoo Finance ticker symbol.
        entry_price :
            If ``None``, the latest closing price is used as the entry
            price.  Otherwise the caller-supplied price is used.
        stop_loss_pct :
            Fraction below entry price for the stop-loss
            (default 4 % → stop = entry × (1 - *stop_loss_pct*)).

        Returns
        -------
        ScreeningResult
        """
        result = ScreeningResult(ticker=ticker)

        # ---- 1. Data Ingestion --------------------------------------
        data = self.ingestion.fetch(ticker)
        if data is None:
            result.error = "Failed to fetch data."
            return result
        result.ohlcv_data = data

        # ---- 2. Trend Template --------------------------------------
        trend_result = self.trend.validate(data)
        result.trend_template = trend_result

        # ---- 3. Fundamental Filter ----------------------------------
        fundamental_result = self.fundamental.screen(ticker)
        result.minervini_fundamental = fundamental_result

        # ---- 4. VCP Engine ------------------------------------------
        vcp_result = self.vcp.analyse(data)
        result.vcp = vcp_result

        # ---- 5. Position Sizing -------------------------------------
        close = float(data.close.iloc[-1])
        ep = entry_price if entry_price is not None else close
        sl = ep * (1.0 - stop_loss_pct)

        try:
            sizing = self.risk.size_position(ticker, entry_price=ep, stop_loss_price=sl)
            result.position_sizing = sizing
        except ValueError as exc:
            logger.warning("Position sizing skipped for %s: %s", ticker, exc)

        # ---- 6. CAN SLIM Profile ------------------------------------
        result.canslim = self.risk.build_canslim_profile(
            ticker, trend_result, fundamental_result,
        )

        # ---- Approval -----------------------------------------------
        tt_pass = trend_result.passed
        vcp_detect = vcp_result.vcp_detected
        ff_pass = fundamental_result.overall_passed

        result.approved = tt_pass and vcp_detect and ff_pass

        logger.info(
            "Screen [%s]: approved=%s (TT=%s, FF=%s, VCP=%s)",
            ticker, result.approved, tt_pass, ff_pass, vcp_detect,
        )
        return result

    # ------------------------------------------------------------------
    def screen_watchlist(
        self,
        tickers: List[str],
        entry_price_overrides: Optional[Dict[str, float]] = None,
    ) -> List[ScreeningResult]:
        """Screen a list of tickers and return results."""
        results: List[ScreeningResult] = []
        overrides = entry_price_overrides or {}
        for t in tickers:
            ep = overrides.get(t)
            results.append(self.screen(t, entry_price=ep))
        return results


# ===================================================================
# Execution Block
# ===================================================================

def main() -> None:
    """Screen tickers and print structured summaries.

    Usage:
        python3 vcp_momentum_engine.py              # full embedded watchlist
        python3 vcp_momentum_engine.py AAPL MSFT COP  # specific tickers
    """
    raw_args = sys.argv[1:]
    if raw_args:
        # check help before upper-casing
        if any(a in ("-h", "--help") for a in raw_args):
            print("Usage: python3 vcp_momentum_engine.py [TICKER ...]")
            print()
            print("  Without arguments: screens the full embedded watchlist")
            print("  With arguments:    screens only the given ticker(s)")
            print()
            print("  Examples:")
            print("    python3 vcp_momentum_engine.py")
            print("    python3 vcp_momentum_engine.py AAPL MSFT COP")
            sys.exit(0)
        watchlist = [t.upper() for t in raw_args]
    else:
        watchlist = [
        "A", "AAPL", "ABBV", "ABEV", "ABNB", "ABT", "ACN", "ADBE", "ADI",
        "ADM", "ADP", "ADSK", "AEE", "AEM", "AEP", "AER", "AFL", "AFRM",
        "AIG", "AJG", "ALAB", "ALC", "ALNY", "AMAT", "AMD", "AME", "AMGN",
        "AMP", "AMRZ", "AMT", "AMZN", "ANET", "AON", "APD", "APH", "APP",
        "ARGX", "ARM", "AS", "ASX", "ATO", "AU", "AVB", "AVGO", "AWK",
        "AXON", "AXP", "AZO", "B", "BAM", "BAP", "BCH", "BDX", "BE",
        "BEKE", "BG", "BIIB", "BKNG", "BKR", "BLK", "BNS", "BNTX", "BR",
        "BRO", "BSBR", "BSQKZ", "BSX", "BURL", "BX", "CAH", "CARR", "CASY",
        "CAT", "CB", "CBOE", "CBRE", "CCEP", "CCI", "CCJ", "CCL", "CDNS",
        "CDW", "CEG", "CHD", "CHKP", "CHLSY", "CHRW", "CHT", "CHTR", "CI",
        "CIEN", "CINF", "CKHGF", "CL", "CLS", "CM", "CME", "CMG", "CMI",
        "CNC", "CNI", "CNP", "CNQ", "COHR", "COIN", "COP", "COR", "COST",
        "CP", "CPAY", "CPNG", "CPRT", "CQP", "CRCL", "CRDO", "CRH", "CRM",
        "CRWD", "CRWV", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH",
        "CTVA", "CVNA", "CVS", "CVX", "CW", "CYBR", "D", "DAL", "DASH",
        "DDOG", "DE", "DELL", "DG", "DGX", "DHI", "DHR", "DIA", "DIDIY",
        "DIS", "DKS", "DLTR", "DOV", "DRI", "DVN", "DXCM", "EA", "EBAY",
        "EC", "ECL", "ED", "EFX", "EIX", "EL", "ELV", "EME", "EMR", "EOG",
        "EQIX", "EQR", "EQT", "ES", "ESLT", "ESS", "ETN", "ETR", "EVRG",
        "EW", "EXAS", "EXC", "EXPD", "EXPE", "EXR", "FANG", "FAST", "FCX",
        "FDX", "FE", "FER", "FERG", "FICO", "FIG", "FIS", "FISV", "FIX",
        "FLEX", "FLUT", "FMX", "FNV", "FSLR", "FTI", "FTNT", "FTV", "FUTU",
        "GBTC", "GD", "GDDY", "GE", "GEHC", "GEV", "GFL", "GFS", "GIB",
        "GILD", "GIS", "GLD", "GLW", "GM", "GPC", "GPN", "GRMN", "GWRE",
        "GWW", "HAL", "HCA", "HD", "HDB", "HLT", "HOLX", "HON", "HOOD",
        "HPQ", "HSY", "HUBB", "HUBS", "HUM", "IAU", "IBKR", "IBM", "IBN",
        "ICE", "IDXX", "IFF", "ILMN", "IMO", "INCY", "INFY", "INSM",
        "INTC", "INTU", "INVH", "IOT", "IQV", "IR", "IRM", "ISRG", "IT",
        "ITUB", "ITW", "JBL", "JCI", "JNJ", "K", "KB", "KDP", "KEP",
        "KEYS", "KHC", "KLAC", "KMB", "KO", "KR", "KVUE", "L", "LDOS",
        "LH", "LHX", "LIN", "LITE", "LLY", "LMT", "LNG", "LNT", "LOW",
        "LPLA", "LRCX", "LULU", "LVS", "LYV", "MA", "MAR", "MCD", "MCK",
        "MCO", "MDB", "MDLZ", "MDT", "MDY", "MEDP", "MELI", "META", "MKL",
        "MLM", "MMC", "MMM", "MNST", "MO", "MPC", "MPWR", "MRK", "MRVL",
        "MSCI", "MSFT", "MSI", "MTD", "MU", "NBIS", "NDAQ", "NET", "NFLX",
        "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTR", "NTRA",
        "NU", "NUE", "NVDA", "NVO", "NVR", "NXPI", "O", "ODFL", "OKE",
        "ON", "ORCL", "ORLY", "OTIS", "OWL", "PANW", "PAYX", "PCAR", "PDD",
        "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PINS",
        "PKG", "PLTR", "PM", "PNC", "PNR", "PODD", "PPG", "PPL", "PSKY",
        "PSTG", "PSX", "PTC", "PWR", "PYPL", "QCOM", "QQQ", "QSR", "RACE",
        "RBA", "RBLX", "RCL", "RDDT", "REGN", "RIVN", "RKLB", "RKT", "RL",
        "ROK", "ROL", "ROP", "ROST", "RPRX", "RSG", "RTNTF", "RTX", "SATS",
        "SBAC", "SBS", "SBUX", "SCCO", "SE", "SGI", "SHG", "SHOP", "SHW",
        "SLB", "SMCI", "SNA", "SNDK", "SNOW", "SNPS", "SOFI", "SPGI",
        "SPOT", "SPY", "SQM", "SSNC", "STE", "STLA", "STLD", "STX", "STZ",
        "SU", "SW", "SYK", "SYM", "SYY", "TDG", "TDY", "TEAM", "TEL",
        "TER", "TGT", "THC", "TJX", "TLK", "TLN", "TMO", "TOST", "TPL",
        "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT",
        "TTD", "TTWO", "TU", "TW", "TWLO", "TXN", "TYL", "UAL", "UBER",
        "UBS", "UI", "ULS", "ULTA", "UMC", "UNH", "UNP", "UPS", "URI",
        "UTHR", "UTX", "V", "VALE", "VEEV", "VG", "VICI", "VIK", "VIV",
        "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRT", "VRTX", "VST", "VTR",
        "VZ", "WAB", "WAT", "WBD", "WCN", "WDAY", "WDC", "WEC", "WELL",
        "WIT", "WM", "WMB", "WMT", "WPM", "WSM", "WST", "WTW", "XOM",
        "XYL", "YPF", "YUM", "YUMC", "ZBH", "ZM", "ZS", "ZTS",
    ]
    pipeline = Pipeline(equity=100_000.0, risk_pct=0.003)

    print(f"{'='*72}")
    print(f"  VCP Momentum Engine — Minervini + CAN SLIM Screening")
    print(f"  Watchlist: {', '.join(watchlist)}")
    print(f"  Account Equity: $100,000  |  Risk/Trade: 0.3%")
    print(f"{'='*72}")
    print()

    results = pipeline.screen_watchlist(watchlist)

    approved = [r for r in results if r.approved]
    failed = [r for r in results if not r.approved]

    # --- Approved setups ---------------------------------------------
    print(f"\n{'─'*72}")
    print(f"  ✅ APPROVED SETUPS ({len(approved)})")
    print(f"{'─'*72}")
    for r in approved:
        print(f"  {r.ticker}")
        js = r.to_dict()
        for k, v in js.items():
            if k == "ticker":
                continue
            print(f"    {k}: {v}")
        print()

    # --- Failed / skipped --------------------------------------------
    print(f"\n{'─'*72}")
    print(f"  ❌ NOT APPROVED ({len(failed)})")
    print(f"{'─'*72}")
    for r in failed:
        print(f"  {r.ticker}")
        js = r.to_dict()
        for k, v in js.items():
            if k == "ticker":
                continue
            print(f"    {k}: {v}")
        print()

    # --- Summary statistics ------------------------------------------
    total = len(results)
    print(f"\n{'─'*72}")
    print(f"  SUMMARY")
    print(f"{'─'*72}")
    print(f"  Screened : {total}")
    print(f"  Approved : {len(approved)}")
    print(f"  Failures : {len(failed)}")
    print(f"  Pass rate: {len(approved)/total*100:.0f}%" if total else "  No data.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()