#!/usr/bin/env python3
"""
Minervini Relative Strength (RS) Rating Module.

Computes a percentile RS Rank (0-99) for each ticker in a watchlist
using weighted multi-period price returns, ranking within the list.

RS Rank >= 70 is Minervini's minimum for superperformance candidates.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("rs_rating")
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
DEFAULT_CACHE_DIR: str = "rs_cache"
CACHE_FILE: str = "rs_ratings.json"
CACHE_TTL_HOURS: int = 24
# Minimum tickers required for a reliable percentile ranking
MIN_TICKERS_FOR_RANKING: int = 20

# RS Rating gate threshold (Minervini minimum)
RS_MINIMUM: float = 70.0

# Weight windows (trading days)
WINDOW_6M: int = 126
WINDOW_3M: int = 63
WINDOW_1M: int = 21

# Weights for multi-period return
W_6M: float = 0.50
W_3M: float = 0.30
W_1M: float = 0.20


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RSResult:
    """Result for one ticker's RS rating."""
    ticker: str
    rs_rank: float  # 0-99, NaN if unavailable
    weighted_return: Optional[float] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# RSModule
# ---------------------------------------------------------------------------


class RSModule:
    """Compute Minervini-style RS Ratings for a watchlist.

    RS Rating = weighted multi-period return, ranked as a percentile
    against all tickers in the watchlist.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self._cache_path = os.path.join(cache_dir, CACHE_FILE)

    # ------------------------------------------------------------------
    def fetch_returns(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Download OHLCV data and compute weighted returns for *tickers*.

        Uses a single batch ``yfinance.download`` call. For any tickers
        that fail (missing data, network error), logs a warning and stores
        None — they are silently skipped during ranking.

        Returns
        -------
        Dict[str, Optional[float]]
            ``{ticker: weighted_return}`` where values are None for tickers
            with insufficient price history (< 21 trading days available).
        """
        if not tickers:
            logger.warning("fetch_returns called with empty ticker list")
            return {}

        logger.info("Downloading %d tickers for RS computation...", len(tickers))

        try:
            raw = yf.download(
                tickers=tickers,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.error("yfinance batch download failed: %s", exc)
            # Fall back to individual downloads for each ticker
            return self._fetch_returns_individual(tickers)

        if raw is None or raw.empty:
            logger.warning("yfinance batch download returned empty data")
            return self._fetch_returns_individual(tickers)

        results: Dict[str, Optional[float]] = {}

        # yfinance with group_by="ticker" returns a MultiIndex DataFrame
        # where columns are (ticker, field). For a single ticker, it may
        # return a flat DataFrame.
        if isinstance(raw.columns, pd.MultiIndex):
            tickers_found = set(raw.columns.get_level_values(0))
            for t in tickers:
                if t not in tickers_found:
                    logger.warning("%s: not found in batch download", t)
                    results[t] = None
                    continue
                try:
                    close = raw[(t, "Close")].dropna()
                    results[t] = self._compute_weighted_return(close)
                except Exception as exc:
                    logger.warning("%s: failed to compute return: %s", t, exc)
                    results[t] = None
        else:
            # Single ticker — flat columns
            close = raw["Close"].dropna() if "Close" in raw.columns else pd.Series()
            results[tickers[0]] = self._compute_weighted_return(close)

        return results

    # ------------------------------------------------------------------
    def _fetch_returns_individual(
        self, tickers: List[str]
    ) -> Dict[str, Optional[float]]:
        """Fallback: download each ticker individually."""
        logger.info("Falling back to individual downloads for %d tickers...", len(tickers))
        results: Dict[str, Optional[float]] = {}
        for t in tickers:
            try:
                data = yf.download(
                    tickers=t,
                    period="6mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if data is not None and not data.empty:
                    close = data["Close"].dropna() if "Close" in data.columns else pd.Series()
                    results[t] = self._compute_weighted_return(close)
                else:
                    results[t] = None
            except Exception as exc:
                logger.warning("%s: individual download failed: %s", t, exc)
                results[t] = None
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_weighted_return(close: pd.Series) -> Optional[float]:
        """Compute weighted multi-period return from a close price series.

        Full windows (>= 126 trading days):
            return = P6 x 0.50 + P3 x 0.30 + P1 x 0.20

        Partial windows (63-125 trading days):
            return = P3 x 0.60 + P1 x 0.40

        Minimal (21-62 trading days):
            return = P1 x 1.0

        Insufficient (< 21 trading days):
            return None

        Parameters
        ----------
        close : pd.Series
            Daily closing prices, oldest-first (index sorted chronologically).

        Returns
        -------
        Optional[float]
            Weighted return, or None if insufficient data.
        """
        vals = close.values.astype(float)
        n = len(vals)

        if n < WINDOW_1M:  # 21 trading days minimum
            return None

        current = vals[-1]

        p6: Optional[float] = None
        p3: Optional[float] = None
        p1: Optional[float] = None

        if n >= WINDOW_6M:  # 126 trading days
            prev_6m = vals[-WINDOW_6M]
            p6 = (current - prev_6m) / prev_6m if prev_6m != 0 else 0.0

        if n >= WINDOW_3M:  # 63 trading days
            prev_3m = vals[-WINDOW_3M]
            p3 = (current - prev_3m) / prev_3m if prev_3m != 0 else 0.0

        if n >= WINDOW_1M:  # 21 trading days
            prev_1m = vals[-WINDOW_1M]
            p1 = (current - prev_1m) / prev_1m if prev_1m != 0 else 0.0

        # Redistribute weights based on available windows
        if p6 is not None and p3 is not None and p1 is not None:
            return p6 * W_6M + p3 * W_3M + p1 * W_1M
        elif p3 is not None and p1 is not None:
            return p3 * (W_3M / (W_3M + W_1M)) + p1 * (W_1M / (W_3M + W_1M))
        elif p1 is not None:
            return p1
        else:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def compute_percentile_ranks(
        returns: Dict[str, Optional[float]],
    ) -> Dict[str, float]:
        """Rank weighted returns into percentiles 0-99.

        The ticker with the highest return gets RS = 99.0
        The ticker with the lowest return gets RS = 0.0
        Ties share the same percentile.

        Parameters
        ----------
        returns : Dict[str, Optional[float]]
            ``{ticker: weighted_return_or_None}`` from ``fetch_returns()``.

        Returns
        -------
        Dict[str, float]
            ``{ticker: rs_rank}`` — only tickers with non-None returns.
            Returns empty dict if fewer than MIN_TICKERS_FOR_RANKING (20)
            valid returns — the ranking is unreliable below that threshold.
        """
        # Filter out None values (IPOs, failed fetches)
        valid: Dict[str, float] = {
            t: r for t, r in returns.items() if r is not None
        }

        if len(valid) < MIN_TICKERS_FOR_RANKING:
            logger.warning(
                "Only %d valid returns (need >= %d for reliable ranking). "
                "Returning empty.",
                len(valid),
                MIN_TICKERS_FOR_RANKING,
            )
            return {}

        # Sort by weighted return descending (best performer first)
        sorted_tickers = sorted(valid.keys(), key=lambda t: valid[t], reverse=True)
        n = len(sorted_tickers)

        ranks: Dict[str, float] = {}
        for i, ticker in enumerate(sorted_tickers):
            # Percentile: top stock = 99, bottom = 0
            # Formula: (N - rank_position) / N * 99
            percentile = (n - i) / n * 99
            ranks[ticker] = round(percentile, 1)

        logger.info(
            "RS ranking complete: %d tickers ranked (best=%.1f, worst=%.1f)",
            len(ranks),
            max(ranks.values()),
            min(ranks.values()),
        )
        return ranks
