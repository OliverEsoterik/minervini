#!/usr/bin/env python3
"""
Minervini Relative Strength (RS) Rating Module.

Computes a percentile RS Rank (0-99) for each ticker in a watchlist
using weighted multi-period price returns, ranking within the list.

RS Rank >= 70 is Minervini's minimum for superperformance candidates.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
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
        The ticker with the lowest return gets RS = 99/n (approaches 0.0 as universe grows)
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
        i = 0
        while i < n:
            # Find all tickers in this tie group
            group_returns = valid[sorted_tickers[i]]
            j = i
            while j < n and valid[sorted_tickers[j]] == group_returns:
                j += 1
            # Same percentile for the whole group
            percentile = (n - i) / n * 99
            for k in range(i, j):
                ranks[sorted_tickers[k]] = round(percentile, 1)
            i = j  # Skip ahead by group size

        logger.info(
            "RS ranking complete: %d tickers ranked (best=%.1f, worst=%.1f)",
            len(ranks),
            max(ranks.values()),
            min(ranks.values()),
        )
        return ranks

    # ------------------------------------------------------------------
    def _load_cache(self) -> Optional[Dict[str, float]]:
        """Load cached RS ratings if the cache exists and is fresh enough.

        Cache is valid if its ``fetched_at`` timestamp is less than
        ``CACHE_TTL_HOURS`` (24) hours old.

        Returns
        -------
        Optional[Dict[str, float]]
            ``{ticker: rs_rank}`` if cache was fresh, None otherwise.
        """
        if not os.path.exists(self._cache_path):
            return None

        try:
            with open(self._cache_path, "r") as f:
                data = json.load(f)

            fetched_at_str = data.get("fetched_at")
            if not fetched_at_str:
                return None

            fetched_at = datetime.fromisoformat(fetched_at_str)
            age = datetime.now() - fetched_at

            if age.total_seconds() < CACHE_TTL_HOURS * 3600:
                logger.info(
                    "Using cached RS ratings (age: %.0f min)",
                    age.total_seconds() / 60,
                )
                return data.get("ratings", {})
            else:
                logger.info(
                    "Cache expired (age: %.0f hours)",
                    age.total_seconds() / 3600,
                )
                return None

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.warning("Cache load failed for %s: %s", self._cache_path, exc)
            return None

    # ------------------------------------------------------------------
    def _save_cache(
        self,
        ratings: Dict[str, float],
        scores: Dict[str, Optional[float]],
    ) -> None:
        """Save RS ratings and raw scores to a JSON cache file.

        Parameters
        ----------
        ratings : Dict[str, float]
            ``{ticker: rs_rank}``
        scores : Dict[str, Optional[float]]
            ``{ticker: weighted_return_or_None}`` — includes failures (None)
            for diagnostics.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare scores for JSON (round floats, replace None with null)
        clean_scores: Dict[str, Optional[float]] = {
            k: (round(v, 6) if v is not None else None)
            for k, v in scores.items()
        }

        data = {
            "fetched_at": datetime.now().isoformat(),
            "ratings": ratings,
            "scores": clean_scores,
        }

        try:
            with open(self._cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("RS ratings cached to %s", self._cache_path)
        except OSError as exc:
            logger.error("Failed to write cache %s: %s", self._cache_path, exc)

    # ------------------------------------------------------------------
    def get_rs_ratings(
        self,
        tickers: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, float]:
        """Public entry point: fetch, rank, and return RS ratings.

        1. Check cache (unless ``force_refresh=True``).
        2. If cache miss or expired: download data, compute returns, rank.
        3. Cache the result.
        4. Return ``{ticker: rs_rank}``.

        Parameters
        ----------
        tickers : List[str]
            Ticker symbols to rank.
        force_refresh : bool, optional
            If True, bypass cache and re-download (default False).

        Returns
        -------
        Dict[str, float]
            ``{ticker: rs_rank}``.  May be empty if ranking failed.
        """
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                return cached

        logger.info("Computing RS ratings for %d tickers...", len(tickers))
        returns = self.fetch_returns(tickers)
        ranks = self.compute_percentile_ranks(returns)

        if ranks:
            self._save_cache(ranks, returns)

        valid_count = len(ranks)
        logger.info(
            "RS ratings ready: %d/%d tickers ranked",
            valid_count,
            len(tickers),
        )
        return ranks


# ===================================================================
# Standalone execution
# ===================================================================


def main() -> None:
    """Compute and print RS ratings for tickers passed on the command line."""
    import sys
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 rs_rating.py TICKER [TICKER ...]")
        print("  Computes and prints RS ratings for the given tickers.")
        sys.exit(0)

    module = RSModule()
    ranks = module.get_rs_ratings(args, force_refresh=True)
    if ranks:
        print(f"{'Ticker':8s} {'RS Rank':>8s}")
        print("-" * 18)
        for t in sorted(args):
            r = ranks.get(t, None)
            if r is not None:
                print(f"{t:8s} {r:>7.0f}/99")
            else:
                print(f"{t:8s} {'N/A':>8s}")
    else:
        print("No RS ratings computed. Check logs for details.")


if __name__ == "__main__":
    main()
