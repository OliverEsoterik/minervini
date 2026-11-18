# Minervini RS Rating Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Minervini-style Relative Strength (RS) Rating as a hard screening gate (RS ≥ 70) to `batch_screen.py`.

**Architecture:** New standalone `rs_rating.py` module computes weighted multi-period returns, ranks them into percentiles within the watchlist, and caches results. `Pipeline.screen()` accepts an optional `rs_rank` parameter to gate on. `batch_screen.py` runs a pre-phase to compute RS before the parallel screening workers.

**Tech Stack:** Python 3.10+, yfinance, pandas, numpy, json (stdlib)

## Global Constraints

- All data from `yfinance` — no mock data, no hardcoded prices
- RS Rating is a percentile 0-99, NOT RSI
- Watchlist-only ranking (~500 large-cap tickers)
- RS gate: hard reject if < 70 (Minervini minimum)
- Cache RS results to JSON with 1-day TTL (no re-download on every run)
- `rs_rank=None` skips the RS gate (backward compatibility for CLI usage)
- Follow existing code style: dataclass results, structured logging, type hints

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `rs_rating.py` | **Create** | RSModule: batch download → weighted returns → percentile ranking → cache |
| `vcp_momentum_engine.py` | **Modify** | Pipeline.screen() gets `rs_rank` parameter + gate. ScreeningResult gets `rs_rank` field. |
| `batch_screen.py` | **Modify** | Pre-phase RS computation + pass ranks to screen_one + output |
| `README.md` | **Modify** | Document RS module, threshold, usage |

---

### Task 1: Create `rs_rating.py` — RSModule with fetch_returns() and _compute_weighted_return()

**Files:**
- Create: `rs_rating.py`

**Interfaces:**
- Consumes: nothing (standalone)
- Produces: `RSModule.fetch_returns(tickers: List[str]) -> Dict[str, Optional[float]]` — raw weighted score per ticker (or None if insufficient data)

- [ ] **Step 1: Write `rs_rating.py` with imports, constants, RSResult dataclass, and RSModule class shell**

Create the file with the following code. Only the scaffolding + fetch_returns:

```python
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
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
```

- [ ] **Step 2: Run syntax check**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('rs_rating.py').read()); print('✅ Syntax OK')"
```
Expected: `✅ Syntax OK`

- [ ] **Step 3: Run import check**

```bash
cd /home/oliver/minervi && python3 -c "from rs_rating import RSModule, RSResult; print('✅ Import OK')"
```
Expected: `✅ Import OK`

- [ ] **Step 4: Commit**

```bash
cd /home/oliver/minervi && git add rs_rating.py && git commit -m "feat: add RSModule.fetch_returns() with weighted return computation"
```

---

### Task 2: Add `compute_percentile_ranks()` to RSModule

**Files:**
- Modify: `rs_rating.py` — add method to RSModule class

**Interfaces:**
- Consumes: `RSModule.fetch_returns()` output (Dict[str, Optional[float]])
- Produces: `RSModule.compute_percentile_ranks(returns) -> Dict[str, float]` — percentile 0-99 per ticker

- [ ] **Step 1: Add `compute_percentile_ranks` method to RSModule**

Append to the RSModule class (after `_compute_weighted_return`):

```python
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
```

- [ ] **Step 2: Run syntax check**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('rs_rating.py').read()); print('✅ Syntax OK')"
```
Expected: `✅ Syntax OK`

- [ ] **Step 3: Quick functional test with dummy data**

```bash
cd /home/oliver/minervi && python3 -c """
from rs_rating import RSModule

# Test with 5 dummy returns
returns = {
    'AAPL': 0.25,  # 25% return
    'MSFT': 0.15,
    'GOOG': 0.05,
    'AMZN': -0.05,
    'META': -0.15,
}
ranks = RSModule.compute_percentile_ranks(returns)
print(f'AAPL: {ranks[\"AAPL\"]} (expected ~99.0)')
print(f'META: {ranks[\"META\"]} (expected ~0.0)')
print(f'GOOG: {ranks[\"GOOG\"]} (expected ~59.4)')
assert ranks['AAPL'] >= ranks['GOOG'] >= ranks['META']
assert ranks['AAPL'] == 99.0
assert ranks['META'] >= 0.0
print('✅ Ranking correct')
"""
```
Expected: Ranking values plus `✅ Ranking correct`

- [ ] **Step 4: Commit**

```bash
cd /home/oliver/minervi && git add rs_rating.py && git commit -m "feat: add compute_percentile_ranks() to RSModule"
```

---

### Task 3: Add caching + `get_rs_ratings()` orchestrator to RSModule

**Files:**
- Modify: `rs_rating.py` — add caching methods and `get_rs_ratings()` orchestrator

**Interfaces:**
- Consumes: `RSModule.fetch_returns()` + `RSModule.compute_percentile_ranks()`
- Produces: `RSModule.get_rs_ratings(tickers, force_refresh) -> Dict[str, float]` — the public entry point

- [ ] **Step 1: Add `_load_cache`, `_save_cache`, and `get_rs_ratings` methods**

Append to the RSModule class (after `compute_percentile_ranks`):

```python
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
```

- [ ] **Step 2: Run syntax check**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('rs_rating.py').read()); print('✅ Syntax OK')"
```
Expected: `✅ Syntax OK`

- [ ] **Step 3: Run import check**

```bash
cd /home/oliver/minervi && python3 -c "from rs_rating import RSModule; print('✅ Import OK')"
```
Expected: `✅ Import OK`

- [ ] **Step 4: Quick cache test (with dummy path)**

```bash
cd /home/oliver/minervi && python3 -c """
import json, os
from rs_rating import RSModule
# Test that cache save/load works with a temp path
m = RSModule(cache_dir='/tmp/rs_cache_test')
m._save_cache({'AAPL': 99.0}, {'AAPL': 0.25})
loaded = m._load_cache()
assert loaded == {'AAPL': 99.0}, f'Got {loaded}'
# Clean up
import shutil
shutil.rmtree('/tmp/rs_cache_test', ignore_errors=True)
print('✅ Cache test passed')
"""
```
Expected: `✅ Cache test passed`

- [ ] **Step 5: Commit**

```bash
cd /home/oliver/minervi && git add rs_rating.py && git commit -m "feat: add RS rating cache and get_rs_ratings() orchestrator"
```

---

### Task 4: Add RS gate to Pipeline.screen() + ScreeningResult

**Files:**
- Modify: `vcp_momentum_engine.py` — `Pipeline.screen()` signature, `ScreeningResult` dataclass, `ScreeningResult.to_dict()`

**Interfaces:**
- Consumes: `rs_rank: Optional[float]` from caller (`batch_screen.py`)
- Produces: `ScreeningResult.rs_rank` field, RS gate rejection in `Pipeline.screen()`

- [ ] **Step 1: Add `rs_rank: Optional[float] = None` to ScreeningResult**

Find the `ScreeningResult` dataclass definition in `vcp_momentum_engine.py` and add the field:

```python
class ScreeningResult:
    """Aggregate result for one ticker through the full pipeline."""
    ticker: str
    error: Optional[str] = None
    rs_rank: Optional[float] = None   # NEW — add after `error`
    ohlcv_data: Optional[OHLCVData] = None
    ...
```

- [ ] **Step 2: Verify the insertion point**

Run:
```bash
cd /home/oliver/minervi && grep -n "class ScreeningResult" vcp_momentum_engine.py
```
Expected: shows line number. Read around that line to find where to insert.

- [ ] **Step 3: Edit ScreeningResult to add the rs_rank field**

Use the edit tool. Add `rs_rank: Optional[float] = None` after the `error` field. For example, if the current text is:

```
class ScreeningResult:
    """Aggregate result for one ticker through the full pipeline."""
    ticker: str
    error: Optional[str] = None
    ohlcv_data: Optional[OHLCVData] = None
```

Change to:

```
class ScreeningResult:
    """Aggregate result for one ticker through the full pipeline."""
    ticker: str
    error: Optional[str] = None
    rs_rank: Optional[float] = None
    ohlcv_data: Optional[OHLCVData] = None
```

- [ ] **Step 4: Add rs_rank to ScreeningResult.to_dict()**

Find the `to_dict()` method in ScreeningResult. Add rs_rank to the dict. The exact block depends on the current code. Look for the section that builds the dict. Add after the error block:

```python
d["rs_rank"] = self._to_native(self.rs_rank) if self.rs_rank is not None else None
```

- [ ] **Step 5: Add `rs_rank` parameter to Pipeline.screen() and insert the RS gate**

Find the `Pipeline.screen()` method signature. Currently:

```python
    def screen(
        self,
        ticker: str,
        entry_price: Optional[float] = None,
        stop_loss_pct: float = 0.04,
    ) -> ScreeningResult:
```

Change the signature to add `rs_rank`:

```python
    def screen(
        self,
        ticker: str,
        entry_price: Optional[float] = None,
        stop_loss_pct: float = 0.04,
        rs_rank: Optional[float] = None,
    ) -> ScreeningResult:
```

Then find the RS gate insertion point — right after the 52-week high proximity gate. Look for the block that checks `high_52` (approx line 1500). After the 52-week high gate closes, add the RS gate:

After the 52-week high gate block:
```python
        # ---- 0c. RS Rating Gate ----------------------------------------
        if rs_rank is not None and rs_rank < 70:
            result.error = (
                f"RS Rating {rs_rank:.0f} < 70 (Minervini minimum)"
            )
            logger.info(
                "RS gate [%s]: FAIL — rank %.0f < 70", ticker, rs_rank,
            )
            return result
        result.rs_rank = rs_rank
        if rs_rank is not None:
            logger.info(
                "RS gate [%s]: PASS (rank %.0f)", ticker, rs_rank,
            )
```

- [ ] **Step 6: Run syntax check**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('vcp_momentum_engine.py').read()); print('✅ Syntax OK')"
```
Expected: `✅ Syntax OK`

- [ ] **Step 7: Run import check**

```bash
cd /home/oliver/minervi && python3 -c "from vcp_momentum_engine import Pipeline; print('✅ Import OK')"
```
Expected: `✅ Import OK`

- [ ] **Step 8: Verify backward compatibility (CLI usage without RS)**

```bash
cd /home/oliver/minervi && python3 vcp_momentum_engine.py AAPL 2>&1 | head -20
```
Expected: Works without RS gate (AAPL shows as not approved due to other filters, but no RS error).

- [ ] **Step 9: Commit**

```bash
cd /home/oliver/minervi && git add vcp_momentum_engine.py && git commit -m "feat: add rs_rank parameter and RS gate to Pipeline.screen()"
```

---

### Task 5: Add RS pre-phase to batch_screen.py

**Files:**
- Modify: `batch_screen.py` — add RS pre-computation, modified `screen_one()`, RS output

**Interfaces:**
- Consumes: `RSModule.get_rs_ratings()` from Task 3, `Pipeline.screen()` with rs_rank from Task 4
- Produces: RS-filtered screening results with RS rank displayed

- [ ] **Step 1: Add import and RS pre-phase before ThreadPoolExecutor**

Find the `main()` function in `batch_screen.py`. Right after the opening banner print, before the ThreadPoolExecutor, add:

```python
    # ---- RS Rating pre-phase ---------------------------------------
    from rs_rating import RSModule
    print("  Pre-computing RS ratings for watchlist...")
    rs_module = RSModule()
    rs_ranks = rs_module.get_rs_ratings(WATCHLIST)
    valid_rs = {t: r for t, r in rs_ranks.items() if r is not None}
    print(f"  RS computed: {len(valid_rs)}/{total} tickers")
```

- [ ] **Step 2: Modify `screen_one()` to accept and pass rs_rank**

Find the existing `screen_one()` function. Change:

```python
def screen_one(ticker: str, pipeline: Pipeline) -> ScreeningResult:
    """Screen a single ticker — returns result with error handling."""
    try:
        return pipeline.screen(ticker)
    except Exception as exc:
        from vcp_momentum_engine import ScreeningResult
        r = ScreeningResult(ticker=ticker)
        r.error = f"Exception: {exc}"
        return r
```

To:

```python
def screen_one(
    ticker: str,
    pipeline: Pipeline,
    rs_ranks: Dict[str, float],
) -> ScreeningResult:
    """Screen a single ticker — returns result with error handling."""
    try:
        rs_rank = rs_ranks.get(ticker)
        return pipeline.screen(ticker, rs_rank=rs_rank)
    except Exception as exc:
        from vcp_momentum_engine import ScreeningResult
        r = ScreeningResult(ticker=ticker)
        r.error = f"Exception: {exc}"
        return r
```

- [ ] **Step 3: Add `from typing import Dict` if not already imported**

Check if `Dict` is already imported in `batch_screen.py`. If not, add it to the existing import:

```python
from typing import List, Dict
```

- [ ] **Step 4: Update the ThreadPoolExecutor submission to pass rs_ranks**

Find:
```python
        fut_map = {
            pool.submit(screen_one, t, pipeline): t for t in WATCHLIST
        }
```

Change to:
```python
        fut_map = {
            pool.submit(screen_one, t, pipeline, rs_ranks): t
            for t in WATCHLIST
        }
```

- [ ] **Step 5: Show RS rank in approved setups output**

Find the approved setups print section. In the block that prints each approved ticker, add RS rank. Look for:
```python
            print(f"    RS Rank      : {r.rs_rank:.0f}")  # after CAN SLIM score
```

Find the section:
```python
            if ll:
                print(f"    ─────────────────────────────────────────────")
                print(f"    Entry Price    : ${ll.entry_price:.2f}")
                ...
                print(f"    CAN SLIM score : {r.canslim.overall_score}/7")
```

Add RS rank after the CAN SLIM line:
```python
            if r.rs_rank is not None:
                print(f"    RS Rank        : {r.rs_rank:.0f}/99")
```

- [ ] **Step 6: Show RS gate near-misses in the output**

In the error section, the RS gate errors will appear automatically (they set `result.error`). But also add a near-miss section for RS failures specifically. Add after the near-miss sections:

```python
    # ---- RS Gate Rejections ------------------------------------------
    rs_rejected = [
        r for r in failed
        if r.error and "RS Rating" in (r.error or "")
    ]
    if rs_rejected:
        print(f"\n{'─'*72}")
        print(f"  📊 RS GATE REJECTED ({len(rs_rejected)}) — rank < 70")
        print(f"{'─'*72}")
        for r in sorted(rs_rejected, key=lambda x: x.ticker):
            print(f"  {r.ticker:8s} → {r.error}")
```

- [ ] **Step 7: Run syntax check**

```bash
cd /home/oliver/minervi && python3 -c "import ast; ast.parse(open('batch_screen.py').read()); print('✅ Syntax OK')"
```
Expected: `✅ Syntax OK`

- [ ] **Step 8: Run quick import check**

```bash
cd /home/oliver/minervi && python3 -c "from batch_screen import screen_one; print('✅ Import OK')"
```
Expected: `✅ Import OK`

- [ ] **Step 9: Commit**

```bash
cd /home/oliver/minervi && git add batch_screen.py && git commit -m "feat: add RS rating pre-phase to batch_screen.py"
```

---

### Task 6: Update README

**Files:**
- Modify: `README.md` — document RS module, threshold, usage

- [ ] **Step 1: Add RS Rating section to README**

Open `README.md` and find the section that describes the filters/features. Add a new section or update the table. For example, update the filter table:

```markdown
| Filter | What it checks | Method |
|--------|----------------|--------|
| **Trend Template** | Is the stock in a structural bull market? | Close > SMA-50 > SMA-200, both SMAs sloping up |
| **Fundamental Filter** | Is revenue accelerating YoY? | Quarterly revenue growth > 20% baseline |
| **VCP Engine** | Is there a volatility contraction pattern? | Tightening drawdowns, ATR compression, volume dry-up |
| **RS Rating** | Is the stock outperforming peers? | Weighted multi-period return percentile (≥ 70 required) |
```

And add a new subsection:

```markdown
### RS Rating (Minervini)

The engine computes a Relative Strength (RS) Rating for every ticker in the
watchlist before screening. The RS Rating is a percentile rank (0-99)
based on weighted multi-period price returns:

- **6-month return** × 50%
- **3-month return** × 30%
- **1-month return** × 20%

Stocks with RS Rank **< 70** are rejected immediately — they are
underperforming their large-cap peers and do not qualify as Minervini
superperformance candidates.

RS ratings are cached to `rs_cache/rs_ratings.json` with a 1-day TTL
to avoid re-downloading ~500 tickers on every run.

**Usage:** The RS pre-phase runs automatically in `batch_screen.py`.
Standalone `vcp_momentum_engine.py` CLI usage skips RS (backward
compatibility).
```

- [ ] **Step 2: Commit**

```bash
cd /home/oliver/minervi && git add README.md && git commit -m "docs: document RS Rating filter in README"
```

---

### Task 7: Full pipeline verification

**Files:**
- Run: `batch_screen.py` on a 10-ticker subset (not the full 500)

- [ ] **Step 1: Create a small test watchlist and run screening**

```bash
cd /home/oliver/minervi && python3 -c """
from rs_rating import RSModule
from vcp_momentum_engine import Pipeline

# Test on a small subset
watchlist = ['AAPL', 'MSFT', 'NVDA', 'META', 'AMD', 'INTC', 'KO', 'PEP', 'XOM', 'CVX']

# RS pre-phase
rs = RSModule()
ranks = rs.get_rs_ratings(watchlist, force_refresh=True)
print('RS Ratings:')
for t, r in sorted(ranks.items(), key=lambda x: -x[1]):
    print(f'  {t}: {r:.0f}/99')

# Pipeline with RS
pipeline = Pipeline()
for t in watchlist:
    r = pipeline.screen(t, rs_rank=ranks.get(t))
    status = '✅ APPROVED' if r.approved else '❌ REJECTED'
    gate = ''
    if r.error and 'RS Rating' in r.error:
        gate = ' (RS gate)'
    elif not r.approved and not r.error:
        gate = ' (other filters)'
    elif r.error:
        gate = f' ({r.error[:60]})'
    print(f'  {status} {t}{gate}')
"""
```

Expected: Some stocks rejected by RS gate (< 70), some pass. At least one shows APPROVED with RS gate passing.

- [ ] **Step 2: Verify the RS module has a clean `__main__` for standalone testing**

First check if it already has one. If not, add a simple standalone mode to `rs_rating.py` at the end:

```python
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
```

- [ ] **Step 3: Run standalone rs_rating.py**

```bash
cd /home/oliver/minervi && python3 rs_rating.py AAPL MSFT NVDA META
```
Expected: Shows a table with tickers and RS ranks.

- [ ] **Step 4: Verify the rs_cache directory was created**

```bash
cd /home/oliver/minervi && ls -la rs_cache/
```
Expected: Shows `rs_ratings.json` file.

- [ ] **Step 5: Run second time (should use cache)**

```bash
cd /home/oliver/minervi && python3 rs_rating.py AAPL MSFT NVDA META
```
Expected: Log shows "Using cached RS ratings" — much faster.

- [ ] **Step 6: Force refresh to verify re-download**

```bash
cd /home/oliver/minervi && python3 -c "
from rs_rating import RSModule

RSModule().get_rs_ratings(['AAPL', 'MSFT'], force_refresh=True)
"
```
Expected: Downloads fresh data, logs "Computing RS ratings..."

- [ ] **Step 7: Final state check**

```bash
cd /home/oliver/minervi && echo "=== Git status ===" && git status && echo && echo "=== Syntax ===" && python3 -c "import ast; ast.parse(open('rs_rating.py').read()); ast.parse(open('vcp_momentum_engine.py').read()); ast.parse(open('batch_screen.py').read()); print('All files syntax OK')" && echo && echo "=== Imports ===" && python3 -c "from rs_rating import RSModule; from vcp_momentum_engine import Pipeline; print('All imports OK')"
```

- [ ] **Step 8: Commit any remaining changes**

```bash
cd /home/oliver/minervi && git add -A && git commit -m "feat: add standalone main() to rs_rating.py, final adjustments"
```
