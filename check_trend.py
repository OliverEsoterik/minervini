#!/usr/bin/env python3
"""Check Trend Template status — HOLD or SELL."""

import sys
from typing import List, Optional, Tuple

from vcp_momentum_engine import DataIngestion, TrendTemplateValidator


def check_ticker(
    ticker: str,
    entry_price: Optional[float] = None,
    high_watermark: Optional[float] = None,
) -> None:
    """Check one ticker and print a clear HOLD / SELL verdict."""
    ingestion = DataIngestion()
    trend = TrendTemplateValidator()

    print(f"\n{'─'*60}")
    print(f"  {ticker}")
    print(f"{'─'*60}")

    # --- fetch data ------------------------------------------------
    data = ingestion.fetch(ticker)
    if data is None:
        print(f"  ❌ ERROR: Could not fetch data for {ticker}")
        return

    close = float(data.close.iloc[-1])
    print(f"  Current Close : ${close:.2f}")
    if entry_price:
        pnl = (close - entry_price) / entry_price * 100
        print(f"  Entry Price   : ${entry_price:.2f}  (P&L: {pnl:+.1f}%)")
    if high_watermark:
        drawdown = (close - high_watermark) / high_watermark * 100
        print(f"  High Watermark: ${high_watermark:.2f}  (drawdown: {drawdown:+.1f}%)")
    print()

    # --- run trend template ----------------------------------------
    result = trend.validate(data)
    checks = [
        ("Close > SMA-50", result.current_close > result.sma_50),
        ("SMA-50 > SMA-200", result.sma_50 > result.sma_200),
        ("SMA-50 slope > 0", result.sma_50_slope > 0),
    ]
    # 200-day slope is informative but not a hard sell signal alone
    sma200_ok = result.sma_200_slope > 0

    print(f"  Trend Template:")
    for label, ok in checks:
        print(f"    {'✅' if ok else '❌'}  {label}")
    print(f"    {'✅' if sma200_ok else '⚠'}  SMA-200 slope > 0  (informational)")

    print()
    print(f"  SMA-50  : ${result.sma_50:.2f}  (slope: {result.sma_50_slope:+.4f})")
    print(f"  SMA-200 : ${result.sma_200:.2f}  (slope: {result.sma_200_slope:+.4f})")
    print()

    # --- verdict ---------------------------------------------------
    passed = all(c[1] for c in checks)
    if passed:
        print(f"  ✅ VERDICT: HOLD — Trend Template intact")
        if high_watermark:
            trail = close / high_watermark
            if trail < 0.75:
                print(f"  ❌ Trailing stop triggered — 25%+ below high (${high_watermark:.2f})")
            elif trail < 0.85:
                print(f"  ⚠  Within trailing-stop range — 15-25% below high")
            else:
                print(f"  →  Trailing stop safe (< 15% below high)")
    else:
        print(f"  🛑  VERDICT: SELL — Trend Template broken")
        failed = [label for label, ok in checks if not ok]
        print(f"  Reason(s): {', '.join(failed)}")

    if entry_price and not passed:
        suggested_stop = max(close * 0.96, result.sma_50)
        print(f"  →  Suggested stop-loss: ${suggested_stop:.2f}")


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print("Usage: python3 check_trend.py TICKER [TICKER ...]")
        print()
        print("Optional environment overrides:")
        print("  ENTRY=BUYPRICE     Your entry price for P&L calculation")
        print("  HIGH=PEAKPRICE     Highest price since entry (trailing stop check)")
        print()
        print("Examples:")
        print("  python3 check_trend.py AAPL MSFT COP")
        print("  ENTRY=142.28 HIGH=165.00 python3 check_trend.py COP")
        sys.exit(0)

    entry = _env_float("ENTRY")
    high = _env_float("HIGH")

    for ticker in args:
        check_ticker(ticker.upper(), entry_price=entry, high_watermark=high)


def _env_float(name: str) -> Optional[float]:
    import os
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        print(f"  ⚠  ${name}={v} is not a valid number, ignoring", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()