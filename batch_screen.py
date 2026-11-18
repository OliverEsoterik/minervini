#!/usr/bin/env python3
"""Batch screening runner — processes the full watchlist in parallel."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from vcp_momentum_engine import Pipeline, ScreeningResult

# ── Full watchlist ──────────────────────────────────────────────────
WATCHLIST: List[str] = [
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

# Remove duplicates while preserving order
SEEN = set()
WATCHLIST = [x for x in WATCHLIST if not (x in SEEN or SEEN.add(x))]


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


def main() -> None:
    total = len(WATCHLIST)
    print(f"\n{'='*72}")
    print(f"  VCP Momentum Engine — Batch Screening")
    print(f"  Watchlist: {total} tickers  |  Workers: 8× parallel")
    print(f"  Equity: $100,000  |  Risk/Trade: 0.3%  |  Stop: -4%")
    print(f"{'='*72}\n")

    # ---- RS Rating pre-phase ---------------------------------------
    from rs_rating import RSModule
    print("  Pre-computing RS ratings for watchlist...")
    rs_module = RSModule()
    rs_ranks = rs_module.get_rs_ratings(WATCHLIST)
    valid_rs = {t: r for t, r in rs_ranks.items() if r is not None}
    print(f"  RS computed: {len(valid_rs)}/{total} tickers")

    pipeline = Pipeline()
    results: List[ScreeningResult] = []

    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {
            pool.submit(screen_one, t, pipeline, rs_ranks): t
            for t in WATCHLIST
        }
        for fut in as_completed(fut_map):
            r = fut.result()
            results.append(r)
            done += 1
            if r.approved:
                print(f"  ✅ APPROVED: {r.ticker}")
            # Progress: print every 10%
            if done % max(1, total // 10) == 0:
                pct = done * 100 // total
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done)
                print(f"  [{pct}%] {done}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remain)")

    elapsed = time.time() - t0

    # ── Separate approved vs failed ────────────────────────────────
    approved = [r for r in results if r.approved]
    failed = [r for r in results if not r.approved]
    errored = [r for r in failed if r.error]

    # ── Print approved setups ──────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  ✅ APPROVED SETUPS ({len(approved)} of {total})")
    print(f"{'='*72}")
    if approved:
        for r in sorted(approved, key=lambda x: x.ticker):
            d = r.to_dict()
            ll = r.position_sizing
            print(f"\n  {r.ticker}")
            print(f"    Trend Template : PASS")
            print(f"    Fundamental    : PASS")
            print(f"    VCP Pattern    : DETECTED")

            # ── VCP breakout level (last swing high) ───────────────
            v = r.vcp
            pivot = None
            if v and v.contractions:
                pivot = v.contractions[-1]["peak"]
                close = ll.entry_price if ll else None
                if close:
                    pct_to_pivot = (pivot / close - 1) * 100
                    pivot_note = f"(needs +{pct_to_pivot:.1f}% to clear)"
                    print(f"    VCP Pivot      : ${pivot:.2f}  {pivot_note}")

            if ll:
                print(f"    ─────────────────────────────────────────────")
                print(f"    Last Close     : ${ll.entry_price:.2f}  (position sizing reference)")
                print(f"    Stop-Loss      : ${ll.stop_loss_price:.2f}  ({((ll.stop_loss_price/ll.entry_price)-1)*100:+.1f}%)")
                print(f"    Shares         : {ll.shares_to_buy}")
                print(f"    Risk/Trade     : ${ll.total_risk_deployed:.0f}  ({ll.risk_pct_of_equity*100:.3f}% of equity)")
                print(f"    CAN SLIM score : {r.canslim.overall_score}/7")
            if r.rs_rank is not None:
                print(f"    RS Rank        : {r.rs_rank:.0f}/99")
            if pivot:
                print(f"    >> Enter only on a confirmed breakout above ${pivot:.2f} with volume.")
    else:
        print(f"\n  No stocks passed all three filters in this run.")

    # ── Near-misses (2/3 filters) ──────────────────────────────────
    near_miss = [
        r for r in failed
        if not r.error
        and (r.trend_template and r.trend_template.passed)
        and (r.minervini_fundamental and r.minervini_fundamental.overall_passed)
        and not (r.vcp and r.vcp.vcp_detected)
    ]
    if near_miss:
        print(f"\n\n{'─'*72}")
        print(f"  ⚠  NEAR-MISSES (TT+FUND PASS, VCP FAIL)")
        print(f"{'─'*72}")
        for r in sorted(near_miss, key=lambda x: x.ticker):
            v = r.vcp
            vcp_note = (
                f"ATR={v.atr_pct:.1f}%, "
                f"consol={v.latest_consolidation_pct:.1f}%, "
                f"{len(v.contractions)} swings"
                if v else "no VCP data"
            ) if v else "no VCP data"
            print(f"  {r.ticker:8s} → {vcp_note}")

    # Also show TT+VCP but FF fail
    near_miss_ff = [
        r for r in failed
        if not r.error
        and (r.trend_template and r.trend_template.passed)
        and (r.vcp and r.vcp.vcp_detected)
        and not (r.minervini_fundamental and r.minervini_fundamental.overall_passed)
    ]
    if near_miss_ff:
        print(f"\n\n{'─'*72}")
        print(f"  ⚠  NEAR-MISSES (TT+VCP PASS, FUND FAIL)")
        print(f"{'─'*72}")
        for r in sorted(near_miss_ff, key=lambda x: x.ticker):
            mf = r.minervini_fundamental
            if mf:
                dims = [
                    f"EPS:{'✅' if mf.eps_growth.passed else '❌'}"
                        + (f"{mf.eps_growth.value*100:.0f}%" if mf.eps_growth.value is not None else ""),
                    f"Rev:{'✅' if mf.revenue_growth.passed else '❌'}"
                        + (f"{mf.revenue_growth.value*100:.0f}%" if mf.revenue_growth.value is not None else ""),
                    f"Marg:{'✅' if mf.profit_margin.passed else '❌'}",
                    f"Accel:{mf.score}/{mf.total_dims}",
                ]
                print(f"  {r.ticker:8s} → {' | '.join(dims)}")
            else:
                print(f"  {r.ticker:8s} → no fundamental data")

    # ── Errors ─────────────────────────────────────────────────────
    if errored:
        print(f"\n\n{'─'*72}")
        print(f"  ❌ ERRORS ({len(errored)} tickers)")
        print(f"{'─'*72}")
        for r in errored:
            print(f"  {r.ticker}: {r.error}")

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

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Total scan  : {total} tickers")
    print(f"  Time        : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Approved    : {len(approved)}")
    print(f"  Near-misses : {len(near_miss)} (VCP fail) + {len(near_miss_ff)} (FUND fail)")
    print(f"  Errors      : {len(errored)}")
    print(f"{'='*72}\n")

    if approved:
        print(f"  >> Found {len(approved)} VCP setups. Review the pivot levels above.")
        for r in approved:
            v = r.vcp
            pivot = v.contractions[-1]["peak"] if v and v.contractions else None
            if pivot:
                print(f"     {r.ticker}: buy on breakout above ${pivot:.2f}")
    else:
        if near_miss_ff:
            print(f"  >> The near-misses with TT+VCP pass are your best candidates.")
            print(f"     If revenue growth is above 10%, consider lowering the threshold.")
        if near_miss:
            print(f"  >> The near-misses with TT+FUND pass need VCP to finish forming.")


if __name__ == "__main__":
    main()