# stock_diagnostics.py
# =====================================================================
# Stock-Level Diagnostics
# =====================================================================
# Two checks on the listed/returns panel:
#
#   1. SHORT-LIVED STOCKS  — stocks with fewer than 52 weeks of listed
#      returns.  These stocks can never receive a momentum score because
#      the rolling window requires TOTAL = 52 weeks of history.
#
#   2. TRADING GAPS  — stocks that have at least one gap in their
#      listed period.  A "gap" means the stock was listed (live=1),
#      then became not-listed (live=0) for one or more weeks, then
#      reappeared as listed (live=1) again later.
#
# OUTPUT (to output_data/):
#   stock_diagnostics.log           — full log of both checks
#   stocks_short_lived.csv          — list of stocks with < 52 listed weeks
#   stocks_with_trading_gaps.csv    — list of stocks that have gaps
#   combined_data_verification.csv  — full Date × Stock × Return × Live × Flag
#                                     panel for manual verification
#
# Standalone:   python stock_diagnostics.py
# From loader:  from stock_diagnostics import run_stock_diagnostics
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import os
from logger_setup import _setup_logger

log = _setup_logger()


# ── Constants (match compute_momentum_signal.py) ─────────────────────
LOOKBACK = 48
SKIP     = 4
TOTAL    = LOOKBACK + SKIP   # 52


def find_short_lived_stocks(live, names, dates, output_dir='output_data'):
    """
    Identifies stocks with fewer than TOTAL (52) weeks of listed returns.

    These stocks can NEVER receive a momentum score because the rolling
    window needs at least 52 consecutive-or-scattered listed weeks
    within the lookback range.

    Returns a DataFrame with one row per short-lived stock.
    """
    T, N = live.shape
    weeks_listed = np.sum(live == 1, axis=0)   # (N,) count of listed weeks

    short_mask = weeks_listed < TOTAL
    n_short = int(np.sum(short_mask))

    log.info("=" * 65)
    log.info("DIAGNOSTIC 1: Stocks with fewer than 52 listed weeks")
    log.info("=" * 65)
    log.info(f"  Minimum weeks required for momentum score : {TOTAL}")
    log.info(f"  Total stocks in universe                  : {N:,}")
    log.info(f"  Stocks with >= {TOTAL} listed weeks          : {N - n_short:,}")
    log.info(f"  Stocks with <  {TOTAL} listed weeks          : {n_short:,}  "
             f"({n_short / N * 100:.2f}%)")

    rows = []
    for j in np.where(short_mask)[0]:
        listed_weeks = int(weeks_listed[j])
        # Find first and last listed dates
        listed_idx = np.where(live[:, j] == 1)[0]
        if len(listed_idx) > 0:
            first_date = dates[listed_idx[0]]
            last_date  = dates[listed_idx[-1]]
        else:
            first_date = pd.NaT
            last_date  = pd.NaT

        rows.append({
            'Stock_Index':   j,
            'Stock_Name':    names[j],
            'Weeks_Listed':  listed_weeks,
            'First_Listed':  first_date,
            'Last_Listed':   last_date,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('Weeks_Listed', ascending=True).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'stocks_short_lived.csv')
    df.to_csv(path, index=False)
    log.info(f"  Saved: {path}")

    # Log a few examples
    if len(df) > 0:
        log.info(f"\n  Shortest-lived stocks (bottom 10):")
        log.info(f"  {'Name':<45} {'Weeks':>6}  {'First Listed':<12} {'Last Listed':<12}")
        log.info(f"  {'-'*45} {'-'*6}  {'-'*12} {'-'*12}")
        for _, row in df.head(10).iterrows():
            fd = row['First_Listed'].strftime('%Y-%m-%d') if pd.notna(row['First_Listed']) else 'N/A'
            ld = row['Last_Listed'].strftime('%Y-%m-%d') if pd.notna(row['Last_Listed']) else 'N/A'
            log.info(f"  {row['Stock_Name']:<45} {row['Weeks_Listed']:>6}  {fd:<12} {ld:<12}")

        # Distribution summary
        bins = [0, 1, 5, 10, 20, 30, 40, 51]
        labels = ['0', '1-4', '5-9', '10-19', '20-29', '30-39', '40-51']
        df['Bin'] = pd.cut(df['Weeks_Listed'], bins=bins, labels=labels,
                           right=False, include_lowest=True)
        dist = df['Bin'].value_counts().sort_index()
        log.info(f"\n  Distribution of short-lived stocks by weeks listed:")
        for b, cnt in dist.items():
            if cnt > 0:
                log.info(f"    {b:>8} weeks : {cnt:>4} stocks")
    else:
        log.info("  No short-lived stocks found — every stock has >= 52 listed weeks.")

    log.info("-" * 65)
    return df


def find_trading_gaps(live, names, dates, output_dir='output_data'):
    """
    Identifies stocks that have GAPS in their listed period.

    A gap means the stock transitions from listed (1) -> not-listed (0)
    -> listed (1) at least once.  In other words, there are at least
    TWO separate stretches of consecutive listed weeks, separated by
    one or more not-listed weeks.

    Returns a DataFrame with one row per gapped stock, including the
    number of separate listed stretches, total gap weeks, and details
    of each gap.
    """
    T, N = live.shape

    log.info("=" * 65)
    log.info("DIAGNOSTIC 2: Stocks with gaps in trading (re-listings)")
    log.info("=" * 65)

    rows = []
    for j in range(N):
        col = live[:, j]
        listed_idx = np.where(col == 1)[0]

        if len(listed_idx) < 2:
            continue   # 0 or 1 listed week — can't have a gap

        # Consecutive differences in the listed indices
        # If all listed weeks are contiguous, diffs are all 1
        diffs = np.diff(listed_idx)
        gap_positions = np.where(diffs > 1)[0]

        if len(gap_positions) == 0:
            continue   # no gaps — single contiguous block

        # This stock has at least one gap
        n_stretches = len(gap_positions) + 1
        total_gap_weeks = 0
        gap_details = []

        for gp in gap_positions:
            gap_start_idx = listed_idx[gp]       # last listed index before gap
            gap_end_idx   = listed_idx[gp + 1]   # first listed index after gap
            gap_length    = int(gap_end_idx - gap_start_idx - 1)
            total_gap_weeks += gap_length
            gap_details.append({
                'gap_start': dates[gap_start_idx].strftime('%Y-%m-%d'),
                'gap_end':   dates[gap_end_idx].strftime('%Y-%m-%d'),
                'gap_weeks': gap_length,
            })

        rows.append({
            'Stock_Index':       j,
            'Stock_Name':        names[j],
            'Weeks_Listed':      int(len(listed_idx)),
            'First_Listed':      dates[listed_idx[0]],
            'Last_Listed':       dates[listed_idx[-1]],
            'N_Stretches':       n_stretches,
            'N_Gaps':            len(gap_positions),
            'Total_Gap_Weeks':   total_gap_weeks,
            'Gap_Details':       gap_details,
        })

    n_gapped = len(rows)
    n_clean  = N - n_gapped

    log.info(f"  Total stocks in universe      : {N:,}")
    log.info(f"  Stocks with NO gaps (clean)   : {n_clean:,}  "
             f"({n_clean / N * 100:.2f}%)")
    log.info(f"  Stocks with >= 1 gap          : {n_gapped:,}  "
             f"({n_gapped / N * 100:.2f}%)")

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values('Total_Gap_Weeks', ascending=False).reset_index(drop=True)

        # Save (expand gap details into a readable string)
        df_save = df.copy()
        df_save['Gap_Details'] = df_save['Gap_Details'].apply(
            lambda glist: ' | '.join(
                f"{g['gap_start']} to {g['gap_end']} ({g['gap_weeks']}w)"
                for g in glist
            )
        )
        path = os.path.join(output_dir, 'stocks_with_trading_gaps.csv')
        df_save.to_csv(path, index=False)
        log.info(f"  Saved: {path}")

        # Log the worst offenders
        log.info(f"\n  Top 15 stocks by total gap weeks:")
        log.info(f"  {'Name':<40} {'Listed':>6} {'Gaps':>5} {'Gap Wks':>8}  "
                 f"{'Stretches':>9}")
        log.info(f"  {'-'*40} {'-'*6} {'-'*5} {'-'*8}  {'-'*9}")
        for _, row in df.head(15).iterrows():
            log.info(f"  {row['Stock_Name']:<40} "
                     f"{row['Weeks_Listed']:>6} "
                     f"{row['N_Gaps']:>5} "
                     f"{row['Total_Gap_Weeks']:>8}  "
                     f"{row['N_Stretches']:>9}")

        # Summary statistics
        log.info(f"\n  Gap statistics across {n_gapped} gapped stocks:")
        log.info(f"    Mean gaps per stock     : {df['N_Gaps'].mean():.2f}")
        log.info(f"    Max gaps in one stock   : {df['N_Gaps'].max()}")
        log.info(f"    Mean total gap weeks    : {df['Total_Gap_Weeks'].mean():.1f}")
        log.info(f"    Max total gap weeks     : {df['Total_Gap_Weeks'].max()}")
        log.info(f"    Mean stretches per stock: {df['N_Stretches'].mean():.2f}")

        # Distribution by number of gaps
        gap_dist = df['N_Gaps'].value_counts().sort_index()
        log.info(f"\n  Distribution by number of gaps:")
        for ng, cnt in gap_dist.items():
            log.info(f"    {ng:>3} gap(s) : {cnt:>4} stocks")
    else:
        log.info("  No trading gaps found — every stock has a single contiguous listed period.")
        path = os.path.join(output_dir, 'stocks_with_trading_gaps.csv')
        pd.DataFrame().to_csv(path, index=False)
        log.info(f"  Saved: {path}  (empty)")

    log.info("-" * 65)
    return df


def build_combined_verification(returns, live, names, dates,
                                df_short, df_gaps,
                                output_dir='output_data'):
    """
    Combines US_Returns, US_Names, and US_Dates into a single CSV
    for manual verification.

    Layout (wide format, mirrors the raw data):
        - First column  : Date  (from US_Dates.xlsx)
        - Column headers: Stock names (from US_Names.xlsx)
        - Cell values   : Weekly returns (from US_Returns.csv)

    This is the full 1,513 × 7,261 panel so you can look up any
    stock by name and check its return series week by week.
    """
    log.info("=" * 65)
    log.info("BUILDING COMBINED VERIFICATION FILE")
    log.info("=" * 65)

    T, N = returns.shape

    df = pd.DataFrame(returns, index=dates, columns=names)
    df.index.name = 'Date'

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'combined_data_verification.csv')
    df.to_csv(path)

    log.info(f"  Shape : {T:,} rows (weeks) × {N:,} columns (stocks)")
    log.info(f"  Rows  : {dates[0].strftime('%Y-%m-%d')} to "
             f"{dates[-1].strftime('%Y-%m-%d')}")
    log.info(f"  Saved : {path}")
    log.info("-" * 65)

    return df


def run_stock_diagnostics(data, output_dir='output_data'):
    """
    Runs both stock-level diagnostics.

    INPUT:
        data       : dict from load_all_data()
        output_dir : str — folder for output files

    OUTPUT:
        df_short  : DataFrame — stocks with < 52 listed weeks
        df_gaps   : DataFrame — stocks with trading gaps
        df_combined : DataFrame — full combined panel for verification
    """
    live    = data['live']
    returns = data['returns']
    names   = data['names']
    dates   = data['dates']

    log.info("\n")
    log.info("#" * 65)
    log.info("  STOCK-LEVEL DIAGNOSTICS")
    log.info("#" * 65)

    df_short = find_short_lived_stocks(live, names, dates, output_dir)
    df_gaps  = find_trading_gaps(live, names, dates, output_dir)

    # Cross-reference: how many short-lived stocks also have gaps?
    if len(df_short) > 0 and len(df_gaps) > 0:
        short_set = set(df_short['Stock_Index'])
        gap_set   = set(df_gaps['Stock_Index'])
        overlap   = short_set & gap_set
        log.info(f"\n  CROSS-REFERENCE:")
        log.info(f"    Short-lived stocks       : {len(short_set):,}")
        log.info(f"    Stocks with gaps         : {len(gap_set):,}")
        log.info(f"    Both (short + gapped)    : {len(overlap):,}")
        log.info(f"    Short-only               : {len(short_set - gap_set):,}")
        log.info(f"    Gapped-only              : {len(gap_set - short_set):,}")

    # Build combined verification file
    df_combined = build_combined_verification(
        returns, live, names, dates, df_short, df_gaps, output_dir
    )

    log.info("\n" + "#" * 65)
    log.info("  STOCK DIAGNOSTICS COMPLETE")
    log.info("#" * 65)

    return df_short, df_gaps, df_combined


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    df_short, df_gaps, df_combined = run_stock_diagnostics(data)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Stocks with < 52 listed weeks : {len(df_short):,}")
    print(f"  Stocks with trading gaps       : {len(df_gaps):,}")
    print(f"  Combined verification rows     : {len(df_combined):,}")
    print(f"\nCSVs saved to output_data/")
    print(f"  stocks_short_lived.csv")
    print(f"  stocks_with_trading_gaps.csv")
    print(f"  combined_data_verification.csv")
