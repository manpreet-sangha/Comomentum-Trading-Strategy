# test_zero_returns.py
# =====================================================================
# Zero-Return Diagnostic Module
# =====================================================================
# Checks whether the raw TxN return matrix contains any exact-zero
# values (0.0).  A return of exactly 0.0 is valid (the stock's price
# was unchanged that week), but it is extremely rare.  A high
# concentration of zeros in one stock may indicate stale or padded
# data from the provider.
#
# This module provides a reusable function check_zero_returns() that
# is called by data_loader.py before clean_returns() runs.  It can
# also be executed as a standalone script:
#     python test_zero_returns.py
#
# OUTPUT (via logger):
#   - Total zero count and percentage
#   - Per-stock zero counts
#   - Top 20 stocks with most zeros
#   - Year-by-year zero rates
#   - Final verdict (pass / flag)
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from logger_setup import _setup_logger

log = _setup_logger()


def check_zero_returns(returns, dates, names):
    """
    Scans the raw return matrix for exact-zero values and logs a
    detailed diagnostic report.

    INPUT:
        returns : np.ndarray (TxN) - raw weekly stock returns
        dates   : pd.DatetimeIndex  - weekly date series (length T)
        names   : np.ndarray (N,)   - stock name labels

    OUTPUT:
        total_zeros : int - number of exact-zero cells found
    """

    T, N = returns.shape
    total_cells = T * N

    log.info("=" * 65)
    log.info("  ZERO-RETURN DIAGNOSTIC  (raw returns matrix)")
    log.info("=" * 65)
    log.info(f"  Dataset dimensions : T={T:,} weeks  x  N={N:,} stocks")
    log.info(f"  Total cells        : {total_cells:,}")

    # ------------------------------------------------------------------
    # 1. Cell breakdown
    # ------------------------------------------------------------------
    zero_mask    = (returns == 0.0)
    total_zeros  = int(np.sum(zero_mask))
    total_nan    = int(np.sum(np.isnan(returns)))
    total_finite = int(np.sum(np.isfinite(returns)))

    log.info(f"  {'─' * 55}")
    log.info(f"  Cell breakdown")
    log.info(f"  {'─' * 55}")
    log.info(f"    Finite values  : {total_finite:,}  "
             f"({total_finite / total_cells * 100:.2f}%)")
    log.info(f"    NaN values     : {total_nan:,}  "
             f"({total_nan / total_cells * 100:.2f}%)")
    log.info(f"    Exact zeros    : {total_zeros:,}  "
             f"({total_zeros / total_cells * 100:.4f}%)")

    # ------------------------------------------------------------------
    # 2. Per-stock zero count
    # ------------------------------------------------------------------
    zeros_per_stock = np.sum(zero_mask, axis=0)   # shape (N,)
    stocks_with_zeros = int(np.sum(zeros_per_stock > 0))

    log.info(f"  {'─' * 55}")
    log.info(f"  Per-stock summary")
    log.info(f"  {'─' * 55}")
    log.info(f"    Stocks with no zeros  : {N - stocks_with_zeros:,}  "
             f"({(N - stocks_with_zeros) / N * 100:.1f}%)")
    log.info(f"    Stocks with ≥1 zero   : {stocks_with_zeros:,}  "
             f"({stocks_with_zeros / N * 100:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Top 20 stocks with most zeros
    # ------------------------------------------------------------------
    if stocks_with_zeros > 0:
        log.info(f"  {'─' * 55}")
        log.info(f"  Top 20 stocks with most zeros")
        log.info(f"  {'─' * 55}")
        log.info(f"    {'Rank':<6}{'Stock Name':<30}{'Zeros':>10}"
                 f"{'of T':>10}{'Zero %':>10}")
        log.info(f"    {'─'*6}{'─'*30}{'─'*10}{'─'*10}{'─'*10}")

        order = np.argsort(-zeros_per_stock)
        for rank, idx in enumerate(order[:20], start=1):
            if zeros_per_stock[idx] == 0:
                break
            pct = zeros_per_stock[idx] / T * 100
            name = str(names[idx])[:28]
            log.info(f"    {rank:<6}{name:<30}{zeros_per_stock[idx]:>10,}"
                     f"{T:>10,}{pct:>9.2f}%")

    # ------------------------------------------------------------------
    # 4. Year-by-year breakdown
    # ------------------------------------------------------------------
    log.info(f"  {'─' * 55}")
    log.info(f"  Zero rate by year")
    log.info(f"  {'─' * 55}")
    log.info(f"    {'Year':<8}{'Cells':>14}{'Zeros':>14}{'Zero %':>10}")
    log.info(f"    {'─'*8}{'─'*14}{'─'*14}{'─'*10}")

    years = dates.year
    for yr in np.sort(np.unique(years)):
        yr_rows = (years == yr)
        cells_yr = int(np.sum(yr_rows)) * N
        zeros_yr = int(np.sum(zero_mask[yr_rows, :]))
        pct = zeros_yr / cells_yr * 100 if cells_yr > 0 else 0
        log.info(f"    {yr:<8}{cells_yr:>14,}{zeros_yr:>14,}{pct:>9.4f}%")

    # ------------------------------------------------------------------
    # 5. Final verdict
    # ------------------------------------------------------------------
    log.info("=" * 65)
    if total_zeros == 0:
        log.info("  RESULT: No exact-zero values found in raw returns matrix.")
    else:
        log.info(f"  RESULT: {total_zeros:,} exact-zero values found")
        log.info(f"  across {stocks_with_zeros:,} stocks "
                 f"({stocks_with_zeros / N * 100:.1f}% of universe).")
    log.info("=" * 65)

    return total_zeros


def plot_zero_diagnostic(returns, dates, output_dir='output_data'):
    """
    Produces a compact, report-ready figure summarising the zero-return
    diagnostic.  Two panels side by side:

        Left  – horizontal bar chart showing the cell breakdown
                (non-zero finite, NaN, exact zero) with counts & %.
        Right – year-by-year zero rate as a bar chart (confirms 0 for
                every year when the dataset is clean).

    INPUT:
        returns    : np.ndarray (TxN) - raw weekly stock returns
        dates      : pd.DatetimeIndex - weekly date series (length T)
        output_dir : str              - folder for the saved PNG

    OUTPUT:
        Saves  plot9_zero_diagnostic.png  to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    T, N = returns.shape
    total_cells = T * N

    total_zeros  = int(np.sum(returns == 0.0))
    total_nan    = int(np.sum(np.isnan(returns)))
    total_nonzero_finite = int(np.sum(np.isfinite(returns))) - total_zeros

    # ── colour palette (academic / muted) ────────────────────────────
    CLR_FINITE = '#2563EB'   # royal blue  – non-zero finite returns
    CLR_NAN    = '#94A3B8'   # slate grey  – NaN (not-listed)
    CLR_ZERO   = '#DC2626'   # crimson red – exact zeros
    CLR_BAR    = '#3B82F6'   # lighter blue for year bars
    CLR_BG     = '#F8FAFC'   # very light grey figure background

    # ── figure setup ─────────────────────────────────────────────────
    fig, (ax_bar, ax_yr) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={'width_ratios': [1.1, 1.6]},
    )
    fig.patch.set_facecolor(CLR_BG)
    for ax in (ax_bar, ax_yr):
        ax.set_facecolor(CLR_BG)

    # ================================================================
    # LEFT PANEL – horizontal bar: cell breakdown
    # ================================================================
    categories = ['Non-zero\nfinite', 'NaN', 'Exact\nzero']
    counts     = [total_nonzero_finite, total_nan, total_zeros]
    colours    = [CLR_FINITE, CLR_NAN, CLR_ZERO]

    bars = ax_bar.barh(categories, counts, color=colours, height=0.55,
                       edgecolor='white', linewidth=0.8)

    # annotate each bar with count + percentage
    for bar, cnt in zip(bars, counts):
        pct = cnt / total_cells * 100
        # place label outside for readability
        ax_bar.text(bar.get_width() + total_cells * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{cnt:,}  ({pct:.2f}%)',
                    va='center', ha='left', fontsize=8.5, fontweight='bold',
                    color='#334155')

    ax_bar.set_xlim(0, total_cells * 1.35)
    ax_bar.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax_bar.set_xlabel('Number of cells', fontsize=9, color='#475569')
    ax_bar.set_title('Cell Breakdown', fontsize=11, fontweight='bold',
                     color='#1E293B', pad=10)
    ax_bar.tick_params(axis='y', labelsize=8.5, colors='#334155')
    ax_bar.tick_params(axis='x', labelsize=8, colors='#64748B')
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.spines[['bottom', 'left']].set_color('#CBD5E1')

    # ================================================================
    # RIGHT PANEL – year-by-year zero rate
    # ================================================================
    years = dates.year
    unique_years = np.sort(np.unique(years))
    zero_mask = (returns == 0.0)

    zero_pct_by_year = []
    for yr in unique_years:
        yr_rows = (years == yr)
        cells_yr = int(np.sum(yr_rows)) * N
        zeros_yr = int(np.sum(zero_mask[yr_rows, :]))
        zero_pct_by_year.append(zeros_yr / cells_yr * 100 if cells_yr else 0)

    ax_yr.bar(unique_years.astype(str), zero_pct_by_year,
              color=CLR_BAR, width=0.7, edgecolor='white', linewidth=0.4)

    ax_yr.set_ylabel('Exact-zero rate (%)', fontsize=9, color='#475569')
    ax_yr.set_title('Zero Rate by Year', fontsize=11, fontweight='bold',
                    color='#1E293B', pad=10)
    ax_yr.tick_params(axis='x', rotation=45, labelsize=7, colors='#64748B')
    ax_yr.tick_params(axis='y', labelsize=8, colors='#64748B')
    ax_yr.spines[['top', 'right']].set_visible(False)
    ax_yr.spines[['bottom', 'left']].set_color('#CBD5E1')

    # if all zeros are 0 %, set y-axis to a small range so bars are visible as "nothing"
    max_pct = max(zero_pct_by_year) if zero_pct_by_year else 0
    if max_pct == 0:
        ax_yr.set_ylim(0, 0.05)
        ax_yr.text(0.5, 0.55, '0.0 % in every year',
                   transform=ax_yr.transAxes, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='#16A34A',
                   bbox=dict(boxstyle='round,pad=0.4', fc='#DCFCE7',
                             ec='#86EFAC', linewidth=1.2))

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'plot9_zero_diagnostic.png')
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)

    log.info(f"    Saved {save_path}")


# =====================================================================
# Run as standalone script: python test_zero_returns.py
# =====================================================================
if __name__ == '__main__':
    import pandas as pd
    from data.read_returns import load_returns
    from data.read_dates import load_dates
    from data.read_names import load_names

    returns = load_returns('input_data/')
    dates   = load_dates('input_data/')
    names   = load_names('input_data/')

    check_zero_returns(returns, dates, names)
    plot_zero_diagnostic(returns, dates)
