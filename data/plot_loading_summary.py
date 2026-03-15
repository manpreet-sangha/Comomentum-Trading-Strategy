# plot_loading_summary.py
# =====================================================================
# Data Loading Summary – Table-Only Report
# =====================================================================
# Produces a single summary table combining:
#   • Dataset dimensions (T, N, date range)
#   • Input file shapes
#   • Cleaning breakdown based on the US_live flag
#     (live==1 → listed / trading;  live==0 → not listed / not trading)
#   • NaN counts — NaN in the returns file is NOT synonymous with
#     "not listed".  NaN can arise from holidays, trading halts,
#     or other data gaps.  Listing status is determined solely by
#     the live indicator in US_live.csv.
#
# Standalone:   python plot_loading_summary.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_loading_summary(data, output_dir='output_data'):
    """
    Single summary table (no graphs) combining dataset dimensions,
    file shapes, and cleaning breakdown.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """

    os.makedirs(output_dir, exist_ok=True)

    returns       = data['returns']
    live          = data['live']
    dates         = data['dates']
    ff_factors    = data['ff_factors']
    T             = data['T']
    N             = data['N']

    # ── derived counts ───────────────────────────────────────────────
    total_cells   = T * N
    listed_cells  = int(np.sum(live == 1))
    not_listed    = int(np.sum(live == 0))

    # Among listed cells, how many have a finite return?
    finite_and_listed = int(np.sum(np.isfinite(returns) & (live == 1)))
    nan_in_listed     = listed_cells - finite_and_listed   # genuine data gaps

    # Among not-listed cells, did any have a numeric value that was
    # forced to NaN during cleaning?
    forced_nan = int(np.sum((live == 0) & np.isfinite(returns)))
    notlisted_already_nan = int(np.sum((live == 0) & ~np.isfinite(returns)))

    date_start = dates[0].strftime('%Y-%m-%d')
    date_end   = dates[-1].strftime('%Y-%m-%d')
    date_range = f'{date_start} to {date_end}'
    ff_cols    = ff_factors.shape[1] + 1                  # 3 factors + RF

    # ── helper: format percentage ────────────────────────────────────
    def pct(part, whole):
        return f'({part / whole * 100:.1f}%)' if whole else ''

    # ── table rows ───────────────────────────────────────────────────
    rows = [
        # Section 1 — Dataset dimensions
        ['Period',              date_range],
        ['Weeks (T)',           f'{T:,}'],
        ['Stocks (N)',          f'{N:,}'],
        ['Total cells (T×N)',   f'{total_cells:,}'],
        ['', ''],
        # Section 2 — Input file shapes + coverage
        ['US_Returns.csv',      f'{T:,} × {N:,}  |  {date_range}'],
        ['US_live.csv',         f'{T:,} × {N:,}  |  {date_range}'],
        ['US_Dates.xlsx',       f'{T:,} dates  |  {date_range}'],
        ['US_Names.xlsx',       f'{N:,} stock names'],
        ['FamaFrench.csv',      f'{T:,} × {ff_cols}  |  {date_range}'],
        ['', ''],
        # Section 3 — Cleaning breakdown
        ['Listed cells (live=1)',
         f'{listed_cells:,}  {pct(listed_cells, total_cells)}'],
        ['    with valid return',
         f'{finite_and_listed:,}'],
        ['    NaN (data gaps)',
         f'{nan_in_listed:,}'],
        ['Not-listed cells (live=0)',
         f'{not_listed:,}  {pct(not_listed, total_cells)}'],
        ['    already NaN in raw file',
         f'{notlisted_already_nan:,}  (no action needed)'],
        ['    had numeric value → set to NaN',
         f'{forced_nan:,}  (forced to NaN)'],
    ]

    # ── colours ──────────────────────────────────────────────────────
    CLR_BG      = '#FFFFFF'
    CLR_TEXT    = '#000000'
    CLR_HDR     = '#000000'
    CLR_HDR_BG  = '#E2E8F0'

    # ── figure ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'

    n_rows = len(rows) + 1          # +1 for the header row
    row_height = 0.38               # inches per row
    fig_height = n_rows * row_height
    fig, ax = plt.subplots(figsize=(9, fig_height))
    fig.patch.set_facecolor(CLR_BG)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])   # fill the entire figure

    tbl = ax.table(
        cellText=rows,
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        loc='upper center',
        colWidths=[0.45, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.0, 1.8)

    # Style header row
    for ci in range(2):
        cell = tbl[0, ci]
        cell.set_facecolor(CLR_HDR_BG)
        cell.set_text_props(fontweight='bold', color=CLR_HDR, fontsize=14,
                            fontfamily='Arial')
        cell.set_edgecolor('#CBD5E1')
        cell.set_linewidth(0.5)

    # Style data rows
    stripe = ['#FFFFFF', '#F1F5F9']
    for ri in range(len(rows)):
        for ci in range(2):
            cell = tbl[ri + 1, ci]
            cell.set_facecolor(stripe[ri % 2])
            cell.set_edgecolor('#CBD5E1')
            cell.set_linewidth(0.4)
            cell.set_text_props(color=CLR_TEXT)

        # Blank separator rows — hide borders
        if rows[ri][0] == '':
            for ci in range(2):
                tbl[ri + 1, ci].set_facecolor(CLR_BG)
                tbl[ri + 1, ci].set_edgecolor(CLR_BG)

    save_path = os.path.join(output_dir, 'plot10_loading_summary.png')
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_loading_summary(data)
