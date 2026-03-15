# plot_dimensions.py
# =====================================================================
# Input-File Dimension Summary Table
# =====================================================================
# Renders a clean summary table showing the shape, date coverage and
# content description for every input file.  Designed for direct
# inclusion in a coursework report.
#
# Standalone:   python plot_dimensions.py
# From loader:  from plot_dimensions import plot_input_dimensions
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_input_dimensions(data, output_dir='output_data'):
    """
    Draws a formatted summary table of input-file dimensions.

    INPUT:
        data       : dict : dictionary returned by load_all_data()
        output_dir : str  : folder for the saved PNG
    """

    os.makedirs(output_dir, exist_ok=True)

    T = data['T']
    N = data['N']
    dates = data['dates']
    ff_cols = data['ff_factors'].shape[1] + 1   # 3 factors + RF = 4

    date_start = dates[0].strftime('%Y-%m-%d')
    date_end   = dates[-1].strftime('%Y-%m-%d')

    # ── table rows: (File, Rows, Columns, Shape, Coverage) ───────────
    rows = [
        ['US_Returns.csv',  f'{T:,}',  f'{N:,}',      f'{T:,} x {N:,}',      f'{date_start}  to  {date_end}'],
        ['US_live.csv', f'{T:,}',  f'{N:,}',      f'{T:,} x {N:,}',      f'{date_start}  to  {date_end}'],
        ['US_Dates.xlsx',   f'{T:,}',  '1',           f'{T:,} x 1',          f'{date_start}  to  {date_end}'],
        ['US_Names.xlsx',   '1',       f'{N:,}',      f'1 x {N:,}',          'Cross-sectional (no time axis)'],
        ['FamaFrench.csv',  f'{T:,}',  f'{ff_cols}',  f'{T:,} x {ff_cols}',  f'{date_start}  to  {date_end}'],
    ]
    col_labels = ['File', 'Rows', 'Columns', 'Shape', 'Coverage']

    # ── colours ──────────────────────────────────────────────────────
    CLR_BG      = '#F8FAFC'
    CLR_TEXT    = '#1E293B'
    CLR_HEADER = '#1E3A5F'
    CLR_HDR_BG = '#E2E8F0'
    row_colours = ['#FFFFFF', '#F1F5F9']          # alternating stripes

    # ── figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.55)

    # ── style header row ─────────────────────────────────────────────
    for col_idx in range(len(col_labels)):
        cell = tbl[0, col_idx]
        cell.set_facecolor(CLR_HDR_BG)
        cell.set_text_props(fontweight='bold', color=CLR_HEADER, fontsize=9.5)
        cell.set_edgecolor('#CBD5E1')
        cell.set_linewidth(0.6)

    # ── style data rows ──────────────────────────────────────────────
    for row_idx in range(len(rows)):
        stripe = row_colours[row_idx % 2]
        for col_idx in range(len(col_labels)):
            cell = tbl[row_idx + 1, col_idx]
            cell.set_facecolor(stripe)
            cell.set_edgecolor('#CBD5E1')
            cell.set_linewidth(0.6)
            cell.set_text_props(color=CLR_TEXT)

        # file-name column in black bold
        tbl[row_idx + 1, 0].set_text_props(
            fontweight='bold', color='#000000')

    ax.set_title('Input File Dimensions',
                 fontsize=13, fontweight='bold', color=CLR_TEXT, pad=14)

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'plot10_input_dimensions.png')
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_input_dimensions(data)
