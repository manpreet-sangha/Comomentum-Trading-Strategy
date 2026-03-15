# test_clean_returns.py
# =====================================================================
# Input Data Dimension Diagnostic
# =====================================================================
# Produces a single 3-panel figure for the coursework report:
#
#   Panel A – Timeline Coverage
#             Horizontal bars showing each file's date span on a
#             shared time axis.  Answers: do files overlap?
#
#   Panel B – Matrix Shape Diagram
#             Proportionally-scaled rectangles (height ∝ rows,
#             width ∝ columns) so you can see at a glance which
#             files are tall-skinny vs short-wide vs full TxN.
#
#   Panel C – Summary Table
#             Formatted table listing dataset, rows, columns, and
#             a mini density bar showing relative T coverage.
#
# Standalone:   python test_clean_returns.py
# From loader:  from test_clean_returns import plot_data_dimensions
#               plot_data_dimensions(data, output_dir)
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from logger_setup import _setup_logger

log = _setup_logger()


def plot_data_dimensions(data, output_dir='output_data'):
    """
    Creates a 3-panel diagnostic figure showing the dimensions
    and time coverage of every input file.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """

    os.makedirs(output_dir, exist_ok=True)

    T = data['T']
    N = data['N']
    dates = data['dates']
    ff_cols = data['ff_factors'].shape[1] + 1   # 3 factors + RF = 4
    date_start = dates[0]
    date_end   = dates[-1]

    # ── file definitions ─────────────────────────────────────────────
    # (label, rows, cols, start_date, end_date, colour)
    # All five files share the same T date range in this dataset.
    files = [
        ('US_Returns.csv',   T, N,       date_start, date_end, '#2563EB'),
        ('US_live,csv.csv',  T, N,       date_start, date_end, '#7C3AED'),
        ('US_Dates.xlsx',    T, 1,       date_start, date_end, '#0891B2'),
        ('US_Names.xlsx',    1, N,       None,       None,     '#D97706'),
        ('FamaFrench.csv',   T, ff_cols, date_start, date_end, '#059669'),
    ]

    # ── colours / style ──────────────────────────────────────────────
    CLR_BG   = '#F8FAFC'
    CLR_TEXT = '#1E293B'
    CLR_DIM  = '#64748B'
    CLR_EDGE = '#CBD5E1'
    CLR_GRID = '#E2E8F0'

    # ══════════════════════════════════════════════════════════════════
    # FIGURE: 3 panels stacked vertically
    # ══════════════════════════════════════════════════════════════════
    fig, (ax_time, ax_shape, ax_table) = plt.subplots(
        3, 1, figsize=(11, 9.5),
        gridspec_kw={'height_ratios': [1.0, 1.4, 1.0]},
    )
    fig.patch.set_facecolor(CLR_BG)
    for ax in (ax_time, ax_shape, ax_table):
        ax.set_facecolor(CLR_BG)

    # ==================================================================
    # PANEL A – Timeline Coverage
    # ==================================================================
    time_files = [(f[0], f[3], f[4], f[5]) for f in files if f[3] is not None]
    y_labels_time = [f[0] for f in time_files]

    for i, (label, d0, d1, clr) in enumerate(time_files):
        ax_time.barh(i, (d1 - d0).days, left=d0, height=0.5,
                     color=clr, alpha=0.75, edgecolor='white', linewidth=0.8)
        # date range text inside the bar
        mid = d0 + (d1 - d0) / 2
        ax_time.text(mid, i,
                     f'{d0.strftime("%Y")}–{d1.strftime("%Y")}',
                     ha='center', va='center', fontsize=7.5,
                     color='white', fontweight='bold')

    ax_time.set_yticks(range(len(y_labels_time)))
    ax_time.set_yticklabels(y_labels_time, fontsize=8.5, fontweight='bold',
                            color=CLR_TEXT)
    ax_time.invert_yaxis()
    ax_time.set_title('A.  Timeline Coverage',
                      fontsize=11, fontweight='bold', color=CLR_TEXT,
                      loc='left', pad=8)
    ax_time.spines[['top', 'right']].set_visible(False)
    ax_time.spines[['bottom', 'left']].set_color(CLR_EDGE)
    ax_time.tick_params(axis='x', labelsize=7.5, colors=CLR_DIM)
    ax_time.set_xlabel('')

    # Note for US_Names.xlsx (no time axis)
    ax_time.text(0.99, 0.02,
                 'US_Names.xlsx has no time dimension (1 × N)',
                 transform=ax_time.transAxes, ha='right', va='bottom',
                 fontsize=7, color=CLR_DIM, style='italic')

    # ==================================================================
    # PANEL B – Matrix Shape Diagram
    # ==================================================================
    # Height ∝ rows, Width ∝ columns.  Scaled so the biggest TxN
    # rectangle fills a reference box.
    ref_h = 2.4      # max rectangle height
    ref_w = 2.2      # max rectangle width
    scale_h = ref_h / T
    scale_w = ref_w / N

    gap = 0.7
    x_cursor = 0.5

    for (label, r, c, _d0, _d1, clr) in files:
        w = max(c * scale_w, 0.12)
        h = max(r * scale_h, 0.12)

        rect = mpatches.FancyBboxPatch(
            (x_cursor, ref_h - h), w, h,
            boxstyle='round,pad=0.03',
            facecolor=clr, alpha=0.18,
            edgecolor=clr, linewidth=1.6,
        )
        ax_shape.add_patch(rect)

        cx = x_cursor + w / 2
        cy = ref_h - h / 2

        # Dimension inside
        ax_shape.text(cx, cy, f'{r:,}×{c:,}',
                      ha='center', va='center', fontsize=8,
                      fontweight='bold', color=clr)

        # File name above
        ax_shape.text(cx, ref_h + 0.15, label,
                      ha='center', va='bottom', fontsize=7.5,
                      fontweight='bold', color=CLR_TEXT)

        x_cursor += w + gap

    total_w = x_cursor - gap + 0.5
    ax_shape.set_xlim(0, total_w)
    ax_shape.set_ylim(-0.3, ref_h + 0.5)
    ax_shape.set_aspect('equal')
    ax_shape.axis('off')
    ax_shape.set_title('B.  Matrix Shape  (height ∝ rows, width ∝ columns)',
                       fontsize=11, fontweight='bold', color=CLR_TEXT,
                       loc='left', pad=8)

    # ==================================================================
    # PANEL C – Summary Table
    # ==================================================================
    col_labels = ['File', 'Rows', 'Columns', 'Shape', 'Coverage']
    table_data = []

    for (label, r, c, d0, d1, clr) in files:
        shape_str = f'{r:,} × {c:,}'
        if d0 is not None:
            coverage = f'{d0.strftime("%Y-%m-%d")} → {d1.strftime("%Y-%m-%d")}'
        else:
            coverage = 'N/A  (cross-sectional)'
        table_data.append([label, f'{r:,}', f'{c:,}', shape_str, coverage])

    ax_table.axis('off')
    ax_table.set_title('C.  Dimension Summary',
                       fontsize=11, fontweight='bold', color=CLR_TEXT,
                       loc='left', pad=8)

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.20, 0.10, 0.10, 0.18, 0.35],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#1E293B')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(CLR_EDGE)

    # Style data rows with alternating colours + file-specific left edge
    for i in range(len(files)):
        row_bg = '#FFFFFF' if i % 2 == 0 else '#F1F5F9'
        clr = files[i][5]
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor(CLR_EDGE)
            if j == 0:
                cell.set_text_props(fontweight='bold', color=clr)

    # ── T / N note at bottom ─────────────────────────────────────────
    fig.text(0.5, 0.01,
             f'T = {T:,} weeks  ·  N = {N:,} stocks  ·  '
             f'{date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}',
             ha='center', va='bottom', fontsize=8.5, color=CLR_DIM)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
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
    plot_data_dimensions(data)
