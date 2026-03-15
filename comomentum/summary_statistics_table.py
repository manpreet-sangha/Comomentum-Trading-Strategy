# summary_statistics_table.py
# =====================================================================
# Summary Statistics Table — Replicating Lou & Polk (2021) Table I
# =====================================================================
#
# Produces a three-panel summary statistics table rendered as an image:
#
#   Panel A: Summary Statistics (N, Mean, Std. Dev., Min, Max)
#   Panel B: Correlation matrix among the five variables
#   Panel C: Autocorrelation (non-overlapping 12-month windows)
#
# Variables:
#   CoMOM   — average comomentum (0.5 * (CoMOM_L + CoMOM_W))
#   CoMOM_L — loser-decile comomentum
#   CoMOM_W — winner-decile comomentum
#   MRET    — trailing 2-year compounded market return
#   MVOL    — trailing 2-year monthly market volatility
#
# Standalone:  python summary_statistics_table.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def _annual_means(series, dates):
    """Resample a weekly series to non-overlapping 12-month means."""
    df = pd.DataFrame({'val': series}, index=pd.DatetimeIndex(dates))
    annual = df['val'].resample('YE').mean().dropna()
    return annual


def generate_summary_table(comomentum, comom_winner, comom_loser,
                            mret, mvol, dates,
                            save_path='output_data/summary_statistics_table.png'):
    """
    Produce a three-panel summary statistics table as a PNG image.

    Parameters
    ----------
    comomentum   : T-length array — CoMOM
    comom_winner : T-length array — CoMOM_W
    comom_loser  : T-length array — CoMOM_L
    mret         : T-length array — trailing 2-year market return
    mvol         : T-length array — trailing 2-year market volatility
    dates        : T-length DatetimeIndex
    save_path    : str — output PNG path
    """

    dates = pd.DatetimeIndex(dates)

    # ── Align: use weeks where ALL five variables are finite ─────────
    variables = {
        'CoMOM':   comomentum,
        'CoMOM_L': comom_loser,
        'CoMOM_W': comom_winner,
        'MRET':    mret,
        'MVOL':    mvol,
    }
    var_names = list(variables.keys())

    df_all = pd.DataFrame(variables, index=dates)

    # ================================================================
    # PANEL A: Summary Statistics
    # ================================================================
    panel_a_rows = []
    for name in var_names:
        s = df_all[name].dropna()
        panel_a_rows.append([
            name, len(s),
            f'{s.mean():.3f}', f'{s.std():.3f}',
            f'{s.min():.3f}', f'{s.max():.3f}',
        ])

    # ================================================================
    # PANEL B: Correlation (pairwise, using all overlapping valid obs)
    # ================================================================
    corr_matrix = df_all.corr()
    n_vars = len(var_names)

    panel_b_rows = []
    for i, name in enumerate(var_names):
        row = [name]
        for j in range(n_vars):
            if j > i:
                row.append('')
            else:
                row.append(f'{corr_matrix.iloc[i, j]:.3f}')
        panel_b_rows.append(row)

    # ================================================================
    # PANEL C: Autocorrelation (non-overlapping 12-month windows)
    # ================================================================
    comom_vars = ['CoMOM', 'CoMOM_L', 'CoMOM_W']
    annual = {}
    for name in comom_vars:
        annual[name] = _annual_means(df_all[name].values, dates)

    # Align annual series on common years
    annual_df = pd.DataFrame(annual).dropna()

    # Build t and t+1 columns
    ac_names_t  = [f'{n}_t' for n in comom_vars]
    ac_names_t1 = [f'{n}_t+1' for n in comom_vars]

    ac_df = pd.DataFrame()
    for name in comom_vars:
        vals = annual_df[name].values
        ac_df[f'{name}_t']   = vals[:-1]
        ac_df[f'{name}_t+1'] = vals[1:]

    ac_corr = ac_df.corr()
    all_ac_names = ac_names_t + ac_names_t1

    panel_c_rows = []
    for i, rname in enumerate(all_ac_names):
        row = [rname.replace('_t+1', '$_{t+1}$').replace('_t', '$_t$')]
        for j in range(len(all_ac_names)):
            if j > i:
                row.append('')
            else:
                row.append(f'{ac_corr.iloc[i, j]:.3f}')
        panel_c_rows.append(row)

    # ================================================================
    # RENDER AS IMAGE
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5))
    for ax in axes:
        ax.axis('off')

    # ── Helper to draw a table on an axis ────────────────────────────
    def _draw_table(ax, col_labels, rows, title, italic_col0=True):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.55)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_text_props(fontweight='bold', fontstyle='italic')
            cell.set_facecolor('#f0f0f0')
            cell.set_edgecolor('#cccccc')

        # Style data rows
        for i in range(len(rows)):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_edgecolor('#cccccc')
                if j == 0 and italic_col0:
                    cell.set_text_props(fontstyle='italic')

    # ── Panel A ──────────────────────────────────────────────────────
    _draw_table(
        axes[0],
        ['Variable', 'N', 'Mean', 'Std. Dev.', 'Min', 'Max'],
        panel_a_rows,
        'Panel A: Summary Statistics',
    )

    # ── Panel B ──────────────────────────────────────────────────────
    _draw_table(
        axes[1],
        [''] + var_names,
        panel_b_rows,
        'Panel B: Correlation',
    )

    # ── Panel C ──────────────────────────────────────────────────────
    ac_col_headers = [''] + [
        n.replace('_t+1', '$_{t+1}$').replace('_t', '$_t$')
        for n in all_ac_names
    ]
    _draw_table(
        axes[2],
        ac_col_headers,
        panel_c_rows,
        'Panel C: Autocorrelation',
    )

    plt.tight_layout(h_pad=2.0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"  Saved: {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from comomentum.compute_comomentum import compute_comomentum
    from data.market_variables import compute_market_variables

    print("Loading data...")
    data = load_all_data('input_data/')

    print("Computing momentum signal...")
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )

    print("Computing comomentum...")
    comomentum, comom_w, comom_l = compute_comomentum(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'], data['dates']
    )

    print("Computing market variables...")
    mret, mvol = compute_market_variables(
        data['ff_factors'], data['rf'], data['dates']
    )

    print("Generating summary statistics table...")
    generate_summary_table(
        comomentum, comom_w, comom_l, mret, mvol, data['dates'],
        save_path='output_data/summary_statistics_table.png'
    )
