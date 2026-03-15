# plot_comom_time_series.py
# =====================================================================
# Time-Series Plot of Comomentum — Replicating Lou & Polk (2021) Fig 1
# =====================================================================
#
# Plots the average pairwise abnormal return correlation (comomentum)
# sampled at 6-month intervals.  Three series are shown:
#
#   1. CoMOM       (red solid)  — average of winner & loser comomentum
#   2. CoMOM_W     (blue dashed) — winner decile only
#   3. CoMOM_L     (grey dashed) — loser  decile only
#
# The x-axis shows dates labelled every 6 months (e.g. Mar-93, Sep-93).
#
# Standalone:  python plot_comom_time_series.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import WEEKS_PER_YEAR


def plot_comom_time_series(comomentum, comom_winner, comom_loser, dates,
                           sample_months=6,
                           save_path='output_data/plot_comom_time_series.png'):
    """
    Plot the comomentum time series sampled at fixed intervals,
    replicating the style of Lou & Polk (2021) Figure 1.

    INPUTS:
        comomentum   : T-length array — CoMOM = 0.5*(CoMOM_W + CoMOM_L)
        comom_winner : T-length array — winner-decile comomentum
        comom_loser  : T-length array — loser-decile  comomentum
        dates        : T-length DatetimeIndex
        sample_months: int — sampling interval in months (default 6)
        save_path    : str — output file path
    """

    # ── Build a DataFrame with weekly data ───────────────────────────
    dates = pd.DatetimeIndex(dates)
    df = pd.DataFrame({
        'date': dates,
        'comom': comomentum,
        'comom_w': comom_winner,
        'comom_l': comom_loser,
    }).set_index('date')

    # ── Resample to 6-month periods (mean within each period) ────────
    rule = f'{sample_months}ME'
    df_semi = df.resample(rule).mean()

    # Drop rows where all three are NaN
    df_semi = df_semi.dropna(how='all')

    n_points = len(df_semi)
    print(f"  Comomentum time-series plot: {n_points} sample points "
          f"({sample_months}-month intervals)")

    if n_points == 0:
        print("  WARNING: No valid sample points — cannot plot.")
        return

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df_semi.index, df_semi['comom'],
            color='red', linewidth=1.8, label='CoMOM')

    ax.plot(df_semi.index, df_semi['comom_w'],
            color='#2980B9', linewidth=1.2, linestyle='--',
            label='CoMOM$^W$ (winners)')

    ax.plot(df_semi.index, df_semi['comom_l'],
            color='#7F8C8D', linewidth=1.2, linestyle='--',
            label='CoMOM$^L$ (losers)')

    # ── Formatting ───────────────────────────────────────────────────
    ax.set_ylabel('Average Pairwise Abnormal Correlation', fontsize=12)
    ax.set_title('Comomentum Time Series',
                 fontsize=13, fontweight='bold')

    # X-axis: show labels every 6 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 9]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha='right',
             fontsize=8)

    # Y-axis
    y_min = min(0, np.nanmin(df_semi[['comom', 'comom_w', 'comom_l']].values) - 0.01)
    y_max = np.nanmax(df_semi[['comom', 'comom_w', 'comom_l']].values) * 1.15
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from comomentum.compute_comomentum import compute_comomentum

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

    print("Plotting time series...")
    plot_comom_time_series(
        comomentum, comom_w, comom_l, data['dates'],
        sample_months=6,
        save_path='output_data/plot_comom_time_series.png'
    )
