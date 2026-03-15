# plot_comom_event_study.py
# =====================================================================
# Event-Study Plot: Average Comomentum Around Portfolio Formation
# =====================================================================
#
# Reproduces Figure 2 from Lou & Polk (2021).
#
# TWO PANELS:
#   Top panel    — Avg COMOM across all formation dates
#   Bottom panel — Avg COMOM split by High-CoMOM vs Low-CoMOM periods
#                  (above/below the time-series median at Year 0)
#
# METHODOLOGY:
#   1. Each week with a valid comomentum value is treated as a
#      potential "formation date" (Year 0).
#   2. For each formation date t, we look up comomentum at relative
#      horizons: Year -4, Year -3, …, Year 0, …, Year +3, Year +4.
#      A "year" = 52 weeks.
#   3. Top panel: average across ALL formation dates.
#   4. Bottom panel: split formation dates into two groups —
#      "High COMOM" (Year 0 value ≥ median) and "Low COMOM"
#      (Year 0 value < median) — and average each group separately.
#
# Standalone:  python plot_comom_event_study.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WEEKS_PER_YEAR


def plot_comom_event_study(comomentum, dates,
                           max_years=8,
                           save_path='output_data/plot_comom_event_study.png'):
    """
    Produce a two-panel event-study plot of comomentum around
    portfolio formation (Year 0), replicating Lou & Polk (2021) Fig 2.

    INPUTS:
        comomentum : T-length np.ndarray — weekly comomentum series
        dates      : T-length array-like  — weekly dates
        max_years  : int — how many years before/after to plot (default 4)
        save_path  : str — where to save the figure

    RETURNS:
        rel_years  : list of ints  [-max_years … +max_years]
        avg_comom  : list of floats  (one per relative year)
    """

    T = len(comomentum)
    weeks_per_year = WEEKS_PER_YEAR  # 52

    # Relative year offsets in weeks
    rel_years = list(range(-max_years, max_years + 1))
    rel_offsets = [y * weeks_per_year for y in rel_years]
    year0_col = rel_years.index(0)  # column index of Year 0

    # ── Identify all valid formation weeks ───────────────────────────
    min_offset = min(rel_offsets)
    max_offset = max(rel_offsets)

    formation_weeks = []
    for t in range(T):
        if not np.isfinite(comomentum[t]):
            continue
        if (t + min_offset) < 0 or (t + max_offset) >= T:
            continue
        formation_weeks.append(t)

    n_formations = len(formation_weeks)
    print(f"  Event study: {n_formations} valid formation weeks")

    if n_formations == 0:
        print("  WARNING: No valid formation weeks — cannot plot.")
        return rel_years, [np.nan] * len(rel_years)

    # ── Collect comomentum at each relative horizon ──────────────────
    n_horizons = len(rel_years)
    comom_matrix = np.full((n_formations, n_horizons), np.nan)

    for i, t in enumerate(formation_weeks):
        for j, offset in enumerate(rel_offsets):
            comom_matrix[i, j] = comomentum[t + offset]

    # ── Overall average (top panel) ──────────────────────────────────
    avg_comom = np.nanmean(comom_matrix, axis=0)

    # ── Split by High / Low COMOM at Year 0 (bottom panel) ──────────
    year0_values = comom_matrix[:, year0_col]
    median_comom = np.nanmedian(year0_values)

    high_mask = year0_values >= median_comom
    low_mask  = year0_values < median_comom

    avg_high = np.nanmean(comom_matrix[high_mask, :], axis=0)
    avg_low  = np.nanmean(comom_matrix[low_mask, :], axis=0)

    n_high = int(np.sum(high_mask))
    n_low  = int(np.sum(low_mask))

    # ── Print summary table ──────────────────────────────────────────
    print(f"  Median CoMOM at Year 0: {median_comom:.6f}")
    print(f"  High-COMOM formations : {n_high}")
    print(f"  Low-COMOM  formations : {n_low}")
    print(f"\n  {'Rel Year':>10}  {'Avg All':>9}  {'Avg High':>9}  {'Avg Low':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}")
    for y, a_all, a_hi, a_lo in zip(rel_years, avg_comom, avg_high, avg_low):
        label = f"Year {y:+d}" if y != 0 else "Year  0"
        print(f"  {label:>10}  {a_all:9.6f}  {a_hi:9.6f}  {a_lo:9.6f}")

    # ── Create two-panel figure ──────────────────────────────────────
    x_labels = [f"Year {y}" for y in rel_years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11))

    # ────────── TOP PANEL: Avg COMOM ─────────────────────────────────
    ax1.plot(rel_years, avg_comom,
             color='black', linewidth=2,
             marker='o', markersize=7,
             markerfacecolor='black', markeredgecolor='black',
             zorder=3, label='Avg COMOM')

    ax1.set_xlim(-max_years - 0.5, max_years + 0.5)
    ax1.set_xticks(rel_years)
    ax1.set_xticklabels(x_labels, fontsize=11)

    y_max_1 = max(0.10, np.nanmax(avg_comom) * 1.15)
    ax1.set_ylim(0, y_max_1)
    ax1.set_yticks(np.arange(0, y_max_1 + 0.005, 0.01))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    ax1.legend(loc='lower center', fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, -0.13))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # ────────── BOTTOM PANEL: High vs Low COMOM ──────────────────────
    ax2.plot(rel_years, avg_high,
             color='#C0392B', linewidth=2,
             marker='o', markersize=7,
             markerfacecolor='#C0392B', markeredgecolor='#C0392B',
             zorder=3, label='High COMOM')

    ax2.plot(rel_years, avg_low,
             color='#7F8C8D', linewidth=2, linestyle='--',
             marker='s', markersize=7,
             markerfacecolor='#7F8C8D', markeredgecolor='#7F8C8D',
             zorder=3, label='Low COMOM')

    ax2.set_xlim(-max_years - 0.5, max_years + 0.5)
    ax2.set_xticks(rel_years)
    ax2.set_xticklabels(x_labels, fontsize=11)

    y_max_2 = max(0.16, np.nanmax(avg_high) * 1.15)
    ax2.set_ylim(0, y_max_2)
    ax2.set_yticks(np.arange(0, y_max_2 + 0.005, 0.02))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    ax2.legend(loc='lower center', fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(h_pad=4.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {save_path}")

    return rel_years, avg_comom.tolist()


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

    print("\nGenerating event-study plot...")
    rel_years, avg_comom = plot_comom_event_study(
        comomentum, data['dates'],
        max_years=4,
        save_path='output_data/plot_comom_event_study.png'
    )
