# plot_cleaning_impact.py
# =====================================================================
# Cleaning Impact Diagnostic Plot
# =====================================================================
# Visualises the effect of removing non-listed observations from the
# raw return matrix.  From a finance perspective this answers:
#
#   1. How large is the investable universe at each point in time?
#   2. What fraction of the panel is discarded as non-investable?
#   3. How does the cleaning reshape the cross-sectional distribution
#      that feeds into momentum scoring and portfolio construction?
#
# The figure has three panels:
#   (A) Investable universe over time — shows how many stocks a
#       portfolio manager could actually trade each week (listed)
#       vs how many are in the file but not tradeable (non-listed).
#   (B) Cumulative cleaning rate — the running share of all cells
#       that have been set to NaN, highlighting whether the data
#       becomes cleaner or dirtier as the sample grows.
#   (C) Per-year bar chart — annual counts of listed vs non-listed
#       cells, showing the balance between usable and discarded data
#       year by year.
#
# Standalone:   python plot_cleaning_impact.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from logger_setup import _setup_logger

log = _setup_logger()


def plot_cleaning_impact(data, output_dir='output_data'):
    """
    Three-panel figure showing the impact of removing non-listed
    observations from the raw return matrix.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """

    os.makedirs(output_dir, exist_ok=True)

    live   = data['live']
    dates  = data['dates']
    T      = data['T']
    N      = data['N']

    # ── derived series ───────────────────────────────────────────────
    n_listed     = np.sum(live == 1, axis=1)        # (T,) listed per week
    n_not_listed = N - n_listed                     # (T,) non-listed per week
    pct_listed   = n_listed / N * 100               # (T,) % listed

    # Cumulative cleaning rate (running fraction set to NaN)
    cum_not_listed = np.cumsum(n_not_listed)
    cum_total      = np.arange(1, T + 1) * N
    cum_clean_pct  = cum_not_listed / cum_total * 100

    # Per-year aggregates
    years        = dates.year
    unique_years = np.sort(np.unique(years))
    yr_listed    = []
    yr_not_listed = []
    for yr in unique_years:
        mask = (years == yr)
        yr_listed.append(np.sum(live[mask, :] == 1))
        yr_not_listed.append(np.sum(live[mask, :] == 0))

    # ── colours ──────────────────────────────────────────────────────
    CLR_LISTED     = '#2563EB'   # blue  – investable
    CLR_NOT_LISTED = '#EF4444'   # red   – discarded
    CLR_BG         = '#F8FAFC'
    CLR_TEXT       = '#1E293B'

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    fig.patch.set_facecolor(CLR_BG)

    # ── Panel A: Investable universe over time ───────────────────────
    ax = axes[0]
    ax.fill_between(dates, 0, n_listed, color=CLR_LISTED,
                    alpha=0.25, label='Listed (investable)')
    ax.fill_between(dates, n_listed, N, color=CLR_NOT_LISTED,
                    alpha=0.20, label='Not listed (removed)')
    ax.plot(dates, n_listed, color=CLR_LISTED, linewidth=1.0)
    ax.axhline(N, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.text(dates[-1], N, f'  N = {N:,}', va='bottom', ha='right',
            fontsize=8, color='grey')

    # Key annotations
    peak_idx = np.argmax(n_listed)
    trough_idx = np.argmin(n_listed)
    ax.annotate(f'Peak: {n_listed[peak_idx]:,} stocks\n'
                f'({pct_listed[peak_idx]:.1f}% of universe)',
                xy=(dates[peak_idx], n_listed[peak_idx]),
                xytext=(40, -30), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', color=CLR_LISTED, lw=1.0),
                color=CLR_LISTED)
    ax.annotate(f'Trough: {n_listed[trough_idx]:,} stocks\n'
                f'({pct_listed[trough_idx]:.1f}%)',
                xy=(dates[trough_idx], n_listed[trough_idx]),
                xytext=(40, 20), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', color=CLR_NOT_LISTED, lw=1.0),
                color=CLR_NOT_LISTED)

    ax.set_title('(A)  Investable Universe After Cleaning',
                 fontsize=13, fontweight='bold', color=CLR_TEXT)
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_facecolor(CLR_BG)

    # ── Panel B: Cumulative cleaning rate ────────────────────────────
    ax = axes[1]
    ax.plot(dates, cum_clean_pct, color=CLR_NOT_LISTED, linewidth=1.3)
    ax.fill_between(dates, 0, cum_clean_pct, color=CLR_NOT_LISTED, alpha=0.12)

    # Final rate annotation
    final_pct = cum_clean_pct[-1]
    ax.axhline(final_pct, color=CLR_NOT_LISTED, linewidth=0.6,
               linestyle=':', alpha=0.6)
    total_removed = int(np.sum(live == 0))
    total_cells   = T * N
    ax.text(dates[T // 2], final_pct + 1.5,
            f'Final: {final_pct:.1f}% of all cells removed  '
            f'({total_removed:,} / {total_cells:,})',
            ha='center', fontsize=9, color=CLR_NOT_LISTED,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=CLR_NOT_LISTED, alpha=0.85, linewidth=0.6))

    ax.set_title('(B)  Cumulative Share of Panel Marked as Not Listed',
                 fontsize=13, fontweight='bold', color=CLR_TEXT)
    ax.set_ylabel('Cumulative % Removed')
    ax.set_ylim(0, max(cum_clean_pct) + 8)
    ax.set_facecolor(CLR_BG)

    # ── Panel C: Per-year listed vs not-listed cells ─────────────────
    ax = axes[2]
    x = np.arange(len(unique_years))
    width = 0.38

    ax.bar(x - width / 2, yr_listed, width, label='Listed (kept)',
           color=CLR_LISTED, alpha=0.80)
    ax.bar(x + width / 2, yr_not_listed, width, label='Not listed (removed)',
           color=CLR_NOT_LISTED, alpha=0.80)

    ax.set_xticks(x)
    ax.set_xticklabels(unique_years, rotation=45, ha='right', fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_title('(C)  Annual Cell Counts: Kept vs Removed',
                 fontsize=13, fontweight='bold', color=CLR_TEXT)
    ax.set_ylabel('Number of Stock-Weeks')
    ax.set_xlabel('Year')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_facecolor(CLR_BG)

    # ── save ─────────────────────────────────────────────────────────
    fig.suptitle('Impact of Removing Non-Listed Observations from US_Returns',
                 fontsize=15, fontweight='bold', color=CLR_TEXT, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_dir, 'plot10_cleaning_impact.png')
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_cleaning_impact(data)
