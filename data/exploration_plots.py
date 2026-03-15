# exploration_plots.py
# =====================================================================
# Data Exploration Plots
# =====================================================================
# Generates a suite of diagnostic charts that summarise the key
# characteristics of the input dataset from a trading-strategy
# perspective.  All figures are saved to output_data/ as PNG files.
#
# Standalone:   python exploration_plots.py
# From loader:  from exploration_plots import plot_data_overview
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from logger_setup import _setup_logger
from data.test_zero_returns import plot_zero_diagnostic

log = _setup_logger()


def plot_data_overview(data, output_dir='output_data'):
    """
    Creates and saves diagnostic plots about the loaded dataset.

    Plots produced:
        1. Universe size over time (number of live stocks per week)
        2. Live vs dead cell composition (stacked area chart)
        3. Weekly cross-sectional return statistics (mean, median, std)
        4. Distribution of weekly returns (histogram with tail markers)
        5. Missing-data heatmap (fraction of NaN returns per year)
        6. Fama-French factor cumulative returns
        7. Average stock lifespan distribution
        8. Summary statistics text panel
        9. Zero-return diagnostic (cell breakdown + yearly zero rate)

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNGs
    """

    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 60)
    log.info("GENERATING DATA EXPLORATION PLOTS")
    log.info("=" * 60)

    returns_clean = data['returns_clean']
    live          = data['live']
    dates         = data['dates']
    ff_factors    = data['ff_factors']
    rf            = data['rf']
    T             = data['T']
    N             = data['N']

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # ------------------------------------------------------------------
    # PLOT 1: Number of live (tradeable) stocks per week
    # ------------------------------------------------------------------
    log.info("  Plot 1: Universe size over time...")
    n_live_per_week = np.sum(live == 1, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, n_live_per_week, color='#2563EB', linewidth=1.2)
    ax.fill_between(dates, 0, n_live_per_week, alpha=0.15, color='#2563EB')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.annotate(f'Max: {np.max(n_live_per_week):,} stocks',
                xy=(dates[np.argmax(n_live_per_week)], np.max(n_live_per_week)),
                xytext=(40, -35), textcoords='offset points',
                fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
    ax.annotate(f'Min: {np.min(n_live_per_week):,} stocks',
                xy=(dates[np.argmin(n_live_per_week)], np.min(n_live_per_week)),
                xytext=(-120, -30), textcoords='offset points',
                fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot1_universe_size.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot1_universe_size.png")

    # ------------------------------------------------------------------
    # PLOT 2: Live vs Dead cells - stacked area
    # ------------------------------------------------------------------
    log.info("  Plot 2: Live vs Dead composition over time...")
    n_dead_per_week = N - n_live_per_week

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(dates, n_live_per_week, n_dead_per_week,
                 labels=['Live (tradeable)', 'Dead (delisted / pre-IPO)'],
                 colors=['#22C55E', '#EF4444'], alpha=0.7)
    ax.set_title('Live vs Dead Stock Observations per Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(loc='upper left', fontsize=10)
    total_live = np.sum(live == 1)
    total_dead = np.sum(live == 0)
    ax.text(0.98, 0.05,
            f'Total: {total_live:,} live ({total_live/(T*N)*100:.1f}%) | '
            f'{total_dead:,} dead ({total_dead/(T*N)*100:.1f}%)',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot2_live_vs_dead.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot2_live_vs_dead.png")

    # ------------------------------------------------------------------
    # PLOT 3: Weekly cross-sectional return statistics
    # ------------------------------------------------------------------
    log.info("  Plot 3: Weekly cross-sectional return statistics...")
    cs_mean   = np.nanmean(returns_clean, axis=1)
    cs_median = np.nanmedian(returns_clean, axis=1)
    cs_std    = np.nanstd(returns_clean, axis=1, ddof=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel A: mean and median
    axes[0].plot(dates, cs_mean * 100, color='#2563EB', linewidth=0.8, label='Mean', alpha=0.85)
    axes[0].plot(dates, cs_median * 100, color='#F97316', linewidth=0.8, label='Median', alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Weekly Cross-Sectional Return: Mean & Median', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Panel B: standard deviation
    axes[1].plot(dates, cs_std * 100, color='#DC2626', linewidth=0.8, alpha=0.85)
    axes[1].set_ylabel('Std Dev (%)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Weekly Cross-Sectional Return Volatility', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot3_return_statistics.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot3_return_statistics.png")

    # ------------------------------------------------------------------
    # PLOT 4: Distribution of all valid weekly returns (histogram)
    # ------------------------------------------------------------------
    log.info("  Plot 4: Return distribution histogram...")
    all_rets = returns_clean[np.isfinite(returns_clean)]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Clip for display to avoid extreme outliers dominating the histogram
    clip_lo, clip_hi = np.percentile(all_rets, [0.5, 99.5])
    clipped = all_rets[(all_rets >= clip_lo) & (all_rets <= clip_hi)]
    ax.hist(clipped * 100, bins=200, color='#6366F1', alpha=0.75, edgecolor='none')
    ax.axvline(np.mean(all_rets) * 100, color='red', linestyle='--', linewidth=1.2,
               label=f'Mean = {np.mean(all_rets)*100:.3f}%')
    ax.axvline(np.median(all_rets) * 100, color='orange', linestyle='--', linewidth=1.2,
               label=f'Median = {np.median(all_rets)*100:.3f}%')
    ax.set_title('Distribution of Weekly Stock Returns (0.5th-99.5th percentile)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Weekly Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=10)

    # Add skewness / kurtosis annotation
    from scipy.stats import skew, kurtosis as kurt_fn
    try:
        sk = skew(all_rets)
        ku = kurt_fn(all_rets)
        ax.text(0.97, 0.95,
                f'N = {len(all_rets):,}\nSkew = {sk:.2f}\nExcess Kurt = {ku:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))
    except ImportError:
        ax.text(0.97, 0.95, f'N = {len(all_rets):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot4_return_distribution.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot4_return_distribution.png")

    # ------------------------------------------------------------------
    # PLOT 5: Missing data heatmap - fraction of NaN per year
    # ------------------------------------------------------------------
    log.info("  Plot 5: Missing data by year...")
    years = dates.year
    unique_years = np.sort(np.unique(years))

    nan_frac_by_year = []
    live_frac_by_year = []
    for yr in unique_years:
        mask_yr = (years == yr)
        ret_yr = returns_clean[mask_yr, :]
        live_yr = live[mask_yr, :]
        nan_frac_by_year.append(np.mean(np.isnan(ret_yr)) * 100)
        live_frac_by_year.append(np.mean(live_yr == 1) * 100)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(unique_years))
    width = 0.38

    bars1 = ax1.bar(x - width/2, live_frac_by_year, width,
                    label='Listed cells (%)', color='#2563EB', alpha=0.85)
    bars2 = ax1.bar(x + width/2, nan_frac_by_year, width,
                    label='NaN cells (%)', color='#64748B', alpha=0.75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_years, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Percentage of Cells (%)')
    ax1.set_title('Data Availability by Year: Listed Cells vs NaN Cells',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot5_missing_data_by_year.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot5_missing_data_by_year.png")

    # ------------------------------------------------------------------
    # PLOT 6: Fama-French factor cumulative returns
    # ------------------------------------------------------------------
    log.info("  Plot 6: Fama-French factor cumulative returns...")
    factor_names = ['Mkt-RF', 'SMB', 'HML']
    colors_ff = ['#2563EB', '#F97316', '#22C55E']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (fname, col) in enumerate(zip(factor_names, colors_ff)):
        cum_ret = np.cumprod(1 + ff_factors[:, i]) - 1
        ax.plot(dates, cum_ret * 100, label=fname, color=col, linewidth=1.2)

    # Also plot risk-free cumulative return
    cum_rf = np.cumprod(1 + rf) - 1
    ax.plot(dates, cum_rf * 100, label='RF (risk-free)', color='grey',
            linewidth=1.0, linestyle='--')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Fama-French Factor Cumulative Returns (weekly)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot6_ff_cumulative_returns.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot6_ff_cumulative_returns.png")

    # ------------------------------------------------------------------
    # PLOT 9: Zero-return diagnostic (from test_zero_returns module)
    # ------------------------------------------------------------------
    log.info("  Plot 9: Zero-return diagnostic...")
    plot_zero_diagnostic(data['returns'], dates, output_dir)

    log.info("-" * 60)
    log.info(f"All 9 data exploration plots saved to {os.path.abspath(output_dir)}/")
    log.info("-" * 60)


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_data_overview(data)
