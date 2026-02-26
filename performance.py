# performance.py
# =====================================================================
# Performance Analytics & Plotting Module
# =====================================================================
# This module provides functions to:
#   1. Compute summary statistics for a factor-return time series
#      (annualised return, volatility, Sharpe ratio, t-stat, etc.)
#   2. Plot cumulative factor returns, comomentum, scaling factors,
#      and side-by-side bar-chart comparisons.
#
# Keeping these utilities separate from the main script makes them
# easy to reuse for any factor strategy, not just momentum.
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

WEEKS_PER_YEAR = 52  # used to annualise weekly statistics


# =====================================================================
# 1. Summary Statistics
# =====================================================================

def compute_stats(gamma_series, label, weeks_per_year=WEEKS_PER_YEAR):
    """
    Computes and prints a comprehensive set of summary statistics
    for a factor-return time series.

    INPUTS:
        gamma_series  : T-length np.ndarray - weekly factor returns (may contain NaN)
        label         : str - descriptive name for this strategy (used in printout)
        weeks_per_year: int - number of trading weeks per year (default 52)

    OUTPUTS:
        stats : dict - dictionary containing all computed metrics:
            'label'    : str   - strategy name
            'n'        : int   - number of valid (non-NaN) weeks
            'mean_w'   : float - mean weekly return
            'std_w'    : float - std dev of weekly returns
            'mean_ann' : float - annualised mean return
            'std_ann'  : float - annualised std dev
            'sharpe'   : float - annualised Sharpe ratio
            'tstat'    : float - t-statistic (mean / standard error)
            'skew'     : float - skewness of weekly returns
            'kurt'     : float - excess kurtosis of weekly returns
            'max_dd'   : float - maximum drawdown of cumulative returns
    """

    # Filter out NaN entries
    valid = gamma_series[np.isfinite(gamma_series)]
    n = len(valid)

    # ----- Basic weekly statistics -----
    mean_w = np.mean(valid)
    std_w = np.std(valid, ddof=1)

    # ----- Annualise -----
    # Mean scales linearly with time; std dev scales with sqrt(time)
    mean_ann = mean_w * weeks_per_year
    std_ann = std_w * np.sqrt(weeks_per_year)

    # ----- Sharpe ratio (annualised) -----
    sharpe = mean_ann / std_ann if std_ann > 0 else np.nan

    # ----- T-statistic -----
    # Tests H0: mean factor return = 0
    tstat = mean_w / (std_w / np.sqrt(n)) if (std_w > 0 and n > 1) else np.nan

    # ----- Maximum drawdown -----
    # Drawdown = how far the cumulative return has fallen from its
    # running peak. Max drawdown = worst such decline.
    cum = np.cumsum(valid)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = np.min(drawdown)

    # ----- Higher moments -----
    skew = pd.Series(valid).skew()
    kurt = pd.Series(valid).kurtosis()  # excess kurtosis

    # ----- Print results -----
    print(f"\n  --- {label} ---")
    print(f"  Number of valid weeks:       {n}")
    print(f"  Mean weekly return:          {mean_w * 100:.4f}%")
    print(f"  Std weekly return:           {std_w * 100:.4f}%")
    print(f"  Annualised mean return:      {mean_ann * 100:.2f}%")
    print(f"  Annualised std deviation:    {std_ann * 100:.2f}%")
    print(f"  Annualised Sharpe ratio:     {sharpe:.3f}")
    print(f"  T-statistic:                 {tstat:.3f}")
    print(f"  Skewness:                    {skew:.3f}")
    print(f"  Excess kurtosis:             {kurt:.3f}")
    print(f"  Max drawdown (cumulative):   {max_dd * 100:.2f}%")

    return {
        'label': label, 'n': n,
        'mean_w': mean_w, 'std_w': std_w,
        'mean_ann': mean_ann, 'std_ann': std_ann,
        'sharpe': sharpe, 'tstat': tstat,
        'skew': skew, 'kurt': kurt, 'max_dd': max_dd
    }


def print_summary_table(stats_std, stats_adj):
    """
    Prints a formatted side-by-side comparison table of the standard
    and adjusted momentum strategies.

    INPUTS:
        stats_std : dict - statistics for the standard momentum factor
        stats_adj : dict - statistics for the adjusted momentum factor
    """
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<35} {'Standard Mom':>15} {'Adjusted Mom':>15}")
    print("-" * 65)
    print(f"{'Ann. Mean Return (%)':<35} {stats_std['mean_ann']*100:>15.2f} {stats_adj['mean_ann']*100:>15.2f}")
    print(f"{'Ann. Std Dev (%)':<35} {stats_std['std_ann']*100:>15.2f} {stats_adj['std_ann']*100:>15.2f}")
    print(f"{'Ann. Sharpe Ratio':<35} {stats_std['sharpe']:>15.3f} {stats_adj['sharpe']:>15.3f}")
    print(f"{'T-Statistic':<35} {stats_std['tstat']:>15.3f} {stats_adj['tstat']:>15.3f}")
    print(f"{'Skewness':<35} {stats_std['skew']:>15.3f} {stats_adj['skew']:>15.3f}")
    print(f"{'Excess Kurtosis':<35} {stats_std['kurt']:>15.3f} {stats_adj['kurt']:>15.3f}")
    print(f"{'Max Drawdown (%)':<35} {stats_std['max_dd']*100:>15.2f} {stats_adj['max_dd']*100:>15.2f}")
    print("=" * 70)


# =====================================================================
# 2. Plotting Functions
# =====================================================================

def plot_main_results(dates, gamma_std, gamma_adj, comomentum, scaling,
                      save_path='momentum_results.png'):
    """
    Creates a 3-panel figure showing:
        Panel 1: Cumulative factor returns - standard vs. adjusted momentum
        Panel 2: Comomentum time series
        Panel 3: Momentum scaling factor over time

    INPUTS:
        dates      : pd.DatetimeIndex - weekly date vector
        gamma_std  : T-length array - standard momentum factor returns
        gamma_adj  : T-length array - adjusted momentum factor returns
        comomentum : T-length array - comomentum measure
        scaling    : T-length array - time-varying scaling factor
        save_path  : str - file path to save the figure
    """

    # Cumulative returns (sum of weekly log-like returns)
    cum_std = np.nancumsum(gamma_std)
    cum_adj = np.nancumsum(gamma_adj)

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # ---- Panel 1: Cumulative Factor Returns ----
    ax1 = axes[0]
    ax1.plot(dates, cum_std, label='Standard Momentum',
             linewidth=1.2, color='steelblue')
    ax1.plot(dates, cum_adj, label='Adjusted Momentum (Comomentum)',
             linewidth=1.2, color='darkorange')
    ax1.set_title('Cumulative Factor Returns: Standard vs. Adjusted Momentum',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))

    # ---- Panel 2: Comomentum Time Series ----
    ax2 = axes[1]
    ax2.plot(dates, comomentum, linewidth=0.8, color='purple')
    ax2.set_title('Comomentum (Average Pairwise Abnormal Return Correlation)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Comomentum')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))

    # ---- Panel 3: Scaling Factor ----
    ax3 = axes[2]
    ax3.plot(dates, scaling, linewidth=0.8, color='green')
    ax3.set_title('Momentum Scaling Factor (Inverse Comomentum Signal)',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Scaling Factor')
    ax3.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5,
                label='Neutral (1.5)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Main results chart saved as '{save_path}'")


def plot_comparison_bars(stats_std, stats_adj,
                         save_path='momentum_comparison.png'):
    """
    Creates a 3-panel bar chart comparing the standard and adjusted
    momentum strategies on:
        - Annualised mean return
        - Annualised Sharpe ratio
        - T-statistic (with a 5% significance threshold line)

    INPUTS:
        stats_std : dict - statistics for the standard strategy
        stats_adj : dict - statistics for the adjusted strategy
        save_path : str  - file path to save the figure
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ['Standard\nMomentum', 'Adjusted\nMomentum']
    colors = ['steelblue', 'darkorange']

    # Bar 1: Annualised mean return
    axes[0].bar(labels,
                [stats_std['mean_ann'] * 100, stats_adj['mean_ann'] * 100],
                color=colors)
    axes[0].set_title('Annualised Mean Return (%)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('%')

    # Bar 2: Annualised Sharpe ratio
    axes[1].bar(labels,
                [stats_std['sharpe'], stats_adj['sharpe']],
                color=colors)
    axes[1].set_title('Annualised Sharpe Ratio',
                      fontsize=12, fontweight='bold')

    # Bar 3: T-statistic
    axes[2].bar(labels,
                [stats_std['tstat'], stats_adj['tstat']],
                color=colors)
    axes[2].set_title('T-Statistic',
                      fontsize=12, fontweight='bold')
    axes[2].axhline(y=1.96, color='red', linestyle='--', alpha=0.7,
                    label='1.96 (5% significance)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Comparison chart saved as '{save_path}'")
