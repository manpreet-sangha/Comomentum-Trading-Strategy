# step2_plots.py
# =====================================================================
# Step 2 — Momentum Diagnostic Plots
# =====================================================================
# Generates three diagnostic charts for the momentum signal:
#   1. Scatter plot: momentum exposure vs next-week return
#   2. Histogram: distribution of raw momentum
#   3. Factor comparison over time (4-panel, panels 3 & 4 placeholders)
#
# Standalone:   python step2_plots.py
# From loader:  from step2_plots import generate_step2_plots
# =====================================================================

import os
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from logger_setup import _setup_logger

log = _setup_logger()


def plot_scatter_momentum_vs_return(momentum_std, returns_clean, live,
                                    output_dir='output_data'):
    """
    Scatter plot: raw momentum (t) vs next-week return (t+1).
    Subsampled to 50k points for readability, with OLS fit line.
    """
    mom_flat  = momentum_std[:-1, :].ravel()
    ret_flat  = returns_clean[1:, :].ravel()
    live_flat = live[1:, :].ravel()

    mask = np.isfinite(mom_flat) & np.isfinite(ret_flat) & (live_flat == 1)

    np.random.seed(42)
    n_total = int(np.sum(mask))
    n_plot  = min(50_000, n_total)
    idx_all = np.where(mask)[0]
    idx_sample = np.random.choice(idx_all, size=n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(mom_flat[idx_sample], ret_flat[idx_sample] * 100,
               s=2, alpha=0.15, color='steelblue', rasterized=True)

    coeffs = polyfit(mom_flat[mask], ret_flat[mask] * 100, deg=1)
    x_line = np.linspace(-4, 4, 200)
    y_line = coeffs[0] + coeffs[1] * x_line
    ax.plot(x_line, y_line, color='red', linewidth=2,
            label=f'OLS fit (slope = {coeffs[1]:.4f}% per unit)')

    ax.set_xlabel('Momentum Exposure (t)', fontsize=12)
    ax.set_ylabel('Next-Week Stock Return (%, t+1)', fontsize=12)
    ax.set_title('Scatter Plot: Momentum Exposure vs. Next-Week Return',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'step2_scatter_momentum_vs_return.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_histogram_momentum(momentum_std, dates, output_dir='output_data'):
    """
    Two-panel histogram: (a) pooled across all weeks, (b) last week only.
    Uses raw momentum values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Pooled
    all_vals = momentum_std.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]

    axes[0].hist(all_vals, bins=100, color='steelblue', edgecolor='white',
                 alpha=0.85, density=True)
    axes[0].set_xlabel('Momentum', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('(a) Distribution of Raw Momentum\n'
                       '(All Weeks Pooled)', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.6,
                    label='Zero')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Panel (b): Last week
    last_vals = momentum_std[-1, :]
    last_vals = last_vals[np.isfinite(last_vals)]

    axes[1].hist(last_vals, bins=60, color='darkorange', edgecolor='white',
                 alpha=0.85, density=True)
    axes[1].set_xlabel('Momentum', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'(b) Cross-Sectional Distribution\n'
                       f'(Last Week: {dates[-1].strftime("%Y-%m-%d")})',
                       fontsize=13, fontweight='bold')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.6,
                    label='Zero')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'step2_histogram_momentum.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_factor_comparison(momentum, momentum_std, dates, T,
                           output_dir='output_data'):
    """
    4-panel time series: raw mean, standardised mean, comomentum
    (placeholder), adjusted momentum (placeholder).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Factor Comparison Over Time\n'
                 '(Cross-Sectional Mean per Week)',
                 fontsize=15, fontweight='bold', y=0.98)

    raw_mean = np.nanmean(momentum, axis=1)
    std_mean = np.nanmean(momentum_std, axis=1)

    # Panel 1: Raw momentum
    axes[0].plot(dates, raw_mean * 100, color='steelblue', linewidth=0.8)
    axes[0].axhline(0, color='black', linewidth=0.6, linestyle='--',
                    alpha=0.5)
    axes[0].set_ylabel('Mean raw return (%)', fontsize=10)
    axes[0].set_title('(1) Raw Momentum Factor — cross-sectional mean of '
                       '48-week compounded return',
                       fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.25)
    axes[0].fill_between(dates, raw_mean * 100, 0,
                         where=(raw_mean >= 0), alpha=0.15, color='green',
                         label='Positive mean (bull momentum)')
    axes[0].fill_between(dates, raw_mean * 100, 0,
                         where=(raw_mean < 0), alpha=0.15, color='red',
                         label='Negative mean (bear momentum)')
    axes[0].legend(fontsize=9, loc='upper right')

    # Panel 2: Placeholder (standardisation removed)
    axes[1].text(0.5, 0.5,
                 'Panel removed\n(standardisation no longer applied)',
                 transform=axes[1].transAxes,
                 ha='center', va='center', fontsize=11, color='gray',
                 style='italic',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightyellow',
                           edgecolor='orange', alpha=0.8))
    axes[1].set_ylabel('(unused)', fontsize=10)
    axes[1].set_title('(2) Standardised Momentum — removed (paper replication)',
                       fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.25)
    axes[1].set_yticks([])

    # Panel 3: Comomentum (placeholder)
    axes[2].text(0.5, 0.5,
                 'Comomentum (Step 4)\n\nNot yet computed.\n'
                 'Will show the average pairwise residual correlation\n'
                 'among momentum stocks each week.',
                 transform=axes[2].transAxes,
                 ha='center', va='center', fontsize=11, color='gray',
                 style='italic',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightyellow',
                           edgecolor='orange', alpha=0.8))
    axes[2].set_ylabel('Comomentum', fontsize=10)
    axes[2].set_title('(3) Comomentum — average residual correlation '
                       'among momentum stocks (placeholder: Step 4)',
                       fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.25)
    axes[2].set_yticks([])

    # Panel 4: Adjusted momentum (placeholder)
    axes[3].text(0.5, 0.5,
                 'Adjusted Momentum Factor (Step 5)\n\nNot yet computed.\n'
                 'Will show the comomentum-scaled standardised momentum\n'
                 '(inverse comomentum weighting applied, then '
                 're-standardised).',
                 transform=axes[3].transAxes,
                 ha='center', va='center', fontsize=11, color='gray',
                 style='italic',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightyellow',
                           edgecolor='purple', alpha=0.8))
    axes[3].set_ylabel('Mean z-score', fontsize=10)
    axes[3].set_title('(4) Adjusted Momentum Factor — cross-sectional '
                       'mean after comomentum adjustment '
                       '(placeholder: Step 5)',
                       fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.25)
    axes[3].set_yticks([])
    axes[3].set_xlabel('Date', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, 'step2_factor_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def generate_step2_plots(momentum, momentum_std, data,
                         output_dir='output_data'):
    """
    Runs all Step 2 diagnostic plots.

    INPUT:
        momentum     : np.ndarray (TxN) — raw momentum scores
        momentum_std : np.ndarray (TxN) — standardised momentum scores
        data         : dict — data dictionary from load_all_data()
        output_dir   : str  — folder for output PNGs
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 2 PLOTS: Generating momentum diagnostics")
    log.info("=" * 60)

    log.info("  Chart 1: Scatter (momentum vs next-week return)...")
    plot_scatter_momentum_vs_return(
        momentum_std, data['returns_clean'], data['live'], output_dir
    )

    log.info("  Chart 2: Histogram of momentum exposures...")
    plot_histogram_momentum(momentum_std, data['dates'], output_dir)

    log.info("  Chart 3: Factor comparison over time (4-panel)...")
    plot_factor_comparison(
        momentum, momentum_std, data['dates'], data['T'], output_dir
    )

    log.info("-" * 60)
    log.info("STEP 2 PLOTS COMPLETE")
    log.info("-" * 60)


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )
    generate_step2_plots(momentum, momentum_std, data)
