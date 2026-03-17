# plot3_return_statistics.py
# =====================================================================
# Plot 3: Weekly Cross-Sectional Return Statistics
# =====================================================================
# Two-panel figure:
#   (A) Mean and median weekly return across all listed stocks.
#   (B) Cross-sectional standard deviation (dispersion) per week.
#
# Standalone:   python plot3_return_statistics.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_return_statistics(data, output_dir='output_data'):
    """
    Two-panel chart of weekly cross-sectional mean, median, and
    standard deviation of cleaned returns.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    returns_clean = data['returns_clean']
    dates         = data['dates']

    cs_mean   = np.nanmean(returns_clean, axis=1)
    cs_median = np.nanmedian(returns_clean, axis=1)
    cs_std    = np.nanstd(returns_clean, axis=1, ddof=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # Panel A: mean and median
    axes[0].plot(dates, cs_mean * 100, color='#2563EB', linewidth=0.6,
                 label='Mean', alpha=0.85)
    axes[0].plot(dates, cs_median * 100, color='#F97316', linewidth=0.6,
                 label='Median', alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.4, linestyle='--')
    axes[0].set_ylabel('Return (%)', fontsize=8)
    axes[0].set_title('Mean \& Median',
                       fontsize=9, fontweight='bold')
    axes[0].legend(fontsize=7)
    axes[0].tick_params(labelsize=7)

    # Panel B: standard deviation
    axes[1].plot(dates, cs_std * 100, color='#DC2626', linewidth=0.6,
                 alpha=0.85)
    axes[1].set_ylabel('Std Dev (%)', fontsize=8)
    axes[1].set_xlabel('Date', fontsize=8)
    axes[1].set_title('Volatility',
                       fontsize=9, fontweight='bold')
    axes[1].tick_params(labelsize=7)

    fig.tight_layout()

    save_path = os.path.join(output_dir, 'plot3_return_statistics.png')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_return_statistics(data)
