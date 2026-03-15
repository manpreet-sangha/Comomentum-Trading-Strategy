# plot4_return_distribution.py
# =====================================================================
# Plot 4: Distribution of Weekly Stock Returns
# =====================================================================
# Histogram of all valid (finite) weekly returns clipped to the
# 0.5th–99.5th percentile range, with mean/median lines and
# skewness / excess kurtosis annotation.
#
# Standalone:   python plot4_return_distribution.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_return_distribution(data, output_dir='output_data'):
    """
    Histogram of all valid weekly returns (0.5th–99.5th percentile).

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    returns_clean = data['returns_clean']
    all_rets = returns_clean[np.isfinite(returns_clean)]

    clip_lo, clip_hi = np.percentile(all_rets, [0.5, 99.5])
    clipped = all_rets[(all_rets >= clip_lo) & (all_rets <= clip_hi)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clipped * 100, bins=200, color='#6366F1', alpha=0.75,
            edgecolor='none')
    ax.axvline(np.mean(all_rets) * 100, color='red', linestyle='--',
               linewidth=1.2,
               label=f'Mean = {np.mean(all_rets)*100:.3f}%')
    ax.axvline(np.median(all_rets) * 100, color='orange', linestyle='--',
               linewidth=1.2,
               label=f'Median = {np.median(all_rets)*100:.3f}%')
    ax.set_title(
        'Distribution of Weekly Stock Returns (0.5th-99.5th percentile)',
        fontsize=14, fontweight='bold')
    ax.set_xlabel('Weekly Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=10)

    # Skewness / kurtosis annotation
    from scipy.stats import skew, kurtosis as kurt_fn
    try:
        sk = skew(all_rets)
        ku = kurt_fn(all_rets)
        ax.text(0.97, 0.95,
                f'N = {len(all_rets):,}\nSkew = {sk:.2f}\n'
                f'Excess Kurt = {ku:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))
    except ImportError:
        ax.text(0.97, 0.95, f'N = {len(all_rets):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    fig.tight_layout()

    save_path = os.path.join(output_dir, 'plot4_return_distribution.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_return_distribution(data)
