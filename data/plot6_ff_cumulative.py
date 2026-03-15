# plot6_ff_cumulative.py
# =====================================================================
# Plot 6: Fama-French Factor Cumulative Returns
# =====================================================================
# Cumulative return lines for Mkt-RF, SMB, HML and the risk-free
# rate over the full sample period.
#
# Standalone:   python plot6_ff_cumulative.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_ff_cumulative(data, output_dir='output_data'):
    """
    Line chart of Fama-French factor cumulative returns.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    ff_factors = data['ff_factors']
    rf         = data['rf']
    dates      = data['dates']

    factor_names = ['Mkt-RF', 'SMB', 'HML']
    colors_ff    = ['#2563EB', '#F97316', '#22C55E']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (fname, col) in enumerate(zip(factor_names, colors_ff)):
        cum_ret = np.cumprod(1 + ff_factors[:, i]) - 1
        ax.plot(dates, cum_ret * 100, label=fname, color=col,
                linewidth=1.2)

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

    save_path = os.path.join(output_dir, 'plot6_ff_cumulative_returns.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_ff_cumulative(data)
