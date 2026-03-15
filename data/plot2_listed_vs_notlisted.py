# plot2_listed_vs_notlisted.py
# =====================================================================
# Plot 2: Listed vs Not-Listed Composition Over Time
# =====================================================================
# Stacked area chart showing the weekly split between listed
# (investable) and not-listed (pre-IPO / delisted) stock-weeks.
#
# Standalone:   python plot2_listed_vs_notlisted.py
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


def plot_listed_vs_notlisted(data, output_dir='output_data'):
    """
    Stacked area chart of listed vs not-listed stocks per week.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    live  = data['live']
    dates = data['dates']
    T     = data['T']
    N     = data['N']

    n_listed     = np.sum(live == 1, axis=1)
    n_not_listed = N - n_listed

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(dates, n_listed, n_not_listed,
                 labels=['Listed (investable)',
                         'Not listed (pre-IPO / delisted)'],
                 colors=['#2563EB', '#64748B'], alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(loc='upper left', fontsize=10)

    total_listed     = int(np.sum(live == 1))
    total_not_listed = int(np.sum(live == 0))
    ax.text(0.98, 0.05,
            f'Total: {total_listed:,} listed '
            f'({total_listed / (T * N) * 100:.1f}%)  |  '
            f'{total_not_listed:,} not listed '
            f'({total_not_listed / (T * N) * 100:.1f}%)',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'plot2_listed_vs_notlisted.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_listed_vs_notlisted(data)
