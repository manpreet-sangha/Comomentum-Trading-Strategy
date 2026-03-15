# plot1_universe_size.py
# =====================================================================
# Plot 1: Investable Universe Size Over Time
# =====================================================================
# Shows the number of listed (tradeable) stocks per week across the
# full sample period.  The blue area highlights how the investable
# universe grows then shrinks — reflecting IPO waves and delistings.
#
# Standalone:   python plot1_universe_size.py
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


def plot_universe_size(data, output_dir='output_data'):
    """
    Line chart of the number of listed stocks per week.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    live  = data['live']
    dates = data['dates']

    n_live_per_week = np.sum(live == 1, axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, n_live_per_week, color='#2563EB', linewidth=1.2)
    ax.fill_between(dates, 0, n_live_per_week, alpha=0.15, color='#2563EB')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Force y-axis to start at 0 with clean round ticks
    y_max = int(np.max(n_live_per_week) * 1.08)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500))

    # Remove the offset text box that matplotlib adds to axes
    ax.yaxis.get_offset_text().set_visible(False)
    ax.xaxis.get_offset_text().set_visible(False)

    # Remove spines that double up with the grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate(f'Max: {np.max(n_live_per_week):,} stocks',
                xy=(dates[np.argmax(n_live_per_week)],
                    np.max(n_live_per_week)),
                xytext=(40, -35), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='grey'))
    ax.annotate(f'Min: {np.min(n_live_per_week):,} stocks',
                xy=(dates[np.argmin(n_live_per_week)],
                    np.min(n_live_per_week)),
                xytext=(-120, -30), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='grey'))
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'plot1_universe_size.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_universe_size(data)
