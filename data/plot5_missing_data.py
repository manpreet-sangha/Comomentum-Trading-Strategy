# plot5_missing_data.py
# =====================================================================
# Plot 5: Data Availability by Year
# =====================================================================
# Grouped bar chart showing, for each calendar year, the percentage
# of cells that are listed (have live==1) vs NaN in the cleaned
# return matrix.
#
# Standalone:   python plot5_missing_data.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import matplotlib.pyplot as plt
from logger_setup import _setup_logger

log = _setup_logger()


def plot_missing_data(data, output_dir='output_data'):
    """
    Per-year bar chart of listed cells vs NaN cells.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    returns_clean = data['returns_clean']
    live          = data['live']
    dates         = data['dates']

    years        = dates.year
    unique_years = np.sort(np.unique(years))

    nan_frac_by_year  = []
    live_frac_by_year = []
    for yr in unique_years:
        mask_yr = (years == yr)
        ret_yr  = returns_clean[mask_yr, :]
        live_yr = live[mask_yr, :]
        nan_frac_by_year.append(np.mean(np.isnan(ret_yr)) * 100)
        live_frac_by_year.append(np.mean(live_yr == 1) * 100)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    x     = np.arange(len(unique_years))
    width = 0.38

    ax.bar(x - width / 2, live_frac_by_year, width,
           label='Listed cells (%)', color='#2563EB', alpha=0.85)
    ax.bar(x + width / 2, nan_frac_by_year, width,
           label='NaN cells (%)', color='#64748B', alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_years, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage of Cells (%)')
    ax.set_title('Data Availability by Year: Listed Cells vs NaN Cells',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'plot5_missing_data_by_year.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved {save_path}")


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    plot_missing_data(data)
