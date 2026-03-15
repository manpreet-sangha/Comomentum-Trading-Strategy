# dataplots.py
# =====================================================================
# Data Exploration Plots – Orchestrator
# =====================================================================
# Imports all individual plot modules and runs them in sequence.
# Each plot module can also be run standalone.
#
# Standalone:   python dataplots.py
# From loader:  from dataplots import generate_all_plots
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
from logger_setup import _setup_logger

from data.plot1_universe_size       import plot_universe_size
from data.plot2_listed_vs_notlisted import plot_listed_vs_notlisted
from data.plot3_return_statistics   import plot_return_statistics
from data.plot4_return_distribution import plot_return_distribution
from data.plot5_missing_data        import plot_missing_data
from data.plot6_ff_cumulative       import plot_ff_cumulative
from data.plot_loading_summary      import plot_loading_summary

log = _setup_logger()


def generate_all_plots(data, output_dir='output_data'):
    """
    Runs every data exploration plot and saves PNGs to output_dir.

    INPUT:
        data       : dict  – dictionary returned by load_all_data()
        output_dir : str   – folder for the saved PNGs
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("GENERATING DATA EXPLORATION PLOTS")
    log.info("=" * 60)

    log.info("  Plot 1: Universe size over time...")
    plot_universe_size(data, output_dir)

    log.info("  Plot 2: Listed vs not-listed composition...")
    plot_listed_vs_notlisted(data, output_dir)

    log.info("  Plot 3: Weekly cross-sectional return statistics...")
    plot_return_statistics(data, output_dir)

    log.info("  Plot 4: Return distribution histogram...")
    plot_return_distribution(data, output_dir)

    log.info("  Plot 5: Missing data by year...")
    plot_missing_data(data, output_dir)

    log.info("  Plot 6: Fama-French factor cumulative returns...")
    plot_ff_cumulative(data, output_dir)

    log.info("  Plot 10: Data loading & cleaning summary table...")
    plot_loading_summary(data, output_dir)

    log.info("-" * 60)
    log.info(f"All data exploration plots saved to "
             f"{os.path.abspath(output_dir)}/")
    log.info("-" * 60)


# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    generate_all_plots(data)
