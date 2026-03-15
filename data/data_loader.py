# data_loader.py
# =====================================================================
# Data Loading Module
# =====================================================================
# This module handles loading all input data files required by the
# momentum trading strategy. It reads stock returns, listed/not-listed
# flags, dates, company names, and Fama-French factor data from the
# input_data folder and returns them in a clean dictionary for
# downstream use.
#
# INPUT FILES AND THEIR SIGNIFICANCE:
#
#   US_Returns.csv (TxN)
#       Weekly total returns for N=7261 US stocks over T=1513 weeks.
#       Returns are in decimal form (e.g. 0.02 = +2%).
#       This is the core dataset: every step of the analysis (momentum
#       computation, Fama-MacBeth regressions, performance evaluation)
#       ultimately depends on these return series.
#
#   US_live.csv (TxN)
#       A binary indicator matrix (1 or 0) with the same shape as
#       US_Returns.csv.
#
#       "LISTED" (value = 1):
#           The stock was actively listed and traded on that date.
#           Its return is a real, investable number.
#
#       "NOT LISTED" (value = 0):
#           The stock was NOT yet listed (pre-IPO) or had already been
#           delisted (bankruptcy, merger, acquisition, going private,
#           etc.) on that date.  The return value in US_Returns.csv for
#           a not-listed observation is not investable and must NOT be
#           used in any computation.
#
#       WHY THIS MATTERS:
#           If we included not-listed returns we would introduce
#           survivorship bias (only using companies that survived the
#           full sample) or contaminate our factor scores with
#           non-investable observations.  Setting not-listed
#           observations to NaN ensures they are automatically excluded
#           from nanmean, nanstd, nanprod, regressions, and all other
#           numerical routines.  This gives us a clean, bias-free
#           panel.
#
#   US_Dates.xlsx (Tx1)
#       Weekly date stamps in YYYYMMDD integer format.  Converted to
#       Python datetime objects for use as time-series indices, chart
#       labels, and output file formatting.
#
#   US_Names.xlsx (1xN or Nx1)
#       Company name labels for each of the N stocks.  Used
#       for labelling output CSVs and charts.  The file may be stored
#       as a single row (1xN) or a single column (Nx1) depending on
#       how it was exported; we handle both layouts.
#
#   FamaFrench.csv (Tx4)
#       Weekly Fama-French three-factor returns plus the risk-free rate:
#         - Mkt-RF : excess market return (market minus risk-free)
#         - SMB    : Small-Minus-Big size factor
#         - HML    : High-Minus-Low value factor
#         - RF     : risk-free rate (T-bill)
#       These are used in two places:
#         (a) Computing comomentum residuals: we regress each momentum
#             stock's returns on the FF3 factors to isolate abnormal
#             (idiosyncratic) returns, whose pairwise correlations
#             form the comomentum measure (Lou & Polk, 2021).
#         (b) Potentially adjusting for risk in performance evaluation.
#
# LOGGING:
#   Every file load is logged with confirmation of success/failure
#   and the shape (rows x columns) of the data read.  A summary log
#   is written to  output_data/data_loading.log  so that a front-end
#   application can display the log when the user clicks "read_data".
#   Logs are also printed to the console for interactive use.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import os
from logger_setup import _setup_logger
from data.read_returns import load_returns
from data.read_live import load_live
from data.read_dates import load_dates
from data.read_names import load_names
from data.read_fama_french import load_fama_french
from data.clean_returns import clean_returns
from data.dimension_checks import check_dimensions, log_loading_summary
from data.dataplots import generate_all_plots


# =====================================================================
# Logger  (module-level, configured once on first import)
# =====================================================================

log = _setup_logger()


def load_all_data(datadir='input_data/'):
    """
    Loads all required data files from the specified directory.

    FILES LOADED:
        US_Returns.csv   - TxN matrix of weekly total stock returns (decimals)
        US_live.csv      - TxN matrix of listed/not-listed dummies (1=listed, 0=not listed)
        US_Dates.xlsx    - Tx1 vector of weekly dates (YYYYMMDD format)
        US_Names.xlsx    - Nx1 vector of stock/company names
        FamaFrench.csv   - Tx4 matrix of weekly FF3 factor returns + risk-free rate

    INPUT:
        datadir : str - path to the folder containing the data files
                        (default: 'input_data/')

    OUTPUT:
        data : dict - dictionary with the following keys:
            'returns'       : np.ndarray (TxN) - raw weekly stock returns
            'returns_clean' : np.ndarray (TxN) - returns with not-listed stocks set to NaN
            'live'          : np.ndarray (TxN) - listed/not-listed indicator matrix
            'dates'         : pd.DatetimeIndex  - weekly date series
            'names'         : np.ndarray (N,)   - stock name labels
            'ff_factors'    : np.ndarray (Tx3)  - Fama-French factor matrix [MktRF, SMB, HML]
            'rf'            : np.ndarray (T,)   - risk-free rate series
            'T'             : int - number of weeks
            'N'             : int - number of stocks
    """

    # ------------------------------------------------------------------
    # 1–5. Load all five input files via dedicated reader modules
    # ------------------------------------------------------------------
    returns    = load_returns(datadir)       # TxN weekly stock returns
    live       = load_live(datadir)          # TxN listed/not-listed indicator
    dates      = load_dates(datadir)         # T dates
    names      = load_names(datadir)         # N stock names
    ff_factors, rf = load_fama_french(datadir)  # Tx3 FF factors + T risk-free rate

    # ------------------------------------------------------------------
    # 6. Clean returns: set non-listed observations to NaN
    # ------------------------------------------------------------------
    returns_clean = clean_returns(returns, live)

    T, N = returns.shape

    # ------------------------------------------------------------------
    # Dimension consistency checks
    # ------------------------------------------------------------------
    check_dimensions(returns, live, dates, names, ff_factors, T, N)

    # ------------------------------------------------------------------
    # Data loading summary (written to log file + console)
    # ------------------------------------------------------------------
    log_loading_summary(returns, live, dates, names, ff_factors, T, N, datadir)

    # ------------------------------------------------------------------
    # Pack everything into a dictionary and return
    # ------------------------------------------------------------------
    data = {
        'returns':       returns,
        'returns_clean': returns_clean,
        'live':          live,
        'dates':         dates,
        'names':         names,
        'ff_factors':    ff_factors,
        'rf':            rf,
        'T':             T,
        'N':             N,
    }

    return data


# =====================================================================
# Run as standalone script: python data_loader.py
# =====================================================================
# When executed directly, this loads all data, prints the full log to
# the console, and saves the log to output_data/data_loading.log.
# This lets you (or the future front-end app) verify that all input
# files are present and correctly shaped BEFORE running the full
# momentum_strategy.py pipeline.
# =====================================================================

if __name__ == '__main__':
    data = load_all_data(datadir='input_data/')

    # Generate data exploration plots
    generate_all_plots(data, output_dir='output_data')

    # Quick sanity print so the user sees the key variables
    print("\nData dictionary keys:", list(data.keys()))
    print(f"returns_clean shape: {data['returns_clean'].shape}")
    print(f"live shape:          {data['live'].shape}")
    print(f"dates length:        {len(data['dates'])}")
    print(f"names length:        {len(data['names'])}")
    print(f"ff_factors shape:    {data['ff_factors'].shape}")
    print(f"rf shape:            {data['rf'].shape}")
    print(f"\nLog file written to: output_data/data_loading.log")
    print(f"Plots saved to:      output_data/")
