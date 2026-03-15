# clean_returns.py
# =====================================================================
# Return-Cleaning Module
# =====================================================================
# Takes the raw TxN return matrix and the TxN live/dead indicator and
# produces a clean return matrix where every non-listed observation
# (live == 0) is set to NaN.
#
# All stocks in the dataset are real companies.  Not every company was
# listed for the full duration of the time series — some IPO'd part-way
# through, others were delisted (merger, acquisition, bankruptcy, etc.)
# before the series ends.  The live flag marks each stock's listing
# window:
#   live = 1  →  the stock was listed on an exchange that week
#   live = 0  →  the stock had not yet IPO'd or had already been delisted
#
# WHY THIS IS NEEDED:
#   The raw US_Returns.csv already contains NaN for all non-listed
#   cells in this dataset.  However, other datasets may not — a
#   non-listed cell could carry any numeric value (including 0.0,
#   which is a perfectly valid return for a listed stock but is not
#   investable for a stock that wasn't trading).
#
#   This module acts as a safety net: it uses the authoritative live
#   indicator from US_live,csv.csv to force every non-listed cell to
#   NaN, regardless of what value the returns file happens to contain:
#     - If live==0 AND the return is already NaN  -> no change
#     - If live==0 AND the return is numeric      -> forced to NaN
#       (not investable since the stock wasn't listed)
#
#   If we skipped this step and a non-listed cell had a numeric value:
#     - Momentum scores would include non-investable observations.
#     - Fama-MacBeth regressions would include non-investable
#       observations, distorting the estimated risk premium.
#     - Performance statistics would overcount the universe.
#
#   By replacing non-listed cells with NaN:
#     - NumPy's nan-aware functions (nanmean, nanstd, nanprod)
#       automatically skip them.
#     - Pandas .corr(min_periods=...) ignores NaN pairs.
#     - Our Fama-MacBeth loop explicitly checks for finite values
#       before including a stock in the cross-sectional regression.
#
#   Result: a clean, survivorship-bias-free panel of returns where
#   only genuinely listed observations participate in the analysis.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from logger_setup import _setup_logger

log = _setup_logger()


def clean_returns(returns, live):
    """
    Sets non-listed observations to NaN using the live indicator.

    All stocks are real companies — live simply flags each stock's
    listing window (1 = listed that week, 0 = not yet IPO'd or
    already delisted).

    INPUT:
        returns : np.ndarray (TxN) - raw weekly stock returns
        live    : np.ndarray (TxN) - listing indicator (1=listed, 0=not listed)

    OUTPUT:
        returns_clean : np.ndarray (TxN) - returns with non-listed cells set to NaN
    """
    returns_clean = returns.copy().astype(float)

    T, N = returns.shape
    total_cells = T * N

    # Count non-listed cells and break them down
    # dead_mask = (live == 0), creates a boolean array the same shape as live (T×N) 
    # where each element is True if the stock was not listed that week, and False if it was listed.
    dead_mask = (live == 0)

    # np.sum() treats True as 1 and False as 0, then adds them all up.
    # for example, if dead_mask = [[False, True,  False],
    #                               [True,  False, False]]
    # np.sum(dead_mask)  →  2

    n_dead_total = np.sum(dead_mask)
    n_dead_already_nan = np.sum(dead_mask & np.isnan(returns_clean))
    n_dead_had_value = n_dead_total - n_dead_already_nan

    # Count NaNs that exist in listed cells (genuine missing data)
    live_mask = (live == 1)
    n_live_nan = np.sum(live_mask & np.isnan(returns_clean))
    n_live_valid = np.sum(live_mask & np.isfinite(returns_clean))

    # Force all non-listed cells to NaN
    returns_clean[dead_mask] = np.nan

    log.info("  --- Return matrix cleaning report ---")
    log.info(f"  Total cells in TxN matrix     : {total_cells:,}")
    log.info(f"  Listed cells  (live==1)       : {np.sum(live_mask):,}  "
             f"({np.sum(live_mask)/total_cells*100:.1f}%)")
    log.info(f"    of which have valid returns  : {n_live_valid:,}")
    log.info(f"    of which are NaN (data gaps) : {n_live_nan:,}")
    log.info(f"  Not-listed    (live==0)        : {n_dead_total:,}  "
             f"({n_dead_total/total_cells*100:.1f}%)")
    log.info(f"    already NaN in raw file      : {n_dead_already_nan:,}  (no action needed)")
    log.info(f"    had a numeric value -> NaN   : {n_dead_had_value:,}  (forced to NaN)")
    if n_dead_had_value > 0:
        log.info(f"  ** {n_dead_had_value:,} numeric values from non-listed periods forced to NaN **")

    return returns_clean
