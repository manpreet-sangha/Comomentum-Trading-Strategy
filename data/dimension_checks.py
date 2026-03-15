# dimension_checks.py
# =====================================================================
# Dimension Consistency Checks & Data Loading Summary
# =====================================================================
# Validates that all loaded input arrays have compatible shapes and
# logs a structured summary of what was loaded.  Called by data_loader
# immediately after all five input files have been read and cleaned.
#
# Standalone:   python dimension_checks.py
# From loader:  from dimension_checks import check_dimensions, log_loading_summary
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
from datetime import datetime
from logger_setup import _setup_logger

log = _setup_logger()


def check_dimensions(returns, live, dates, names, ff_factors, T, N):
    """
    Verifies that every loaded array has the expected shape.

    INPUT:
        returns    : np.ndarray (TxN)
        live       : np.ndarray (TxN)
        dates      : pd.DatetimeIndex (T,)
        names      : np.ndarray (N,)
        ff_factors : np.ndarray (Tx3)
        T          : int - expected number of weeks
        N          : int - expected number of stocks

    OUTPUT:
        checks_passed : bool - True if all checks pass
    """
    checks_passed = True

    if returns.shape != live.shape:
        log.error(f"  MISMATCH: returns shape {returns.shape} != live shape {live.shape}")
        checks_passed = False
    if len(dates) != T:
        log.error(f"  MISMATCH: dates length {len(dates)} != T={T}")
        checks_passed = False
    if len(names) != N:
        log.error(f"  MISMATCH: names length {len(names)} != N={N}")
        checks_passed = False
    if ff_factors.shape[0] != T:
        log.error(f"  MISMATCH: FF factors rows {ff_factors.shape[0]} != T={T}")
        checks_passed = False

    if checks_passed:
        log.info("  All dimension checks passed.")
    else:
        log.warning("  Some dimension checks FAILED - see errors above.")

    return checks_passed


def log_loading_summary(returns, live, dates, names, ff_factors, T, N, datadir):
    """
    Logs a structured summary of the loaded dataset.

    INPUT:
        returns    : np.ndarray (TxN)
        live       : np.ndarray (TxN)
        dates      : pd.DatetimeIndex (T,)
        names      : np.ndarray (N,)
        ff_factors : np.ndarray (Tx3)
        T          : int - number of weeks
        N          : int - number of stocks
        datadir    : str - path to the input data folder
    """
    live_pct = np.sum(live == 1) / live.size * 100
    total_cells = live.size
    listed_cells = int(np.sum(live == 1))
    not_listed   = int(np.sum(live == 0))
    finite_and_listed = int(np.sum(np.isfinite(returns) & (live == 1)))
    nan_in_listed     = listed_cells - finite_and_listed
    notlisted_already_nan = int(np.sum((live == 0) & ~np.isfinite(returns)))
    forced_nan = int(np.sum((live == 0) & np.isfinite(returns)))

    log.info("-" * 60)
    log.info("DATA LOADING SUMMARY")
    log.info("-" * 60)
    log.info(f"  Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Source directory : {os.path.abspath(datadir)}")
    log.info(f"  Number of weeks (T)  : {T}")
    log.info(f"  Number of stocks (N) : {N}")
    log.info(f"  Date range           : {dates[0].strftime('%Y-%m-%d')} to "
             f"{dates[-1].strftime('%Y-%m-%d')}")
    log.info(f"  Files loaded:")
    log.info(f"    US_Returns.csv    : {T} x {N}  (weekly stock returns)")
    log.info(f"    US_live.csv       : {T} x {N}  (listed/not-listed flags)")
    log.info(f"    US_Dates.xlsx     : {T} dates")
    log.info(f"    US_Names.xlsx     : {N} stock names")
    log.info(f"    FamaFrench.csv    : {T} x {ff_factors.shape[1]+1}  "
             f"(Mkt-RF, SMB, HML, RF)")
    log.info(f"  Listed observations  : {listed_cells:,} / {total_cells:,} "
             f"({live_pct:.1f}%)")
    log.info(f"  Not-listed (NaN) obs : {not_listed:,} "
             f"({100 - live_pct:.1f}%)")
    log.info("-" * 60)
    log.info(f"  Total cells in TxN matrix     : {total_cells:,}")
    log.info(f"  Listed cells  (live==1)       : {listed_cells:,}  ({live_pct:.1f}%)")
    log.info(f"    of which have valid returns  : {finite_and_listed:,}")
    log.info(f"    of which are NaN (data gaps) : {nan_in_listed:,}")
    log.info(f"  Not-listed    (live==0)        : {not_listed:,}  ({100 - live_pct:.1f}%)")
    log.info(f"    already NaN in raw file      : {notlisted_already_nan:,}  (no action needed)")
    log.info(f"    had a numeric value -> NaN   : {forced_nan:,}  (forced to NaN)")
    log.info("-" * 60)
    log.info("STEP 1 COMPLETE - all data loaded successfully.")
    log.info("-" * 60)


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    check_dimensions(
        data['returns'], data['live'], data['dates'],
        data['names'], data['ff_factors'], data['T'], data['N']
    )
    log_loading_summary(
        data['returns'], data['live'], data['dates'],
        data['names'], data['ff_factors'], data['T'], data['N'],
        'input_data/'
    )
