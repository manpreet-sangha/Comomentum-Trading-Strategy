# read_dates.py
# =====================================================================
# Loader for US_Dates.xlsx
# =====================================================================
# Reads the Tx1 vector of weekly date stamps stored as integers in
# YYYYMMDD format and converts them to a pandas DatetimeIndex.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def load_dates(datadir='input_data/'):
    """
    Loads US_Dates.xlsx from the specified directory.

    INPUT:
        datadir : str - path to the folder containing the data files

    OUTPUT:
        dates : pd.DatetimeIndex (length T) - weekly date series
    """
    log.info("Loading US_Dates.xlsx ...")
    dates_df = pd.read_excel(datadir + 'US_Dates.xlsx', header=None)
    dates_raw = dates_df.iloc[:, 0].values
    dates = pd.to_datetime(dates_raw.astype(str), format='%Y%m%d')
    log.info(f"  OK  US_Dates.xlsx   ->  {len(dates)} dates")
    log.info(f"       First date: {dates[0]}, Last date: {dates[-1]}")
    return dates
