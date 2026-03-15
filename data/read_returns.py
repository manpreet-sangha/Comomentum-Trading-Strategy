# read_returns.py
# =====================================================================
# Loader for US_Returns.csv
# =====================================================================
# Reads the TxN matrix of weekly total stock returns (decimal format).
# Each row is one week, each column is one stock.
# Example: 0.02 = +2% weekly return.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def load_returns(datadir='input_data/'):
    """
    Loads US_Returns.csv from the specified directory.

    INPUT:
        datadir : str - path to the folder containing the data files

    OUTPUT:
        returns : np.ndarray (TxN) - raw weekly stock returns in decimal form
    """
    log.info("Loading US_Returns.csv ...")
    returns = pd.read_csv(datadir + 'US_Returns.csv', header=None).values
    log.info(f"  OK  US_Returns.csv  ->  {returns.shape[0]} rows x {returns.shape[1]} columns")
    log.info(f"       dtype={returns.dtype}, "
             f"NaN count={np.sum(np.isnan(returns.astype(float)))}, "
             f"min={np.nanmin(returns):.6f}, max={np.nanmax(returns):.6f}")
    return returns
