# read_live.py
# =====================================================================
# Loader for US_live.csv
# =====================================================================
# Reads the TxN binary indicator matrix (1 = live / traded, 0 = dead /
# delisted or pre-IPO).  Same dimensions as US_Returns.csv.
# Used to mask out non-investable observations.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def load_live(datadir='input_data/'):
    """
    Loads US_live,csv.csv from the specified directory.

    INPUT:
        datadir : str - path to the folder containing the data files

    OUTPUT:
        live : np.ndarray (TxN) - binary live/dead indicator matrix
    """
    log.info("Loading US_live.csv ...")
    live = pd.read_csv(datadir + 'US_live.csv', header=None).values
    log.info(f"  OK  US_live.csv ->  {live.shape[0]} rows x {live.shape[1]} columns")
    log.info(f"       Unique values: {np.unique(live)}, "
             f"live=1 count={np.sum(live == 1):,}, dead=0 count={np.sum(live == 0):,}")
    return live
