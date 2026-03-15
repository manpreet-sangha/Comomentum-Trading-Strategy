# read_fama_french.py
# =====================================================================
# Loader for FamaFrench.csv
# =====================================================================
# Reads the weekly Fama-French three-factor returns plus the risk-free
# rate.  Returns a Tx3 factor matrix (Mkt-RF, SMB, HML) and a
# separate T-length risk-free rate vector.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def load_fama_french(datadir='input_data/'):
    """
    Loads FamaFrench.csv from the specified directory.

    INPUT:
        datadir : str - path to the folder containing the data files

    OUTPUT:
        ff_factors : np.ndarray (Tx3) - weekly factor returns [Mkt-RF, SMB, HML]
        rf         : np.ndarray (T,)  - weekly risk-free rate
    """
    log.info("Loading FamaFrench.csv ...")
    ff = pd.read_csv(datadir + 'FamaFrench.csv')
    ff_factors = np.column_stack((
        ff['Mkt-RF'].values,
        ff['SMB'].values,
        ff['HML'].values
    ))
    rf = ff['RF'].values
    log.info(f"  OK  FamaFrench.csv  ->  {ff.shape[0]} rows x {ff.shape[1]} columns "
             f"(factors: Mkt-RF, SMB, HML + RF)")
    log.info(f"       FF factors shape: {ff_factors.shape}, RF shape: {rf.shape}")
    return ff_factors, rf


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    ff_factors, rf = load_fama_french()
    print(f"FF factors shape : {ff_factors.shape}")
    print(f"RF shape         : {rf.shape}")
