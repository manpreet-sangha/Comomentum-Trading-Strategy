# read_names.py
# =====================================================================
# Loader for US_Names.xlsx
# =====================================================================
# Reads the stock / company name labels.  The file may store names as
# a single row (1xN) or a single column (Nx1); both layouts are
# handled and flattened to a 1-D array of length N.
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def load_names(datadir='input_data/'):
    """
    Loads US_Names.xlsx from the specified directory.

    INPUT:
        datadir : str - path to the folder containing the data files

    OUTPUT:
        names : np.ndarray (N,) - stock / company name labels
    """
    log.info("Loading US_Names.xlsx ...")
    names_df = pd.read_excel(datadir + 'US_Names.xlsx', header=None)
    if names_df.shape[0] == 1:
        # Names stored as one row with N columns -> read along columns
        names = names_df.iloc[0, :].values
        log.info(f"       Names file layout: 1 row x {names_df.shape[1]} columns (1xN format)")
    else:
        # Names stored as N rows in a single column -> read along rows
        names = names_df.iloc[:, 0].values
        log.info(f"       Names file layout: {names_df.shape[0]} rows x 1 column (Nx1 format)")
    log.info(f"  OK  US_Names.xlsx   ->  {len(names)} stock names")
    log.info(f"       First 5 names: {list(names[:5])}")
    return names
