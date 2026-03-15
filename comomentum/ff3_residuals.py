# =====================================================================
# Fama-French 3-Factor (FF3) Regression Residuals
# =====================================================================
# WHAT THIS MODULE DOES:
#   For a given (W x K) matrix of weekly returns and the corresponding
#   (W x 3) Fama-French factor matrix [Mkt-RF, SMB, HML], run a
#   cross-sectional OLS regression for each of the K stocks:
#
#       r_{i,w} = alpha_i + beta_MKT * MktRF_w + beta_SMB * SMB_w
#                 + beta_HML * HML_w + epsilon_{i,w}
#
#   and return the (W x K) matrix of residuals epsilon.
#
#   Following Lewellen & Nagel (2006), betas are estimated over
#   a rolling 52-week window, allowing them to vary over time.
#
#   A stock must have at least MIN_RESID_OBS (or CORR_WINDOW if
#   MIN_RESID_OBS is None) valid return observations in the window
#   to be included.  Stocks with fewer valid weeks get all-NaN
#   residuals.
#
# Standalone:  python ff3_residuals.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from config import CORR_WINDOW, MIN_RESID_OBS


def compute_ff3_residuals(returns_window, ff_window):
    """
    Regress each stock's returns on FF3 factors and return residuals.

    INPUT:
        returns_window : (W, K) np.ndarray — weekly returns for K stocks
        ff_window      : (W, 3) np.ndarray — [Mkt-RF, SMB, HML]

    OUTPUT:
        residuals      : (W, K) np.ndarray — OLS residuals (NaN where
                         the return was NaN or the stock had too few
                         valid observations)
    """
    W, K = returns_window.shape
    residuals = np.full((W, K), np.nan)

    # Threshold: use MIN_RESID_OBS if set, else CORR_WINDOW (52)
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    for j in range(K):
        y_j = returns_window[:, j]
        valid_j = np.isfinite(y_j)

        if np.sum(valid_j) < resid_threshold:
            continue

        # Design matrix: intercept + 3 FF factors
        X_j = np.column_stack([
            np.ones(np.sum(valid_j)),
            ff_window[valid_j, :]
        ])
        Y_j = y_j[valid_j]

        # OLS:  Y = X @ beta  →  residual = Y - X @ beta_hat
        coefs, _, _, _ = np.linalg.lstsq(X_j, Y_j, rcond=None)
        residuals[valid_j, j] = Y_j - X_j @ coefs

    return residuals


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data

    data = load_all_data('input_data/')
    returns = data['returns_clean']
    ff = data['ff_factors']

    # Demo: run on a 52-week window ending at the last week
    T = returns.shape[0]
    w_start = T - CORR_WINDOW
    ret_window = returns[w_start:T, :]
    ff_window = ff[w_start:T, :]

    residuals = compute_ff3_residuals(ret_window, ff_window)

    n_stocks = np.sum(np.any(np.isfinite(residuals), axis=0))
    print(f"FF3 residuals: {residuals.shape[0]} weeks x {residuals.shape[1]} stocks")
    print(f"  Stocks with valid residuals: {n_stocks}")
    print(f"  Mean residual: {np.nanmean(residuals):.8f}")
