# fama_macbeth.py
# =====================================================================
# Fama-MacBeth Cross-Sectional Regression Module
# =====================================================================
# This module implements the Fama-MacBeth (1973) procedure for
# estimating factor risk premia via repeated cross-sectional
# regressions. It is adapted from the course code (solveFamaMacBeth-
# Exercise.py) with the addition of robust handling of missing data
# and a live/dead filter, which are essential for real-world datasets.
#
# KEY IDEA:
#   At each week t, we run a cross-sectional OLS regression:
#       r_i,t  =  alpha_t  +  gamma_t * factor_i,t-1  +  epsilon_i,t
#   where r_i,t is the return of stock i in week t, and factor_i,t-1
#   is the stock's lagged factor exposure. The slope gamma_t is the
#   "factor return" for that week. Collecting all gamma_t over time
#   gives us the factor return time series.
# =====================================================================

import numpy as np


def famaMacBeth(factor, returns, live):
    """
    Runs single-factor Fama-MacBeth cross-sectional regressions
    week by week through the entire sample.

    Handles missing data: at each week, any stock with a missing
    return OR missing factor exposure OR that is not live is
    excluded from the cross-sectional regression for that week.

    INPUTS:
        factor  : TxN np.ndarray - factor exposures (used with 1-week lag)
        returns : TxN np.ndarray - weekly stock returns
        live    : TxN np.ndarray - live indicator (1 = live, 0 = dead)

    OUTPUTS:
        gamma : T-length np.ndarray - weekly factor return estimates
                (NaN for weeks where the regression could not be run)
        tstat : float - t-statistic testing whether the mean factor
                return is significantly different from zero
    """

    T, N = factor.shape

    # Pre-allocate the factor-return vector (one entry per week)
    gamma = np.full(T, np.nan)

    for t in range(1, T):

        # ----- Identify the dependent and independent variables -----
        # Y (dependent)  : stock returns at time t
        # X (independent): lagged factor exposure from time t-1
        y = returns[t, :]
        x = factor[t - 1, :]
        lv = live[t, :]

        # ----- Build a validity mask -----
        # A stock must satisfy ALL three conditions to enter the regression:
        #   1. It is live (lv == 1)
        #   2. Its return is not NaN
        #   3. Its factor exposure is not NaN
        valid = (lv == 1) & np.isfinite(y) & np.isfinite(x)

        # Need at least 3 valid stocks to run a meaningful regression
        if np.sum(valid) < 3:
            continue

        # ----- Set up and solve the OLS regression -----
        # Design matrix: column of ones (intercept) + factor exposure
        n_valid = np.sum(valid)
        Y = y[valid, np.newaxis]                                       # (n_valid x 1)
        X = np.hstack((np.ones((n_valid, 1)), x[valid, np.newaxis]))   # (n_valid x 2)

        # Solve via least squares: [alpha_t, gamma_t]
        coefs = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Store the slope coefficient (= factor return for week t)
        gamma[t] = coefs[1, 0]

    # ----- Compute the t-statistic on the mean factor return -----
    # t = mean(gamma) / ( std(gamma) / sqrt(number of valid weeks) )
    valid_gamma = gamma[np.isfinite(gamma)]
    n_weeks = len(valid_gamma)

    if n_weeks > 1:
        tstat = (np.nanmean(gamma)
                 / (np.nanstd(gamma, ddof=1) / np.sqrt(n_weeks)))
    else:
        tstat = np.nan

    return gamma, tstat
