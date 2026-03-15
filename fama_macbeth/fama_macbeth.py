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
import pandas as pd
import os


def famaMacBeth(factor, returns, live, dates=None, save_path=None):
    """
    Runs single-factor Fama-MacBeth cross-sectional regressions
    week by week through the entire sample.

    Handles missing data: at each week, any stock with a missing
    return OR missing factor exposure OR that is not live is
    excluded from the cross-sectional regression for that week.

    INPUTS:
        factor    : TxN np.ndarray - factor exposures (used with 1-week lag)
        returns   : TxN np.ndarray - weekly stock returns
        live      : TxN np.ndarray - live indicator (1 = live, 0 = dead)
        dates     : length-T array-like (optional) - week-ending dates;
                    if provided together with save_path, a CSV of all
                    regression outputs is saved to disk.
        save_path : str (optional) - file path for the output CSV.
                    Only used when dates is also supplied.

    OUTPUTS:
        gamma : T-length np.ndarray - weekly factor return estimates
                (NaN for weeks where the regression could not be run)
        tstat : float - t-statistic testing whether the mean factor
                return is significantly different from zero
    """

    T, N = factor.shape

    # Pre-allocate vectors (one entry per week)
    gamma = np.full(T, np.nan)          # slope  (factor return)
    alpha = np.full(T, np.nan)          # intercept
    n_stocks = np.full(T, np.nan)       # number of valid stocks
    r2 = np.full(T, np.nan)            # R-squared of the cross-sectional fit
    mean_y = np.full(T, np.nan)         # mean return of stocks in regression
    std_y = np.full(T, np.nan)          # std of returns in regression
    mean_x = np.full(T, np.nan)         # mean factor exposure in regression
    std_x = np.full(T, np.nan)          # std of factor exposures in regression

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
        nv = int(np.sum(valid))
        if nv < 3:
            continue

        # ----- Set up and solve the OLS regression -----
        # Design matrix: column of ones (intercept) + factor exposure
        yv = y[valid]
        xv = x[valid]
        Y = yv[:, np.newaxis]                                          # (nv x 1)
        X = np.hstack((np.ones((nv, 1)), xv[:, np.newaxis]))           # (nv x 2)

        # Solve via least squares: [alpha_t, gamma_t]
        coefs = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Store regression outputs for week t
        alpha[t] = coefs[0, 0]
        gamma[t] = coefs[1, 0]
        n_stocks[t] = nv

        # Descriptive statistics of the regression inputs
        mean_y[t] = np.mean(yv)
        std_y[t] = np.std(yv, ddof=1) if nv > 1 else 0.0
        mean_x[t] = np.mean(xv)
        std_x[t] = np.std(xv, ddof=1) if nv > 1 else 0.0

        # R-squared: 1 − SS_res / SS_tot
        fitted = X @ coefs
        ss_res = float(np.sum((Y - fitted) ** 2))
        ss_tot = float(np.sum((Y - np.mean(yv)) ** 2))
        r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # ----- Compute the t-statistic on the mean factor return -----
    # t = mean(gamma) / ( std(gamma) / sqrt(number of valid weeks) )
    valid_gamma = gamma[np.isfinite(gamma)]
    n_weeks = len(valid_gamma)

    if n_weeks > 1:
        tstat = (np.nanmean(gamma)
                 / (np.nanstd(gamma, ddof=1) / np.sqrt(n_weeks)))
    else:
        tstat = np.nan

    # ----- Save CSV if requested -----
    if dates is not None and save_path is not None:
        _save_regression_csv(
            save_path, dates, alpha, gamma, n_stocks,
            r2, mean_y, std_y, mean_x, std_x, tstat, n_weeks
        )

    return gamma, tstat


# =====================================================================
# Internal helper: write the regression CSV with column explanations
# =====================================================================
def _save_regression_csv(path, dates, alpha, gamma, n_stocks,
                         r2, mean_y, std_y, mean_x, std_x,
                         tstat, n_weeks):
    """Write an Excel workbook with two sheets:
      - 'Documentation' : column definitions, model description, summary stats
      - 'Data'          : weekly regression coefficients and inputs
    """

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # ---- Force .xlsx extension ----
    base, _ = os.path.splitext(path)
    xlsx_path = base + '.xlsx'

    # ---- Build the documentation table ----
    doc_rows = [
        ["FAMA-MACBETH CROSS-SECTIONAL REGRESSION OUTPUT", ""],
        ["", ""],
        ["Regression model (run cross-sectionally at each week t):", ""],
        ["    r_i,t  =  alpha_t  +  gamma_t * factor_i,t-1  +  epsilon_i,t", ""],
        ["", ""],
        ["COLUMN DEFINITIONS", ""],
        ["Column", "Description"],
        ["date",
         "Week-ending date (YYYY-MM-DD) for this regression. "
         "Returns are measured at this date; factor exposure is lagged "
         "by one week (from date t-1)."],
        ["alpha",
         "Intercept of the cross-sectional OLS regression. "
         "Represents the average return of a stock with zero factor exposure. "
         "Units: weekly return (decimal)."],
        ["gamma",
         "Slope coefficient (= FACTOR RETURN for this week). "
         "Measures the return earned per unit of lagged factor exposure. "
         "A positive gamma means stocks with higher momentum last week "
         "earned higher returns this week. Units: weekly return (decimal)."],
        ["n_stocks",
         "Number of stocks that entered the regression for this week. "
         "A stock is included only if: (1) live == 1 (listed on the exchange), "
         "(2) return at time t is not NaN, (3) factor exposure at t-1 is not NaN."],
        ["r_squared",
         "R-squared of the cross-sectional regression. "
         "Fraction of cross-sectional return variance explained by the factor exposure. "
         "Typically very small (< 1%) because individual stock returns are noisy."],
        ["mean_return",
         "Mean of the dependent variable (stock returns) across the n_stocks "
         "that entered the regression. Units: weekly return (decimal)."],
        ["std_return",
         "Standard deviation of stock returns across the n_stocks in the regression. "
         "Units: weekly return (decimal)."],
        ["mean_factor",
         "Mean of the lagged factor exposures (x-variable) across n_stocks. "
         "After standardisation this should be approximately 0 each week."],
        ["std_factor",
         "Standard deviation of the lagged factor exposures across n_stocks. "
         "After standardisation this should be approximately 1 each week."],
        ["", ""],
        ["NOTE",
         "Rows where all values (except date) are blank correspond to weeks "
         "where the regression could not be run (e.g. the first 52 weeks, "
         "before any momentum score exists, or weeks with fewer than 3 valid stocks)."],
        ["", ""],
        ["SUMMARY STATISTICS", ""],
        ["Metric", "Value"],
        ["Mean gamma", f"{np.nanmean(gamma):.8f}"],
        ["Std gamma", f"{np.nanstd(gamma, ddof=1):.8f}"],
        ["T-statistic", f"{tstat:.4f}"],
        ["Valid weeks", f"{n_weeks}"],
    ]
    df_doc = pd.DataFrame(doc_rows, columns=["Item", "Details"])

    # ---- Build the data table ----
    df_data = pd.DataFrame({
        'date':        dates,
        'alpha':       alpha,
        'gamma':       gamma,
        'n_stocks':    n_stocks,
        'r_squared':   r2,
        'mean_return': mean_y,
        'std_return':  std_y,
        'mean_factor': mean_x,
        'std_factor':  std_x,
    })

    # Format n_stocks as integer where valid
    df_data['n_stocks'] = df_data['n_stocks'].apply(
        lambda v: '' if np.isnan(v) else str(int(v))
    )

    # ---- Write to Excel with two sheets ----
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_doc.to_excel(writer, sheet_name='Documentation', index=False)
        df_data.to_excel(writer, sheet_name='Data', index=False)

    print(f"  Saved: {xlsx_path}")
