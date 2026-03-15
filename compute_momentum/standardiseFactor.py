import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# standardiseFactor.py
# =====================================================================
# Cross-sectional z-score standardisation utility.
#
# This function is applied to a TxN matrix of raw factor exposures
# (e.g. momentum) and converts them into cross-sectional z-scores
# at each point in time:
#
#     z_{i,t} = (x_{i,t} - mean_t) / std_t
#
# where mean_t and std_t are computed across all N stocks at week t.
#
# WHY DO WE STANDARDISE (z-score) THE MOMENTUM FACTOR?
# -----------------------------------------------------
# 1. COMPARABILITY ACROSS TIME
#    Raw momentum values (cumulative returns) can vary widely in
#    magnitude from one week to the next.  During volatile markets
#    the spread of raw momentum might be huge, while in calm periods
#    it is small.  Standardising ensures that an exposure of +1.0
#    always means "one standard deviation above the cross-sectional
#    average" regardless of the market regime.
#
# 2. FAMA-MACBETH REGRESSION INTERPRETATION
#    In Fama-MacBeth cross-sectional regressions (Step 3), we regress
#    next-week stock returns on the factor exposure:
#        r_{i,t+1} = alpha_t + gamma_t * z_{i,t} + epsilon_{i,t+1}
#    When z is standardised, the estimated slope gamma_t has a clean
#    interpretation: it is the expected return difference (per week)
#    between a stock that is one cross-sectional standard deviation
#    above average momentum and an average-momentum stock.  Without
#    standardisation, gamma would mix the effect of momentum with
#    the time-varying scale of the raw exposures.
#
# 3. PORTFOLIO CONSTRUCTION
#    Standardised exposures let us form dollar-neutral long/short
#    portfolios with consistent leverage.  A stock with z = +2 gets
#    twice the weight of z = +1, ensuring weights are proportional
#    to relative momentum rank, not to raw cumulative return levels.
#
# 4. COMOMENTUM ADJUSTMENT (Step 5)
#    When we later multiply the standardised momentum by a
#    time-varying scaling factor (inverse comomentum signal), the
#    adjustment operates on a unit-free quantity, so the scaling
#    has a uniform effect across all weeks.
#
# 5. NaN HANDLING
#    nanmean and nanstd ignore NaN entries (dead/unlisted stocks),
#    so the standardisation is computed only over the stocks that
#    are alive and have a valid momentum factor at each week.
#    Stocks that are NaN remain NaN after standardisation.
#
# FORMULA (applied independently at each row / week t):
#    factorStd[t, :] = (factorRaw[t, :] - nanmean(factorRaw[t, :])) 
#                      / nanstd(factorRaw[t, :], ddof=1)
#
# INPUTS:  factorRaw = TxN array of raw factor exposures
#          (T = 1513 weekly dates, N = 7261 US stocks).
#
# OUTPUTS: factorStd = TxN array of standardised factor exposures
#          (same 1513 x 7261 dimensions as the input).
#          At each week t, the cross-sectional mean is 0 and the
#          cross-sectional standard deviation is 1.
#
# =====================================================================
# TERMINOLOGY NOTE
# =====================================================================
# Three related but distinct terms are used throughout this project:
#
#   1. RAW MOMENTUM FACTOR
#      The 48-week compounded return for each stock at each week.
#      Computed in momentum_factor.compute_momentum().
#      Variable name: `momentum`   Shape: TxN
#
#   2. STANDARDISED MOMENTUM FACTOR  (output of THIS function)
#      The cross-sectional z-score of the raw momentum factor.
#      Each stock's raw value is re-expressed as "how many standard
#      deviations above or below the cross-sectional average it is"
#      at that week.  This is NOT the same as "standard momentum".
#      Variable name: `momentum_std`   Shape: TxN
#
#   3. STANDARD MOMENTUM FACTOR
#      Refers to the BASELINE momentum strategy (Jegadeesh & Titman,
#      1993) BEFORE any comomentum adjustment.  It encompasses both
#      the raw factor (#1) and its standardised form (#2).  The word
#      "standard" here means "conventional / unadjusted", not
#      "standardised in the statistical sense".
#
#   4. ADJUSTED MOMENTUM FACTOR
#      The standardised momentum factor (#2) scaled by an inverse
#      comomentum signal (Lou & Polk, 2021) and then re-standardised.
#      Variable name: `momentum_adj_std`   Shape: TxN
#
# In summary:
#   "standardised" = z-scored (statistical operation, this file)
#   "standard"     = conventional/baseline (strategic distinction)
# =====================================================================


def standardiseFactor(factorRaw):

    # Standardises a factor cross-sectionally by first subtracting
    # the cross-sectional mean from the raw factor exposures at each
    # point in time and then dividing this difference by the cross-sectional
    # standard deviation. Standardised factor exposures have a cross-sectional
    # mean of zero and a cross-sectional standard deviation of one.
    #
    # INPUTS: factorRaw = TxN array of raw factor exposures (where
    #                     T = 1513 weekly dates and N = 7261 US stocks).
    #                     In this project the factor is the momentum score
    #                     (cumulative return over 48 weeks, skipping the
    #                     most recent 4 weeks) for each stock at each week.
    #
    # OUTPUTS: factorStd = TxN array of standardised factor exposures
    #                      (same 1513 x 7261 dimensions as the input)

    N = factorRaw.shape[1]

    # standardise factor (subtract mean from raw factor exposure and divide this difference
    # by std. dev. cross-sectionally)

    avg = np.tile(np.nanmean(factorRaw,axis=1,keepdims=True),(1,N))
    stdev = np.tile(np.nanstd(factorRaw,axis=1,ddof=1,keepdims=True),(1,N))

    return (factorRaw - avg) / stdev
