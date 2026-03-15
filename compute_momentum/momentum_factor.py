# momentum_factor.py
# =====================================================================
# Momentum Factor Computation Module
# =====================================================================
# This module contains all functions related to computing:
#   1. The STANDARD momentum factor (48-week lookback including current
#      week, with the latest 4 weeks of the dataset skipped once).
#      1st factor at week 1508, last factor at week 47.
#   2. The COMOMENTUM measure (Lou & Polk, 2021)
#   3. The ADJUSTED momentum factor (standard momentum scaled by an
#      inverse-comomentum signal)
#
# These functions are kept separate from the main script so they can
# be tested, reused, or modified independently.
# =====================================================================

import numpy as np
import pandas as pd
import logging
import os


# =====================================================================
# Logger setup  (writes to output_data/momentum_factor.log + console)
# Log file is overwritten each time the code runs.
# =====================================================================

def _setup_logger(log_dir='output_data', log_filename='momentum_factor.log'):
    logger = logging.getLogger('momentum_factor')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, log_filename), mode='w', encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = _setup_logger()


# =====================================================================
# 1. Standard Momentum Factor
# =====================================================================

def compute_momentum(returns_clean, dates, lookback=48, skip=4):
    """
    Computes a standard momentum factor for each stock at each point
    in time.

    INTERPRETATION (as used in this coursework):
        - skip  = 4  means we DISCARD the latest 4 weeks of the entire
          dataset once (a one-time global skip from the end).  The last
          week that receives a momentum factor is therefore T-1-skip
          (index 1508 for T=1513, skip=4).
        - lookback = 48  means the momentum window INCLUDES the current
          week itself.  At week t the window covers weeks [t-47 .. t],
          giving exactly 48 weekly returns.

    RANGE OF VALID MOMENTUM FACTORS:
        1st momentum factor : week 1508 (lookback [1461..1508])
        2nd momentum factor : week 1507 (lookback [1460..1507])
        ...
        Last momentum factor: week 47   (lookback [0..47])
        Total weeks with a momentum factor: 1462

    CALCULATION (at week t, for stock i):
        mom_{i,t} = prod(1 + r_{i,s}  for s in [t-47, t]) - 1
        NaN returns inside the window are skipped by nanprod (treated
        as multiplicative identity = 1).

    INPUTS:
        returns_clean : TxN np.ndarray - weekly returns (dead stocks = NaN)
        dates         : array-like of length T - date labels for each week
        lookback      : int  - width of the return window including the
                        current week (default 48 ~ 11 months)
        skip          : int  - number of most-recent weeks to discard
                        from the END of the dataset (default 4 ~ 1 month)

    OUTPUTS:
        momentum     : TxN np.ndarray - raw momentum factor values
        momentum_std : TxN np.ndarray - same as momentum (raw, no standardisation)
    """

    T, N = returns_clean.shape
    first_t = lookback - 1            # earliest week with a full window (47)
    last_t  = T - 1 - skip            # latest week with a factor (1508)
    n_scored = last_t - first_t + 1   # total weeks with factors (1462)

    # Helper to format a date (handles both Timestamp and datetime)
    def _d(idx):
        return pd.Timestamp(dates[idx]).strftime('%Y-%m-%d')

    log.info("=" * 60)
    log.info("STEP 2: Computing standard momentum factor")
    log.info("=" * 60)
    log.info(f"  Parameters: lookback={lookback}w, skip={skip}w (one-time from end)")
    log.info(f"  Input: T={T} weeks (indices 0..{T-1}), N={N} stocks")
    log.info(f"  Date range: {_d(0)} to {_d(T-1)}")
    log.info(f"  Skipped weeks (end of dataset): "
             f"indices {T-skip}..{T-1}  "
             f"({_d(T-skip)} to {_d(T-1)})")
    log.info(f"  1st momentum factor : week {last_t} ({_d(last_t)})  "
             f"lookback [{last_t-lookback+1}..{last_t}] "
             f"({_d(last_t-lookback+1)} to {_d(last_t)})")
    log.info(f"  Last momentum factor: week {first_t} ({_d(first_t)})  "
             f"lookback [{0}..{first_t}] "
             f"({_d(0)} to {_d(first_t)})")
    log.info(f"  Total weeks with momentum factor: {n_scored}")

    # Pre-allocate output array (NaN = no factor for that week)
    momentum = np.full((T, N), np.nan)

    for t in range(first_t, last_t + 1):

        # Window: weeks [t - lookback + 1 .. t] inclusive = 48 rows
        window_start = t - lookback + 1       # e.g. t=47 -> start=0
        ret_window = returns_clean[window_start: t + 1, :]   # shape (48, N)

        # Compound returns: prod(1 + r) - 1
        # nanprod treats NaN as 1 (multiplicative identity)
        cum_ret = np.nanprod(1 + ret_window, axis=0) - 1

        momentum[t, :] = cum_ret

        # Progress every 200 weeks
        if t % 200 == 0:
            n_valid_t = np.sum(np.isfinite(cum_ret))
            log.info(f"  Week {t:>5} ({_d(t)}) : "
                     f"{n_valid_t:,} stocks with valid momentum factor  "
                     f"lookback [{t-lookback+1}..{t}] "
                     f"({_d(t-lookback+1)} to {_d(t)})")

    # No standardisation — use raw momentum as per Lou & Polk (2021)
    momentum_std = momentum

    # ----- Summary -----
    n_valid = np.sum(np.isfinite(momentum))
    n_cells = T * N
    n_scored_cells = n_scored * N
    log.info("-" * 60)
    log.info("MOMENTUM FACTOR COMPUTATION SUMMARY")
    log.info("-" * 60)
    log.info(f"  Total cells (T x N)                  : {n_cells:,}")
    log.info(f"  Cells with momentum factor ({n_scored} x {N}) : {n_scored_cells:,}")
    log.info(f"  Valid momentum factor values          : {n_valid:,}  "
             f"({n_valid/n_scored_cells*100:.1f}% of factor cells)")
    log.info(f"  NaN within factor range               : {n_scored_cells - n_valid:,}  "
             f"(stocks with all-NaN returns in their window)")
    log.info(f"  Weeks with no factor (pre-week {first_t}) : "
             f"{first_t} weeks  ({first_t * N:,} cells)  "
             f"(before {_d(first_t)})")
    log.info(f"  Weeks with no factor (skip)            : "
             f"{skip} weeks  ({skip * N:,} cells)  "
             f"({_d(T-skip)} to {_d(T-1)})")

    # Spot checks at first, middle, and last factor weeks
    for check_t in [last_t, (first_t + last_t) // 2, first_t]:
        mom_t = momentum[check_t, :]
        nv = np.sum(np.isfinite(mom_t))
        ws = check_t - lookback + 1
        if nv > 0:
            log.info(f"  Spot check week {check_t} ({_d(check_t)}): "
                     f"{nv:,} valid, "
                     f"mean={np.nanmean(mom_t):.4f}, "
                     f"median={np.nanmedian(mom_t):.4f}, "
                     f"std={np.nanstd(mom_t, ddof=1):.4f}  "
                     f"lookback [{ws}..{check_t}] "
                     f"({_d(ws)} to {_d(check_t)})")

    log.info("-" * 60)
    log.info("STEP 2 COMPLETE")
    log.info("-" * 60)

    return momentum, momentum_std


# =====================================================================
# 2. Comomentum Measure (Lou & Polk, 2021)
# =====================================================================

def compute_comomentum(returns_clean, momentum_std, live, ff_factors,
                       corr_window=52, min_resid_obs=20, min_stocks=5):
    """
    Computes the comomentum time series following Lou & Polk (2021).

    INTUITION:
        Comomentum measures how correlated the *abnormal* returns of
        typical momentum stocks are with each other. High comomentum
        signals crowded momentum trades (many arbitrageurs are in the
        same positions) which tends to predict *weak* future momentum
        returns. Low comomentum signals less crowding and predicts
        *strong* future momentum returns.

    PROCEDURE (at each week t):
        1. Identify "momentum stocks": stocks in the top quintile
           (winners) and bottom quintile (losers) of the momentum
           distribution among live stocks.
        2. For each momentum stock, run a time-series regression of
           its last 52 weeks of returns on the three Fama-French
           factors (Mkt-RF, SMB, HML) to obtain abnormal return
           residuals (i.e. the component of returns not explained
           by common risk factors).
        3. Compute the pairwise correlation matrix of these residuals
           across all momentum stocks.
        4. Comomentum = the average of all pairwise correlations
           (upper triangle of the correlation matrix, excluding the
           diagonal).

    Note: The paper also describes an industry adjustment which we
    omit here for simplicity, as instructed by the coursework brief.

    INPUTS:
        returns_clean : TxN np.ndarray  - weekly returns (dead = NaN)
        momentum_std  : TxN np.ndarray  - raw momentum scores
        live          : TxN np.ndarray  - live/dead dummies
        ff_factors    : Tx3 np.ndarray  - [Mkt-RF, SMB, HML] weekly
        corr_window   : int  - rolling window for residual correlations
                        (default 52 weeks = 1 year)
        min_resid_obs : int  - minimum valid residual observations per stock
        min_stocks    : int  - minimum number of momentum stocks needed

    OUTPUTS:
        comomentum : T-length np.ndarray - comomentum time series (NaN
                     where it could not be computed)
    """

    T, N = returns_clean.shape

    # Comomentum requires both momentum scores (available from week 47+)
    # and a corr_window-week return history for residual estimation
    mom_first = 47  # first week with a valid momentum score (lookback-1)
    comom_start = max(mom_first, corr_window)

    log.info("=" * 60)
    log.info("STEP 4: Computing comomentum (Lou & Polk, 2021)")
    log.info("=" * 60)
    log.info(f"  Parameters: corr_window={corr_window}w, "
             f"min_resid_obs={min_resid_obs}, min_stocks={min_stocks}")
    log.info(f"  Computation starts at week {comom_start}")

    comomentum = np.full(T, np.nan)
    n_computed = 0
    n_skipped_few_mom = 0
    n_skipped_few_resid = 0

    for t in range(comom_start, T):

        # ----- Step 1: Identify momentum stocks (winners + losers) -----
        mom_t = momentum_std[t, :]
        lv_t = live[t, :]

        # Only consider live stocks with a valid momentum score
        valid_mom = np.isfinite(mom_t) & (lv_t == 1)

        if np.sum(valid_mom) < 10:
            n_skipped_few_mom += 1
            continue

        # Compute quintile breakpoints (20th and 80th percentiles)
        mom_valid = mom_t[valid_mom]
        q20 = np.nanpercentile(mom_valid, 20)
        q80 = np.nanpercentile(mom_valid, 80)

        # Momentum stocks = losers (bottom 20%) + winners (top 20%)
        mom_stocks = valid_mom & ((mom_t <= q20) | (mom_t >= q80))
        mom_idx = np.where(mom_stocks)[0]

        if len(mom_idx) < min_stocks:
            n_skipped_few_mom += 1
            continue

        # ----- Step 2: Regress returns on FF3 factors, collect residuals -----
        # Use a rolling window of `corr_window` weeks ending at week t
        ret_window = returns_clean[t - corr_window + 1: t + 1, :][:, mom_idx]
        ff_window = ff_factors[t - corr_window + 1: t + 1, :]

        n_mom = len(mom_idx)
        residuals = np.full((corr_window, n_mom), np.nan)

        for j in range(n_mom):
            y_j = ret_window[:, j]
            valid_j = np.isfinite(y_j)

            if np.sum(valid_j) < min_resid_obs:
                continue

            # OLS: r_j = alpha + beta1*MktRF + beta2*SMB + beta3*HML + eps
            X_j = np.hstack((np.ones((np.sum(valid_j), 1)),
                             ff_window[valid_j, :]))
            Y_j = y_j[valid_j]

            coefs_j = np.linalg.lstsq(X_j, Y_j, rcond=None)[0]

            # Store the residual (= abnormal return)
            residuals[valid_j, j] = Y_j - X_j.dot(coefs_j)

        # ----- Step 3: Compute pairwise correlations of residuals -----
        # Only keep stocks with enough residual data points
        valid_stocks = np.sum(np.isfinite(residuals), axis=0) >= min_resid_obs
        if np.sum(valid_stocks) < min_stocks:
            n_skipped_few_resid += 1
            continue

        resid_valid = residuals[:, valid_stocks]

        # Use pandas for pairwise-complete correlation (handles NaNs)
        corr_matrix = pd.DataFrame(resid_valid).corr(min_periods=15).values

        # ----- Step 4: Average the upper-triangle correlations -----
        n_valid = resid_valid.shape[1]
        corr_sum = 0.0
        corr_count = 0

        for ii in range(n_valid):
            for jj in range(ii + 1, n_valid):
                if np.isfinite(corr_matrix[ii, jj]):
                    corr_sum += corr_matrix[ii, jj]
                    corr_count += 1

        if corr_count > 0:
            comomentum[t] = corr_sum / corr_count
            n_computed += 1

        # Progress indicator (logged every 100 weeks)
        if t % 100 == 0:
            log.info(f"    Comomentum: processed week {t}/{T}...")

    # --- Summary ---
    n_valid_comom = np.sum(np.isfinite(comomentum))
    log.info("-" * 50)
    log.info("  Comomentum computation summary:")
    log.info(f"    Total weeks in loop       : {T - comom_start}")
    log.info(f"    Successfully computed      : {n_computed}")
    log.info(f"    Skipped (few mom stocks)   : {n_skipped_few_mom}")
    log.info(f"    Skipped (few residuals)    : {n_skipped_few_resid}")
    log.info(f"    Skipped (other / no corrs) : "
             f"{T - comom_start - n_computed - n_skipped_few_mom - n_skipped_few_resid}")
    log.info(f"    Valid comomentum values    : {n_valid_comom} / {T}")
    if n_valid_comom > 0:
        comom_vals = comomentum[np.isfinite(comomentum)]
        log.info(f"    Mean comomentum            : {np.mean(comom_vals):.6f}")
        log.info(f"    Std  comomentum            : {np.std(comom_vals):.6f}")
        log.info(f"    Min  comomentum            : {np.min(comom_vals):.6f}")
        log.info(f"    Max  comomentum            : {np.max(comom_vals):.6f}")
    log.info("-" * 50)

    return comomentum


# =====================================================================
# 3. Adjusted Momentum Factor (Comomentum-Enhanced)
# =====================================================================

def compute_adjusted_momentum(momentum_std, comomentum, T, N):
    """
    Adjusts the standard momentum factor using the comomentum signal
    following the insight from Lou & Polk (2021).

    RATIONALE:
        The paper finds that when comomentum is LOW (less crowding),
        momentum strategies tend to earn HIGHER future returns, and
        when comomentum is HIGH (more crowding), momentum strategies
        tend to earn LOWER future returns.

    ADJUSTMENT APPROACH:
        We scale the momentum exposure inversely with the *lagged*
        comomentum level. This is done in a look-ahead-bias-free
        manner by using an expanding-window percentile rank:

        1. At each week t, compute the percentile rank of the current
           comomentum value relative to ALL past comomentum values
           (expanding window -> no look-ahead bias).
        2. Define a scaling factor:
               scaling_t = 2.0 - percentile_rank_{t-1}
           This gives:
             - Low comomentum  (rank ~ 0) -> scaling ~ 2.0  (increase bet)
             - High comomentum (rank ~ 1) -> scaling ~ 1.0  (reduce bet)
        3. Multiply the standardised momentum exposures by the scaling
           factor, then re-standardise cross-sectionally.

    INPUTS:
        momentum_std : TxN np.ndarray - raw momentum exposures
        comomentum   : T-length np.ndarray - comomentum time series
        T            : int - number of weeks
        N            : int - number of stocks

    OUTPUTS:
        momentum_adj_std : TxN np.ndarray - adjusted & re-standardised momentum
        scaling          : T-length np.ndarray - scaling factor time series
        comom_pctile     : T-length np.ndarray - comomentum percentile ranks
    """

    # ----- Step A: Expanding-window percentile rank of comomentum -----
    # At time t, rank the current comomentum relative to all historical
    # values. This avoids look-ahead bias since we only use past data.
    log.info("=" * 60)
    log.info("STEP 5: Adjusting momentum using comomentum (Lou & Polk, 2021)")
    log.info("=" * 60)
    log.info(f"  Input momentum_std shape : {momentum_std.shape}")
    log.info(f"  Input comomentum length  : {len(comomentum)}")
    log.info(f"  T={T}, N={N}")

    log.info("  Step A: Computing expanding-window percentile rank of comomentum...")
    comom_pctile = np.full(T, np.nan)
    n_pctile_computed = 0

    for t in range(1, T):
        past_comom = comomentum[:t]  # all values strictly before time t
        valid_past = past_comom[np.isfinite(past_comom)]

        # Need enough history and a valid current value to compute rank
        if len(valid_past) < 10 or not np.isfinite(comomentum[t]):
            continue

        # Percentile rank = fraction of past values <= current value
        comom_pctile[t] = np.mean(valid_past <= comomentum[t])
        n_pctile_computed += 1

    n_valid_pctile = np.sum(np.isfinite(comom_pctile))
    log.info(f"    Percentile ranks computed : {n_pctile_computed} / {T-1}")
    if n_valid_pctile > 0:
        pct_vals = comom_pctile[np.isfinite(comom_pctile)]
        log.info(f"    Percentile mean={np.mean(pct_vals):.4f}, "
                 f"std={np.std(pct_vals):.4f}, "
                 f"min={np.min(pct_vals):.4f}, max={np.max(pct_vals):.4f}")

    # ----- Step B: Compute the scaling factor -----
    # Use the LAGGED percentile rank (t-1) to avoid any contemporaneous bias
    log.info("  Step B: Computing scaling factor = 2.0 - lagged percentile rank...")
    scaling = np.full(T, np.nan)

    for t in range(1, T):
        if np.isfinite(comom_pctile[t - 1]):
            scaling[t] = 2.0 - comom_pctile[t - 1]
        else:
            # When no comomentum data is available yet, use neutral scaling (1.0)
            scaling[t] = 1.0

    n_valid_scaling = np.sum(np.isfinite(scaling))
    scl_vals = scaling[np.isfinite(scaling)]
    log.info(f"    Valid scaling values : {n_valid_scaling} / {T}")
    if len(scl_vals) > 0:
        log.info(f"    Scaling mean={np.mean(scl_vals):.4f}, "
                 f"std={np.std(scl_vals):.4f}, "
                 f"min={np.min(scl_vals):.4f}, max={np.max(scl_vals):.4f}")

    # ----- Step C: Scale the momentum factor -----
    # Multiply each stock's momentum by the time-varying scaling
    log.info("  Step C: Scaling momentum exposures...")
    momentum_adj = np.full((T, N), np.nan)

    for t in range(T):
        if np.isfinite(scaling[t]):
            momentum_adj[t, :] = momentum_std[t, :] * scaling[t]
        else:
            momentum_adj[t, :] = momentum_std[t, :]

    # ----- Step D: No re-standardisation (paper replication) -----
    log.info("  Step D: Skipped — no re-standardisation (raw momentum).")
    momentum_adj_std = momentum_adj

    # --- Final summary ---
    n_finite_adj = np.sum(np.isfinite(momentum_adj_std))
    log.info("-" * 50)
    log.info("  Adjusted momentum summary:")
    log.info(f"    Output shape               : {momentum_adj_std.shape}")
    log.info(f"    Finite values               : {n_finite_adj} / {T * N} "
             f"({100.0 * n_finite_adj / (T * N):.2f}%)")
    # Spot check: verify cross-sectional mean ~ 0, std ~ 1 at a mid-point
    mid_t = T // 2
    row = momentum_adj_std[mid_t, :]
    finite_row = row[np.isfinite(row)]
    if len(finite_row) > 1:
        log.info(f"    Spot check week {mid_t}: "
                 f"mean={np.mean(finite_row):.6f}, std={np.std(finite_row):.6f}")
    log.info("-" * 50)

    return momentum_adj_std, scaling, comom_pctile
