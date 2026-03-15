# pairwise_correlations.py
# =====================================================================
# Pairwise Abnormal Correlations
# =====================================================================
# WHAT THIS MODULE DOES:
#   Given a (W x K) matrix of FF3 residuals for K stocks in one
#   decile over W weeks, compute the average pairwise correlation:
#
#       CoMOM_decile = mean over all K*(K-1)/2 unique pairs of
#                      Corr(resid_i, resid_j)
#
#   Two functions are provided:
#
#   1. decile_comomentum(residuals)
#      Returns the scalar average pairwise correlation plus counts.
#      Used by compute_comomentum.py in the main weekly loop.
#
#   2. build_corr_outputs(residuals, stock_names)
#      Returns the full K×K correlation matrix, long-format pairs
#      DataFrame, average correlation, and counts.  Used by
#      save_pairwise_correlations.py for diagnostic Excel output.
#
# Standalone:  python pairwise_correlations.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from config import CORR_WINDOW, MIN_RESID_OBS, MIN_STOCKS


def _eligible_mask(residuals):
    """
    Identify columns (stocks) with enough valid residual observations.

    Returns:
        eligible_idx : 1-D array of eligible column indices
        K_eligible   : int — number of eligible stocks
    """
    W, K = residuals.shape
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    eligible = np.zeros(K, dtype=bool)
    for i in range(K):
        eligible[i] = np.sum(np.isfinite(residuals[:, i])) >= resid_threshold

    eligible_idx = np.where(eligible)[0]
    return eligible_idx, len(eligible_idx)


# =====================================================================
# Scalar average pairwise correlation (used in the weekly loop)
# =====================================================================
def decile_comomentum(residuals):
    """
    Given a (W x K) matrix of FF3 residuals for K stocks in one decile
    over W weeks, compute the average pairwise abnormal correlation
    as described in Lou & Polk (2021).

    For every unique pair (i, j) where i < j:
        rho_{i,j} = Corr(resid_i, resid_j)
    CoMOM_decile = mean of all K*(K-1)/2 pairwise correlations.

    A stock must have at least MIN_RESID_OBS (or CORR_WINDOW) valid
    residual weeks to be eligible.  Only pairs where BOTH stocks are
    eligible contribute a correlation.

    Returns:
        (comom_value, K_total, K_contributed)
        comom_value    : float or NaN — decile comomentum
        K_total        : int — total stocks in the decile
        K_contributed  : int — stocks that were eligible
    """
    W, K = residuals.shape

    # If MIN_STOCKS is set (not None), enforce it; otherwise accept any K >= 2
    min_k = MIN_STOCKS if MIN_STOCKS is not None else 2
    if K < min_k:
        return np.nan, K, 0

    eligible_idx, K_eligible = _eligible_mask(residuals)

    min_contrib = MIN_STOCKS if MIN_STOCKS is not None else 2
    if K_eligible < min_contrib:
        return np.nan, K, K_eligible

    # Compute full correlation matrix via np.corrcoef
    resid_eligible = residuals[:, eligible_idx]              # (W, K_eligible)
    corr_matrix = np.corrcoef(resid_eligible, rowvar=False)  # (K_eligible, K_eligible)

    # Extract upper triangle (all unique pairs i < j)
    upper_idx = np.triu_indices(K_eligible, k=1)
    pairwise_corrs = corr_matrix[upper_idx]

    # Remove any NaN pairs (can occur if a stock has zero variance)
    valid_corrs = pairwise_corrs[np.isfinite(pairwise_corrs)]

    if len(valid_corrs) == 0:
        return np.nan, K, K_eligible

    return np.mean(valid_corrs), K, K_eligible


# =====================================================================
# Full correlation matrix + long-format pairs (for diagnostic output)
# =====================================================================
def build_corr_outputs(residuals, stock_names):
    """
    Given a (W, K) residual matrix and stock names, compute:
      - The K×K pairwise correlation matrix (DataFrame)
      - A long-format DataFrame of all unique pairs with correlations
      - Summary statistics

    Returns:
        (corr_df, pairs_df, avg_corr, n_pairs, K_eligible)
    """
    W, K = residuals.shape

    eligible_idx, K_eligible = _eligible_mask(residuals)

    if K_eligible < 2:
        return None, None, np.nan, 0, K_eligible

    eligible_names = [stock_names[i] for i in eligible_idx]
    resid_eligible = residuals[:, eligible_idx]

    # Full correlation matrix
    corr_matrix = np.corrcoef(resid_eligible, rowvar=False)
    corr_df = pd.DataFrame(corr_matrix,
                            index=eligible_names,
                            columns=eligible_names)

    # Long-format: all unique pairs (upper triangle)
    pairs_rows = []
    upper_i, upper_j = np.triu_indices(K_eligible, k=1)
    for idx in range(len(upper_i)):
        ii, jj = upper_i[idx], upper_j[idx]
        rho = corr_matrix[ii, jj]
        pairs_rows.append({
            'stock_i': eligible_names[ii],
            'stock_j': eligible_names[jj],
            'correlation': rho
        })

    pairs_df = pd.DataFrame(pairs_rows)

    # Summary
    valid_corrs = corr_matrix[upper_i, upper_j]
    valid_corrs = valid_corrs[np.isfinite(valid_corrs)]
    n_pairs = len(valid_corrs)
    avg_corr = np.mean(valid_corrs) if n_pairs > 0 else np.nan

    return corr_df, pairs_df, avg_corr, n_pairs, K_eligible


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from comomentum.ff3_residuals import compute_ff3_residuals
    from comomentum.decile_sort import sort_deciles

    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )

    # Demo: pairwise correlations at the last week
    T = data['returns_clean'].shape[0]
    t = T - 1

    loser_idx, winner_idx, n_valid = sort_deciles(
        momentum_std[t, :], data['live'][t, :]
    )
    print(f"Week {t}: {n_valid} valid stocks, "
          f"{len(loser_idx)} losers, {len(winner_idx)} winners")

    w_start = t - CORR_WINDOW + 1
    ff_window = data['ff_factors'][w_start:t+1, :]

    for label, idx in [('Loser', loser_idx), ('Winner', winner_idx)]:
        if len(idx) >= 2:
            ret = data['returns_clean'][w_start:t+1, :][:, idx]
            resid = compute_ff3_residuals(ret, ff_window)

            comom, K_total, K_contrib = decile_comomentum(resid)
            print(f"  {label} CoMOM = {comom:.6f} "
                  f"(K_total={K_total}, K_contrib={K_contrib})")

            names = [str(data['names'][i]) for i in idx]
            corr_df, pairs_df, avg, n_pairs, k_elig = build_corr_outputs(
                resid, names)
            print(f"  {label} full output: {n_pairs} pairs, "
                  f"avg_corr={avg:.6f}, K_eligible={k_elig}")
