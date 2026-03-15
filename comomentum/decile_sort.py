# decile_sort.py
# =====================================================================
# WHAT THIS MODULE DOES:
#   Given a cross-section of momentum scores and a live indicator,
#   sort stocks into extreme deciles:
#     - Bottom DECILE_PCT_LO % → loser decile
#     - Top (100 - DECILE_PCT_HI) % → winner decile
#   Returns column indices for each group.
#
#   Pairwise abnormal correlations are in pairwise_correlations.py.
#
# Standalone:  python decile_sort.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from config import DECILE_PCT_LO, DECILE_PCT_HI


# =====================================================================
# Decile sorting
# =====================================================================
def sort_deciles(momentum_t, live_t):
    """
    Sort stocks into extreme loser and winner deciles at one week.

    INPUT:
        momentum_t : (N,) np.ndarray — momentum scores for N stocks
        live_t     : (N,) np.ndarray — live indicator (1 = live, 0 = dead)

    OUTPUT:
        loser_idx  : 1-D np.ndarray of column indices in the loser decile
        winner_idx : 1-D np.ndarray of column indices in the winner decile
        n_valid    : int — number of stocks that had valid momentum + live
    """
    # A stock qualifies only if it has a non-NaN momentum score and 
    # is currently listed. Stocks that are dead or missing momentum are excluded.
    valid_mask = np.isfinite(momentum_t) & (live_t == 1)

    n_valid = int(np.sum(valid_mask))

    # Need at least 2 valid stocks to form any groups.
    if n_valid < 2:
        return np.array([], dtype=int), np.array([], dtype=int), n_valid

    mom_valid = momentum_t[valid_mask]
    
    # With DECILE_PCT_LO=10 and DECILE_PCT_HI=90, this finds the momentum values 
    # that separate the bottom 10% (losers) and top 10% (winners) from the rest.
    q_lo = np.percentile(mom_valid, DECILE_PCT_LO)
    q_hi = np.percentile(mom_valid, DECILE_PCT_HI)

    # Losers: valid stocks whose momentum is at or below the 10th percentile
    loser_mask  = valid_mask & (momentum_t <= q_lo)
    
    # Winners: valid stocks whose momentum is at or above the 90th percentile
    winner_mask = valid_mask & (momentum_t >= q_hi)

    # Converts boolean masks to integer col indices into original N-stock array. 
    loser_idx  = np.where(loser_mask)[0]
    winner_idx = np.where(winner_mask)[0]

    # Example: If there are 800 valid stocks at week t, 
    # roughly 80 land in losers (≤ 10th percentile) 
    # and 80 in winners (≥ 90th percentile). 
    # Their FF3 residuals are then computed and pairwise-correlated 
    # to produce comomentum.
    return loser_idx, winner_idx, n_valid

# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )

    # Demo: sort deciles at the last week
    T = data['returns_clean'].shape[0]
    t = T - 1

    loser_idx, winner_idx, n_valid = sort_deciles(
        momentum_std[t, :], data['live'][t, :]
    )
    print(f"Week {t}: {n_valid} valid stocks, "
          f"{len(loser_idx)} losers, {len(winner_idx)} winners")
