# compute_adjusted_momentum.py
# =====================================================================
# Adjusted Momentum Factor — Inverse Comomentum Timing
# =====================================================================
#
# PAPER REFERENCE:
#   Lou, D. & Polk, C. (2021). "Comomentum: Inferring Arbitrage
#   Activity from Return Correlations."  Review of Financial Studies.
#
# RATIONALE:
#   The paper finds that when comomentum is LOW (less crowding /
#   arbitrage), momentum strategies earn HIGHER future returns.
#   When comomentum is HIGH (more crowding), momentum strategies
#   earn LOWER future returns.
#
#   To exploit this, we TIME the momentum strategy: INCREASE the bet
#   when comomentum is low and DECREASE it when comomentum is high.
#
# WHY SCALE FACTOR RETURNS (gamma), NOT EXPOSURES?
# -------------------------------------------------
#   The scaling factor s_t is the SAME for all stocks at week t.
#   If we multiply the cross-sectional z-scores by s_t and then
#   re-standardise, the effect is perfectly undone:
#
#       z_new_i = (s_t * z_i − mean(s_t * z)) / std(s_t * z)
#               = (s_t * z_i − s_t * 0) / (|s_t| * 1)
#               = z_i
#
#   So the Fama-MacBeth regression would see identical cross-sectional
#   rankings → identical gamma → no change.
#
#   Instead, we apply the scaling to the FACTOR RETURN series:
#       gamma_adj[t] = gamma_std[t] * scaling[t]
#
#   This correctly represents a strategy that increases / decreases
#   its capital allocation to the momentum portfolio depending on
#   the level of crowding (comomentum).  The Fama-MacBeth regression
#   runs ONCE on the standard momentum factor, and the scaling
#   determines how much of the resulting premium the investor captures
#   at each point in time.
#
# PROCEDURE:
#   Step A — Expanding-window percentile rank of comomentum
#       At each week t, compute the percentile rank of comomentum[t]
#       relative to ALL comomentum values from the start up to t-1
#       (strictly past — no look-ahead bias).  Need ≥ 10 past values.
#
#   Step B — Scaling factor
#       scaling[t] = 2.0 - percentile_rank[t-1]
#       This gives:
#         • Low  comomentum (rank ~ 0) → scaling ~ 2.0 (increase bet)
#         • Mid  comomentum (rank ~ 0.5) → scaling ~ 1.5 (neutral-ish)
#         • High comomentum (rank ~ 1) → scaling ~ 1.0 (reduce bet)
#       Note: we use the LAGGED percentile rank (t-1) to ensure the
#       signal is available before the return it is predicting.
#
#   Step C — Scale factor returns (NOT exposures)
#       gamma_adj[t] = gamma_std[t] * scaling[t]
#
# INPUT:
#   gamma_std  : T-length np.ndarray — standard momentum factor returns
#   comomentum : T-length array      — comomentum time series
#
# OUTPUT:
#   gamma_adj    : T-length array  — adjusted (scaled) factor returns
#   scaling      : T-length array  — scaling factor time series
#   comom_pctile : T-length array  — percentile rank series
#
# Standalone:  python compute_adjusted_momentum.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from logger_setup import _setup_logger
from config import MIN_PAST_VALUES

log = _setup_logger()


def compute_adjusted_momentum(gamma_std, comomentum):
    """
    Adjusts the standard momentum FACTOR RETURNS using an inverse
    comomentum timing signal following Lou & Polk (2021).

    This scales the Fama-MacBeth gamma series — NOT the cross-sectional
    exposures — to avoid the z-score identity trap where re-standardising
    a uniformly-scaled z-score matrix recovers the original z-scores.

    Parameters
    ----------
    gamma_std  : T-length np.ndarray
        Weekly factor return series from Fama-MacBeth on standard momentum.
    comomentum : T-length np.ndarray
        Comomentum time series.

    Returns
    -------
    gamma_adj    : T-length np.ndarray — adjusted factor returns
    scaling      : T-length np.ndarray — time-varying scaling factor
    comom_pctile : T-length np.ndarray — expanding-window percentile ranks
    """

    T = len(gamma_std)

    log.info("=" * 60)
    log.info("STEP 5: Adjusting momentum with comomentum (Lou & Polk, 2021)")
    log.info("=" * 60)
    log.info(f"  Input gamma_std    : length {T}, "
             f"{np.sum(np.isfinite(gamma_std))} finite values")
    log.info(f"  Input comomentum   : length {len(comomentum)}, "
             f"{np.sum(np.isfinite(comomentum))} finite values")

    # ── Step A: Expanding-window percentile rank ─────────────────────
    log.info("  Step A: Computing expanding-window percentile rank ...")
    comom_pctile = np.full(T, np.nan)
    n_pctile = 0

    for t in range(1, T):
        past = comomentum[:t]                      # all values before t
        valid_past = past[np.isfinite(past)]

        if len(valid_past) < MIN_PAST_VALUES or not np.isfinite(comomentum[t]):
            continue

        # Percentile rank = fraction of past values ≤ current value
        comom_pctile[t] = np.mean(valid_past <= comomentum[t])
        n_pctile += 1

    log.info(f"    Percentile ranks computed: {n_pctile}")
    if n_pctile > 0:
        pv = comom_pctile[np.isfinite(comom_pctile)]
        log.info(f"    mean={np.mean(pv):.4f}, std={np.std(pv):.4f}, "
                 f"min={np.min(pv):.4f}, max={np.max(pv):.4f}")

    # ── Step B: Scaling factor = 2.0 − lagged percentile rank ────────
    log.info("  Step B: Computing scaling factor = 2.0 - lagged percentile rank ...")
    scaling = np.full(T, np.nan)

    for t in range(1, T):
        if np.isfinite(comom_pctile[t - 1]):
            scaling[t] = 2.0 - comom_pctile[t - 1]
        else:
            # No past comomentum info → neutral scaling (1.0)
            scaling[t] = 1.0

    n_valid_scl = int(np.sum(np.isfinite(scaling)))
    scl_vals = scaling[np.isfinite(scaling)]
    log.info(f"    Valid scaling values: {n_valid_scl}")
    if len(scl_vals) > 0:
        log.info(f"    mean={np.mean(scl_vals):.4f}, std={np.std(scl_vals):.4f}, "
                 f"min={np.min(scl_vals):.4f}, max={np.max(scl_vals):.4f}")

    # ── Step C: Scale the factor returns (NOT exposures) ─────────────
    log.info("  Step C: Scaling factor RETURNS (gamma_std * scaling) ...")
    gamma_adj = np.full(T, np.nan)

    for t in range(T):
        if np.isfinite(gamma_std[t]) and np.isfinite(scaling[t]):
            gamma_adj[t] = gamma_std[t] * scaling[t]
        elif np.isfinite(gamma_std[t]):
            # No scaling available → keep original factor return
            gamma_adj[t] = gamma_std[t]
        # else: both NaN → stays NaN

    n_finite_adj = int(np.sum(np.isfinite(gamma_adj)))

    # ── Summary ──────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("ADJUSTED MOMENTUM SUMMARY")
    log.info("-" * 60)
    log.info(f"  Finite gamma_std   : {int(np.sum(np.isfinite(gamma_std)))}")
    log.info(f"  Finite gamma_adj   : {n_finite_adj}")
    log.info(f"  Finite scaling     : {n_valid_scl}")

    if n_finite_adj > 0:
        va = gamma_adj[np.isfinite(gamma_adj)]
        vs = gamma_std[np.isfinite(gamma_std)]
        log.info(f"  gamma_std  mean={np.mean(vs)*100:.4f}%, "
                 f"std={np.std(vs)*100:.4f}%")
        log.info(f"  gamma_adj  mean={np.mean(va)*100:.4f}%, "
                 f"std={np.std(va)*100:.4f}%")
        # Check they differ
        corr = np.corrcoef(vs[:len(va)], va[:len(va)])[0, 1]
        log.info(f"  Correlation(gamma_std, gamma_adj) = {corr:.6f}")
        log.info(f"  (Should be < 1.0 if scaling has any effect)")

    log.info("-" * 60)
    log.info("STEP 5 COMPLETE — adjusted momentum factor returns computed.")
    log.info("-" * 60)

    return gamma_adj, scaling, comom_pctile


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from comomentum.compute_comomentum import compute_comomentum
    from fama_macbeth.fama_macbeth import famaMacBeth

    data = load_all_data('input_data/')
    mom, mom_std = compute_momentum_signal(data['returns_clean'], data['dates'])

    # Run Fama-MacBeth on standard momentum first
    gamma_std, tstat_std = famaMacBeth(mom_std, data['returns_clean'], data['live'])
    print(f"\nStandard momentum t-stat: {tstat_std:.4f}")

    # Compute comomentum
    comom, _, _ = compute_comomentum(
        data['returns_clean'], mom_std,
        data['live'], data['ff_factors'], data['dates']
    )

    # Adjust factor returns
    gamma_adj, scl, pctile = compute_adjusted_momentum(gamma_std, comom)
    print(f"\ngamma_adj: {np.sum(np.isfinite(gamma_adj))} finite values")
    print(f"gamma_std mean: {np.nanmean(gamma_std)*100:.4f}%")
    print(f"gamma_adj mean: {np.nanmean(gamma_adj)*100:.4f}%")
