# compute_momentum_signal.py
# =====================================================================
# Momentum Signal Construction
# =====================================================================
# Computes the standard momentum factor for each stock at each week
# using a ROLLING window that skips the most recent 4 weeks.
#
# LOGIC (matches momentum_schedule.py):
#   At each computation date t (from week 52 onwards, 0-indexed t=51):
#     - Skip the 4 most recent weeks: indices t, t-1, t-2, t-3
#     - Compound returns over the 48 weeks before that:
#       indices (t - 51) to (t - 4)
#     - Momentum score = product of (1 + r) over those 48 weeks, minus 1
#
#   Parameters:
#     LOOKBACK = 48 weeks  (≈ 11 months of compounded returns)
#     SKIP     = 4  weeks  (≈ 1 month, most recent — avoids reversal)
#     TOTAL    = 52 weeks  (minimum history before first score)
#
#   First score: week index 51 (week 52 in 1-based = 19921225)
#     Lookback: indices 0..47  (19920103 to 19921127)
#     Skipped:  indices 48..51 (19921204 to 19921225)
#
#   Second score: week index 52 (week 53 = 19930101)
#     Lookback: indices 1..48  (19920110 to 19921204)
#     Skipped:  indices 49..52 (19921211 to 19930101)
#
#   ...and so on, rolling forward one week at a time.
#
#   Total scored weeks: T - TOTAL + 1 = 1513 - 52 + 1 = 1462
#   Weeks 0..50 (first 51 weeks) have no momentum score (NaN).
#
# COMPOUNDING:
#   mom_{i,t} = prod(1 + r_{i,s} for s in lookback window) - 1
#
# ELIGIBILITY (per-window, per-stock):
#   A stock receives a momentum score at week t ONLY if ALL 52 weeks
#   in the full window [t-51 .. t] have valid (non-NaN) returns.
#   This means:
#     - The 48 lookback weeks [t-51 .. t-4] must ALL be traded
#     - The 4 skip weeks [t-3 .. t] must ALL be traded
#   If ANY of the 52 weeks is NaN, the stock gets NaN for that week.
#   This naturally handles gapped stocks: they get a score only in
#   periods where they have a continuous 52-week traded stretch.
#
# OUTPUT:
#   momentum     : TxN np.ndarray — raw momentum scores (NaN for t < 51)
#   momentum_std : TxN np.ndarray — same as momentum (raw, no standardisation)
#
# Standalone:   python compute_momentum_signal.py
# From loader:  from compute_momentum_signal import compute_momentum_signal
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from logger_setup import _setup_logger
from config import LOOKBACK, SKIP, TOTAL

log = _setup_logger()


def compute_momentum_signal(returns_clean, dates):
    """
    Computes the rolling momentum signal for every stock at every week.

    At week t the momentum score for stock i is the cumulative return
    over weeks (t-51) to (t-4), skipping the 4 most recent weeks.

    INPUT:
        returns_clean : np.ndarray (TxN) — weekly returns (not-listed = NaN)
        dates         : pd.DatetimeIndex  — T weekly dates

    OUTPUT:
        momentum      : np.ndarray (TxN) — raw momentum factor
        momentum_std  : np.ndarray (TxN) — same as momentum (raw, no standardisation)
    """

    T, N = returns_clean.shape

    # Index of first and last scored weeks (0-based)
    first_t = TOTAL - 1               # 51  (week 52 in 1-based)
    last_t  = T - 1                   # 1512
    n_scored = last_t - first_t + 1   # 1462

    # Helper to format a date
    def _d(idx):
        return pd.Timestamp(dates[idx]).strftime('%Y-%m-%d')

    log.info("=" * 60)
    log.info("STEP 2: Computing momentum signal (rolling window)")
    log.info("=" * 60)
    log.info(f"  Lookback  = {LOOKBACK} weeks")
    log.info(f"  Skip      = {SKIP} weeks (most recent, per rolling window)")
    log.info(f"  Total     = {TOTAL} weeks required before first score")
    log.info(f"  Input     : T={T} weeks, N={N} stocks")
    log.info(f"  Date range: {_d(0)} to {_d(T-1)}")
    log.info(f"  First momentum score: week {first_t+1} (index {first_t}), "
             f"date {_d(first_t)}")
    log.info(f"    Lookback : indices 0..{first_t - SKIP}  "
             f"({_d(0)} to {_d(first_t - SKIP)})")
    log.info(f"    Skipped  : indices {first_t - SKIP + 1}..{first_t}  "
             f"({_d(first_t - SKIP + 1)} to {_d(first_t)})")
    log.info(f"  Last momentum score : week {last_t+1} (index {last_t}), "
             f"date {_d(last_t)}")
    log.info(f"    Lookback : indices {last_t - TOTAL + 1}..{last_t - SKIP}  "
             f"({_d(last_t - TOTAL + 1)} to {_d(last_t - SKIP)})")
    log.info(f"    Skipped  : indices {last_t - SKIP + 1}..{last_t}  "
             f"({_d(last_t - SKIP + 1)} to {_d(last_t)})")
    log.info(f"  Total scored weeks: {n_scored}")
    log.info(f"  Weeks with no score (t < {first_t}): {first_t} weeks")

    # ── Pre-allocate output (NaN = no score for that week) ───────────
    momentum = np.full((T, N), np.nan)

    # ── Rolling computation ──────────────────────────────────────────
    for t in range(first_t, T):
        # Full 52-week window: indices (t - TOTAL + 1) to t
        full_start = t - TOTAL + 1     # e.g. t=51 -> 0
        full_end   = t                 # e.g. t=51 -> 51

        # Lookback window (for compounding): indices (t - TOTAL + 1) to (t - SKIP)
        lb_start = full_start          # e.g. t=51 -> 0
        lb_end   = t - SKIP           # e.g. t=51 -> 47

        # Full window of 52 weeks (used for eligibility check)
        full_window = returns_clean[full_start: full_end + 1, :]  # shape (52, N)

        # ELIGIBILITY: stock must have ALL 52 weeks with valid returns
        # (48 lookback + 4 skip — all must be traded)
        any_nan = np.any(np.isnan(full_window), axis=0)  # True if any week is NaN

        # Lookback window only (for compounding the 48-week return)
        ret_window = returns_clean[lb_start: lb_end + 1, :]  # shape (48, N)

        # Compound returns: prod(1 + r) over the 48 lookback weeks
        mom_t = np.prod(1 + ret_window, axis=0) - 1

        # NaN-out stocks that don't have the full 52 valid weeks
        mom_t[any_nan] = np.nan

        momentum[t, :] = mom_t

        # Progress logging every 200 weeks
        if (t - first_t) % 200 == 0:
            n_valid_t = np.sum(np.isfinite(momentum[t, :]))
            log.info(f"    Week {t+1:>5} ({_d(t)}) : "
                     f"{n_valid_t:,} stocks scored  |  "
                     f"lookback {_d(lb_start)} to {_d(lb_end)}")

    # ── No standardisation — use raw momentum as per Lou & Polk (2021) ──
    momentum_std = momentum

    # ── Summary ──────────────────────────────────────────────────────
    n_valid = int(np.sum(np.isfinite(momentum)))
    n_scored_cells = n_scored * N

    log.info("-" * 60)
    log.info("MOMENTUM SIGNAL SUMMARY")
    log.info("-" * 60)
    log.info(f"  Total cells (T × N)        : {T * N:,}")
    log.info(f"  Scored cells ({n_scored} × {N}) : {n_scored_cells:,}")
    log.info(f"  Valid momentum values       : {n_valid:,}  "
             f"({n_valid / n_scored_cells * 100:.1f}% of scored cells)")
    log.info(f"  NaN within scored range     : {n_scored_cells - n_valid:,}  "
             f"(stocks not listed during their lookback)")

    # Spot checks
    for label, check_t in [("First", first_t),
                           ("Mid",   (first_t + last_t) // 2),
                           ("Last",  last_t)]:
        mom_t = momentum[check_t, :]
        nv = int(np.sum(np.isfinite(mom_t)))
        lb_s = check_t - TOTAL + 1
        lb_e = check_t - SKIP
        if nv > 0:
            log.info(f"  {label} (week {check_t+1}, {_d(check_t)}): "
                     f"{nv:,} valid, "
                     f"mean={np.nanmean(mom_t):.4f}, "
                     f"median={np.nanmedian(mom_t):.4f}, "
                     f"std={np.nanstd(mom_t, ddof=1):.4f}  |  "
                     f"lookback {_d(lb_s)} to {_d(lb_e)}")

    log.info("-" * 60)
    log.info("STEP 2 COMPLETE — momentum signal computed (raw, no z-scoring).")
    log.info("-" * 60)

    return momentum, momentum_std


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )
    print(f"\nMomentum shape: {momentum.shape}")
    print(f"Valid scores:   {np.sum(np.isfinite(momentum)):,}")
    print(f"First scored week: index {TOTAL - 1}")
