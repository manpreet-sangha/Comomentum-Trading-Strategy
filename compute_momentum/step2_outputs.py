# step2_outputs.py
# =====================================================================
# Step 2 — Save Momentum Data to CSV
# =====================================================================
# Saves two CSV files to output_data/:
#   1. momentum_raw.csv               — raw scores, all stocks
#   2. momentum_summary.csv           — weekly cross-sectional stats
#
# Standalone:   python step2_outputs.py
# From loader:  from step2_outputs import save_momentum_outputs
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()


def save_momentum_outputs(momentum, momentum_std, dates, names,
                          output_dir='output_data'):
    """
    Saves momentum factor data to CSV files.

    INPUT:
        momentum     : np.ndarray (TxN) — raw momentum scores
        momentum_std : np.ndarray (TxN) — raw momentum (kept for API compat)
        dates        : pd.DatetimeIndex  — weekly dates
        names        : np.ndarray (N,)   — stock name labels
        output_dir   : str               — folder for output files
    """

    os.makedirs(output_dir, exist_ok=True)
    T, N = momentum.shape

    log.info("=" * 60)
    log.info("STEP 2 OUTPUT: Saving momentum data to CSV")
    log.info("=" * 60)

    # ── 1. Raw momentum (all stocks) ────────────────────────────────
    df_raw = pd.DataFrame(
        momentum,
        index=dates,
        columns=names
    )
    df_raw.index.name = 'Date'
    path1 = os.path.join(output_dir, 'momentum_raw.csv')
    df_raw.to_csv(path1)
    log.info(f"  Saved: {path1}  (raw momentum, all {N} stocks)")

    # ── 2. Weekly cross-sectional summary ────────────────────────────────
    rows = []
    for t in range(T):
        mom_t = momentum[t, :]
        valid = mom_t[np.isfinite(mom_t)]
        if len(valid) > 0:
            rows.append({
                'Date':    dates[t],
                'N_valid': len(valid),
                'Mean':    np.mean(valid),
                'Median':  np.median(valid),
                'Std':     np.std(valid, ddof=1),
                'Min':     np.min(valid),
                'Max':     np.max(valid),
            })
    df_summary = pd.DataFrame(rows)
    path3 = os.path.join(output_dir, 'momentum_summary.csv')
    df_summary.to_csv(path3, index=False)
    log.info(f"  Saved: {path3}  (weekly cross-sectional stats, "
             f"{len(df_summary)} weeks)")

    log.info("-" * 60)

    return df_summary


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
    save_momentum_outputs(momentum, momentum_std,
                          data['dates'], data['names'])
