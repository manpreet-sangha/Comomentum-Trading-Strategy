# save_ff3_residuals.py
# =====================================================================
# Save FF3 Regression Residuals — Step 4 Diagnostic Output
# =====================================================================
#
# At each week t in the comomentum computation, the code regresses
# each stock's last 52 weeks of returns on the Fama-French three
# factors (Mkt-RF, SMB, HML).  The residuals from these regressions
# are the "abnormal returns" used to compute pairwise correlations
# and ultimately the comomentum measure.
#
# This module saves those residuals to an Excel workbook so the
# intermediate computations can be inspected.
#
# WHAT IS SAVED:
#   An .xlsx file with three sheets:
#     1. 'Documentation'  — explains every column + methodology
#     2. 'Loser_Residuals' — (52 x K_losers) residuals for the
#         loser decile at the chosen snapshot week
#     3. 'Winner_Residuals' — (52 x K_winners) residuals for the
#         winner decile at the chosen snapshot week
#
#   Since residuals are recomputed at every single week (rolling
#   52-week window), saving ALL 1,462 weeks would produce a huge
#   file.  Instead, this module saves a SNAPSHOT for a user-chosen
#   week (defaulting to the last computable week).  To save multiple
#   weeks, call save_ff3_residuals() in a loop.
#
# Standalone:  python save_ff3_residuals.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import os

from config import (CORR_WINDOW, DECILE_PCT_LO, DECILE_PCT_HI,
                     MIN_RESID_OBS)
from comomentum.ff3_residuals import compute_ff3_residuals
from comomentum.decile_sort import sort_deciles


def save_ff3_residuals(returns_clean, momentum_std, live,
                       ff_factors, dates, names,
                       snapshot_week=None,
                       save_path='output_data/ff3_residuals.xlsx'):
    """
    Compute and save FF3 residuals for loser and winner deciles at
    one snapshot week.

    INPUTS:
        returns_clean : TxN np.ndarray
        momentum_std  : TxN np.ndarray  (raw momentum, kept for API compat)
        live          : TxN np.ndarray
        ff_factors    : Tx3 np.ndarray
        dates         : length-T array-like
        names         : length-N array-like  (stock names / tickers)
        snapshot_week : int index (0-based). Defaults to last week.
        save_path     : output .xlsx path

    OUTPUTS:
        Writes an Excel file.  Returns (resid_losers, resid_winners,
        loser_names, winner_names) for further use.
    """
    T, N = returns_clean.shape

    # ── Default to the last week ─────────────────────────────────────
    if snapshot_week is None:
        snapshot_week = T - 1

    t = snapshot_week
    date_str = pd.Timestamp(dates[t]).strftime('%Y-%m-%d')

    # ── Decile sort at week t ────────────────────────────────────────
    loser_idx, winner_idx, _ = sort_deciles(momentum_std[t, :], live[t, :])
    valid_mask = np.isfinite(momentum_std[t, :]) & (live[t, :] == 1)

    # ── Rolling 52-week window ───────────────────────────────────────
    w_start = t - CORR_WINDOW + 1
    w_end   = t + 1
    window_dates = [pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
                    for i in range(w_start, w_end)]
    ff_window = ff_factors[w_start:w_end, :]

    # ── Compute residuals ────────────────────────────────────────────
    ret_losers  = returns_clean[w_start:w_end, :][:, loser_idx]
    ret_winners = returns_clean[w_start:w_end, :][:, winner_idx]
    resid_losers  = compute_ff3_residuals(ret_losers, ff_window)
    resid_winners = compute_ff3_residuals(ret_winners, ff_window)

    loser_names  = [str(names[i]) for i in loser_idx]
    winner_names = [str(names[i]) for i in winner_idx]

    # ── Build documentation sheet ────────────────────────────────────
    doc_rows = [
        ["FF3 REGRESSION RESIDUALS — STEP 4 SNAPSHOT", ""],
        ["", ""],
        ["Snapshot week index", str(t)],
        ["Snapshot date", date_str],
        ["Rolling window", f"{window_dates[0]} to {window_dates[-1]} "
                           f"({CORR_WINDOW} weeks)"],
        ["", ""],
        ["REGRESSION MODEL (per stock, over the 52-week window):", ""],
        ["  r_i,w = alpha_i + beta_MKT * MktRF_w + beta_SMB * SMB_w "
         "+ beta_HML * HML_w + epsilon_i,w", ""],
        ["  Residual = epsilon_i,w = r_i,w − fitted values", ""],
        ["", ""],
        ["Loser decile", f"Bottom {DECILE_PCT_LO}% of momentum scores"],
        ["Winner decile", f"Top {100 - DECILE_PCT_HI}% of momentum scores"],
        ["N losers", str(len(loser_idx))],
        ["N winners", str(len(winner_idx))],
        ["N valid stocks at this date", str(int(np.sum(valid_mask)))],
        ["", ""],
        ["SHEET: Loser_Residuals", ""],
        ["  Rows", f"{CORR_WINDOW} weeks ({window_dates[0]} to "
                   f"{window_dates[-1]})"],
        ["  Columns", "One per loser-decile stock (labelled by stock name)"],
        ["  Values", "FF3 regression residual (decimal weekly return). "
                     "NaN means that stock did not have a valid return in "
                     "that week or failed the minimum-observations filter."],
        ["", ""],
        ["SHEET: Winner_Residuals", ""],
        ["  Rows", f"{CORR_WINDOW} weeks"],
        ["  Columns", "One per winner-decile stock"],
        ["  Values", "Same as Loser_Residuals"],
        ["", ""],
        ["PURPOSE", ""],
        ["", "These residuals are the 'abnormal returns' stripped of "
             "Fama-French three-factor exposure. They are used to "
             "compute pairwise correlations within each decile, which "
             "are then averaged to produce the comomentum measure "
             "(Lou & Polk, 2021)."],
    ]
    df_doc = pd.DataFrame(doc_rows, columns=["Item", "Details"])

    # ── Build data sheets ────────────────────────────────────────────
    df_losers = pd.DataFrame(resid_losers, columns=loser_names,
                              index=window_dates)
    df_losers.index.name = 'date'

    df_winners = pd.DataFrame(resid_winners, columns=winner_names,
                               index=window_dates)
    df_winners.index.name = 'date'

    # ── Write Excel ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path)
                else '.', exist_ok=True)
    base, _ = os.path.splitext(save_path)
    xlsx_path = base + '.xlsx'

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_doc.to_excel(writer, sheet_name='Documentation', index=False)
        df_losers.to_excel(writer, sheet_name='Loser_Residuals')
        df_winners.to_excel(writer, sheet_name='Winner_Residuals')

    print(f"  Saved FF3 residuals: {xlsx_path}")
    return resid_losers, resid_winners, loser_names, winner_names


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

    resid_l, resid_w, names_l, names_w = save_ff3_residuals(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'],
        data['dates'], data['names'],
        snapshot_week=None,  # last week
        save_path='output_data/ff3_residuals.xlsx'
    )
    print(f"\n  Loser residuals  : {resid_l.shape[0]} weeks x "
          f"{resid_l.shape[1]} stocks")
    print(f"  Winner residuals : {resid_w.shape[0]} weeks x "
          f"{resid_w.shape[1]} stocks")
