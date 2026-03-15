# save_pairwise_correlations.py
# =====================================================================
# Save Pairwise Correlation Matrices — Step 4 Diagnostic Output
# =====================================================================
#
# After computing FF3 residuals for the loser and winner deciles,
# the comomentum procedure computes pairwise abnormal correlations
# (Lou & Polk, 2021):
#     Corr(resid_i, resid_j)  for all unique pairs (i, j) in each decile
#
# This module saves the full K×K correlation matrices for both
# deciles to an Excel workbook so the computations can be inspected.
#
# WHAT IS SAVED:
#   An .xlsx file with five sheets:
#     1. 'Documentation'       — explains methodology + summary stats
#     2. 'Loser_CorrMatrix'    — K_L × K_L pairwise correlation matrix
#     3. 'Winner_CorrMatrix'   — K_W × K_W pairwise correlation matrix
#     4. 'Loser_Pairwise'      — long-format table of all unique pairs
#     5. 'Winner_Pairwise'     — long-format table of all unique pairs
#
#   Like the residual file, this is a SNAPSHOT for one chosen week.
#
# Standalone:  python save_pairwise_correlations.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import os

from config import (CORR_WINDOW, DECILE_PCT_LO, DECILE_PCT_HI,
                     MIN_RESID_OBS, MIN_STOCKS)
from comomentum.ff3_residuals import compute_ff3_residuals
from comomentum.decile_sort import sort_deciles
from comomentum.pairwise_correlations import build_corr_outputs


def save_pairwise_correlations(returns_clean, momentum_std, live,
                                ff_factors, dates, names,
                                snapshot_week=None,
                                save_path='output_data/pairwise_correlations.xlsx'):
    """
    Compute and save pairwise correlation matrices for loser and
    winner deciles at one snapshot week.

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
        Writes an Excel file.  Returns a dict with keys:
            corr_losers, corr_winners, pairs_losers, pairs_winners,
            avg_corr_losers, avg_corr_winners, comomentum
    """
    T, N = returns_clean.shape

    if snapshot_week is None:
        snapshot_week = T - 1

    t = snapshot_week
    date_str = pd.Timestamp(dates[t]).strftime('%Y-%m-%d')

    # ── Decile sort ──────────────────────────────────────────────────
    loser_idx, winner_idx, _ = sort_deciles(momentum_std[t, :], live[t, :])

    loser_names  = [str(names[i]) for i in loser_idx]
    winner_names = [str(names[i]) for i in winner_idx]

    # ── Rolling window ───────────────────────────────────────────────
    w_start = t - CORR_WINDOW + 1
    w_end   = t + 1
    window_start_str = pd.Timestamp(dates[w_start]).strftime('%Y-%m-%d')
    window_end_str   = date_str
    ff_window = ff_factors[w_start:w_end, :]

    # ── Compute residuals ────────────────────────────────────────────
    ret_losers  = returns_clean[w_start:w_end, :][:, loser_idx]
    ret_winners = returns_clean[w_start:w_end, :][:, winner_idx]
    resid_losers  = compute_ff3_residuals(ret_losers, ff_window)
    resid_winners = compute_ff3_residuals(ret_winners, ff_window)

    # ── Pairwise correlations ────────────────────────────────────
    corr_l, pairs_l, avg_l, n_pairs_l, k_elig_l = build_corr_outputs(
        resid_losers, loser_names)
    corr_w, pairs_w, avg_w, n_pairs_w, k_elig_w = build_corr_outputs(
        resid_winners, winner_names)

    comom = 0.5 * (avg_w + avg_l) if (np.isfinite(avg_w) and
                                       np.isfinite(avg_l)) else np.nan

    # ── Documentation sheet ──────────────────────────────────────────
    doc_rows = [
        ["PAIRWISE ABNORMAL CORRELATION MATRICES — STEP 4 SNAPSHOT", ""],
        ["", ""],
        ["Snapshot week index", str(t)],
        ["Snapshot date", date_str],
        ["Rolling window",
         f"{window_start_str} to {window_end_str} ({CORR_WINDOW} weeks)"],
        ["", ""],
        ["METHODOLOGY (Lou & Polk, 2021):", ""],
        ["  1. Sort all live stocks with valid momentum into deciles.", ""],
        ["  2. For each loser/winner stock, regress its 52-week returns "
         "on FF3 factors (Mkt-RF, SMB, HML) → collect residuals.", ""],
        ["  3. Compute Corr(resid_i, resid_j) for every unique pair "
         "(i, j) within each decile.", ""],
        ["  4. CoMOM_decile = mean of all K*(K-1)/2 pairwise correlations.", ""],
        ["  5. CoMOM = 0.5 * (CoMOM_winners + CoMOM_losers).", ""],
        ["", ""],
        ["LOSER DECILE", ""],
        ["  Criterion", f"Bottom {DECILE_PCT_LO}% of momentum scores"],
        ["  N stocks in decile", str(len(loser_idx))],
        ["  N eligible (≥{} valid weeks)".format(CORR_WINDOW), str(k_elig_l)],
        ["  N unique pairs", str(n_pairs_l)],
        ["  Mean pairwise correlation (= CoMOM_L)", f"{avg_l:.6f}"
         if np.isfinite(avg_l) else "NaN"],
        ["", ""],
        ["WINNER DECILE", ""],
        ["  Criterion", f"Top {100 - DECILE_PCT_HI}% of momentum scores"],
        ["  N stocks in decile", str(len(winner_idx))],
        ["  N eligible (≥{} valid weeks)".format(CORR_WINDOW), str(k_elig_w)],
        ["  N unique pairs", str(n_pairs_w)],
        ["  Mean pairwise correlation (= CoMOM_W)", f"{avg_w:.6f}"
         if np.isfinite(avg_w) else "NaN"],
        ["", ""],
        ["COMOMENTUM", f"{comom:.6f}" if np.isfinite(comom) else "NaN"],
        ["  = 0.5 * (CoMOM_W + CoMOM_L)", ""],
        ["", ""],
        ["SHEET: Loser_CorrMatrix", ""],
        ["  Shape", f"{k_elig_l} x {k_elig_l}"],
        ["  Values", "Pearson correlation of FF3 residuals between each "
                     "pair of loser-decile stocks. Diagonal = 1.0."],
        ["", ""],
        ["SHEET: Winner_CorrMatrix", ""],
        ["  Shape", f"{k_elig_w} x {k_elig_w}"],
        ["  Values", "Same, for winner-decile stocks."],
        ["", ""],
        ["SHEET: Loser_Pairwise", ""],
        ["  Columns", "stock_i, stock_j, correlation"],
        ["  Rows", f"All {n_pairs_l} unique pairs (i < j) in the "
                   "loser decile."],
        ["", ""],
        ["SHEET: Winner_Pairwise", ""],
        ["  Columns", "stock_i, stock_j, correlation"],
        ["  Rows", f"All {n_pairs_w} unique pairs in the winner decile."],
    ]
    df_doc = pd.DataFrame(doc_rows, columns=["Item", "Details"])

    # ── Write Excel ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path)
                else '.', exist_ok=True)
    base, _ = os.path.splitext(save_path)
    xlsx_path = base + '.xlsx'

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_doc.to_excel(writer, sheet_name='Documentation', index=False)
        if corr_l is not None:
            corr_l.to_excel(writer, sheet_name='Loser_CorrMatrix')
            pairs_l.to_excel(writer, sheet_name='Loser_Pairwise',
                             index=False)
        if corr_w is not None:
            corr_w.to_excel(writer, sheet_name='Winner_CorrMatrix')
            pairs_w.to_excel(writer, sheet_name='Winner_Pairwise',
                             index=False)

    print(f"  Saved pairwise correlations: {xlsx_path}")

    return {
        'corr_losers':      corr_l,
        'corr_winners':     corr_w,
        'pairs_losers':     pairs_l,
        'pairs_winners':    pairs_w,
        'avg_corr_losers':  avg_l,
        'avg_corr_winners': avg_w,
        'comomentum':       comom,
    }


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

    result = save_pairwise_correlations(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'],
        data['dates'], data['names'],
        snapshot_week=None,  # last week
        save_path='output_data/pairwise_correlations.xlsx'
    )

    print(f"\n  CoMOM_L = {result['avg_corr_losers']:.6f}")
    print(f"  CoMOM_W = {result['avg_corr_winners']:.6f}")
    print(f"  CoMOM   = {result['comomentum']:.6f}")
    if result['pairs_losers'] is not None:
        print(f"  Loser pairs  : {len(result['pairs_losers'])}")
    if result['pairs_winners'] is not None:
        print(f"  Winner pairs : {len(result['pairs_winners'])}")
