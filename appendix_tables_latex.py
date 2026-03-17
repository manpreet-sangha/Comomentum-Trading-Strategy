# appendix_tables_latex.py
# =====================================================================
# Generates Appendix LaTeX tables for all remaining calculation windows:
#   - Comomentum (FF3 residuals + decile pairwise correlations)
#   - Market variables (MRET trailing 104w, MVOL trailing 24 months)
#   - Fama-MacBeth cross-sectional regressions
#   - Adjusted momentum (expanding-window percentile rank scaling)
# =====================================================================

import os
import numpy as np
import pandas as pd
from config import (CORR_WINDOW, DECILE_PCT_LO, DECILE_PCT_HI,
                    MIN_PAST_VALUES, WEEKS_PER_YEAR)


# ── Constants from market_variables.py (mirror them here) ────────────
MRET_WINDOW = 2 * WEEKS_PER_YEAR   # 104 weeks
MVOL_MONTHS = 24                    # 24 months


def _fmt(d):
    return d.strftime('%Y-%m-%d')


def _head_tail_rows(all_indices, dates, row_fn, n_head=5, n_tail=3):
    """Build LaTeX rows for first n_head and last n_tail indices with ellipsis."""
    head = all_indices[:n_head]
    tail = all_indices[-n_tail:]
    if head[-1] >= tail[0]:
        selected = sorted(set(head + tail))
        ellipsis = False
    else:
        selected = head + tail
        ellipsis = True

    rows = []
    for idx_pos, t in enumerate(selected):
        rows.append(row_fn(t, dates))
        if ellipsis and idx_pos == len(head) - 1:
            ncols = rows[-1].count('&') + 1
            rows.append(f'    \\multicolumn{{{ncols}}}{{c}}{{$\\vdots$}} \\\\')
    return rows


# =====================================================================
# 1. Comomentum calculation windows
# =====================================================================
def generate_comomentum_windows_latex(comomentum, dates,
                                       save_path='latex_report/table_comom_calc.tex'):
    dates = pd.DatetimeIndex(dates)
    T = len(dates)
    first_t = CORR_WINDOW - 1  # index 51

    valid_indices = [t for t in range(first_t, T) if np.isfinite(comomentum[t])]
    n_valid = len(valid_indices)

    def row_fn(t, dates):
        w = t + 1
        resid_start = dates[t - CORR_WINDOW + 1]
        resid_end = dates[t]
        return (f'    {w} & {_fmt(resid_start)} & {_fmt(resid_end)} '
                f'& {_fmt(dates[t])} \\\\')

    rows = _head_tail_rows(valid_indices, dates, row_fn)

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Comomentum calculation windows. '
               rf'FF3 residual window = {CORR_WINDOW} weeks. '
               rf'Decile thresholds: bottom {DECILE_PCT_LO}\% (losers), '
               rf'top {100 - DECILE_PCT_HI}\% (winners). '
               rf'Valid weeks: {n_valid:,}.}}')
    tex.append(r'\label{tab:comom_calc}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{r l l l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & FF3 Window Start & FF3 Window End & CoMOM Date \\')
    tex.append(r'\midrule')
    tex.extend(rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(rf'\textit{{Note.}} At each week $t$, stocks in the extreme loser '
               rf'(bottom {DECILE_PCT_LO}\%) and winner (top {100 - DECILE_PCT_HI}\%) '
               rf'momentum deciles have their last {CORR_WINDOW} weeks of returns '
               r'regressed on the Fama--French three factors. '
               r'Comomentum is the average pairwise correlation of the resulting '
               r'residuals within each decile, averaged across both legs.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")


# =====================================================================
# 2. Market variables calculation windows
# =====================================================================
def generate_market_variables_windows_latex(mret, mvol, dates,
                                             save_path='latex_report/table_mktvar_calc.tex'):
    dates = pd.DatetimeIndex(dates)
    T = len(dates)

    # MRET valid indices
    mret_valid = [t for t in range(T) if np.isfinite(mret[t])]
    n_mret = len(mret_valid)
    # MVOL valid indices
    mvol_valid = [t for t in range(T) if np.isfinite(mvol[t])]
    n_mvol = len(mvol_valid)

    # ── MRET rows ────────────────────────────────────────────────────
    def mret_row(t, dates):
        w = t + 1
        start = dates[t - MRET_WINDOW + 1]
        end = dates[t]
        return f'    {w} & {_fmt(start)} & {_fmt(end)} & {_fmt(dates[t])} \\\\'

    mret_rows = _head_tail_rows(mret_valid, dates, mret_row)

    # ── MVOL rows ────────────────────────────────────────────────────
    def mvol_row(t, dates):
        w = t + 1
        return f'    {w} & {_fmt(dates[t])} \\\\'

    mvol_rows = _head_tail_rows(mvol_valid, dates, mvol_row)

    # First and last MVOL dates for caption
    mvol_first = _fmt(dates[mvol_valid[0]]) if mvol_valid else 'N/A'
    mvol_last = _fmt(dates[mvol_valid[-1]]) if mvol_valid else 'N/A'

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Market variable calculation windows.}')
    tex.append(r'\label{tab:mktvar_calc}')
    tex.append(r'\small')
    tex.append('')

    # MRET sub-table
    tex.append(rf'\textbf{{Panel A: MRET}} --- trailing {MRET_WINDOW}-week '
               rf'compounded market return ({n_mret:,} valid weeks)\\[4pt]')
    tex.append(r'\begin{tabular}{r l l l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & Window Start & Window End & MRET Date \\')
    tex.append(r'\midrule')
    tex.extend(mret_rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{12pt}')
    tex.append('')

    # MVOL sub-table
    tex.append(rf'\textbf{{Panel B: MVOL}} --- trailing {MVOL_MONTHS}-month '
               rf'rolling standard deviation of monthly market returns '
               rf'({n_mvol:,} valid weeks)\\[4pt]')
    tex.append(r'\begin{tabular}{r l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & MVOL Date \\')
    tex.append(r'\midrule')
    tex.extend(mvol_rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(rf'\textit{{Note.}} MRET at week $t$ is the compounded gross '
               rf'market return (Mkt-RF + RF) over the preceding {MRET_WINDOW} weeks. '
               rf'MVOL at week $t$ is computed by first compounding weekly market returns '
               rf'to monthly frequency, then taking a {MVOL_MONTHS}-month rolling '
               r'standard deviation. Each week inherits the value of its calendar month.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")


# =====================================================================
# 3. Fama-MacBeth regression windows
# =====================================================================
def generate_famamacbeth_windows_latex(gamma, dates,
                                        save_path='latex_report/table_fm_calc.tex'):
    dates = pd.DatetimeIndex(dates)
    T = len(dates)

    valid_indices = [t for t in range(T) if np.isfinite(gamma[t])]
    n_valid = len(valid_indices)

    def row_fn(t, dates):
        w = t + 1
        factor_date = _fmt(dates[t - 1]) if t >= 1 else 'N/A'
        return (f'    {w} & {factor_date} & {_fmt(dates[t])} '
                f'& {_fmt(dates[t])} \\\\')

    rows = _head_tail_rows(valid_indices, dates, row_fn)

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Fama--MacBeth cross-sectional regression windows. '
               rf'Valid weeks: {n_valid:,}.}}')
    tex.append(r'\label{tab:fm_calc}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{r l l l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & Factor Date ($t{-}1$) & Return Date ($t$) '
               r'& $\gamma_t$ Date \\')
    tex.append(r'\midrule')
    tex.extend(rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(r'\textit{Note.} At each week $t$, a cross-sectional OLS regression '
               r'$r_{i,t} = \alpha_t + \gamma_t \, \text{MOM}^{*}_{i,t-1} + \varepsilon_{i,t}$ '
               r'is run across all eligible stocks. The momentum exposure is lagged by one week. '
               r'A minimum of 3 valid stocks is required per cross-section.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")


# =====================================================================
# 4. Adjusted momentum calculation windows
# =====================================================================
def generate_adjusted_momentum_windows_latex(comomentum, gamma_adj, dates,
                                              save_path='latex_report/table_adjmom_calc.tex'):
    dates = pd.DatetimeIndex(dates)
    T = len(dates)

    valid_indices = [t for t in range(T) if np.isfinite(gamma_adj[t])]
    n_valid = len(valid_indices)

    # Find the first week where the percentile rank is actually used
    # (i.e. not just fallback scaling = 1.0)
    n_past_finite = 0
    first_scaled = None
    for t in range(T):
        if np.isfinite(comomentum[t]):
            n_past_finite += 1
        if n_past_finite >= MIN_PAST_VALUES + 1 and first_scaled is None:
            first_scaled = t

    def row_fn(t, dates):
        w = t + 1
        past_count = int(np.sum(np.isfinite(comomentum[:t])))
        return (f'    {w} & {_fmt(dates[t])} & {past_count} '
                f'& {_fmt(dates[t])} \\\\')

    rows = _head_tail_rows(valid_indices, dates, row_fn)

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Adjusted momentum calculation windows. '
               rf'Minimum past CoMOM values for percentile rank: {MIN_PAST_VALUES}. '
               rf'Valid weeks: {n_valid:,}.}}')
    tex.append(r'\label{tab:adjmom_calc}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{r l r l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & Date & Past CoMOM obs. & $\gamma^{\text{adj}}_t$ Date \\')
    tex.append(r'\midrule')
    tex.extend(rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(r'\textit{Note.} At each week $t$, the percentile rank of '
               r'$\text{CoMOM}_t$ is computed relative to all past values '
               r'$\text{CoMOM}_{1}, \ldots, \text{CoMOM}_{t-1}$ (expanding window, '
               rf'minimum {MIN_PAST_VALUES} observations). '
               r'The scaling factor is $s_t = 2 - \text{PctRank}_{t-1}$, '
               r'mapping to the range $[1, 2]$. '
               r'Adjusted factor return: $\gamma^{\text{adj}}_t = s_t \cdot \gamma_t$.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
