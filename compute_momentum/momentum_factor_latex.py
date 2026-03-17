# momentum_factor_latex.py
# =====================================================================
# Generates an Appendix table showing the momentum factor calculation
# windows for the first and last few computation dates.
# =====================================================================

import os
import numpy as np
import pandas as pd
from config import LOOKBACK, SKIP, TOTAL


def generate_momentum_factor_table_latex(dates,
                                          save_path='latex_report/table_momentum_calc.tex',
                                          n_head=5, n_tail=3):
    """
    Produce a LaTeX table illustrating the momentum calculation windows.

    For a selection of computation dates t, shows:
      - Lookback start  : dates[t - TOTAL + 1]  (= t - 51)
      - Lookback end    : dates[t - SKIP]        (= t - 4)
      - Skip start      : dates[t - SKIP + 1]    (= t - 3)
      - Skip end / calc : dates[t]
    """
    dates = pd.DatetimeIndex(dates)
    T = len(dates)
    first_t = TOTAL - 1  # index 51

    # Select row indices to display
    all_t = list(range(first_t, T))
    head = all_t[:n_head]
    tail = all_t[-n_tail:]
    if head[-1] >= tail[0]:
        selected = sorted(set(head + tail))
        insert_ellipsis = False
    else:
        selected = head + tail
        insert_ellipsis = True

    def fmt(d):
        return d.strftime('%Y-%m-%d')

    # Build rows
    rows = []
    for idx, t in enumerate(selected):
        lb_start = dates[t - TOTAL + 1]
        lb_end   = dates[t - SKIP]
        sk_start = dates[t - SKIP + 1]
        sk_end   = dates[t]
        calc     = dates[t]
        week_num = t + 1  # 1-based
        rows.append(
            f'    {week_num} & {fmt(lb_start)} & {fmt(lb_end)} '
            f'& {SKIP} ({fmt(sk_start)} -- {fmt(sk_end)}) & {fmt(calc)} \\\\'
        )
        if insert_ellipsis and idx == len(head) - 1:
            rows.append(
                r'    \multicolumn{5}{c}{$\vdots$} \\'
            )

    n_scored = T - TOTAL + 1

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Momentum factor calculation windows. '
               rf'Lookback = {LOOKBACK} weeks, skip = {SKIP} weeks, '
               rf'total history = {TOTAL} weeks. '
               rf'Scored weeks: {n_scored:,} (weeks {first_t + 1}--{T}).}}')
    tex.append(r'\label{tab:momentum_calc}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{r l l l l}')
    tex.append(r'\toprule')
    tex.append(r'    Week & Lookback Start & Lookback End '
               r'& Skipped Obs & Calculation Date \\')
    tex.append(r'\midrule')
    tex.extend(rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(rf'\textit{{Note.}} The lookback window spans {LOOKBACK} weeks '
               rf'of compounded returns (indices $t-{TOTAL - 1}$ to $t-{SKIP}$). '
               rf'The {SKIP} most recent weeks (indices $t-{SKIP - 1}$ to $t$) are '
               r'skipped to avoid short-term reversal contamination.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
