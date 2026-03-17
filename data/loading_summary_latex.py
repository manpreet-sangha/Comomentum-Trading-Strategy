# loading_summary_latex.py
# =====================================================================
# Generates the Data Loading Summary as a standalone LaTeX file
# that can be \input{} into the main report.
# =====================================================================

import os
import numpy as np


def generate_loading_summary_latex(data, save_path='latex_report/table_loading.tex'):
    """
    Produce the data loading summary as a .tex file with data from the pipeline.
    """
    returns    = data['returns']
    live       = data['live']
    dates      = data['dates']
    ff_factors = data['ff_factors']
    T          = data['T']
    N          = data['N']

    total_cells        = T * N
    listed_cells       = int(np.sum(live == 1))
    not_listed         = int(np.sum(live == 0))
    finite_and_listed  = int(np.sum(np.isfinite(returns) & (live == 1)))
    nan_in_listed      = listed_cells - finite_and_listed
    forced_nan         = int(np.sum((live == 0) & np.isfinite(returns)))
    notlisted_already_nan = int(np.sum((live == 0) & ~np.isfinite(returns)))

    date_start = dates[0].strftime('%Y-%m-%d')
    date_end   = dates[-1].strftime('%Y-%m-%d')
    date_range = f'{date_start} to {date_end}'
    ff_cols    = ff_factors.shape[1] + 1  # 3 factors + RF

    def pct(part, whole):
        return f'({part / whole * 100:.1f}\\%)' if whole else ''

    tex = []
    tex.append(r'\begin{table}[ht]')
    tex.append(r'\centering')
    tex.append(r'\caption{Data loading summary}')
    tex.append(r'\label{tab:loading_summary}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l l}')
    tex.append(r'\toprule')
    tex.append(r'\textbf{Metric} & \textbf{Value} \\')
    tex.append(r'\midrule')

    # Section 1 — Dataset dimensions
    tex.append(r'\multicolumn{2}{l}{\textit{Dataset Dimensions}} \\[2pt]')
    tex.append(f'Period & {date_start} to {date_end} \\\\')
    tex.append(f'Weeks ($T$) & {T:,} \\\\')
    tex.append(f'Stocks ($N$) & {N:,} \\\\')
    tex.append(f'Total cells ($T \\times N$) & {total_cells:,} \\\\')
    tex.append(r'\addlinespace')

    # Section 2 — Input files
    tex.append(r'\multicolumn{2}{l}{\textit{Input Files}} \\[2pt]')
    tex.append(f'US\\_Returns.csv & ${T:,} \\times {N:,}$ \\,|\\, {date_range} \\\\')
    tex.append(f'US\\_live.csv & ${T:,} \\times {N:,}$ \\,|\\, {date_range} \\\\')
    tex.append(f'US\\_Dates.xlsx & {T:,} dates \\,|\\, {date_range} \\\\')
    tex.append(f'US\\_Names.xlsx & {N:,} stock names \\\\')
    tex.append(f'FamaFrench.csv & ${T:,} \\times {ff_cols}$ \\,|\\, {date_range} \\\\')
    tex.append(r'\addlinespace')

    # Section 3 — Cleaning breakdown
    tex.append(r'\multicolumn{2}{l}{\textit{Cleaning Breakdown}} \\[2pt]')
    tex.append(f'Listed cells (live$=1$) & {listed_cells:,} {pct(listed_cells, total_cells)} \\\\')
    tex.append(f'\\quad with valid return & {finite_and_listed:,} \\\\')
    tex.append(f'\\quad NaN (data gaps) & {nan_in_listed:,} \\\\')
    tex.append(f'Not-listed cells (live$=0$) & {not_listed:,} {pct(not_listed, total_cells)} \\\\')
    tex.append(f'\\quad already NaN in raw file & {notlisted_already_nan:,} \\\\')
    tex.append(f'\\quad numeric value $\\to$ NaN & {forced_nan:,} \\\\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
