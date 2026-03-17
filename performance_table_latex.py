# performance_table_latex.py
# =====================================================================
# Generates a LaTeX comparison table for Standard vs. Adjusted Momentum
# factor return statistics.
# =====================================================================

import os


def generate_performance_table_latex(stats_std, stats_adj,
                                      save_path='latex_report/table_performance.tex'):
    """
    Generates a LaTeX table comparing standard and adjusted momentum
    factor return statistics.

    INPUTS:
        stats_std : dict - output of compute_stats() for standard momentum
        stats_adj : dict - output of compute_stats() for adjusted momentum
        save_path : str  - path to write the .tex file
    """

    rows = [
        ('Valid weeks',            'n',        'd',  0),
        ('Ann.\\ mean return (\\%)', 'mean_ann', 'pct', 2),
        ('Ann.\\ std.\\ dev.\\ (\\%)', 'std_ann', 'pct', 2),
        ('Ann.\\ Sharpe ratio',    'sharpe',   'f',  3),
        ('$t$-statistic',          'tstat',    'f',  3),
        ('Skewness',               'skew',     'f',  3),
        ('Excess kurtosis',        'kurt',     'f',  3),
        ('Max drawdown (\\%)',     'max_dd',   'pct', 2),
    ]

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Standard vs.\ Adjusted Momentum Factor Returns --- '
               r'annualised performance comparison over the full sample period.}')
    tex.append(r'\label{tab:performance}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l c c}')
    tex.append(r'\toprule')
    tex.append(r'    & Standard Momentum & Adjusted Momentum \\')
    tex.append(r'\midrule')

    for label, key, fmt, dec in rows:
        v_std = stats_std[key]
        v_adj = stats_adj[key]
        if fmt == 'd':
            c_std = f'{int(v_std)}'
            c_adj = f'{int(v_adj)}'
        elif fmt == 'pct':
            c_std = f'{v_std * 100:.{dec}f}'
            c_adj = f'{v_adj * 100:.{dec}f}'
        else:
            c_std = f'{v_std:.{dec}f}'
            c_adj = f'{v_adj:.{dec}f}'
        tex.append(f'    {label} & {c_std} & {c_adj} \\\\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
