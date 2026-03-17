# summary_statistics_latex.py
# =====================================================================
# Generates Table I (Summary Statistics) as a standalone LaTeX file
# that can be \input{} into the main report.
# =====================================================================

import numpy as np
import pandas as pd


def _annual_means(series, dates):
    """Resample a weekly series to non-overlapping 12-month means."""
    df = pd.DataFrame({'val': series}, index=pd.DatetimeIndex(dates))
    return df['val'].resample('YE').mean().dropna()


def _tex_varname(name):
    """Convert e.g. 'CoMOM_L' -> '$\\text{CoMOM}_L$', 'MRET' -> 'MRET'."""
    if '_' in name:
        parts = name.split('_', 1)
        return rf'$\text{{{parts[0]}}}_{{{parts[1]}}}$'
    return name


def generate_summary_table_latex(comomentum, comom_winner, comom_loser,
                                  mret, mvol, dates,
                                  save_path='latex_report/table1.tex'):
    """
    Produce Table I (3 panels) as a .tex file with data from the pipeline.
    """
    dates = pd.DatetimeIndex(dates)
    variables = {
        'CoMOM':   comomentum,
        'CoMOM_L': comom_loser,
        'CoMOM_W': comom_winner,
        'MRET':    mret,
        'MVOL':    mvol,
    }
    var_names = list(variables.keys())
    df_all = pd.DataFrame(variables, index=dates)

    # ── Panel A ──────────────────────────────────────────────────────
    pa_rows = []
    for name in var_names:
        s = df_all[name].dropna()
        label = _tex_varname(name)
        pa_rows.append(
            f"    {label} & {len(s):,} & {s.mean():.3f} & {s.std():.3f} "
            f"& {s.min():.3f} & {s.max():.3f} \\\\"
        )

    # ── Panel B ──────────────────────────────────────────────────────
    corr = df_all.corr()
    n = len(var_names)
    pb_rows = []
    for i in range(n):
        label = _tex_varname(var_names[i])
        cells = [label]
        for j in range(n):
            if j > i:
                cells.append('')
            else:
                cells.append(f"{corr.iloc[i, j]:.3f}")
        pb_rows.append('    ' + ' & '.join(cells) + ' \\\\')

    # ── Panel C ──────────────────────────────────────────────────────
    comom_vars = ['CoMOM', 'CoMOM_L', 'CoMOM_W']
    annual = {}
    for name in comom_vars:
        annual[name] = _annual_means(df_all[name].values, dates)
    annual_df = pd.DataFrame(annual).dropna()
    n_years = len(annual_df)

    ac_df = pd.DataFrame()
    for name in comom_vars:
        vals = annual_df[name].values
        ac_df[f'{name}_t'] = vals[:-1]
        ac_df[f'{name}_t1'] = vals[1:]
    ac_corr = ac_df.corr()

    col_keys = [f'{n}_t' for n in comom_vars] + [f'{n}_t1' for n in comom_vars]
    col_labels_tex = []
    for k in col_keys:
        base = k.replace('_t1', '').replace('_t', '')
        time_sub = 't+1' if k.endswith('_t1') else 't'
        # e.g. CoMOM_L -> \text{CoMOM}_{L,t}
        if '_' in base:
            parts = base.split('_', 1)
            math = rf'$\text{{{parts[0]}}}_{{{parts[1]},{time_sub}}}$'
        else:
            math = rf'$\text{{{base}}}_{{{time_sub}}}$'
        col_labels_tex.append(math)

    pc_rows = []
    for i, rkey in enumerate(col_keys):
        cells = [col_labels_tex[i]]
        for j in range(len(col_keys)):
            if j > i:
                cells.append('')
            else:
                cells.append(f"{ac_corr.iloc[i, j]:.3f}")
        pc_rows.append('    ' + ' & '.join(cells) + ' \\\\')

    # ── Build .tex ───────────────────────────────────────────────────
    pb_header_labels = [''] + [_tex_varname(n) for n in var_names]
    pc_header_labels = [''] + col_labels_tex

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Summary statistics, correlations, and autocorrelations of comomentum and market variables}')
    tex.append(r'\label{tab:summary_statistics}')
    tex.append(r'\small')
    tex.append('')

    # Panel A
    tex.append(r'% --- Panel A ---')
    tex.append(r'\textbf{Panel A: Summary Statistics}\\[4pt]')
    tex.append(r'\begin{tabular}{l r r r r r}')
    tex.append(r'\toprule')
    tex.append(r'    Variable & N & Mean & Std.\ Dev. & Min & Max \\')
    tex.append(r'\midrule')
    tex.extend(pa_rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{12pt}')
    tex.append('')

    # Panel B
    tex.append(r'% --- Panel B ---')
    tex.append(r'\textbf{Panel B: Correlation}\\[4pt]')
    tex.append(r'\begin{tabular}{l' + ' r' * n + '}')
    tex.append(r'\toprule')
    tex.append('    ' + ' & '.join(pb_header_labels) + ' \\\\')
    tex.append(r'\midrule')
    tex.extend(pb_rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{12pt}')
    tex.append('')

    # Panel C
    tex.append(r'% --- Panel C ---')
    tex.append(rf'\textbf{{Panel C: Autocorrelation}} ({n_years} years)\\[4pt]')
    tex.append(r'\begin{tabular}{l' + ' r' * len(col_keys) + '}')
    tex.append(r'\toprule')
    tex.append('    ' + ' & '.join(pc_header_labels) + ' \\\\')
    tex.append(r'\midrule')
    tex.extend(pc_rows)
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append('')

    tex.append(r'\end{table}')

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
