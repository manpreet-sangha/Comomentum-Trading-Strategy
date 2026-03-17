# determinants_table_latex.py
# =====================================================================
# Generates Table II (Determinants of Comomentum) as a standalone
# LaTeX file that can be \input{} into the main report.
# =====================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _detrend(series):
    valid = series.dropna()
    if len(valid) < 3:
        return series
    trend = np.arange(len(valid), dtype=float)
    X = sm.add_constant(trend)
    model = sm.OLS(valid.values, X).fit()
    return pd.Series(model.resid, index=valid.index, name=series.name)


def _stars(pval):
    if pval < 0.01:
        return '^{***}'
    elif pval < 0.05:
        return '^{**}'
    elif pval < 0.10:
        return '^{*}'
    return ''


def _run_ols_nw(y, X, maxlags=12):
    data = pd.concat([y.rename('_y_'), X], axis=1).dropna()
    if len(data) < X.shape[1] + 2:
        return None
    y_clean = data['_y_']
    X_clean = sm.add_constant(data.drop(columns='_y_'))
    model = sm.OLS(y_clean, X_clean).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags}
    )
    regressor_names = [c for c in X_clean.columns if c != 'const']
    return {
        'coefs': {n: model.params[n] for n in regressor_names},
        'se':    {n: model.bse[n] for n in regressor_names},
        'tvals': {n: model.tvalues[n] for n in regressor_names},
        'pvals': {n: model.pvalues[n] for n in regressor_names},
        'adj_r2': model.rsquared_adj,
        'nobs': int(model.nobs),
    }


def generate_determinants_table_latex(comomentum, gamma_std, mret, mvol, dates,
                                       save_path='latex_report/table2.tex'):
    dates = pd.DatetimeIndex(dates)

    df_weekly = pd.DataFrame({
        'CoMOM': comomentum, 'MOM': gamma_std,
        'MRET': mret, 'MVOL': mvol,
    }, index=dates)

    annual = pd.DataFrame(index=df_weekly.resample('YE').mean().index)
    annual['CoMOM'] = df_weekly['CoMOM'].resample('YE').mean()
    annual['MOM'] = df_weekly['MOM'].resample('YE').apply(
        lambda x: np.prod(1.0 + x.dropna().values) - 1.0
        if len(x.dropna()) > 0 else np.nan
    )
    annual['MRET'] = df_weekly['MRET'].resample('YE').mean()
    annual['MVOL'] = df_weekly['MVOL'].resample('YE').mean()
    annual = annual.dropna(subset=['CoMOM'])

    detrended = pd.DataFrame(index=annual.index)
    for col in annual.columns:
        detrended[col] = _detrend(annual[col])

    dep_var = detrended['CoMOM']
    regressors = pd.DataFrame(index=detrended.index)
    regressors['MOM_lag'] = detrended['MOM'].shift(1)
    regressors['MRET_lag'] = detrended['MRET'].shift(1)
    regressors['MVOL_lag'] = detrended['MVOL'].shift(1)

    specs = [
        ('[1]', ['MOM_lag']),
        ('[2]', ['MRET_lag', 'MVOL_lag']),
        ('[3]', ['MOM_lag', 'MRET_lag', 'MVOL_lag']),
    ]

    display = {
        'MOM_lag':  r'$\text{MOM}_{t-1}$',
        'MRET_lag': r'$\text{MRET}_{t-1}$',
        'MVOL_lag': r'$\text{MVOL}_{t-1}$',
    }
    all_regs = ['MOM_lag', 'MRET_lag', 'MVOL_lag']
    n_specs = len(specs)

    results = []
    for spec_name, spec_vars in specs:
        res = _run_ols_nw(dep_var, regressors[spec_vars], maxlags=12)
        results.append((spec_name, spec_vars, res))

    # ── Build .tex ───────────────────────────────────────────────────
    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Determinants of Comomentum --- OLS regressions with Newey--West standard errors (12 lags). '
               r'Dependent variable: detrended $\text{CoMOM}_t$.}')
    tex.append(r'\label{tab:determinants}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l' + ' c' * n_specs + '}')
    tex.append(r'\toprule')
    tex.append('    ' + ' & '.join([''] + [s[0] for s in specs]) + r' \\')
    tex.append(r'\midrule')

    for reg_key in all_regs:
        # Coefficient row
        cells = [display[reg_key]]
        for _, spec_vars, res in results:
            if res and reg_key in res['coefs']:
                c = res['coefs'][reg_key]
                s = _stars(res['pvals'][reg_key])
                cells.append(f'${c:.4f}{s}$')
            else:
                cells.append('')
        tex.append('    ' + ' & '.join(cells) + r' \\')

        # SE row
        cells = ['']
        for _, spec_vars, res in results:
            if res and reg_key in res['se']:
                cells.append(f'$({res["se"][reg_key]:.4f})$')
            else:
                cells.append('')
        tex.append('    ' + ' & '.join(cells) + r' \\')

        # t-statistic row
        cells = ['']
        for _, spec_vars, res in results:
            if res and reg_key in res['tvals']:
                cells.append(f'$t = {res["tvals"][reg_key]:.2f}$')
            else:
                cells.append('')
        tex.append('    ' + ' & '.join(cells) + r' \\')

        # p-value row
        cells = ['']
        for _, spec_vars, res in results:
            if res and reg_key in res['pvals']:
                p = res['pvals'][reg_key]
                if p < 0.001:
                    cells.append('$p < 0.001$')
                else:
                    cells.append(f'$p = {p:.3f}$')
            else:
                cells.append('')
        tex.append('    ' + ' & '.join(cells) + r' \\[4pt]')

    tex.append(r'\midrule')

    # Adj-R²
    cells = [r'Adj.\ $R^2$']
    for _, _, res in results:
        cells.append(f'${res["adj_r2"]:.3f}$' if res else '')
    tex.append('    ' + ' & '.join(cells) + r' \\')

    # N
    cells = ['$N$']
    for _, _, res in results:
        cells.append(f'{res["nobs"]}' if res else '')
    tex.append('    ' + ' & '.join(cells) + r' \\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\vspace{4pt}')
    tex.append(r'\begin{flushleft}')
    tex.append(r'\footnotesize')
    tex.append(r'\textit{Note.} Newey--West standard errors (12 lags) in parentheses. '
               r'*, **, *** denote significance at the 10\%, 5\%, and 1\% levels.')
    tex.append(r'\end{flushleft}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
