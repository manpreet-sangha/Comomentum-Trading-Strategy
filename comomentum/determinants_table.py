# determinants_table.py
# =====================================================================
# Determinants of Comomentum — Replicating Lou & Polk (2021) Table II
# =====================================================================
#
# Runs time-series OLS regressions of detrended CoMOM on variables
# related to arbitrage activity:
#
#   DepVar = Detrended CoMOM_t
#
#   Regressors (available):
#     MOM_{t-1}   — return of the momentum strategy in year t-1
#     MRET_{t-1}  — trailing 2-year compounded market return
#     MVOL_{t-1}  — trailing 2-year monthly market volatility
#
# All variables are detrended (linear time trend removed) and
# resampled to annual frequency before regression, following the
# paper's methodology.
#
# Standard errors are Newey-West (HAC) with 12 lags.
#
# Produces a PNG table with regression coefficients, standard errors,
# significance stars, Adj-R², and number of observations.
#
# Standalone:  python -m comomentum.determinants_table
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# =====================================================================
# Helper: detrend a series (remove linear time trend)
# =====================================================================
def _detrend(series):
    """
    Remove a linear time trend from a pandas Series.
    Returns the residuals (detrended series).
    """
    valid = series.dropna()
    if len(valid) < 3:
        return series
    trend = np.arange(len(valid), dtype=float)
    X = sm.add_constant(trend)
    model = sm.OLS(valid.values, X).fit()
    residuals = pd.Series(model.resid, index=valid.index, name=series.name)
    return residuals


# =====================================================================
# Helper: significance stars
# =====================================================================
def _stars(pval):
    """Return significance stars for a p-value."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''


# =====================================================================
# Helper: run a single OLS regression with Newey-West SEs
# =====================================================================
def _run_ols_nw(y, X, maxlags=12):
    """
    Run OLS with Newey-West (HAC) standard errors.

    Parameters
    ----------
    y : pd.Series — dependent variable
    X : pd.DataFrame — regressors (without constant; added here)

    Returns
    -------
    dict with keys: 'coefs', 'se', 'pvals', 'adj_r2', 'nobs', 'names'
    """
    # Align and drop any NaN rows
    data = pd.concat([y.rename('_y_'), X], axis=1).dropna()
    if len(data) < X.shape[1] + 2:
        return None

    y_clean = data['_y_']
    X_clean = sm.add_constant(data.drop(columns='_y_'))

    model = sm.OLS(y_clean, X_clean).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags}
    )

    # Extract results for non-constant regressors only
    regressor_names = [c for c in X_clean.columns if c != 'const']
    coefs = {name: model.params[name] for name in regressor_names}
    se = {name: model.bse[name] for name in regressor_names}
    pvals = {name: model.pvalues[name] for name in regressor_names}

    return {
        'coefs': coefs,
        'se': se,
        'pvals': pvals,
        'adj_r2': model.rsquared_adj,
        'nobs': int(model.nobs),
        'names': regressor_names,
    }


# =====================================================================
# Main: generate the determinants table
# =====================================================================
def generate_determinants_table(comomentum, gamma_std, mret, mvol, dates,
                                 save_path='output_data/determinants_table.png'):
    """
    Produce an OLS regression table (Table II) as a PNG image.

    Parameters
    ----------
    comomentum : (T,) array — weekly CoMOM series
    gamma_std  : (T,) array — weekly momentum factor return (Fama-MacBeth γ)
    mret       : (T,) array — weekly trailing 2-year market return
    mvol       : (T,) array — weekly trailing 2-year market volatility
    dates      : T-length DatetimeIndex
    save_path  : str — output PNG path
    """
    dates = pd.DatetimeIndex(dates)
    T = len(dates)

    # ── Build weekly DataFrame ───────────────────────────────────────
    df_weekly = pd.DataFrame({
        'CoMOM': comomentum,
        'MOM': gamma_std,
        'MRET': mret,
        'MVOL': mvol,
    }, index=dates)

    # ── Resample to annual frequency ────────────────────────────────
    # CoMOM, MRET, MVOL: annual mean of weekly values
    # MOM: compound weekly factor returns into annual return
    annual = pd.DataFrame(index=df_weekly.resample('YE').mean().index)

    annual['CoMOM'] = df_weekly['CoMOM'].resample('YE').mean()

    # MOM: compound weekly returns → annual return
    annual['MOM'] = df_weekly['MOM'].resample('YE').apply(
        lambda x: np.prod(1.0 + x.dropna().values) - 1.0
            if len(x.dropna()) > 0 else np.nan
    )

    annual['MRET'] = df_weekly['MRET'].resample('YE').mean()
    annual['MVOL'] = df_weekly['MVOL'].resample('YE').mean()

    annual = annual.dropna(subset=['CoMOM'])

    # ── Detrend all variables ────────────────────────────────────────
    detrended = pd.DataFrame(index=annual.index)
    for col in annual.columns:
        detrended[col] = _detrend(annual[col])

    # ── Lag independent variables by 1 year ──────────────────────────
    dep_var = detrended['CoMOM']
    regressors = pd.DataFrame(index=detrended.index)
    regressors['MOM_lag'] = detrended['MOM'].shift(1)
    regressors['MRET_lag'] = detrended['MRET'].shift(1)
    regressors['MVOL_lag'] = detrended['MVOL'].shift(1)

    # ── Define the 4 regression specifications ───────────────────────
    # Column [1]: MOM only
    # Column [2]: MRET + MVOL only
    # Column [3]: MOM + MRET + MVOL
    # Column [4]: All three (same as [3], but matching the paper layout)
    specs = [
        ('[1]', ['MOM_lag']),
        ('[2]', ['MRET_lag', 'MVOL_lag']),
        ('[3]', ['MOM_lag', 'MRET_lag', 'MVOL_lag']),
    ]

    # Display names for the rows
    display_names = {
        'MOM_lag':  'MOM$_{t-1}$',
        'MRET_lag': 'MRET$_{t-1}$',
        'MVOL_lag': 'MVOL$_{t-1}$',
    }

    # All unique regressor keys (preserving order)
    all_regressors = ['MOM_lag', 'MRET_lag', 'MVOL_lag']

    # ── Run regressions ──────────────────────────────────────────────
    results = []
    for spec_name, spec_vars in specs:
        X = regressors[spec_vars]
        res = _run_ols_nw(dep_var, X, maxlags=12)
        results.append((spec_name, spec_vars, res))

    # ── Build table rows ─────────────────────────────────────────────
    # Each regressor gets two rows: coefficient + standard error
    # Then Adj-R² and No. Obs. at the bottom
    n_specs = len(specs)

    table_rows = []
    for reg_key in all_regressors:
        # Coefficient row
        coef_row = [display_names[reg_key]]
        for spec_name, spec_vars, res in results:
            if res is not None and reg_key in res['coefs']:
                coef = res['coefs'][reg_key]
                pval = res['pvals'][reg_key]
                stars = _stars(pval)
                coef_row.append(f'{coef:.3f}{stars}')
            else:
                coef_row.append('')
        table_rows.append(coef_row)

        # Standard error row
        se_row = ['']
        for spec_name, spec_vars, res in results:
            if res is not None and reg_key in res['se']:
                se_row.append(f'[{res["se"][reg_key]:.3f}]')
            else:
                se_row.append('')
        table_rows.append(se_row)

    # Blank separator row
    table_rows.append([''] * (1 + n_specs))

    # Adj-R² row
    r2_row = ['Adj-R\u00B2']
    for spec_name, spec_vars, res in results:
        if res is not None:
            r2_row.append(f'{res["adj_r2"]:.2f}')
        else:
            r2_row.append('')
    table_rows.append(r2_row)

    # No. Obs. row
    nobs_row = ['No. Obs.']
    for spec_name, spec_vars, res in results:
        if res is not None:
            nobs_row.append(f'{res["nobs"]}')
        else:
            nobs_row.append('')
    table_rows.append(nobs_row)

    # ── Render as PNG ────────────────────────────────────────────────
    col_labels = [''] + [s[0] for s in specs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_title(
        'Table II: Determinants of Comomentum\n'
        'DepVar = Detrended CoMOM$_t$',
        fontsize=13, fontweight='bold', pad=20
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight='bold', fontstyle='italic')
        cell.set_facecolor('#f0f0f0')
        cell.set_edgecolor('#cccccc')

    # Style data rows
    for i in range(len(table_rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor('#cccccc')
            if j == 0:
                cell.set_text_props(fontstyle='italic')

    # Add footnote
    fig.text(
        0.5, 0.02,
        'Newey-West standard errors (12 lags) in brackets. '
        '*, **, *** denote significance at 10%, 5%, 1%.',
        ha='center', fontsize=8, fontstyle='italic', color='#555555'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"  Saved: {save_path}")


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from comomentum.compute_comomentum import compute_comomentum
    from data.market_variables import compute_market_variables
    from fama_macbeth.fama_macbeth import famaMacBeth

    print("Loading data...")
    data = load_all_data('input_data/')

    print("Computing momentum signal...")
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )

    print("Running Fama-MacBeth...")
    gamma_std, tstat_std = famaMacBeth(
        momentum_std, data['returns_clean'], data['live'],
        dates=data['dates']
    )

    print("Computing comomentum...")
    comomentum, comom_w, comom_l = compute_comomentum(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'], data['dates']
    )

    print("Computing market variables...")
    mret, mvol = compute_market_variables(
        data['ff_factors'], data['rf'], data['dates']
    )

    print("Generating Table II: Determinants of Comomentum...")
    generate_determinants_table(
        comomentum, gamma_std, mret, mvol, data['dates'],
        save_path='output_data/determinants_table.png'
    )
