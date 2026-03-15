# momentum_strategy.py
# =====================================================================
# Enhanced Momentum Trading Strategy using Lou & Polk (2021) Comomentum
# SMM282 Quantitative Trading — Coursework 2026
# =====================================================================
#
# OVERVIEW:
#   Main pipeline / entry point.  Each step delegates to a dedicated
#   reusable module — this file is purely an orchestrator.
#
# PROJECT STRUCTURE:
#   momentum_strategy.py          <- THIS FILE (pipeline orchestrator)
#   data_loader.py                <- Step 1: Loads all input data
#   stock_diagnostics.py          <- Step 1b: Flags short-lived & gapped stocks
#   compute_momentum_signal.py    <- Step 2: Rolling momentum (48w LB, 4w skip)
#   step2_outputs.py              <- Step 2: Saves momentum CSVs
#   step2_plots.py                <- Step 2: Diagnostic charts (scatter,
#                                            histogram, 4-panel comparison)
#   compute_comomentum.py         <- Step 4: Comomentum (Lou & Polk, 2021)
#   compute_adjusted_momentum.py  <- Step 5: Inverse-comomentum scaling
#   standardiseFactor.py          <- Cross-sectional z-score utility
#   fama_macbeth.py               <- Fama-MacBeth regression engine
#   performance.py                <- Summary statistics & charting
#   logger_setup.py               <- Centralised logging
#   input_data/                   <- Raw CSV / Excel files
#   output_data/                  <- Generated plots, CSVs, logs
#
# STEPS:
#   (1) Load data
#   (2) Compute standard momentum factor
#       — 48-week lookback, 4-week ROLLING skip (most recent 4 weeks
#         relative to each computation date, NOT a one-time global skip)
#       — First score at week 52 (index 51 = 19921225)
#       — 1,462 scored weeks
#   (3) Run Fama-MacBeth regressions on standard momentum
#   (4) Compute comomentum measure (Lou & Polk, 2021)
#   (5) Adjust momentum: scale factor RETURNS (gamma) by inverse
#       comomentum — NOT the exposures (see compute_adjusted_momentum.py)
#   (6) Compare results: plots and summary statistics
# =====================================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Import project modules ───────────────────────────────────────────
from data.data_loader                import load_all_data
from compute_momentum.stock_diagnostics import run_stock_diagnostics
from compute_momentum.compute_momentum_signal import compute_momentum_signal
from compute_momentum.step2_outputs import save_momentum_outputs
from compute_momentum.step2_plots   import generate_step2_plots
from fama_macbeth.fama_macbeth   import famaMacBeth
from comomentum.compute_comomentum import compute_comomentum
from adjusted_momentum.compute_adjusted_momentum import compute_adjusted_momentum
from comomentum.save_ff3_residuals import save_ff3_residuals
from comomentum.save_pairwise_correlations import save_pairwise_correlations
from comomentum.plot_comom_event_study import plot_comom_event_study
from comomentum.plot_comom_time_series import plot_comom_time_series
from data.market_variables import compute_market_variables
from comomentum.summary_statistics_table import generate_summary_table
from performance                import compute_stats, print_summary_table, plot_main_results

# =====================================================================
# (1) LOAD DATA
# =====================================================================
data = load_all_data(datadir='input_data/')

# =====================================================================
# (1b) STOCK DIAGNOSTICS — flag & exclude problematic stocks
#      Only stocks with < 52 total listed weeks are globally excluded
#      (they can NEVER have a valid 52-week window).
#      Gapped stocks are KEPT — they may still have valid 52-week
#      windows during some periods.  The momentum computation itself
#      enforces a per-window check: all 52 weeks (48 lookback + 4 skip)
#      must have valid returns for a stock to receive a score.
# =====================================================================
df_short, df_gaps, df_combined = run_stock_diagnostics(data)

# Collect column indices of short-lived stocks ONLY
exclude_idx = set()
if len(df_short) > 0:
    exclude_idx |= set(df_short['Stock_Index'])

n_excluded = len(exclude_idx)
n_gapped = len(df_gaps) if len(df_gaps) > 0 else 0
print(f"\n  Excluding {n_excluded} stocks globally "
      f"({len(df_short)} short-lived, < 52 total listed weeks).")
print(f"  Keeping {n_gapped} gapped stocks — per-window eligibility "
      f"checked in Step 2.")

# NaN-out excluded columns in returns_clean so momentum loop skips them
for j in exclude_idx:
    data['returns_clean'][:, j] = np.nan

# =====================================================================
# (2) COMPUTE STANDARD MOMENTUM FACTOR
#     48-week lookback, 4-week rolling skip (per computation date)
# =====================================================================
momentum, momentum_std = compute_momentum_signal(
    data['returns_clean'], data['dates']
)

# ── Save momentum CSVs ──────────────────────────────────────────────
save_momentum_outputs(
    momentum, momentum_std, data['dates'], data['names']
)

# ── Diagnostic plots ────────────────────────────────────────────────
generate_step2_plots(momentum, momentum_std, data)

# =====================================================================
# (3) FAMA-MACBETH ON STANDARD MOMENTUM
#     Cross-sectional regression: r_{i,t} = α + γ * mom_{i,t-1} + ε
#     γ_t is the factor return for week t.  Momentum is used raw
#     (no z-scoring) to replicate the paper as-is.
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 3: Fama-MacBeth regressions on STANDARD momentum")
print("=" * 70)
gamma_std, tstat_std = famaMacBeth(
    momentum_std, data['returns_clean'], data['live'],
    dates=data['dates'],
    save_path='output_data/fama_macbeth_standard_momentum.xlsx'
)
n_valid_gamma = int(np.sum(np.isfinite(gamma_std)))
print(f"  Factor return series: {n_valid_gamma} valid weeks out of {data['T']}")
print(f"  t-statistic = {tstat_std:.4f}")

# =====================================================================
# (4) COMPUTE COMOMENTUM (Lou & Polk, 2021)
#     Separate winner & loser decile residual correlations, then average.
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 4: Computing comomentum")
print("=" * 70)
comomentum, comom_winner, comom_loser = compute_comomentum(
    data['returns_clean'], momentum_std,
    data['live'], data['ff_factors'], data['dates']
)
n_valid_comom = int(np.sum(np.isfinite(comomentum)))
print(f"  Comomentum: {n_valid_comom} valid values out of {data['T']} weeks")

# ── Save Step 4 diagnostic outputs (snapshot of last week) ──────────
save_ff3_residuals(
    data['returns_clean'], momentum_std,
    data['live'], data['ff_factors'],
    data['dates'], data['names'],
    snapshot_week=None,
    save_path='output_data/ff3_residuals.xlsx'
)
save_pairwise_correlations(
    data['returns_clean'], momentum_std,
    data['live'], data['ff_factors'],
    data['dates'], data['names'],
    snapshot_week=None,
    save_path='output_data/pairwise_correlations.xlsx'
)
plot_comom_event_study(
    comomentum, data['dates'],
    max_years=4,
    save_path='output_data/plot_comom_event_study.png'
)
plot_comom_time_series(
    comomentum, comom_winner, comom_loser, data['dates'],
    sample_months=6,
    save_path='output_data/plot_comom_time_series.png'
)

# ── Market variables for Table I summary statistics ──────────────────
mret, mvol = compute_market_variables(
    data['ff_factors'], data['rf'], data['dates']
)
n_valid_mret = int(np.sum(np.isfinite(mret)))
n_valid_mvol = int(np.sum(np.isfinite(mvol)))
print(f"  MRET: {n_valid_mret} valid values, "
      f"mean={np.nanmean(mret):.4f}")
print(f"  MVOL: {n_valid_mvol} valid values, "
      f"mean={np.nanmean(mvol):.4f}")

# ── Summary statistics table (Table I) ─────────────────────────────────
generate_summary_table(
    comomentum, comom_winner, comom_loser, mret, mvol, data['dates'],
    save_path='output_data/summary_statistics_table.png'
)

# =====================================================================
# (5) ADJUST MOMENTUM USING INVERSE COMOMENTUM
#     Scale the FACTOR RETURNS (gamma_std), NOT the exposures.
#     Scaling exposures and re-standardising undoes the effect because
#     the scaling factor is the same for all stocks at each week.
#     scaling = 2 − lagged percentile rank of comomentum
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 5: Adjusting momentum with inverse comomentum signal")
print("=" * 70)
gamma_adj, scaling, comom_pctile = compute_adjusted_momentum(
    gamma_std, comomentum
)
n_valid_gamma_adj = int(np.sum(np.isfinite(gamma_adj)))
print(f"  Adjusted factor returns: {n_valid_gamma_adj} valid weeks")
print(f"  Scaling factor: mean={np.nanmean(scaling):.4f}, "
      f"std={np.nanstd(scaling):.4f}")
print(f"  gamma_std mean={np.nanmean(gamma_std)*100:.4f}%  →  "
      f"gamma_adj mean={np.nanmean(gamma_adj)*100:.4f}%")

# =====================================================================
# (6) COMPARISON: Summary statistics & plots
#     (Steps 5+6 combined: scaling already applied to gamma_std)
# =====================================================================
print("\n" + "=" * 70)
print("  STEP 6: Comparing Standard vs. Adjusted Momentum")
print("=" * 70)

stats_std = compute_stats(gamma_std, label='Standard Momentum')
stats_adj = compute_stats(gamma_adj, label='Adjusted Momentum')
print_summary_table(stats_std, stats_adj)

plot_main_results(
    data['dates'], gamma_std, gamma_adj, comomentum, scaling,
    save_path='output_data/momentum_results.png'
)

# =====================================================================
# DONE
# =====================================================================
print("\n" + "=" * 70)
print("ALL STEPS COMPLETE.  Outputs saved to output_data/.")
print("=" * 70)
print("Done!")
