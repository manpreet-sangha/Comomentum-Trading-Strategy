# momentum_strategy.py
# =====================================================================
# Enhanced Momentum Trading Strategy using Lou & Polk (2021) Comomentum
# SMM282 Quantitative Trading - Coursework 2026
# =====================================================================
#
# OVERVIEW:
#   This is the main script that orchestrates the entire analysis.
#   It imports reusable modules for data loading, factor computation,
#   Fama-MacBeth regressions, and performance reporting.
#
# PROJECT STRUCTURE:
#   momentum_strategy.py   <- THIS FILE (main pipeline / entry point)
#   data_loader.py         <- Loads all input CSV/Excel data files
#   fama_macbeth.py        <- Fama-MacBeth cross-sectional regression engine
#   momentum_factor.py     <- Computes momentum, comomentum, adjusted momentum
#   standardiseFactor.py   <- Cross-sectional standardisation utility
#   performance.py         <- Summary statistics and charting utilities
#   input_data/            <- Folder with all raw data files
#
# STEPS:
#   (1) Load data
#   (2) Compute standard momentum factor (48w lookback, 4w skip)
#   (3) Run Fama-MacBeth regressions on standard momentum
#   (4) Compute comomentum measure (Lou & Polk, 2021)
#   (5) Adjust momentum using inverse comomentum signal
#   (6) Re-run Fama-MacBeth on adjusted momentum
#   (7) Compare results: plots and summary statistics
#
# ASSUMPTIONS:
#   - Momentum lookback = 48 weeks (includes current week), skip = 4 weeks
#     (one-time removal of the latest 4 weeks from the end of the dataset)
#   - Momentum scores run from week 47 to week 1508 (1462 scored weeks)
#   - Comomentum uses top/bottom quintile stocks (winners + losers)
#   - Comomentum residual correlations are computed over a rolling 52-week window
#   - The comomentum adjustment uses an expanding-window percentile rank
#     to avoid any look-ahead bias
#   - No industry adjustment is applied to the comomentum measure (as per
#     the coursework instructions)
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ----- Import project modules -----
from data_loader import load_all_data
from momentum_factor import compute_momentum

# =====================================================================
# (1) LOAD DATA
# =====================================================================
# All logging, validation, and summary reporting happens inside
# data_loader.load_all_data().  You can also run data_loader.py
# standalone first to verify inputs before running this pipeline.
# =====================================================================
data = load_all_data(datadir='input_data/')

returns_clean = data['returns_clean']
live           = data['live']
dates          = data['dates']
names          = data['names']
ff_factors     = data['ff_factors']
rf             = data['rf']
T              = data['T']
N              = data['N']


# =====================================================================
# (2) COMPUTE STANDARD MOMENTUM FACTOR
#     Lookback: 48 weeks (includes current week) | Skip: 4 weeks
#     (one-time from end of dataset)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Computing standard momentum factor (48w lookback, 4w skip)...")
print("=" * 70)

momentum, momentum_std = compute_momentum(returns_clean,
                                           dates=dates,
                                           lookback=48,
                                           skip=4)

print(f"  Momentum factor computed from week 1508 to week 47 ({T - 1 - 4 - 47 + 1} weeks).")
print(f"  Non-NaN momentum factor values: {np.sum(np.isfinite(momentum)):,}")
print(f"  Momentum factor standardised cross-sectionally (mean=0, std=1).")


# =====================================================================
# STEP 2 - OUTPUT: Save momentum data to output_data/
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2 - OUTPUT: Saving momentum data & generating plots...")
print("=" * 70)

# Create output folder if it does not already exist
output_dir = 'output_data'
os.makedirs(output_dir, exist_ok=True)

# --- Save raw momentum scores as CSV (TxN, with dates as index) ---
# Each row = one week, each column = one stock.
# Values are cumulative returns over the 48-week lookback window.
# NOTE: The full TxN matrix is large (~1513 x 7261). We save a compact
#       version with only the first 20 stocks for inspection, and a
#       separate summary file with cross-sectional statistics per week.
sample_cols = min(20, N)  # first 20 stocks as a readable sample
momentum_sample_df = pd.DataFrame(
    momentum[:, :sample_cols], index=dates, columns=names[:sample_cols]
)
momentum_sample_df.index.name = 'Date'
momentum_sample_df.to_csv(os.path.join(output_dir, 'momentum_raw_sample.csv'))
print(f"  Saved: {output_dir}/momentum_raw_sample.csv  (raw momentum, first {sample_cols} stocks)")

# --- Save standardised momentum scores (sample) as CSV ---
momentum_std_sample_df = pd.DataFrame(
    momentum_std[:, :sample_cols], index=dates, columns=names[:sample_cols]
)
momentum_std_sample_df.index.name = 'Date'
momentum_std_sample_df.to_csv(os.path.join(output_dir, 'momentum_standardised_sample.csv'))
print(f"  Saved: {output_dir}/momentum_standardised_sample.csv  (standardised, first {sample_cols} stocks)")

# --- Save a summary snapshot: cross-sectional stats at each date ---
# For each week: number of live stocks with valid momentum, mean, median,
# standard deviation, min, and max of the raw momentum scores.
summary_rows = []
for t in range(T):
    mom_t = momentum[t, :]
    valid = mom_t[np.isfinite(mom_t)]
    if len(valid) > 0:
        summary_rows.append({
            'Date':       dates[t],
            'N_valid':    len(valid),
            'Mean':       np.mean(valid),
            'Median':     np.median(valid),
            'Std':        np.std(valid, ddof=1),
            'Min':        np.min(valid),
            'Max':        np.max(valid),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_dir, 'momentum_summary.csv'), index=False)
print(f"  Saved: {output_dir}/momentum_summary.csv  (weekly cross-sectional stats)")


# =====================================================================
# STEP 2 - CHARTS
# =====================================================================

# ----- Chart 1: Scatter plot -----
# Shows, for every (week, stock) pair, the raw momentum value plotted
# against the subsequent 1-week stock return.
# This visualises the cross-sectional relationship that Fama-MacBeth
# regressions will formally estimate in Step 3.

print("\n  Generating scatter plot (momentum vs. next-week return)...")

# Flatten the lagged momentum (t) vs. next-week return (t+1)
# Only use valid (live + non-NaN) observations
mom_flat  = momentum_std[:-1, :].ravel()      # lagged factor exposure
ret_flat  = returns_clean[1:, :].ravel()       # next-week return
live_flat = live[1:, :].ravel()                # live indicator at return date

# Validity mask
mask = np.isfinite(mom_flat) & np.isfinite(ret_flat) & (live_flat == 1)

# Subsample for plotting (full dataset can have millions of points)
np.random.seed(42)
n_total = np.sum(mask)
n_plot  = min(50_000, n_total)   # cap at 50k points for readability
idx_all = np.where(mask)[0]
idx_sample = np.random.choice(idx_all, size=n_plot, replace=False)

fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.scatter(mom_flat[idx_sample], ret_flat[idx_sample] * 100,
            s=2, alpha=0.15, color='steelblue', rasterized=True)

# Add a simple OLS fit line to show the average relationship
from numpy.polynomial.polynomial import polyfit
coeffs = polyfit(mom_flat[mask], ret_flat[mask] * 100, deg=1)
x_line = np.linspace(-4, 4, 200)
y_line = coeffs[0] + coeffs[1] * x_line
ax1.plot(x_line, y_line, color='red', linewidth=2,
         label=f'OLS fit (slope = {coeffs[1]:.4f}% per unit)')

ax1.set_xlabel('Standardised Momentum Exposure (t)', fontsize=12)
ax1.set_ylabel('Next-Week Stock Return (%, t+1)', fontsize=12)
ax1.set_title('Scatter Plot: Momentum Exposure vs. Next-Week Return',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-20, 20)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'step2_scatter_momentum_vs_return.png'),
             dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {output_dir}/step2_scatter_momentum_vs_return.png")


# ----- Chart 2: Histogram of momentum factor exposures -----
# Shows the distribution of cross-sectionally standardised momentum
# at a single snapshot date and pooled across all dates.

print("  Generating histogram of momentum exposures...")

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Pooled across ALL weeks
# Flatten all standardised momentum values and drop NaNs
all_mom_std = momentum_std.ravel()
all_mom_std = all_mom_std[np.isfinite(all_mom_std)]

axes[0].hist(all_mom_std, bins=100, color='steelblue', edgecolor='white',
             alpha=0.85, density=True)
axes[0].set_xlabel('Standardised Momentum', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('(a) Distribution of Standardised Momentum\n(All Weeks Pooled)',
                   fontsize=13, fontweight='bold')
axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.6, label='Mean = 0')
axes[0].legend(fontsize=10)
axes[0].set_xlim(-5, 5)
axes[0].grid(True, alpha=0.3)

# Panel (b): Single snapshot - the LAST available week
# Useful to see the cross-sectional spread at one point in time
last_week_mom = momentum_std[-1, :]
last_week_mom = last_week_mom[np.isfinite(last_week_mom)]

axes[1].hist(last_week_mom, bins=60, color='darkorange', edgecolor='white',
             alpha=0.85, density=True)
axes[1].set_xlabel('Standardised Momentum', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'(b) Cross-Sectional Distribution\n(Last Week: {dates[-1].strftime("%Y-%m-%d")})',
                   fontsize=13, fontweight='bold')
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.6, label='Mean = 0')
axes[1].legend(fontsize=10)
axes[1].set_xlim(-5, 5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'step2_histogram_momentum.png'),
             dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {output_dir}/step2_histogram_momentum.png")


# ----- Chart 3: Factor comparison over time -----
# Shows the cross-sectional mean of each available factor plotted as a
# time series.  Comparing factors over time is useful because:
#
#   (a) RAW vs STANDARDISED: Confirms the z-score rescales the raw
#       factor to mean=0 at every week, while preserving the same
#       time-series shape (both lines track market momentum cycles).
#
#   (b) RAW SCALE: The raw cross-sectional mean drifts above/below
#       zero over time, reflecting market-wide bull/bear momentum
#       regimes (e.g. the mean surges during the 1990s tech boom and
#       drops sharply after the 2001 and 2008 crashes).
#
#   (c) STANDARDISED: The cross-sectional mean of the standardised
#       factor is identically zero by construction at every week
#       (this is a built-in property of z-scoring).  Plotting it
#       confirms the standardisation is working correctly.
#
#   NOTE: The ADJUSTED momentum factor (comomentum-scaled) and
#   COMOMENTUM itself are NOT available at this stage -- they are
#   computed in Steps 4 & 5.  Placeholder panels are included here
#   to show where they will appear; they will be populated later.
#
# The 4-panel layout mirrors the four factors in the TERMINOLOGY NOTE:
#   Panel 1: Raw momentum factor       (cross-sectional mean over time)
#   Panel 2: Standardised momentum     (cross-sectional mean over time)
#   Panel 3: Comomentum                (placeholder - Step 4)
#   Panel 4: Adjusted momentum factor  (placeholder - Step 5)

print("\n  Generating 4-factor comparison chart...")

fig3, axes3 = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
fig3.suptitle('Factor Comparison Over Time\n(Cross-Sectional Mean per Week)',
              fontsize=15, fontweight='bold', y=0.98)

# Compute cross-sectional mean at each week (NaN-safe)
raw_mean  = np.nanmean(momentum,     axis=1)   # shape (T,)
std_mean  = np.nanmean(momentum_std, axis=1)   # shape (T,) -- will be ~0 always

# Panel 1: Raw momentum factor
axes3[0].plot(dates, raw_mean * 100, color='steelblue', linewidth=0.8)
axes3[0].axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
axes3[0].set_ylabel('Mean raw return (%)', fontsize=10)
axes3[0].set_title('(1) Raw Momentum Factor  —  cross-sectional mean of 48-week compounded return',
                    fontsize=11, fontweight='bold')
axes3[0].grid(True, alpha=0.25)
axes3[0].fill_between(dates, raw_mean * 100, 0,
                       where=(raw_mean >= 0), alpha=0.15, color='green',
                       label='Positive mean (bull momentum)')
axes3[0].fill_between(dates, raw_mean * 100, 0,
                       where=(raw_mean < 0),  alpha=0.15, color='red',
                       label='Negative mean (bear momentum)')
axes3[0].legend(fontsize=9, loc='upper right')

# Panel 2: Standardised momentum factor
axes3[1].plot(dates, std_mean, color='darkorange', linewidth=0.8)
axes3[1].axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
axes3[1].set_ylabel('Mean z-score', fontsize=10)
axes3[1].set_title('(2) Standardised Momentum Factor  —  cross-sectional mean of z-scores '
                    '(identically 0 by construction)',
                    fontsize=11, fontweight='bold')
axes3[1].grid(True, alpha=0.25)
axes3[1].set_ylim(-0.05, 0.05)   # should be essentially flat at 0
axes3[1].text(dates[T // 2], 0.03,
              'Always = 0 by construction (z-score property)',
              ha='center', va='center', fontsize=9, color='gray',
              style='italic')

# Panel 3: Comomentum (placeholder)
axes3[2].text(0.5, 0.5,
              'Comomentum (Step 4)\n\nNot yet computed.\nWill show the average pairwise '
              'residual correlation\namong momentum stocks each week.',
              transform=axes3[2].transAxes,
              ha='center', va='center', fontsize=11, color='gray',
              style='italic',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='orange', alpha=0.8))
axes3[2].set_ylabel('Comomentum', fontsize=10)
axes3[2].set_title('(3) Comomentum  —  average residual correlation among momentum stocks '
                    '(placeholder: Step 4)',
                    fontsize=11, fontweight='bold')
axes3[2].grid(True, alpha=0.25)
axes3[2].set_yticks([])

# Panel 4: Adjusted momentum factor (placeholder)
axes3[3].text(0.5, 0.5,
              'Adjusted Momentum Factor (Step 5)\n\nNot yet computed.\nWill show the '
              'comomentum-scaled standardised momentum\n(inverse comomentum weighting '
              'applied, then re-standardised).',
              transform=axes3[3].transAxes,
              ha='center', va='center', fontsize=11, color='gray',
              style='italic',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='purple', alpha=0.8))
axes3[3].set_ylabel('Mean z-score', fontsize=10)
axes3[3].set_title('(4) Adjusted Momentum Factor  —  cross-sectional mean after comomentum '
                    'adjustment (placeholder: Step 5)',
                    fontsize=11, fontweight='bold')
axes3[3].grid(True, alpha=0.25)
axes3[3].set_yticks([])

# Shared x-axis label
axes3[3].set_xlabel('Date', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig(os.path.join(output_dir, 'step2_factor_comparison.png'),
             dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {output_dir}/step2_factor_comparison.png")
print("  Note: Panels 3 & 4 are placeholders - will be populated in Steps 4 & 5.")


# =====================================================================
# Done - Steps 3-7 are commented out for now (run only Step 1 & 2)
# =====================================================================
print("\n" + "=" * 70)
print("Steps 1 & 2 complete. Outputs saved to output_data/.")
print("=" * 70)
print("Done!")
