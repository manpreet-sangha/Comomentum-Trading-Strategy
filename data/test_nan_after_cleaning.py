# test_nan_after_cleaning.py
# =====================================================================
# Standalone Diagnostic Script
# =====================================================================
# All stocks in the dataset are real companies.  Not every company was
# listed for the full duration of the time series — some IPO'd part-way
# through, others were delisted (merger, acquisition, bankruptcy, etc.)
# before the series ends.
#
# US_live flags the listing window for each stock:
#   live = 1  →  the stock was listed on an exchange that week
#   live = 0  →  the stock had not yet IPO'd or had already been delisted
#
# After setting all live=0 cells to NaN (they are not investable), this
# script checks whether any NaN values remain WITHIN each stock's
# listing period (live = 1).  Such NaNs represent genuine missing data
# — weeks where the stock was listed but its return was not recorded.
#
# Understanding their prevalence helps decide how to handle them in
# downstream steps (e.g. requiring a minimum observation count when
# computing momentum scores).
#
# OUTPUT (console):
#   - Total listed-period NaN count and percentage
#   - Number (and %) of stocks that have at least one listed-period NaN
#   - Top 20 worst-affected stocks by NaN count
#   - Year-by-year breakdown of listed-period NaN rates
# =====================================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
print("=" * 65)
print("  NaN-AFTER-CLEANING DIAGNOSTIC")
print("=" * 65)

returns = pd.read_csv('input_data/US_Returns.csv', header=None).values
live    = pd.read_csv('input_data/US_live,csv.csv', header=None).values
dates   = pd.to_datetime(
    pd.read_excel('input_data/US_Dates.xlsx', header=None).iloc[:, 0]
    .values.astype(str), format='%Y%m%d'
)
names_df = pd.read_excel('input_data/US_Names.xlsx', header=None)
if names_df.shape[0] == 1:
    names = names_df.iloc[0, :].values
else:
    names = names_df.iloc[:, 0].values

T, N = returns.shape
print(f"\nDataset dimensions : T={T:,} weeks  x  N={N:,} stocks")

# ------------------------------------------------------------------
# 2. Clean returns (set non-listed periods to NaN)
#    live=0 means the stock was not listed that week (pre-IPO or
#    post-delist), so any return value there is not investable.
# ------------------------------------------------------------------
returns_clean = returns.copy().astype(float)
returns_clean[live == 0] = np.nan

# ------------------------------------------------------------------
# 3. Identify NaNs that fall WITHIN listed periods
#    live==1 AND return is NaN  =>  stock was listed but return is
#    missing from the dataset (genuine data gap)
# ------------------------------------------------------------------
live_mask    = (live == 1)
live_nan_mask = live_mask & np.isnan(returns_clean)

total_live_cells = np.sum(live_mask)
total_live_nans  = np.sum(live_nan_mask)

print(f"\nTotal listed cells        : {total_live_cells:,}")
print(f"Listed with valid return  : {total_live_cells - total_live_nans:,}")
print(f"Listed but return missing : {total_live_nans:,}  "
      f"({total_live_nans / total_live_cells * 100:.4f}%)")

# ------------------------------------------------------------------
# 4. Per-stock NaN count (only within each stock's listing period)
# ------------------------------------------------------------------
nans_per_stock  = np.sum(live_nan_mask, axis=0)   # shape (N,)
live_per_stock  = np.sum(live_mask, axis=0)        # shape (N,)

stocks_with_nan = np.sum(nans_per_stock > 0)
stocks_clean    = N - stocks_with_nan

print(f"\nStocks with complete returns : {stocks_clean:,}  "
      f"({stocks_clean / N * 100:.1f}%)")
print(f"Stocks with missing returns : {stocks_with_nan:,}  "
      f"({stocks_with_nan / N * 100:.1f}%)")

# ------------------------------------------------------------------
# 5. Top 20 worst-affected stocks
# ------------------------------------------------------------------
if stocks_with_nan > 0:
    print(f"\n{'─' * 65}")
    print(f"  Top 20 stocks with most missing returns (during listed period)")
    print(f"{'─' * 65}")
    print(f"  {'Rank':<6}{'Stock Name':<30}{'Listed':>10}{'Missing':>10}{'Miss %':>10}")
    print(f"  {'─'*6}{'─'*30}{'─'*10}{'─'*10}{'─'*10}")

    order = np.argsort(-nans_per_stock)  # descending
    for rank, idx in enumerate(order[:20], start=1):
        if nans_per_stock[idx] == 0:
            break
        pct = nans_per_stock[idx] / live_per_stock[idx] * 100 if live_per_stock[idx] > 0 else 0
        name = str(names[idx])[:28]
        print(f"  {rank:<6}{name:<30}{live_per_stock[idx]:>10,}{nans_per_stock[idx]:>10,}"
              f"{pct:>9.2f}%")

# ------------------------------------------------------------------
# 6. Year-by-year breakdown
# ------------------------------------------------------------------
print(f"\n{'─' * 65}")
print(f"  Listed-period NaN rate by year")
print(f"{'─' * 65}")
print(f"  {'Year':<8}{'Listed Cells':>14}{'Missing':>14}{'Miss %':>10}")
print(f"  {'─'*8}{'─'*14}{'─'*14}{'─'*10}")

years = dates.year
for yr in np.sort(np.unique(years)):
    yr_mask = (years == yr)
    live_yr     = np.sum(live_mask[yr_mask, :])
    live_nan_yr = np.sum(live_nan_mask[yr_mask, :])
    pct = live_nan_yr / live_yr * 100 if live_yr > 0 else 0
    print(f"  {yr:<8}{live_yr:>14,}{live_nan_yr:>14,}{pct:>9.2f}%")

# ------------------------------------------------------------------
# 7. Final verdict
# ------------------------------------------------------------------
print(f"\n{'=' * 65}")
if total_live_nans == 0:
    print("  RESULT: No missing returns found during any stock's listing period.")
    print("  Every stock has a complete return for every week it was listed.")
else:
    print(f"  RESULT: {total_live_nans:,} missing returns found during listing periods")
    print(f"  across {stocks_with_nan:,} stocks ({stocks_with_nan/N*100:.1f}% of universe).")
    print(f"  These are genuine data gaps — the stock was listed but its")
    print(f"  return was not recorded in the dataset.")
    print(f"  Downstream code should use min_obs thresholds to handle them.")
print("=" * 65)
