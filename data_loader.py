# data_loader.py
# =====================================================================
# Data Loading Module
# =====================================================================
# This module handles loading all input data files required by the
# momentum trading strategy. It reads stock returns, live/dead flags,
# dates, company names, and Fama-French factor data from the input_data
# folder and returns them in a clean dictionary for downstream use.
#
# INPUT FILES AND THEIR SIGNIFICANCE:
#
#   US_Returns.csv (TxN)
#       Weekly total returns for N=7261 US stocks over T=1513 weeks.
#       Returns are in decimal form (e.g. 0.02 = +2%).
#       This is the core dataset: every step of the analysis (momentum
#       computation, Fama-MacBeth regressions, performance evaluation)
#       ultimately depends on these return series.
#
#   US_live,csv.csv (TxN)
#       A binary indicator matrix (1 or 0) with the same shape as
#       US_Returns.csv.
#
#       "LIVE" (value = 1):
#           The stock was actively listed and traded on that date.
#           Its return is a real, investable number.
#
#       "DEAD" (value = 0):
#           The stock was NOT yet listed (pre-IPO) or had already been
#           delisted (bankruptcy, merger, acquisition, going private,
#           etc.) on that date.  The return value in US_Returns.csv for
#           a dead observation is meaningless (often 0 or stale) and
#           must NOT be used in any computation.
#
#       WHY THIS MATTERS:
#           If we included dead-stock returns we would introduce
#           survivorship bias (only using companies that survived the
#           full sample) or contaminate our factor scores with fake
#           zero returns.  Setting dead observations to NaN ensures
#           they are automatically excluded from nanmean, nanstd,
#           nanprod, regressions, and all other numerical routines.
#           This gives us a clean, bias-free panel.
#
#   US_Dates.xlsx (Tx1)
#       Weekly date stamps in YYYYMMDD integer format.  Converted to
#       Python datetime objects for use as time-series indices, chart
#       labels, and output file formatting.
#
#   US_Names.xlsx (1xN or Nx1)
#       Ticker / company name labels for each of the N stocks.  Used
#       for labelling output CSVs and charts.  The file may be stored
#       as a single row (1xN) or a single column (Nx1) depending on
#       how it was exported; we handle both layouts.
#
#   FamaFrench.csv (Tx4)
#       Weekly Fama-French three-factor returns plus the risk-free rate:
#         - Mkt-RF : excess market return (market minus risk-free)
#         - SMB    : Small-Minus-Big size factor
#         - HML    : High-Minus-Low value factor
#         - RF     : risk-free rate (T-bill)
#       These are used in two places:
#         (a) Computing comomentum residuals: we regress each momentum
#             stock's returns on the FF3 factors to isolate abnormal
#             (idiosyncratic) returns, whose pairwise correlations
#             form the comomentum measure (Lou & Polk, 2021).
#         (b) Potentially adjusting for risk in performance evaluation.
#
# LOGGING:
#   Every file load is logged with confirmation of success/failure
#   and the shape (rows x columns) of the data read.  A summary log
#   is written to  output_data/data_loading.log  so that a front-end
#   application can display the log when the user clicks "read_data".
#   Logs are also printed to the console for interactive use.
# =====================================================================

import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime


# =====================================================================
# Logger setup  (module-level, configured once on first import)
# =====================================================================

def _setup_logger(log_dir='output_data', log_filename='data_loading.log'):
    """
    Creates and returns a logger that writes to BOTH:
      1. A log file  (output_data/data_loading.log)  -- for the future app
      2. The console  (stdout)                        -- for interactive use

    Each run overwrites the log file so the front-end always reads
    the latest data-loading result.
    """
    logger = logging.getLogger('data_loader')

    # Avoid adding duplicate handlers if this module is re-imported
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatter shared by both handlers
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- File handler (append mode) ---
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, log_filename), mode='w', encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = _setup_logger()


def load_all_data(datadir='input_data/'):
    """
    Loads all required data files from the specified directory.

    FILES LOADED:
        US_Returns.csv   - TxN matrix of weekly total stock returns (decimals)
        US_live,csv.csv  - TxN matrix of live/dead dummies (1=live, 0=dead)
        US_Dates.xlsx    - Tx1 vector of weekly dates (YYYYMMDD format)
        US_Names.xlsx    - Nx1 vector of stock/company names
        FamaFrench.csv   - Tx4 matrix of weekly FF3 factor returns + risk-free rate

    INPUT:
        datadir : str - path to the folder containing the data files
                        (default: 'input_data/')

    OUTPUT:
        data : dict - dictionary with the following keys:
            'returns'       : np.ndarray (TxN) - raw weekly stock returns
            'returns_clean' : np.ndarray (TxN) - returns with dead stocks set to NaN
            'live'          : np.ndarray (TxN) - live/dead indicator matrix
            'dates'         : pd.DatetimeIndex  - weekly date series
            'names'         : np.ndarray (N,)   - stock name labels
            'ff_factors'    : np.ndarray (Tx3)  - Fama-French factor matrix [MktRF, SMB, HML]
            'rf'            : np.ndarray (T,)   - risk-free rate series
            'T'             : int - number of weeks
            'N'             : int - number of stocks
    """

    # ------------------------------------------------------------------
    # 1. Weekly stock returns (TxN)
    #    Each row is one week, each column is one stock.
    #    Values are decimal returns (e.g. 0.02 = 2%).
    # ------------------------------------------------------------------
    log.info("Loading US_Returns.csv ...")
    returns = pd.read_csv(datadir + 'US_Returns.csv', header=None).values
    log.info(f"  OK  US_Returns.csv  ->  {returns.shape[0]} rows x {returns.shape[1]} columns")
    log.info(f"       dtype={returns.dtype}, "
             f"NaN count={np.sum(np.isnan(returns.astype(float)))}, "
             f"min={np.nanmin(returns):.6f}, max={np.nanmax(returns):.6f}")

    # ------------------------------------------------------------------
    # 2. Live/dead indicator (TxN)
    #    1 = company was actively traded that week
    #    0 = company was not yet listed or had been delisted
    # ------------------------------------------------------------------
    log.info("Loading US_live,csv.csv ...")
    live = pd.read_csv(datadir + 'US_live,csv.csv', header=None).values
    log.info(f"  OK  US_live,csv.csv ->  {live.shape[0]} rows x {live.shape[1]} columns")
    log.info(f"       Unique values: {np.unique(live)}, "
             f"live=1 count={np.sum(live == 1):,}, dead=0 count={np.sum(live == 0):,}")

    # ------------------------------------------------------------------
    # 3. Dates vector (Tx1)
    #    Stored as integers in YYYYMMDD format; converted to datetime.
    # ------------------------------------------------------------------
    log.info("Loading US_Dates.xlsx ...")
    dates_df = pd.read_excel(datadir + 'US_Dates.xlsx', header=None)
    dates_raw = dates_df.iloc[:, 0].values
    dates = pd.to_datetime(dates_raw.astype(str), format='%Y%m%d')
    log.info(f"  OK  US_Dates.xlsx   ->  {len(dates)} dates")
    log.info(f"       First date: {dates[0]}, Last date: {dates[-1]}")

    # ------------------------------------------------------------------
    # 4. Company / stock names (N)
    #    The file stores names as a single row (1xN) or a single column (Nx1).
    #    We flatten to a 1-D array of length N either way.
    # ------------------------------------------------------------------
    log.info("Loading US_Names.xlsx ...")
    names_df = pd.read_excel(datadir + 'US_Names.xlsx', header=None)
    if names_df.shape[0] == 1:
        # Names stored as one row with N columns -> read along columns
        names = names_df.iloc[0, :].values
        log.info(f"       Names file layout: 1 row x {names_df.shape[1]} columns (1xN format)")
    else:
        # Names stored as N rows in a single column -> read along rows
        names = names_df.iloc[:, 0].values
        log.info(f"       Names file layout: {names_df.shape[0]} rows x 1 column (Nx1 format)")
    log.info(f"  OK  US_Names.xlsx   ->  {len(names)} stock names")
    log.info(f"       First 5 names: {list(names[:5])}")

    # ------------------------------------------------------------------
    # 5. Fama-French 3-factor data + risk-free rate
    #    Columns: Mkt-RF, SMB, HML, RF (all in weekly decimal returns)
    # ------------------------------------------------------------------
    log.info("Loading FamaFrench.csv ...")
    ff = pd.read_csv(datadir + 'FamaFrench.csv')
    ff_factors = np.column_stack((
        ff['Mkt-RF'].values,
        ff['SMB'].values,
        ff['HML'].values
    ))
    rf = ff['RF'].values
    log.info(f"  OK  FamaFrench.csv  ->  {ff.shape[0]} rows x {ff.shape[1]} columns "
             f"(factors: Mkt-RF, SMB, HML + RF)")
    log.info(f"       FF factors shape: {ff_factors.shape}, RF shape: {rf.shape}")

    # ------------------------------------------------------------------
    # 6. Clean returns: set dead-stock observations to NaN
    #
    #    WHY?  The raw US_Returns.csv already contains NaN for MOST
    #    dead-stock cells -- but not necessarily ALL of them.  Some
    #    dead observations may carry a stale value (e.g. 0.0) that
    #    looks like a real return but is not genuinely investable.
    #
    #    This step catches those non-NaN imposters by using the
    #    authoritative live/dead indicator from US_live,csv.csv:
    #      - If live==0 AND the return is already NaN  -> no change
    #      - If live==0 AND the return is a real number -> forced to NaN
    #        (these are the dangerous ones we must remove)
    #
    #    If we left the imposters in:
    #      - Momentum scores would be polluted by fake zero returns.
    #      - Fama-MacBeth regressions would include non-investable
    #        observations, distorting the estimated risk premium.
    #      - Performance statistics would overcount the universe.
    #
    #    By replacing dead cells with NaN:
    #      - NumPy's nan-aware functions (nanmean, nanstd, nanprod)
    #        automatically skip them.
    #      - Pandas .corr(min_periods=...) ignores NaN pairs.
    #      - Our Fama-MacBeth loop explicitly checks for finite values
    #        before including a stock in the cross-sectional regression.
    #
    #    Result: a clean, survivorship-bias-free panel of returns
    #    where only genuinely traded observations participate in the
    #    analysis.
    # ------------------------------------------------------------------
    returns_clean = returns.copy().astype(float)

    T, N = returns.shape
    total_cells = T * N

    # Count dead cells and break them down
    dead_mask = (live == 0)
    n_dead_total = np.sum(dead_mask)
    n_dead_already_nan = np.sum(dead_mask & np.isnan(returns_clean))
    n_dead_had_value = n_dead_total - n_dead_already_nan

    # Also count NaNs that exist in live cells (data gaps in traded stocks)
    live_mask = (live == 1)
    n_live_nan = np.sum(live_mask & np.isnan(returns_clean))
    n_live_valid = np.sum(live_mask & np.isfinite(returns_clean))

    # Now force all dead cells to NaN
    returns_clean[dead_mask] = np.nan

    log.info("  --- Return matrix cleaning report ---")
    log.info(f"  Total cells in TxN matrix     : {total_cells:,}")
    log.info(f"  Live cells  (live==1)          : {np.sum(live_mask):,}  "
             f"({np.sum(live_mask)/total_cells*100:.1f}%)")
    log.info(f"    of which have valid returns  : {n_live_valid:,}")
    log.info(f"    of which are NaN (data gaps) : {n_live_nan:,}")
    log.info(f"  Dead cells  (live==0)          : {n_dead_total:,}  "
             f"({n_dead_total/total_cells*100:.1f}%)")
    log.info(f"    already NaN in raw file      : {n_dead_already_nan:,}  (no action needed)")
    log.info(f"    had a numeric value -> NaN   : {n_dead_had_value:,}  (forced to NaN)")
    if n_dead_had_value > 0:
        log.info(f"  ** {n_dead_had_value:,} stale/fake return values were removed **")

    # ------------------------------------------------------------------
    # Dimension consistency checks
    # ------------------------------------------------------------------
    checks_passed = True
    if returns.shape != live.shape:
        log.error(f"  MISMATCH: returns shape {returns.shape} != live shape {live.shape}")
        checks_passed = False
    if len(dates) != T:
        log.error(f"  MISMATCH: dates length {len(dates)} != T={T}")
        checks_passed = False
    if len(names) != N:
        log.error(f"  MISMATCH: names length {len(names)} != N={N}")
        checks_passed = False
    if ff_factors.shape[0] != T:
        log.error(f"  MISMATCH: FF factors rows {ff_factors.shape[0]} != T={T}")
        checks_passed = False

    if checks_passed:
        log.info("  All dimension checks passed.")
    else:
        log.warning("  Some dimension checks FAILED - see errors above.")

    # ------------------------------------------------------------------
    # Data loading summary (written to log file + console)
    # ------------------------------------------------------------------
    live_pct = np.sum(live == 1) / live.size * 100
    log.info("-" * 60)
    log.info("DATA LOADING SUMMARY")
    log.info("-" * 60)
    log.info(f"  Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Source directory : {os.path.abspath(datadir)}")
    log.info(f"  Number of weeks (T)  : {T}")
    log.info(f"  Number of stocks (N) : {N}")
    log.info(f"  Date range           : {dates[0].strftime('%Y-%m-%d')} to "
             f"{dates[-1].strftime('%Y-%m-%d')}")
    log.info(f"  Files loaded:")
    log.info(f"    US_Returns.csv    : {T} x {N}  (weekly stock returns)")
    log.info(f"    US_live,csv.csv   : {T} x {N}  (live/dead flags)")
    log.info(f"    US_Dates.xlsx     : {T} dates")
    log.info(f"    US_Names.xlsx     : {N} stock names")
    log.info(f"    FamaFrench.csv    : {T} x {ff_factors.shape[1]+1}  "
             f"(Mkt-RF, SMB, HML, RF)")
    log.info(f"  Live observations    : {np.sum(live == 1):,} / {live.size:,} "
             f"({live_pct:.1f}%)")
    log.info(f"  Dead (NaN) obs       : {np.sum(live == 0):,} "
             f"({100 - live_pct:.1f}%)")
    log.info("-" * 60)
    log.info("STEP 1 COMPLETE - all data loaded successfully.")
    log.info("-" * 60)

    # ------------------------------------------------------------------
    # Pack everything into a dictionary and return
    # ------------------------------------------------------------------
    data = {
        'returns':       returns,
        'returns_clean': returns_clean,
        'live':          live,
        'dates':         dates,
        'names':         names,
        'ff_factors':    ff_factors,
        'rf':            rf,
        'T':             T,
        'N':             N,
    }

    return data


# =====================================================================
# Data Exploration Plots
# =====================================================================
# Generates a suite of diagnostic charts that summarise the key
# characteristics of the input dataset from a trading-strategy
# perspective.  All figures are saved to output_data/ as PNG files.
# =====================================================================

def plot_data_overview(data, output_dir='output_data'):
    """
    Creates and saves diagnostic plots about the loaded dataset.

    Plots produced:
        1. Universe size over time (number of live stocks per week)
        2. Live vs dead cell composition (stacked area chart)
        3. Weekly cross-sectional return statistics (mean, median, std)
        4. Distribution of weekly returns (histogram with tail markers)
        5. Missing-data heatmap (fraction of NaN returns per year)
        6. Fama-French factor cumulative returns
        7. Average stock lifespan distribution
        8. Summary statistics text panel
    """

    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 60)
    log.info("GENERATING DATA EXPLORATION PLOTS")
    log.info("=" * 60)

    returns_clean = data['returns_clean']
    live          = data['live']
    dates         = data['dates']
    ff_factors    = data['ff_factors']
    rf            = data['rf']
    T             = data['T']
    N             = data['N']

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # ------------------------------------------------------------------
    # PLOT 1: Number of live (tradeable) stocks per week
    # ------------------------------------------------------------------
    log.info("  Plot 1: Universe size over time...")
    n_live_per_week = np.sum(live == 1, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, n_live_per_week, color='#2563EB', linewidth=1.2)
    ax.fill_between(dates, 0, n_live_per_week, alpha=0.15, color='#2563EB')
    ax.set_title('Number of Live (Tradeable) Stocks per Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.annotate(f'Max: {np.max(n_live_per_week):,} stocks',
                xy=(dates[np.argmax(n_live_per_week)], np.max(n_live_per_week)),
                xytext=(30, -20), textcoords='offset points',
                fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
    ax.annotate(f'Min: {np.min(n_live_per_week):,} stocks',
                xy=(dates[np.argmin(n_live_per_week)], np.min(n_live_per_week)),
                xytext=(30, 20), textcoords='offset points',
                fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot1_universe_size.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot1_universe_size.png")

    # ------------------------------------------------------------------
    # PLOT 2: Live vs Dead cells - stacked area
    # ------------------------------------------------------------------
    log.info("  Plot 2: Live vs Dead composition over time...")
    n_dead_per_week = N - n_live_per_week

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(dates, n_live_per_week, n_dead_per_week,
                 labels=['Live (tradeable)', 'Dead (delisted / pre-IPO)'],
                 colors=['#22C55E', '#EF4444'], alpha=0.7)
    ax.set_title('Live vs Dead Stock Observations per Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(loc='upper left', fontsize=10)
    total_live = np.sum(live == 1)
    total_dead = np.sum(live == 0)
    ax.text(0.98, 0.05,
            f'Total: {total_live:,} live ({total_live/(T*N)*100:.1f}%) | '
            f'{total_dead:,} dead ({total_dead/(T*N)*100:.1f}%)',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot2_live_vs_dead.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot2_live_vs_dead.png")

    # ------------------------------------------------------------------
    # PLOT 3: Weekly cross-sectional return statistics
    # ------------------------------------------------------------------
    log.info("  Plot 3: Weekly cross-sectional return statistics...")
    cs_mean   = np.nanmean(returns_clean, axis=1)
    cs_median = np.nanmedian(returns_clean, axis=1)
    cs_std    = np.nanstd(returns_clean, axis=1, ddof=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel A: mean and median
    axes[0].plot(dates, cs_mean * 100, color='#2563EB', linewidth=0.8, label='Mean', alpha=0.85)
    axes[0].plot(dates, cs_median * 100, color='#F97316', linewidth=0.8, label='Median', alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Weekly Cross-Sectional Return: Mean & Median', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Panel B: standard deviation
    axes[1].plot(dates, cs_std * 100, color='#DC2626', linewidth=0.8, alpha=0.85)
    axes[1].set_ylabel('Std Dev (%)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Weekly Cross-Sectional Return Volatility', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot3_return_statistics.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot3_return_statistics.png")

    # ------------------------------------------------------------------
    # PLOT 4: Distribution of all valid weekly returns (histogram)
    # ------------------------------------------------------------------
    log.info("  Plot 4: Return distribution histogram...")
    all_rets = returns_clean[np.isfinite(returns_clean)]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Clip for display to avoid extreme outliers dominating the histogram
    clip_lo, clip_hi = np.percentile(all_rets, [0.5, 99.5])
    clipped = all_rets[(all_rets >= clip_lo) & (all_rets <= clip_hi)]
    ax.hist(clipped * 100, bins=200, color='#6366F1', alpha=0.75, edgecolor='none')
    ax.axvline(np.mean(all_rets) * 100, color='red', linestyle='--', linewidth=1.2,
               label=f'Mean = {np.mean(all_rets)*100:.3f}%')
    ax.axvline(np.median(all_rets) * 100, color='orange', linestyle='--', linewidth=1.2,
               label=f'Median = {np.median(all_rets)*100:.3f}%')
    ax.set_title('Distribution of Weekly Stock Returns (0.5th-99.5th percentile)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Weekly Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=10)

    # Add skewness / kurtosis annotation
    from scipy.stats import skew, kurtosis as kurt_fn
    try:
        sk = skew(all_rets)
        ku = kurt_fn(all_rets)
        ax.text(0.97, 0.95,
                f'N = {len(all_rets):,}\nSkew = {sk:.2f}\nExcess Kurt = {ku:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))
    except ImportError:
        ax.text(0.97, 0.95, f'N = {len(all_rets):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot4_return_distribution.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot4_return_distribution.png")

    # ------------------------------------------------------------------
    # PLOT 5: Missing data heatmap - fraction of NaN per year
    # ------------------------------------------------------------------
    log.info("  Plot 5: Missing data by year...")
    years = dates.year
    unique_years = np.sort(np.unique(years))

    nan_frac_by_year = []
    live_frac_by_year = []
    for yr in unique_years:
        mask_yr = (years == yr)
        ret_yr = returns_clean[mask_yr, :]
        live_yr = live[mask_yr, :]
        nan_frac_by_year.append(np.mean(np.isnan(ret_yr)) * 100)
        live_frac_by_year.append(np.mean(live_yr == 1) * 100)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(unique_years))
    width = 0.38

    bars1 = ax1.bar(x - width/2, live_frac_by_year, width,
                    label='Live cells (%)', color='#22C55E', alpha=0.8)
    bars2 = ax1.bar(x + width/2, nan_frac_by_year, width,
                    label='NaN cells (%)', color='#EF4444', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_years, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Percentage of Cells (%)')
    ax1.set_title('Data Availability by Year: Live Cells vs NaN Cells',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot5_missing_data_by_year.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot5_missing_data_by_year.png")

    # ------------------------------------------------------------------
    # PLOT 6: Fama-French factor cumulative returns
    # ------------------------------------------------------------------
    log.info("  Plot 6: Fama-French factor cumulative returns...")
    factor_names = ['Mkt-RF', 'SMB', 'HML']
    colors_ff = ['#2563EB', '#F97316', '#22C55E']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (fname, col) in enumerate(zip(factor_names, colors_ff)):
        cum_ret = np.cumprod(1 + ff_factors[:, i]) - 1
        ax.plot(dates, cum_ret * 100, label=fname, color=col, linewidth=1.2)

    # Also plot risk-free cumulative return
    cum_rf = np.cumprod(1 + rf) - 1
    ax.plot(dates, cum_rf * 100, label='RF (risk-free)', color='grey',
            linewidth=1.0, linestyle='--')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Fama-French Factor Cumulative Returns (weekly)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot6_ff_cumulative_returns.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot6_ff_cumulative_returns.png")

    # ------------------------------------------------------------------
    # PLOT 7: Stock lifespan distribution (how many weeks each stock is live)
    # ------------------------------------------------------------------
    log.info("  Plot 7: Stock lifespan distribution...")
    weeks_live_per_stock = np.sum(live == 1, axis=0)  # shape (N,)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weeks_live_per_stock, bins=80, color='#8B5CF6', alpha=0.75, edgecolor='none')
    ax.axvline(np.mean(weeks_live_per_stock), color='red', linestyle='--', linewidth=1.2,
               label=f'Mean = {np.mean(weeks_live_per_stock):.0f} weeks')
    ax.axvline(np.median(weeks_live_per_stock), color='orange', linestyle='--', linewidth=1.2,
               label=f'Median = {np.median(weeks_live_per_stock):.0f} weeks')
    ax.set_title('Distribution of Stock Lifespans (weeks listed)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Weeks Live')
    ax.set_ylabel('Number of Stocks')
    ax.legend(fontsize=10)

    # Annotate how many stocks are live the entire sample
    n_full_sample = np.sum(weeks_live_per_stock == T)
    ax.text(0.97, 0.85,
            f'Total stocks: {N:,}\n'
            f'Full-sample survivors: {n_full_sample:,}\n'
            f'({n_full_sample/N*100:.1f}% of universe)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot7_stock_lifespan.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot7_stock_lifespan.png")

    # ------------------------------------------------------------------
    # PLOT 8: Summary statistics text panel
    # ------------------------------------------------------------------
    log.info("  Plot 8: Summary statistics panel...")
    all_valid = returns_clean[np.isfinite(returns_clean)]
    n_live_total = np.sum(live == 1)
    n_dead_total = np.sum(live == 0)

    summary_lines = [
        f"Dataset: US Weekly Stock Returns",
        f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
        f"",
        f"Dimensions:",
        f"  Weeks (T)  = {T:,}",
        f"  Stocks (N) = {N:,}",
        f"  Total cells = {T*N:,}",
        f"",
        f"Coverage:",
        f"  Live observations   = {n_live_total:,}  ({n_live_total/(T*N)*100:.1f}%)",
        f"  Dead observations   = {n_dead_total:,}  ({n_dead_total/(T*N)*100:.1f}%)",
        f"  Valid return values  = {len(all_valid):,}",
        f"",
        f"Universe (per week):",
        f"  Mean live stocks  = {np.mean(n_live_per_week):,.0f}",
        f"  Max live stocks   = {np.max(n_live_per_week):,}  "
        f"(week {dates[np.argmax(n_live_per_week)].strftime('%Y-%m-%d')})",
        f"  Min live stocks   = {np.min(n_live_per_week):,}  "
        f"(week {dates[np.argmin(n_live_per_week)].strftime('%Y-%m-%d')})",
        f"",
        f"Return Statistics (weekly, all valid obs):",
        f"  Mean   = {np.mean(all_valid)*100:+.4f}%",
        f"  Median = {np.median(all_valid)*100:+.4f}%",
        f"  Std    = {np.std(all_valid, ddof=1)*100:.4f}%",
        f"  Min    = {np.min(all_valid)*100:.2f}%",
        f"  Max    = {np.max(all_valid)*100:.2f}%",
        f"",
        f"Fama-French Factors (annualised mean, weekly x 52):",
        f"  Mkt-RF = {np.mean(ff_factors[:,0])*52*100:+.2f}%",
        f"  SMB    = {np.mean(ff_factors[:,1])*52*100:+.2f}%",
        f"  HML    = {np.mean(ff_factors[:,2])*52*100:+.2f}%",
        f"  RF     = {np.mean(rf)*52*100:+.2f}%",
    ]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    ax.text(0.05, 0.95, '\n'.join(summary_lines),
            transform=ax.transAxes, fontsize=10.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#F8FAFC', ec='#CBD5E1'))
    ax.set_title('Data Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot8_summary_statistics.png'), dpi=150)
    plt.close(fig)
    log.info("    Saved plot8_summary_statistics.png")

    log.info("-" * 60)
    log.info(f"All 8 data exploration plots saved to {os.path.abspath(output_dir)}/")
    log.info("-" * 60)


# =====================================================================
# Run as standalone script: python data_loader.py
# =====================================================================
# When executed directly, this loads all data, prints the full log to
# the console, and saves the log to output_data/data_loading.log.
# This lets you (or the future front-end app) verify that all input
# files are present and correctly shaped BEFORE running the full
# momentum_strategy.py pipeline.
# =====================================================================

if __name__ == '__main__':
    data = load_all_data(datadir='input_data/')

    # Generate data exploration plots
    plot_data_overview(data, output_dir='output_data')

    # Quick sanity print so the user sees the key variables
    print("\nData dictionary keys:", list(data.keys()))
    print(f"returns_clean shape: {data['returns_clean'].shape}")
    print(f"live shape:          {data['live'].shape}")
    print(f"dates length:        {len(data['dates'])}")
    print(f"names length:        {len(data['names'])}")
    print(f"ff_factors shape:    {data['ff_factors'].shape}")
    print(f"rf shape:            {data['rf'].shape}")
    print(f"\nLog file written to: output_data/data_loading.log")
    print(f"Plots saved to:      output_data/plot1_..plot8_*.png")
