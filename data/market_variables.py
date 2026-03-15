# market_variables.py
# =====================================================================
# Market Return (MRET) and Market Volatility (MVOL)
# =====================================================================
#
# Computes two market-level variables used in Table I of
# Lou & Polk (2021) for summary statistics and correlations
# with comomentum:
#
#   MRET  — trailing 2-year (104-week) compounded market return
#   MVOL  — trailing 2-year standard deviation of monthly market returns
#
# These variables are NOT used in the trading strategy itself.
# They are descriptive statistics showing that comomentum is
# negatively correlated with market returns and positively
# correlated with market volatility.
#
# INPUT:
#   ff_factors : (T, 3) np.ndarray — [Mkt-RF, SMB, HML] weekly
#   rf         : (T,)   np.ndarray — weekly risk-free rate
#   dates      : T-length DatetimeIndex
#
# OUTPUT:
#   mret : (T,) np.ndarray — trailing 104-week compounded market return
#   mvol : (T,) np.ndarray — trailing 24-month market return volatility
#
# Standalone:  python market_variables.py
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from logger_setup import _setup_logger
from config import WEEKS_PER_YEAR

log = _setup_logger()

MRET_WINDOW = 2 * WEEKS_PER_YEAR   # 104 weeks (2 years)
MVOL_MONTHS = 24                     # 24 months (2 years)


def compute_market_variables(ff_factors, rf, dates):
    """
    Compute trailing 2-year market return (MRET) and trailing 2-year
    market volatility (MVOL) from weekly Fama-French data.

    Parameters
    ----------
    ff_factors : (T, 3) np.ndarray — [Mkt-RF, SMB, HML]
    rf         : (T,)   np.ndarray — weekly risk-free rate
    dates      : T-length DatetimeIndex

    Returns
    -------
    mret : (T,) np.ndarray — trailing 104-week compounded market return
    mvol : (T,) np.ndarray — trailing 24-month std dev of monthly returns
    """
    T = len(rf)
    dates = pd.DatetimeIndex(dates)

    # Total weekly market return = MktRF + RF
    mkt_weekly = ff_factors[:, 0] + rf

    log.info("=" * 60)
    log.info("MARKET VARIABLES: MRET and MVOL")
    log.info("=" * 60)
    log.info(f"  Weekly market return series: {T} weeks")
    log.info(f"  MRET window: {MRET_WINDOW} weeks (2 years)")
    log.info(f"  MVOL window: {MVOL_MONTHS} months (2 years)")

    # ── MRET: trailing 104-week compounded return ────────────────────
    mret = np.full(T, np.nan)
    gross = 1.0 + mkt_weekly

    for t in range(MRET_WINDOW - 1, T):
        window = gross[t - MRET_WINDOW + 1 : t + 1]
        if np.all(np.isfinite(window)):
            mret[t] = np.prod(window) - 1.0

    n_mret = int(np.sum(np.isfinite(mret)))
    log.info(f"  MRET: {n_mret} valid values")
    if n_mret > 0:
        v = mret[np.isfinite(mret)]
        log.info(f"    mean={np.mean(v):.4f}, std={np.std(v):.4f}, "
                 f"min={np.min(v):.4f}, max={np.max(v):.4f}")

    # ── MVOL: trailing 24-month std dev of monthly returns ───────────
    # Step 1: compound weekly returns into monthly returns
    df = pd.DataFrame({'date': dates, 'mkt': mkt_weekly}).set_index('date')
    monthly = df['mkt'].resample('ME').apply(
        lambda x: np.prod(1.0 + x.values) - 1.0 if len(x) > 0 else np.nan
    )

    # Step 2: rolling 24-month standard deviation
    mvol_monthly = monthly.rolling(window=MVOL_MONTHS, min_periods=MVOL_MONTHS).std()

    # Step 3: map back to weekly frequency (each week gets its month's value)
    mvol = np.full(T, np.nan)
    week_months = dates.to_period('M')
    mvol_dict = {}
    for dt, val in mvol_monthly.items():
        if np.isfinite(val):
            mvol_dict[dt.to_period('M')] = val

    for t in range(T):
        m = week_months[t]
        if m in mvol_dict:
            mvol[t] = mvol_dict[m]

    n_mvol = int(np.sum(np.isfinite(mvol)))
    log.info(f"  MVOL: {n_mvol} valid values")
    if n_mvol > 0:
        v = mvol[np.isfinite(mvol)]
        log.info(f"    mean={np.mean(v):.4f}, std={np.std(v):.4f}, "
                 f"min={np.min(v):.4f}, max={np.max(v):.4f}")

    log.info("=" * 60)
    log.info("MARKET VARIABLES COMPLETE")
    log.info("=" * 60)

    return mret, mvol


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data

    data = load_all_data('input_data/')
    mret, mvol = compute_market_variables(
        data['ff_factors'], data['rf'], data['dates']
    )

    print(f"\nMRET: {np.sum(np.isfinite(mret))} valid values")
    print(f"  mean={np.nanmean(mret):.4f}, std={np.nanstd(mret):.4f}")
    print(f"\nMVOL: {np.sum(np.isfinite(mvol))} valid values")
    print(f"  mean={np.nanmean(mvol):.4f}, std={np.nanstd(mvol):.4f}")
