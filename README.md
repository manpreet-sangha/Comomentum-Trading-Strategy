# Momentum Trading Strategy
### SMM282 Quantitative Trading — Coursework 2026
**Enhancing a Standard Momentum Factor using Lou & Polk (2021) Comomentum**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub repo](https://img.shields.io/badge/GitHub-momentum__trading__strategy-black?logo=github)](https://github.com/manpreet-sangha/momentum_trading_strategy)

---

## Project Overview

This project implements and enhances a standard equity momentum trading strategy
using the comomentum measure introduced by Lou & Polk (2021). The core idea is that
crowded momentum trades (high comomentum) predict weaker future momentum returns,
and we can improve performance by scaling down momentum bets when crowding is high.

---

## Project Structure

```
momentum_strategy.py      Main pipeline / entry point (runs Steps 1-7)
data_loader.py            Loads and validates all input data files
momentum_factor.py        Computes momentum, comomentum, adjusted momentum
standardiseFactor.py      Cross-sectional z-score standardisation utility
fama_macbeth.py           Fama-MacBeth cross-sectional regression engine
performance.py            Summary statistics and charting utilities
input_data/               Raw input files
output_data/              Generated outputs (CSVs, plots, log files)
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load and validate all input data |
| 2 | Compute standard (baseline) momentum factor |
| 3 | Run Fama-MacBeth regressions on standard momentum |
| 4 | Compute comomentum measure (Lou & Polk, 2021) |
| 5 | Adjust momentum using inverse comomentum signal |
| 6 | Re-run Fama-MacBeth on adjusted momentum |
| 7 | Compare results: plots and summary statistics |

---

## Input Data

| File | Description |
|------|-------------|
| `US_Returns.csv` | T x N matrix of weekly stock returns (T=1513 weeks, N=7261 stocks) |
| `US_live,csv.csv` | T x N matrix of live/dead dummies (1 = stock is listed, 0 = delisted) |
| `US_Dates.xlsx` | T x 1 vector of weekly date labels (1992-01-03 to 2020-12-25) |
| `US_Names.xlsx` | 1 x N vector of stock ticker names |
| `FamaFrench.csv` | T x 4 matrix of Fama-French factors (Mkt-RF, SMB, HML, RF) |

---

## Momentum Factor Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lookback` | 48 weeks | Width of the return window, **including the current week** |
| `skip` | 4 weeks | Latest 4 weeks of the dataset are excluded **once globally** |

### Momentum Factor Date Range

| Event | Week Index | Date |
|-------|-----------|------|
| Dataset start | 0 | 1992-01-03 |
| Dataset end | 1512 | 2020-12-25 |
| Skipped weeks | 1509–1512 | 2020-12-04 to 2020-12-25 |
| **1st momentum factor** | **1508** | **2020-11-27** — lookback [1461..1508] |
| 2nd momentum factor | 1507 | 2020-11-20 — lookback [1460..1507] |
| ... | ... | ... |
| **Last momentum factor** | **47** | **1992-11-27** — lookback [0..47] |
| **Total weeks with factors** | **1462** | weeks 47 to 1508 |

---

## Terminology

Three related but distinct terms are used throughout this project:

### 1. Raw Momentum Factor
The 48-week compounded return for each stock at each week:

```
mom_{i,t} = prod(1 + r_{i,s}  for s in [t-47, t]) - 1
```

- Computed in `momentum_factor.compute_momentum()`
- Variable name: `momentum`  —  Shape: T x N

### 2. Standardised Momentum Factor
The **cross-sectional z-score** of the raw momentum factor:

```
z_{i,t} = (mom_{i,t} - mean_t) / std_t
```

where `mean_t` and `std_t` are computed across all N stocks at week t.

- Computed in `standardiseFactor.standardiseFactor()`
- Variable name: `momentum_std`  —  Shape: T x N
- At each week t: cross-sectional mean = 0, std = 1
- **"Standardised" = z-scored (a statistical operation)**

### 3. Standard Momentum Factor
Refers to the **baseline / conventional** momentum strategy
(Jegadeesh & Titman, 1993) **before** any comomentum adjustment.
It encompasses both the raw factor and its standardised form.

- **"Standard" = conventional/unadjusted (a strategic distinction)**
- This is NOT the same as "standardised"

### 4. Adjusted Momentum Factor
The standardised momentum factor scaled by an inverse comomentum signal
(Lou & Polk, 2021) and then re-standardised cross-sectionally.

- Computed in `momentum_factor.compute_adjusted_momentum()`
- Variable name: `momentum_adj_std`  —  Shape: T x N

### Summary Table

| Term | Statistical meaning | Variable |
|------|-------------------|----------|
| Raw momentum factor | 48-week compounded return | `momentum` |
| **Standardised** momentum factor | z-score of raw (mean=0, std=1) | `momentum_std` |
| **Standard** momentum factor | Baseline strategy (before adjustment) | `momentum` + `momentum_std` |
| Adjusted momentum factor | Comomentum-scaled & re-standardised | `momentum_adj_std` |

---

## Why Standardise (z-score) the Momentum Factor?

### 1. Comparability Across Time
Raw momentum values (cumulative returns) vary widely in magnitude across market
regimes. During volatile markets the spread of raw momentum is large; in calm
periods it is small. Standardising ensures that an exposure of +1.0 always means
"one standard deviation above the cross-sectional average" regardless of the
market environment.

### 2. Fama-MacBeth Regression Interpretation
In Fama-MacBeth cross-sectional regressions (Step 3):

```
r_{i,t+1} = alpha_t + gamma_t * z_{i,t} + epsilon_{i,t+1}
```

When `z` is standardised, the slope `gamma_t` has a clean interpretation: the
expected return difference per week between a stock that is **one standard
deviation above** average momentum and an average-momentum stock. Without
standardisation, `gamma` would mix the momentum effect with the time-varying
scale of raw exposures.

### 3. Portfolio Construction
Standardised exposures enable dollar-neutral long/short portfolios with consistent
leverage. A stock with z = +2 gets twice the weight of z = +1, ensuring weights
are proportional to **relative momentum rank**, not to raw cumulative return levels.

### 4. Comomentum Adjustment
When multiplying the standardised momentum by a time-varying scaling factor
(inverse comomentum signal in Step 5), the adjustment operates on a **unit-free
quantity**, so the scaling has a uniform effect across all weeks.

### 5. NaN Handling
`nanmean` and `nanstd` ignore NaN entries (dead/delisted stocks), so the
standardisation is computed only over stocks that are alive and have a valid
momentum factor at each week. Dead stocks remain NaN after standardisation.

---

## Comomentum (Lou & Polk, 2021)

Comomentum measures how correlated the **abnormal returns** of momentum stocks
(winners and losers) are with each other. High comomentum signals crowded momentum
trades, which predicts **weaker** future momentum returns. Low comomentum signals
less crowding and predicts **stronger** future momentum returns.

### Procedure (at each week t):
1. Identify momentum stocks: top and bottom quintile by standardised momentum
2. For each momentum stock, regress its last 52 weeks of returns on Fama-French
   3 factors to obtain abnormal return residuals
3. Compute pairwise correlations of residuals across all momentum stocks
4. Comomentum = average of all upper-triangle pairwise correlations

### Adjustment:
```
scaling_t = 2.0 - percentile_rank(comomentum_{t-1})
```
- Low comomentum (rank ~ 0) → scaling ~ 2.0 → increase momentum bet
- High comomentum (rank ~ 1) → scaling ~ 1.0 → reduce momentum bet

The percentile rank uses an **expanding window** to avoid look-ahead bias.

---

## Getting Started

### Prerequisites

```
Python >= 3.11
numpy
pandas
matplotlib
scipy
openpyxl
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

### Input Data

Place the following files in the `input_data/` folder before running
(files are excluded from the repo due to size — obtain from course materials):

| File | Description |
|------|-------------|
| `US_Returns.csv` | T×N weekly stock returns |
| `US_live,csv.csv` | T×N live/dead dummies |
| `US_Dates.xlsx` | T×1 weekly date labels |
| `US_Names.xlsx` | 1×N stock ticker names |
| `FamaFrench.csv` | T×4 Fama-French factors |

### Run

```bash
python momentum_strategy.py
```

All outputs (plots, CSVs, logs) are written to `output_data/`.

---

## Pipeline Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load & validate all input data | ✅ Complete |
| 2 | Compute standard momentum factor | ✅ Complete |
| 3 | Fama-MacBeth on standard momentum | ⏳ In progress |
| 4 | Compute comomentum (Lou & Polk, 2021) | ⏳ In progress |
| 5 | Adjust momentum using inverse comomentum | ⏳ In progress |
| 6 | Fama-MacBeth on adjusted momentum | ⏳ In progress |
| 7 | Comparison plots & summary statistics | ⏳ In progress |

---

## Output Files

| File | Generated by | Description |
|------|-------------|-------------|
| `output_data/data_loading.log` | `data_loader.py` | Data validation log |
| `output_data/momentum_factor.log` | `momentum_factor.py` | Factor computation log |
| `output_data/plot1_universe_size.png` | `data_loader.py` | Stock universe over time |
| `output_data/plot2_live_vs_dead.png` | `data_loader.py` | Live vs delisted stocks |
| `output_data/plot3_return_statistics.png` | `data_loader.py` | Return stats over time |
| `output_data/plot4_return_distribution.png` | `data_loader.py` | Return distribution |
| `output_data/plot5_missing_data_by_year.png` | `data_loader.py` | Missing data heatmap |
| `output_data/plot6_ff_cumulative_returns.png` | `data_loader.py` | FF factor cumulative returns |
| `output_data/plot7_stock_lifespan.png` | `data_loader.py` | Stock lifespan histogram |
| `output_data/plot8_summary_statistics.png` | `data_loader.py` | Summary statistics table |
| `output_data/step2_scatter_momentum_vs_return.png` | `momentum_strategy.py` | Momentum vs next-week return |
| `output_data/step2_histogram_momentum.png` | `momentum_strategy.py` | Momentum factor distribution |
| `output_data/step2_factor_comparison.png` | `momentum_strategy.py` | 4-panel factor comparison |
| `output_data/momentum_raw_sample.csv` | `momentum_strategy.py` | Sample of raw momentum matrix |
| `output_data/momentum_standardised_sample.csv` | `momentum_strategy.py` | Sample of standardised matrix |
| `output_data/momentum_summary.csv` | `momentum_strategy.py` | Cross-sectional summary stats |

---

## Logging

Each module writes a log file to `output_data/` (overwritten each run):

| Module | Log file |
|--------|----------|
| `data_loader.py` | `output_data/data_loading.log` |
| `momentum_factor.py` | `output_data/momentum_factor.log` |

Logs are also printed to the console. Format:
```
YYYY-MM-DD HH:MM:SS | INFO    | message
```

---

## References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*.
  Journal of Finance, 48(1), 65–91.
- Lou, D. & Polk, C. (2021). *Comomentum: Inferring Arbitrage Activity from Return
  Correlations*. Review of Financial Studies, 35(7), 3272–3302.
- Fama, E. & MacBeth, J. (1973). *Risk, Return, and Equilibrium: Empirical Tests*.
  Journal of Political Economy, 81(3), 607–636.
