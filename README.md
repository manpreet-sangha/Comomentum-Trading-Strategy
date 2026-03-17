# Momentum Trading Strategy

### SMM282 Quantitative Trading — Coursework 2026

**Enhancing a Standard Momentum Factor using Lou & Polk (2021) Comomentum**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub repo](https://img.shields.io/badge/GitHub-Comomentum--Trading--Strategy-black?logo=github)](https://github.com/manpreet-sangha/Comomentum-Trading-Strategy)

---

## Project Overview

This project implements and enhances a standard equity momentum trading strategy
using the **comomentum** measure introduced by Lou & Polk (2021). The core idea is
that crowded momentum trades (high comomentum) predict weaker future momentum
returns, and performance can be improved by scaling down momentum bets when
crowding is high.

**Key features:**

- Modular pipeline — each step is a standalone, reusable `.py` module
- True **pairwise abnormal correlations** (paper-accurate, not leave-one-out)
- Rolling 52-week FF3 residual estimation (time-varying betas per Lewellen & Nagel, 2006)
- Fama-MacBeth regressions with Excel output (Documentation + Data sheets)
- Event-study plot replicating Figure 2 from the paper
- Diagnostic exports: FF3 residuals and pairwise correlation matrices to Excel

---

## Project Structure

### Core Pipeline

| File | Description |
|------|-------------|
| `momentum_strategy.py` | **Main entry point** — orchestrates Steps 1–6 |
| `config.py` | Single source of truth for all tuneable parameters |
| `data_loader.py` | Step 1: Loads and validates all input data files |
| `stock_diagnostics.py` | Step 1b: Flags short-lived & gapped stocks |
| `clean_returns.py` | Return cleaning utilities |
| `compute_momentum_signal.py` | Step 2: Rolling momentum (48w lookback, 4w skip) |
| `compute_comomentum.py` | Step 4: Comomentum via pairwise abnormal correlations |
| `compute_adjusted_momentum.py` | Step 5: Inverse-comomentum scaling of factor returns |
| `standardiseFactor.py` | Cross-sectional z-score utility |
| `fama_macbeth.py` | Fama-MacBeth cross-sectional regression engine |
| `performance.py` | Summary statistics and comparison charting |
| `logger_setup.py` | Centralised logging configuration |

### Output & Diagnostic Modules

| File | Description |
|------|-------------|
| `step2_outputs.py` | Saves momentum CSVs (raw, standardised, summary) |
| `step2_plots.py` | Step 2 diagnostic charts (scatter, histogram, 4-panel) |
| `save_ff3_residuals.py` | Exports FF3 residuals snapshot to Excel |
| `save_pairwise_correlations.py` | Exports pairwise correlation matrices to Excel |
| `plot_comom_event_study.py` | Two-panel event-study plot (Figure 2 from paper) |

### Data-Loading Helpers

| File | Description |
|------|-------------|
| `read_returns.py` | Reads `US_Returns.csv` |
| `read_live.py` | Reads `US_live,csv.csv` |
| `read_dates.py` | Reads `US_Dates.xlsx` |
| `read_names.py` | Reads `US_Names.xlsx` |
| `read_fama_french.py` | Reads `FamaFrench.csv` |

### Exploratory / Diagnostic Plots

| File | Description |
|------|-------------|
| `plot1_universe_size.py` | Stock universe size over time |
| `plot2_listed_vs_notlisted.py` | Listed vs not-listed stocks |
| `plot3_return_statistics.py` | Return statistics over time |
| `plot4_return_distribution.py` | Return distribution |
| `plot5_missing_data.py` | Missing data heatmap |
| `plot6_ff_cumulative.py` | FF factor cumulative returns |
| `plot_cleaning_impact.py` | Before/after cleaning comparison |
| `plot_dimensions.py` | Data dimension checks |
| `plot_loading_summary.py` | Loading summary dashboard |
| `dataplots.py` | General exploratory plots |
| `exploration_plots.py` | Additional exploration |

### Other

| File | Description |
|------|-------------|
| `dimension_checks.py` | Data dimension validation |
| `momentum_schedule.py` | Momentum computation schedule |
| `momentum_factor.py` | Legacy momentum module (superseded by pipeline) |
| `solveCAPMExercise.py` | CAPM exercise (course reference) |
| `solveFamaMacBethExercise.py` | FM exercise (course reference) |
| `solveFamaMacBethMultiExercise.py` | Multi-factor FM exercise |
| `test_clean_returns.py` | Unit test: return cleaning |
| `test_nan_after_cleaning.py` | Unit test: NaN handling |
| `test_zero_returns.py` | Unit test: zero returns |
| `_diag_minpct.py` | Diagnostic: minimum percentile tuning |
| `_diag_weeks.py` | Diagnostic: weeks tuning |

---

## Pipeline Steps

| Step | Description | Module | Status |
|------|-------------|--------|--------|
| 1 | Load & validate all input data | `data_loader.py` | ✅ Complete |
| 1b | Flag & exclude short-lived stocks (< 52 weeks) | `stock_diagnostics.py` | ✅ Complete |
| 2 | Compute standard momentum factor (48w LB, 4w rolling skip) | `compute_momentum_signal.py` | ✅ Complete |
| 3 | Fama-MacBeth regressions on standard momentum | `fama_macbeth.py` | ✅ Complete |
| 4 | Compute comomentum — pairwise abnormal correlations | `compute_comomentum.py` | ✅ Complete |
| 5 | Adjust momentum: scale factor returns by inverse comomentum | `compute_adjusted_momentum.py` | ✅ Complete |
| 6 | Compare Standard vs Adjusted: summary stats & plots | `performance.py` | ✅ Complete |

---

## Configuration Parameters (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LOOKBACK` | 48 | Weeks of compounded returns (≈ 11 months) |
| `SKIP` | 4 | Most recent weeks excluded (short-term reversal) |
| `TOTAL` | 52 | `LOOKBACK + SKIP` — weeks of history before first score |
| `CORR_WINDOW` | 52 | Rolling FF3 regression window (weeks) |
| `DECILE_PCT_LO` | 10 | Bottom 10% = extreme loser decile |
| `DECILE_PCT_HI` | 90 | Top 10% = extreme winner decile |
| `MIN_PAST_VALUES` | 10 | Min past comomentum values for percentile rank |
| `WEEKS_PER_YEAR` | 52 | Annualisation factor |
| `MIN_RESID_OBS` | None | Reserved — uses `CORR_WINDOW` as threshold |
| `MIN_STOCKS` | None | Reserved — accepts any decile size ≥ 2 |

---

## Input Data

| File | Description |
|------|-------------|
| `US_Returns.csv` | T × N matrix of weekly stock returns (T=1,513 weeks, N=7,261 stocks) |
| `US_live,csv.csv` | T × N live/dead indicator (1 = listed, 0 = not listed) |
| `US_Dates.xlsx` | T × 1 weekly date labels (1992-01-03 to 2020-12-25) |
| `US_Names.xlsx` | 1 × N stock ticker names |
| `FamaFrench.csv` | T × 4 Fama-French factors (Mkt-RF, SMB, HML, RF) |

---

## Methodology

### Momentum Signal (Step 2)

At each week *t*, for each stock *i*:

```
mom_{i,t} = ∏(1 + r_{i,s}  for s in [t-51, t-4]) − 1
```

- **48-week lookback** starting 52 weeks back, skipping the most recent 4 weeks
- The skip is **rolling** — always the 4 weeks immediately before *t*
- A stock must have all 52 weekly returns present to receive a score
- First momentum score: week index 51 (1992-12-25)
- 1,462 scored weeks (indices 51–1512)
- Cross-sectionally z-scored → `momentum_std`

### Comomentum (Step 4) — Lou & Polk (2021)

At each week *t*:

1. **Sort** all live stocks with valid momentum into deciles (top/bottom 10%)
2. **FF3 residuals**: For each loser/winner stock, regress its last 52 weeks of
   returns on Mkt-RF, SMB, HML using OLS. Collect residuals (abnormal returns).
   Betas vary over time (rolling 52-week window, per Lewellen & Nagel 2006).
3. **Pairwise correlations**: For each decile, compute the full K × K
   correlation matrix of FF3 residuals using `np.corrcoef`. Extract all
   K(K−1)/2 unique upper-triangle pairs.
4. **Decile comomentum**: Average of all pairwise correlations
5. **Comomentum**: CoMOM = 0.5 × (CoMOM\_Winners + CoMOM\_Losers)

### Adjusted Momentum (Step 5)

The **factor returns** (Fama-MacBeth γ) are scaled — NOT the exposures:

```
scaling_t = 2.0 − percentile_rank(comomentum_{t−1})
gamma_adj_t = gamma_std_t × scaling_t
```

- Low comomentum (rank ≈ 0) → scaling ≈ 2.0 → increase momentum bet
- High comomentum (rank ≈ 1) → scaling ≈ 1.0 → reduce momentum bet
- Percentile rank uses an **expanding window** (no look-ahead bias)

### Why Scale Returns, Not Exposures?

Scaling all stocks' exposures by the same time-varying factor and then
re-standardising cross-sectionally perfectly undoes the scaling (since z-scoring
removes any constant multiplier). The comomentum signal must therefore be applied
to the **factor return series** (γ\_t), where it genuinely modulates the strategy's
time-varying bet size.

---

## Terminology

| Term | Meaning | Variable |
|------|---------|----------|
| Raw momentum | 48-week compounded return | `momentum` |
| Standardised momentum | Cross-sectional z-score (mean=0, std=1) | `momentum_std` |
| Standard momentum | Baseline strategy (before comomentum adjustment) | `gamma_std` |
| Adjusted momentum | Factor returns scaled by inverse comomentum | `gamma_adj` |

- **"Standardised"** = z-scored (a statistical operation)
- **"Standard"** = conventional/unadjusted (a strategic distinction)

---

## Output Files

### Excel Reports

| File | Module | Description |
|------|--------|-------------|
| `fama_macbeth_standard_momentum.xlsx` | `fama_macbeth.py` | FM regression results (Documentation + Data sheets) |
| `ff3_residuals.xlsx` | `save_ff3_residuals.py` | FF3 residuals snapshot (Loser + Winner sheets) |
| `pairwise_correlations.xlsx` | `save_pairwise_correlations.py` | Correlation matrices + long-format pairs |
| `stocks_short_lived.xlsx` | `stock_diagnostics.py` | Short-lived stocks flagged |
| `stocks_with_trading_gaps.xlsx` | `stock_diagnostics.py` | Gapped stocks flagged |
| `combined_data_verification.xlsx` | `stock_diagnostics.py` | Combined diagnostics |

### Plots

| File | Module | Description |
|------|--------|-------------|
| `plot_comom_event_study.png` | `plot_comom_event_study.py` | Two-panel event study (Figure 2) |
| `momentum_results.png` | `performance.py` | Standard vs Adjusted comparison |
| `plot1_universe_size.png` | `plot1_universe_size.py` | Stock universe over time |
| `plot2_live_vs_dead.png` | `plot2_listed_vs_notlisted.py` | Listed vs not-listed |
| `plot3_return_statistics.png` | `plot3_return_statistics.py` | Return statistics |
| `plot4_return_distribution.png` | `plot4_return_distribution.py` | Return distribution |
| `plot5_missing_data_by_year.png` | `plot5_missing_data.py` | Missing data heatmap |
| `plot6_ff_cumulative_returns.png` | `plot6_ff_cumulative.py` | FF cumulative returns |
| `step2_scatter_momentum_vs_return.png` | `step2_plots.py` | Momentum vs next-week return |
| `step2_histogram_momentum.png` | `step2_plots.py` | Momentum distribution |

### CSVs

| File | Module | Description |
|------|--------|-------------|
| `momentum_raw_sample.csv` | `step2_outputs.py` | Sample of raw momentum matrix |
| `momentum_standardised_sample.csv` | `step2_outputs.py` | Sample of standardised momentum |
| `momentum_summary.csv` | `step2_outputs.py` | Cross-sectional summary stats |

### Logs

| File | Module |
|------|--------|
| `data_loading.log` | `data_loader.py` |
| `momentum_factor.log` | `compute_momentum_signal.py` |

All output files are written to `output_data/`.

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

Place the following files in `input_data/` before running (obtain from course
materials):

- `US_Returns.csv`
- `US_live,csv.csv`
- `US_Dates.xlsx`
- `US_Names.xlsx`
- `FamaFrench.csv`

### Run

```bash
python momentum_strategy.py
```

All outputs are written to `output_data/`.

---

## Disclaimer

### AI Assistance

This project was developed with the assistance of AI tools (GitHub Copilot). AI was used to support coding, debugging, and documentation throughout the development process. All outputs were reviewed, validated, and approved by the author. The academic analysis, methodology decisions, and interpretation of results are the author's own.

### Academic Integrity — Do Not Copy

> **⚠️ This repository is published for reference and transparency only.**
>
> This work was submitted as assessed coursework for **SMM282 Quantitative Trading** at **City St George's, University of London**. **Do not copy, reproduce, or submit any part of this material as your own work.** Doing so constitutes **academic misconduct** and may result in disciplinary action under your institution's regulations.
>
> If you find this project useful for learning, you are welcome to study the methodology and code structure, but you must produce your own original implementation.

---

## References

- Lou, D. & Polk, C. (2021). *Comomentum: Inferring Arbitrage Activity from Return
  Correlations*. Review of Financial Studies, 35(7), 3272–3302.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*.
  Journal of Finance, 48(1), 65–91.
- Fama, E. & MacBeth, J. (1973). *Risk, Return, and Equilibrium: Empirical Tests*.
  Journal of Political Economy, 81(3), 607–636.
- Lewellen, J. & Nagel, S. (2006). *The Conditional CAPM Does Not Explain
  Asset-Pricing Anomalies*. Journal of Financial Economics, 82(2), 289–314.
