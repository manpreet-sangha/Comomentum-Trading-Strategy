# streamlit_app/page_data_loading.py
# =====================================================================
# Data Loading Page — Streamlit UI
# =====================================================================

import os
import glob
import logging
import numpy as np
import streamlit as st


def _reset_log_file(project_root: str) -> None:
    """Clear the log file and reset the logger's file handler so each
    Load click produces a fresh log (no carry-over from previous runs)."""
    log_path = os.path.join(project_root, "output_data", "data_loading.log")
    logger = logging.getLogger("data_loader")
    # Close and remove existing file handlers, then add a fresh one (mode='w')
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


def render(project_root: str) -> None:
    """Render the full Data Loading page."""

    st.header("Step 1 — Data Loading & Cleaning")
    st.markdown(
        "Load all required input files, clean returns by masking "
        "non-listed observations, run dimension checks, and generate "
        "exploration plots."
    )

    # --- Check which input files exist -----------------------------------
    datadir = os.path.join(project_root, "input_data")
    required_files = {
        "US_Returns.csv":  "TxN weekly stock returns",
        "US_live.csv":     "TxN listed/not-listed indicator",
        "US_Dates.xlsx":   "Tx1 weekly date stamps",
        "US_Names.xlsx":   "Nx1 stock/company names",
        "FamaFrench.csv":  "Tx4 Fama-French factors + RF",
    }

    st.subheader("Input Files")
    all_present = True
    for fname, desc in required_files.items():
        exists = os.path.isfile(os.path.join(datadir, fname))
        icon = "✅" if exists else "❌"
        st.markdown(f"{icon}  **{fname}** — {desc}")
        if not exists:
            all_present = False

    if not all_present:
        st.warning(
            "Some required files are missing from `input_data/`. "
            "Please add them before loading."
        )

    # --- View Raw Data (opens in new browser tab) ------------------------
    st.markdown(
        '<a href="?view=raw_data" target="_blank" '
        'style="display:inline-block; padding:0.4rem 1rem; '
        'background-color:#262730; color:white; border-radius:0.5rem; '
        'text-decoration:none; font-size:0.875rem;">'
        '\U0001f441 View Raw Data</a>',
        unsafe_allow_html=True,
    )

    # --- Auto-load data (runs once, cached in session state) ---------------
    if not all_present:
        return

    if "data" not in st.session_state:
        datadir_sep = os.path.join(datadir, "")  # ensure trailing separator

        # Reset the log file so only the latest run is kept
        _reset_log_file(project_root)

        # Placeholders for live-updating UI elements
        progress_bar = st.progress(0, text="Starting …")
        log_container = st.empty()
        log_lines: list[str] = []

        def _log(msg: str, pct: int) -> None:
            """Append a log line (newest on top) and advance the progress bar."""
            log_lines.insert(0, msg)
            log_container.code("\n".join(log_lines), language="text")
            progress_bar.progress(pct, text=msg)

        try:
            from data.read_returns import load_returns
            from data.read_live import load_live
            from data.read_dates import load_dates
            from data.read_names import load_names
            from data.read_fama_french import load_fama_french
            from data.clean_returns import clean_returns
            from data.dimension_checks import check_dimensions, log_loading_summary

            # Step 1 — Returns
            _log("Loading US_Returns.csv …", 5)
            returns = load_returns(datadir_sep)
            _log(f"✔ US_Returns.csv  →  {returns.shape[0]} × {returns.shape[1]}", 15)

            # Step 2 — Live indicator
            _log("Loading US_live.csv …", 20)
            live = load_live(datadir_sep)
            _log(f"✔ US_live.csv  →  {live.shape[0]} × {live.shape[1]}", 30)

            # Step 3 — Dates
            _log("Loading US_Dates.xlsx …", 35)
            dates = load_dates(datadir_sep)
            _log(f"✔ US_Dates.xlsx  →  {len(dates)} dates", 45)

            # Step 4 — Names
            _log("Loading US_Names.xlsx …", 50)
            names = load_names(datadir_sep)
            _log(f"✔ US_Names.xlsx  →  {len(names)} names", 55)

            # Step 5 — Fama-French factors
            _log("Loading FamaFrench.csv …", 60)
            ff_factors, rf = load_fama_french(datadir_sep)
            _log(f"✔ FamaFrench.csv  →  {ff_factors.shape[0]} × {ff_factors.shape[1]} + RF", 70)

            # Step 6 — Clean returns
            _log("Cleaning returns (masking non-listed → NaN) …", 75)
            returns_clean = clean_returns(returns, live)
            T, N = returns.shape
            _log(f"✔ Returns cleaned  ({T} × {N})", 80)

            # Step 7 — Dimension checks
            _log("Running dimension consistency checks …", 85)
            check_dimensions(returns, live, dates, names, ff_factors, T, N)
            _log("✔ All dimension checks passed", 90)

            # Step 8 — Summary
            _log("Writing loading summary …", 92)
            log_loading_summary(returns, live, dates, names, ff_factors, T, N, datadir_sep)

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

            # Step 9 — Generate exploration plots
            _log("Generating exploration plots …", 95)
            from data.dataplots import generate_all_plots
            output_dir = os.path.join(project_root, "output_data")
            generate_all_plots(data, output_dir)
            _log("✔ Exploration plots saved", 98)

            st.session_state["data"] = data

            _log("✅ All data loaded successfully!", 100)
            progress_bar.progress(100, text="Done!")
            st.success("All data loaded successfully!")

        except Exception as e:
            _log(f"❌ ERROR: {e}", 0)
            st.error(f"Data loading failed: {e}")

    # --- Display results if data is loaded -------------------------------
    if "data" not in st.session_state:
        return

    data = st.session_state["data"]

    # ---- Log file (full, from disk) -------------------------------------
    st.subheader("📄 Loading Log")
    log_path = os.path.join(project_root, "output_data", "data_loading.log")
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            # Reverse lines so newest entries appear first
            log_text = "\n".join(reversed(f.read().splitlines()))
        st.code(log_text, language="text")

    # ---- Exploration plots ----------------------------------------------
    st.subheader("Exploration Plots")

    _PLOT_INFO = [
        (
            "plot1_universe_size.png",
            "Universe Size Over Time",
            "Number of stocks in the dataset at each weekly date. "
            "Shows how the investment universe grows or shrinks over the sample period.",
        ),
        (
            "plot2_listed_vs_notlisted.png",
            "Listed vs Not-Listed Composition",
            "Stacked area chart showing the proportion of listed (actively traded) "
            "versus not-listed stocks each week.",
        ),
        (
            "plot3_return_statistics.png",
            "Weekly Cross-Sectional Return Statistics",
            "Time series of weekly cross-sectional mean, median, and standard deviation "
            "of stock returns (computed across all listed stocks each week).",
        ),
        (
            "plot4_return_distribution.png",
            "Return Distribution",
            "Histogram of all weekly stock returns for listed observations, "
            "illustrating the overall distributional shape, heavy tails, and skewness.",
        ),
        (
            "plot5_missing_data_by_year.png",
            "Missing Data by Year",
            "Percentage of missing (NaN) return observations per year, "
            "highlighting data quality across the sample period.",
        ),
        (
            "plot6_ff_cumulative_returns.png",
            "Fama-French Factor Cumulative Returns",
            "Cumulative returns of the three Fama-French factors (Mkt-RF, SMB, HML) "
            "and the risk-free rate over the full sample period.",
        ),
        (
            "plot10_loading_summary.png",
            "Data Loading & Cleaning Summary",
            "Summary table of the data loading and cleaning pipeline — "
            "dimensions, date ranges, listed counts, and cleaning impact.",
        ),
    ]

    plot_dir = os.path.join(project_root, "output_data")
    for png_name, title, description in _PLOT_INFO:
        png_path = os.path.join(plot_dir, png_name)
        if not os.path.isfile(png_path):
            continue
        with st.expander(title, expanded=False):
            st.markdown(description)
            st.image(png_path, use_container_width=True)
