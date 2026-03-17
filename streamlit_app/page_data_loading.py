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

    st.header("Compute Comomentum")
    st.markdown(
        "Run the full comomentum pipeline on the default input data. "
        "Review the detected input files below, then click **Run Pipeline**."
    )

    # --- Check which input files exist & show dimensions -----------------
    datadir = os.path.join(project_root, "input_data")

    import pandas as pd

    _FILE_INFO = [
        ("US_Returns.csv",  "Weekly stock returns",        "csv"),
        ("US_live.csv",     "Listed/not-listed indicator",  "csv"),
        ("US_Dates.xlsx",   "Weekly date stamps",           "xlsx"),
        ("US_Names.xlsx",   "Stock/company names",          "xlsx"),
        ("FamaFrench.csv",  "Fama-French factors + RF",     "csv"),
    ]

    st.subheader("Input Files")
    all_present = True
    for fname, desc, fmt in _FILE_INFO:
        fpath = os.path.join(datadir, fname)
        exists = os.path.isfile(fpath)
        if not exists:
            st.markdown(f"❌  **{fname}** — {desc} — *missing*")
            all_present = False
            continue
        try:
            size_kb = os.path.getsize(fpath) / 1024
            if fmt == "csv":
                df_peek = pd.read_csv(fpath, header=None if fname != "FamaFrench.csv" else "infer", nrows=5)
                df_peek = df_peek.dropna(axis=1, how="all")
                ncols = df_peek.shape[1]
                nrows = sum(1 for _ in open(fpath, encoding="utf-8")) - (1 if fname == "FamaFrench.csv" else 0)
                dim = f"{nrows:,} × {ncols:,}"
            else:
                df = pd.read_excel(fpath, header=None)
                dim = f"{df.shape[0]:,} × {df.shape[1]:,}"
            st.markdown(f"✅  **{fname}** — {desc} — `{dim}` ({size_kb:,.0f} KB)")
        except Exception:
            st.markdown(f"✅  **{fname}** — {desc}")

    if not all_present:
        st.warning(
            "Some required files are missing from `input_data/`. "
            "Please add them before running."
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

    # --- Run full pipeline on button click ---------------------------------
    if not all_present:
        return

    from page_upload_and_run import _run_pipeline, _reset_pipeline_state

    # Show the Run Pipeline button when pipeline hasn't run yet
    if not st.session_state.get("pipeline_complete"):
        st.divider()
        if not st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            st.info("Click **Run Pipeline** to execute the full comomentum pipeline.")
            return

        # Clear any stale state and run
        _reset_pipeline_state()

        st.subheader("Pipeline Progress")
        status_container = st.empty()
        progress_bar = st.progress(0, text="Initialising …")
        results_container = st.container()

        _run_pipeline(project_root, status_container, progress_bar,
                      results_container)
