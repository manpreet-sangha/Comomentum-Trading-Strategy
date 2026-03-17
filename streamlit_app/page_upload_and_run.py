# streamlit_app/page_upload_and_run.py
# =====================================================================
# Upload Files & Run Pipeline — Streamlit UI
# =====================================================================
# Allows users to upload their own 5 input files and run the full
# comomentum pipeline with live status updates.
# =====================================================================

import os
import shutil
import logging
import traceback
import numpy as np
import pandas as pd
import streamlit as st

_MAX_PREVIEW_ROWS = 20


# ── File specifications ──────────────────────────────────────────────
_FILE_SPECS = [
    {
        "key": "US_Dates",
        "filename": "US_Dates.xlsx",
        "label": "Upload US_Dates.xlsx file",
        "type": ["xlsx"],
        "desc": "Weekly date stamps (Tx1, YYYYMMDD integer format, no header).",
    },
    {
        "key": "US_Names",
        "filename": "US_Names.xlsx",
        "label": "Upload US_Names.xlsx file",
        "type": ["xlsx"],
        "desc": "Stock/company names (1xN or Nx1, no header).",
    },
    {
        "key": "US_Returns",
        "filename": "US_Returns.csv",
        "label": "Upload US_Returns.csv file",
        "type": ["csv"],
        "desc": "Weekly total stock returns (TxN, decimal format, no header).",
    },
    {
        "key": "US_live",
        "filename": "US_live.csv",
        "label": "Upload US_live.csv file",
        "type": ["csv"],
        "desc": "Listed/not-listed indicator (TxN, binary 0/1, no header).",
    },
    {
        "key": "FamaFrench",
        "filename": "FamaFrench.csv",
        "label": "Upload FamaFrench.csv file",
        "type": ["csv"],
        "desc": "Fama-French 3 factors + RF (Tx4, with column headers: Mkt-RF, SMB, HML, RF).",
    },
]


# ── Pipeline step definitions ────────────────────────────────────────
_PIPELINE_STEPS = [
    ("Step 1 — Load Data",                       "load_data"),
    ("Step 1b — Stock Diagnostics",               "stock_diagnostics"),
    ("Step 2 — Compute Standard Momentum",        "compute_momentum"),
    ("Step 3 — Fama-MacBeth Regressions",         "fama_macbeth"),
    ("Step 4 — Compute Comomentum",               "compute_comomentum"),
    ("Step 5 — Adjust Momentum (Inverse Comom.)", "adjust_momentum"),
    ("Step 6 — Performance Comparison",           "performance"),
]


def _reset_pipeline_state():
    """Clear all pipeline-related session state so a fresh run starts."""
    for k in list(st.session_state.keys()):
        if k.startswith("pipeline_") or k in ("data", "momentum", "momentum_std",
                                                "momentum_summary"):
            del st.session_state[k]


def _save_uploaded_files(project_root: str, uploaded: dict) -> str:
    """Write uploaded files to input_data/ and return the directory path."""
    datadir = os.path.join(project_root, "input_data")
    os.makedirs(datadir, exist_ok=True)
    for spec in _FILE_SPECS:
        file_obj = uploaded[spec["key"]]
        dest = os.path.join(datadir, spec["filename"])
        with open(dest, "wb") as f:
            f.write(file_obj.getbuffer())
    return datadir


def _run_pipeline(project_root: str, status_container, progress_bar,
                  results_container):
    """Execute the full pipeline with live status updates and per-step results."""

    datadir = os.path.join(project_root, "input_data", "")
    output_dir = os.path.join(project_root, "output_data")
    os.makedirs(output_dir, exist_ok=True)

    step_statuses: dict[str, str] = {}

    def _update(msg: str, pct: int, step_key: str | None = None,
                step_status: str | None = None):
        if step_key and step_status:
            step_statuses[step_key] = step_status
        progress_bar.progress(pct, text=msg)
        _render_step_tracker(status_container, step_statuses)

    def _render_step_tracker(container, statuses):
        lines = []
        for label, key in _PIPELINE_STEPS:
            s = statuses.get(key, "pending")
            if s == "running":
                icon = "🔄"
            elif s == "done":
                icon = "✅"
            elif s == "error":
                icon = "❌"
            else:
                icon = "⬜"
            lines.append(f"{icon}  {label}")
        container.markdown("\n\n".join(lines))

    def _show_image(path: str, caption: str = ""):
        """Display an image if it exists."""
        if os.path.isfile(path):
            st.image(path, caption=caption, use_container_width=True)

    def _show_csv(path: str, label: str = ""):
        """Display a CSV preview inside the current context."""
        if not os.path.isfile(path):
            return
        df = pd.read_csv(path, nrows=_MAX_PREVIEW_ROWS + 1)
        total_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1  # exclude header
        if label:
            st.markdown(f"**{label}**")
        if total_rows > _MAX_PREVIEW_ROWS:
            st.caption(f"Showing first {_MAX_PREVIEW_ROWS} of {total_rows:,} rows "
                       f"(large dataset — {os.path.getsize(path)/1024:.0f} KB)")
            st.dataframe(df.head(_MAX_PREVIEW_ROWS), use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    # Render initial state (all pending)
    _render_step_tracker(status_container, step_statuses)

    try:
        # ── Step 1: Load Data ────────────────────────────────────────
        _update("Loading input data …", 2, "load_data", "running")

        from data.data_loader import load_all_data
        data = load_all_data(datadir=datadir)

        _update(f"✔ Data loaded: {data['T']} weeks × {data['N']} stocks", 10,
                "load_data", "done")

        with results_container:
            with st.expander("Step 1 Results — Data Loading", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Weeks (T)", f"{data['T']:,}")
                c2.metric("Stocks (N)", f"{data['N']:,}")
                c3.metric("Date range", f"{data['dates'][0].strftime('%Y-%m-%d')}  →  "
                          f"{data['dates'][-1].strftime('%Y-%m-%d')}")
                _show_image(os.path.join(output_dir, "plot1_universe_size.png"),
                            "Universe Size Over Time")
                _show_image(os.path.join(output_dir, "plot2_listed_vs_notlisted.png"),
                            "Listed vs Not-Listed")
                _show_image(os.path.join(output_dir, "plot3_return_statistics.png"),
                            "Weekly Return Statistics")
                _show_image(os.path.join(output_dir, "plot6_ff_cumulative_returns.png"),
                            "FF Factor Cumulative Returns")

        # ── Step 1b: Stock Diagnostics ───────────────────────────────
        _update("Running stock diagnostics …", 12, "stock_diagnostics", "running")

        from compute_momentum.stock_diagnostics import run_stock_diagnostics
        df_short, df_gaps, df_combined = run_stock_diagnostics(data)

        exclude_idx = set()
        if len(df_short) > 0:
            exclude_idx |= set(df_short['Stock_Index'])
        for j in exclude_idx:
            data['returns_clean'][:, j] = np.nan

        n_excl = len(exclude_idx)
        n_gapped = len(df_gaps) if len(df_gaps) > 0 else 0
        _update(f"✔ Diagnostics done — {n_excl} short-lived stocks excluded", 15,
                "stock_diagnostics", "done")

        with results_container:
            with st.expander("Step 1b Results — Stock Diagnostics", expanded=False):
                c1, c2 = st.columns(2)
                c1.metric("Short-lived excluded", n_excl)
                c2.metric("Gapped (kept)", n_gapped)
                if len(df_short) > 0:
                    st.caption("Short-lived stocks (< 52 listed weeks):")
                    st.dataframe(df_short.head(20), use_container_width=True)
                if len(df_gaps) > 0:
                    st.caption("Stocks with trading gaps (kept for per-window checks):")
                    st.dataframe(df_gaps.head(20), use_container_width=True)

        # ── Step 2: Compute Standard Momentum ────────────────────────
        _update("Computing standard momentum signal …", 17, "compute_momentum", "running")

        from compute_momentum.compute_momentum_signal import compute_momentum_signal
        from compute_momentum.step2_outputs import save_momentum_outputs
        from compute_momentum.step2_plots import generate_step2_plots

        momentum, momentum_std = compute_momentum_signal(
            data['returns_clean'], data['dates']
        )
        save_momentum_outputs(momentum, momentum_std, data['dates'], data['names'])
        generate_step2_plots(momentum, momentum_std, data)

        n_scored = int(np.sum(np.any(np.isfinite(momentum), axis=1)))
        _update(f"✔ Momentum computed — {n_scored} scored weeks", 30,
                "compute_momentum", "done")

        with results_container:
            with st.expander("Step 2 Results — Standard Momentum", expanded=False):
                valid = np.isfinite(momentum)
                c1, c2, c3 = st.columns(3)
                c1.metric("Scored weeks", f"{n_scored:,}")
                c2.metric("Mean coverage",
                          f"{np.mean(np.sum(valid, axis=1)[np.any(valid, axis=1)]):.0f} "
                          f"stocks/week")
                c3.metric("Momentum range",
                          f"{np.nanmin(momentum):.2f}  →  {np.nanmax(momentum):.2f}")
                _show_image(os.path.join(output_dir, "step2_histogram_momentum.png"),
                            "Momentum Score Distribution")
                _show_image(os.path.join(output_dir, "step2_factor_comparison.png"),
                            "Factor Time Series Comparison")
                st.divider()
                st.markdown("##### Output CSVs")
                _show_csv(os.path.join(output_dir, "momentum_raw.csv"),
                          "momentum_raw.csv")
                _show_csv(os.path.join(output_dir, "momentum_standardised.csv"),
                          "momentum_standardised.csv")
                _show_csv(os.path.join(output_dir, "momentum_summary.csv"),
                          "momentum_summary.csv")

        # ── Step 3: Fama-MacBeth ─────────────────────────────────────
        _update("Running Fama-MacBeth regressions …", 32, "fama_macbeth", "running")

        from fama_macbeth.fama_macbeth import famaMacBeth
        gamma_std, tstat_std = famaMacBeth(
            momentum_std, data['returns_clean'], data['live'],
            dates=data['dates'],
            save_path=os.path.join(output_dir, 'fama_macbeth_standard_momentum.xlsx')
        )

        n_valid_gamma = int(np.sum(np.isfinite(gamma_std)))
        _update(f"✔ Fama-MacBeth done — t-stat = {tstat_std:.4f}", 45,
                "fama_macbeth", "done")

        with results_container:
            with st.expander("Step 3 Results — Fama-MacBeth Regressions", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Valid factor-return weeks", f"{n_valid_gamma:,}")
                c2.metric("t-statistic", f"{tstat_std:.4f}")
                c3.metric("Mean weekly γ",
                          f"{np.nanmean(gamma_std)*100:.4f}%")

        # ── Step 4: Compute Comomentum ───────────────────────────────
        _update("Computing comomentum (this may take a while) …", 47,
                "compute_comomentum", "running")

        from comomentum.compute_comomentum import compute_comomentum
        from comomentum.save_ff3_residuals import save_ff3_residuals
        from comomentum.save_pairwise_correlations import save_pairwise_correlations
        from comomentum.plot_comom_event_study import plot_comom_event_study
        from comomentum.plot_comom_time_series import plot_comom_time_series
        from data.market_variables import compute_market_variables
        from comomentum.summary_statistics_table import generate_summary_table
        from comomentum.determinants_table import generate_determinants_table

        comomentum_arr, comom_winner, comom_loser = compute_comomentum(
            data['returns_clean'], momentum_std,
            data['live'], data['ff_factors'], data['dates']
        )

        save_ff3_residuals(
            data['returns_clean'], momentum_std,
            data['live'], data['ff_factors'],
            data['dates'], data['names'],
            snapshot_week=None,
            save_path=os.path.join(output_dir, 'ff3_residuals.xlsx')
        )
        save_pairwise_correlations(
            data['returns_clean'], momentum_std,
            data['live'], data['ff_factors'],
            data['dates'], data['names'],
            snapshot_week=None,
            save_path=os.path.join(output_dir, 'pairwise_correlations.xlsx')
        )
        plot_comom_event_study(
            comomentum_arr, data['dates'],
            max_years=4,
            save_path=os.path.join(output_dir, 'plot_comom_event_study.png')
        )
        plot_comom_time_series(
            comomentum_arr, comom_winner, comom_loser, data['dates'],
            sample_months=6,
            save_path=os.path.join(output_dir, 'plot_comom_time_series.png')
        )

        mret, mvol = compute_market_variables(
            data['ff_factors'], data['rf'], data['dates']
        )
        generate_summary_table(
            comomentum_arr, comom_winner, comom_loser, mret, mvol, data['dates'],
            save_path=os.path.join(output_dir, 'summary_statistics_table.png')
        )
        generate_determinants_table(
            comomentum_arr, gamma_std, mret, mvol, data['dates'],
            save_path=os.path.join(output_dir, 'determinants_table.png')
        )

        n_comom = int(np.sum(np.isfinite(comomentum_arr)))
        _update(f"✔ Comomentum computed — {n_comom} valid values", 70,
                "compute_comomentum", "done")

        with results_container:
            with st.expander("Step 4 Results — Comomentum", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Valid comomentum values", f"{n_comom:,}")
                c2.metric("Mean comomentum", f"{np.nanmean(comomentum_arr):.4f}")
                c3.metric("Std comomentum", f"{np.nanstd(comomentum_arr):.4f}")
                _show_image(os.path.join(output_dir, "plot_comom_time_series.png"),
                            "Comomentum Time Series")
                _show_image(os.path.join(output_dir, "plot_comom_event_study.png"),
                            "Comomentum Event Study")
                _show_image(os.path.join(output_dir, "summary_statistics_table.png"),
                            "Summary Statistics (Table I)")
                _show_image(os.path.join(output_dir, "determinants_table.png"),
                            "Determinants of Comomentum (Table II)")

        # ── Step 5: Adjust Momentum ──────────────────────────────────
        _update("Adjusting momentum with inverse comomentum …", 72,
                "adjust_momentum", "running")

        from adjusted_momentum.compute_adjusted_momentum import compute_adjusted_momentum
        gamma_adj, scaling, comom_pctile = compute_adjusted_momentum(
            gamma_std, comomentum_arr
        )

        n_valid_adj = int(np.sum(np.isfinite(gamma_adj)))
        _update(f"✔ Adjusted momentum: mean {np.nanmean(gamma_adj)*100:.4f}%", 80,
                "adjust_momentum", "done")

        with results_container:
            with st.expander("Step 5 Results — Adjusted Momentum", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Valid adjusted weeks", f"{n_valid_adj:,}")
                c2.metric("Scaling mean", f"{np.nanmean(scaling):.4f}")
                c3.metric("Scaling std", f"{np.nanstd(scaling):.4f}")
                st.caption(
                    f"Mean weekly γ: **{np.nanmean(gamma_std)*100:.4f}%** (standard) → "
                    f"**{np.nanmean(gamma_adj)*100:.4f}%** (adjusted)"
                )

        # ── Step 6: Performance Comparison ───────────────────────────
        _update("Computing performance statistics …", 82, "performance", "running")

        from performance import compute_stats, print_summary_table, plot_main_results

        stats_std = compute_stats(gamma_std, label='Standard Momentum')
        stats_adj = compute_stats(gamma_adj, label='Adjusted Momentum')
        print_summary_table(stats_std, stats_adj)

        plot_main_results(
            data['dates'], gamma_std, gamma_adj, comomentum_arr, scaling,
            save_path=os.path.join(output_dir, 'momentum_results.png')
        )

        _update("✅ Pipeline complete!", 100, "performance", "done")
        progress_bar.progress(100, text="Done!")

        with results_container:
            with st.expander("Step 6 Results — Performance Comparison", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Standard Momentum**")
                    st.metric("Ann. Mean Return",
                              f"{stats_std['mean_ann']*100:.2f}%")
                    st.metric("Ann. Sharpe Ratio", f"{stats_std['sharpe']:.3f}")
                    st.metric("t-statistic", f"{stats_std['tstat']:.3f}")
                    st.metric("Max Drawdown", f"{stats_std['max_dd']*100:.2f}%")
                with c2:
                    st.markdown("**Adjusted Momentum**")
                    st.metric("Ann. Mean Return",
                              f"{stats_adj['mean_ann']*100:.2f}%")
                    st.metric("Ann. Sharpe Ratio", f"{stats_adj['sharpe']:.3f}")
                    st.metric("t-statistic", f"{stats_adj['tstat']:.3f}")
                    st.metric("Max Drawdown", f"{stats_adj['max_dd']*100:.2f}%")
                _show_image(os.path.join(output_dir, "momentum_results.png"),
                            "Standard vs. Adjusted Momentum Factor Returns")

        # Store in session state
        st.session_state["data"] = data
        st.session_state["pipeline_complete"] = True

    except Exception as e:
        # Mark the currently-running step as error
        for _, key in _PIPELINE_STEPS:
            if step_statuses.get(key) == "running":
                step_statuses[key] = "error"
        _render_step_tracker(status_container, step_statuses)
        progress_bar.progress(0, text="Pipeline failed")
        st.error(f"Pipeline failed: {e}")
        st.code(traceback.format_exc(), language="text")


# =====================================================================
# RENDER
# =====================================================================
def render(project_root: str) -> None:
    """Render the Upload Files & Run Pipeline page."""

    st.header("Upload Input Files")
    st.markdown(
        "Upload your own data files to run the comomentum pipeline. "
        "Files must be in **exactly** the same format as the originals."
    )

    # ── File uploaders (two side-by-side sections) ─────────────────
    # Each slot accepts exactly one file with the correct name & format.
    # Re-uploading a correct file replaces the previous version.
    # Wrong-name files are rejected immediately and the uploader resets.
    uploaded = {}
    all_uploaded = True
    has_errors = False

    def _handle_upload(spec):
        """Process a single file uploader; returns (file_ok, error)."""
        file = st.file_uploader(
            f"📄 {spec['filename']}", type=spec["type"],
            key=f"upload_{spec['key']}", help=spec["desc"],
            accept_multiple_files=False,
        )
        if file is None:
            return None, False

        if file.name != spec["filename"]:
            st.error(f"Expected **{spec['filename']}**, got **{file.name}**. "
                     "Please upload the correct file.")
            return None, True

        # Valid file — show confirmation + remove button
        c1, c2 = st.columns([3, 1])
        c1.caption(f"✅ {file.name} — {file.size:,} bytes")
        if c2.button("🗑️", key=f"clear_{spec['key']}",
                     help=f"Remove {spec['filename']}"):
            del st.session_state[f"upload_{spec['key']}"]
            st.rerun()
        return file, False

    left, right = st.columns(2)

    # Left column: Excel files
    with left:
        for spec in _FILE_SPECS[:2]:
            file, err = _handle_upload(spec)
            if err:
                has_errors = True
            elif file is not None:
                uploaded[spec["key"]] = file
            else:
                all_uploaded = False

    # Right column: CSV files
    with right:
        for spec in _FILE_SPECS[2:]:
            file, err = _handle_upload(spec)
            if err:
                has_errors = True
            elif file is not None:
                uploaded[spec["key"]] = file
            else:
                all_uploaded = False

    st.divider()

    # ── Calculate Comomentum button ──────────────────────────────────
    if all_uploaded and not has_errors:
        if st.button("🚀 Calculate Comomentum", type="primary", use_container_width=True):
            # Clear previous pipeline results
            _reset_pipeline_state()

            # Save uploaded files to input_data/
            with st.spinner("Saving uploaded files …"):
                _save_uploaded_files(project_root, uploaded)

            st.success("Files saved to `input_data/`. Starting pipeline …")

            # Pipeline execution with live status
            st.subheader("Pipeline Progress")
            status_container = st.empty()
            progress_bar = st.progress(0, text="Initialising …")
            results_container = st.container()

            _run_pipeline(project_root, status_container, progress_bar,
                          results_container)
    elif has_errors:
        st.warning("Fix the filename errors above before running the pipeline.")
    else:
        st.info("Upload all 5 files to enable the **Calculate Comomentum** button.")
