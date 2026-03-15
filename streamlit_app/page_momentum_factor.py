# streamlit_app/page_momentum_factor.py
# =====================================================================
# Compute Momentum Factor Page — Streamlit UI
# =====================================================================

import os
import logging
import numpy as np
import streamlit as st


def _reset_log_file(project_root: str) -> None:
    """Clear the momentum log file and reset the logger's file handler."""
    log_path = os.path.join(project_root, "output_data", "data_loading.log")
    logger = logging.getLogger("data_loader")
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
    """Render the Compute Momentum Factor page."""

    st.header("Step 2 — Compute Momentum Factor")
    st.markdown(
        "Computes the standard momentum signal for every stock using a "
        "rolling 48-week lookback window (skipping the 4 most recent weeks "
        "to avoid short-term reversal). Raw cumulative returns are used "
        "as per Lou & Polk (2021) — no cross-sectional standardisation."
    )

    # ── Check prerequisite: data must be loaded ─────────────────────
    if "data" not in st.session_state:
        st.warning(
            "Data has not been loaded yet. "
            "Please go to **Data Loading** first."
        )
        return

    data = st.session_state["data"]

    # ── Show configuration ──────────────────────────────────────────
    from config import LOOKBACK, SKIP, TOTAL

    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lookback", f"{LOOKBACK} weeks")
    col2.metric("Skip (recent)", f"{SKIP} weeks")
    col3.metric("Total window", f"{TOTAL} weeks")

    T, N = data["T"], data["N"]
    first_scored = TOTAL  # 1-based week number
    n_scored = T - TOTAL + 1
    st.markdown(
        f"**Input**: {T:,} weeks × {N:,} stocks &nbsp;|&nbsp; "
        f"**First scored week**: {first_scored} &nbsp;|&nbsp; "
        f"**Scored weeks**: {n_scored:,}"
    )

    # ── Auto-compute (runs once, cached in session state) ───────────
    if "momentum" not in st.session_state:
        _reset_log_file(project_root)

        progress_bar = st.progress(0, text="Starting momentum computation …")
        log_container = st.empty()
        log_lines: list[str] = []

        def _log(msg: str, pct: int) -> None:
            log_lines.insert(0, msg)
            log_container.code("\n".join(log_lines), language="text")
            progress_bar.progress(pct, text=msg)

        try:
            # Step 1 — Compute momentum signal
            _log("Computing rolling momentum signal …", 10)
            from compute_momentum.compute_momentum_signal import compute_momentum_signal
            momentum, momentum_std = compute_momentum_signal(
                data["returns_clean"], data["dates"]
            )
            n_valid = int(np.sum(np.isfinite(momentum)))
            _log(
                f"✔ Momentum computed — {n_valid:,} valid scores "
                f"({n_valid / (n_scored * N) * 100:.1f}% of scored cells)",
                40,
            )

            # Step 2 — Save CSV outputs
            _log("Saving momentum data to CSV …", 50)
            from compute_momentum.step2_outputs import save_momentum_outputs
            output_dir = os.path.join(project_root, "output_data")
            df_summary = save_momentum_outputs(
                momentum, momentum_std,
                data["dates"], data["names"], output_dir,
            )
            _log("✔ CSV files saved (raw, standardised, summary)", 65)

            # Step 3 — Generate diagnostic plots
            _log("Generating diagnostic plots …", 70)
            from compute_momentum.step2_plots import generate_step2_plots
            generate_step2_plots(momentum, momentum_std, data, output_dir)
            _log("✔ Diagnostic plots saved", 90)

            # Store results
            st.session_state["momentum"] = momentum
            st.session_state["momentum_std"] = momentum_std
            st.session_state["momentum_summary"] = df_summary

            _log("✅ Momentum factor computation complete!", 100)
            progress_bar.progress(100, text="Done!")
            st.success("Momentum factor computed successfully!")

        except Exception as e:
            _log(f"❌ ERROR: {e}", 0)
            st.error(f"Momentum computation failed: {e}")
            return

    # ── Display results ─────────────────────────────────────────────
    if "momentum" not in st.session_state:
        return

    momentum = st.session_state["momentum"]
    momentum_std = st.session_state["momentum_std"]
    df_summary = st.session_state["momentum_summary"]

    # ── Summary statistics ──────────────────────────────────────────
    st.subheader("Momentum Signal Summary")

    n_valid = int(np.sum(np.isfinite(momentum)))
    n_scored_cells = n_scored * N
    all_valid = momentum[np.isfinite(momentum)]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Valid Scores", f"{n_valid:,}")
    col_b.metric("Coverage", f"{n_valid / n_scored_cells * 100:.1f}%")
    col_c.metric("Mean (raw)", f"{np.mean(all_valid) * 100:+.2f}%")
    col_d.metric("Std (raw)", f"{np.std(all_valid) * 100:.2f}%")

    # ── Weekly summary table ────────────────────────────────────────
    with st.expander("Weekly Cross-Sectional Summary", expanded=False):
        st.markdown(
            "Weekly statistics across all stocks: number of valid scores, "
            "mean, median, standard deviation, min and max of raw momentum."
        )
        st.dataframe(df_summary, height=400, use_container_width=True)

    # ── Diagnostic plots ────────────────────────────────────────────
    st.subheader("Diagnostic Plots")
    output_dir = os.path.join(project_root, "output_data")

    _PLOT_INFO = [
        (
            "step2_scatter_momentum_vs_return.png",
            "Momentum Exposure vs Next-Week Return",
            "Scatter plot of raw momentum at week *t* against the "
            "realised stock return at week *t+1*. A subsample of 50,000 "
            "observations is shown with an OLS fit line. The near-zero slope "
            "reflects the well-known difficulty of predicting individual "
            "stock returns — the momentum effect is small in magnitude and "
            "only becomes economically meaningful when aggregated across "
            "a large portfolio of stocks over many weeks.",
        ),
        (
            "step2_histogram_momentum.png",
            "Distribution of Momentum",
            "Two-panel histogram: (a) pooled distribution across all weeks, "
            "and (b) cross-sectional distribution for the last week only. "
            "Raw momentum scores are shown (no z-scoring).",
        ),
        (
            "step2_factor_comparison.png",
            "Factor Comparison Over Time",
            "Four-panel time series showing: (1) raw momentum — weekly "
            "cross-sectional mean of the 48-week compounded return; "
            "(2) placeholder (standardisation removed); (3) comomentum "
            "placeholder; (4) adjusted momentum placeholder.",
        ),
    ]

    for png_name, title, description in _PLOT_INFO:
        png_path = os.path.join(output_dir, png_name)
        if not os.path.isfile(png_path):
            continue
        with st.expander(title, expanded=False):
            st.markdown(description)
            st.image(png_path, use_container_width=True)

    # ── Log file ────────────────────────────────────────────────────
    st.subheader("📄 Computation Log")
    log_path = os.path.join(project_root, "output_data", "data_loading.log")
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log_text = "\n".join(reversed(f.read().splitlines()))
        with st.expander("View full log", expanded=False):
            st.code(log_text, language="text")
