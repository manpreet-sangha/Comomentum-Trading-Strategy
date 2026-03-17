# streamlit_app/page_raw_data_preview.py
# =====================================================================
# Raw Data Preview Page — Streamlit UI
# =====================================================================

import os
import pandas as pd
import streamlit as st


# ── Shared CSS: left-align text, scrollable table container ──────────
_TABLE_CSS = """
<style>
    .raw-table-wrap {
        max-height: 400px;
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .raw-table-wrap table {
        border-collapse: collapse;
        white-space: nowrap;
        font-size: 0.85rem;
    }
    .raw-table-wrap td {
        text-align: left;
        padding: 4px 12px;
        border: 1px solid #eee;
        min-width: 120px;
    }
</style>
"""

# Files to preview: (filename, description, extension)
_RAW_FILES = [
    ("US_Returns.csv",  "Weekly stock returns",          "csv"),
    ("US_live.csv",     "Listed / not-listed indicator",  "csv"),
    ("US_Dates.xlsx",   "Weekly date stamps",             "xlsx"),
    ("US_Names.xlsx",   "Stock / company names",          "xlsx"),
    ("FamaFrench.csv",  "FF3 factors + risk-free rate",  "csv"),
]


_PREVIEW_ROWS = 20


def _csv_dimensions(fpath: str) -> tuple[int, int]:
    """Return (rows, cols) for a headerless CSV without loading the data."""
    with open(fpath, "r") as f:
        first_line = f.readline()
        cols = first_line.count(",") + 1
        rows = 1 + sum(1 for _ in f)
    return rows, cols


def render(project_root: str) -> None:
    """Render the Raw Data Preview page."""

    st.header("Raw Data Preview")
    st.markdown(f"Showing first **{_PREVIEW_ROWS}** rows of each file (all columns).")

    # Inject table CSS
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

    datadir = os.path.join(project_root, "input_data")

    for fname, desc, ftype in _RAW_FILES:
        fpath = os.path.join(datadir, fname)
        if not os.path.isfile(fpath):
            st.warning(f"**{fname}** not found in `input_data/`.")
            continue

        bar = st.empty()
        bar.progress(0.2, text=f"Reading {fname}…")

        if ftype == "csv":
            total_rows, total_cols = _csv_dimensions(fpath)
            df_preview = pd.read_csv(fpath, header=None, nrows=_PREVIEW_ROWS, dtype=str)
        else:
            df_full = pd.read_excel(fpath, header=None, dtype=str)
            total_rows, total_cols = df_full.shape
            df_preview = df_full.head(_PREVIEW_ROWS).copy()
            del df_full

        heading = f"{fname}  —  {desc}  ({total_rows:,} × {total_cols:,})"

        bar.progress(0.7, text=f"Rendering {fname}…")
        with st.expander(heading, expanded=True):
            html = df_preview.to_html(index=False, header=False, na_rep="")
            del df_preview
            st.markdown(
                f'<div class="raw-table-wrap">{html}</div>',
                unsafe_allow_html=True,
            )
            del html

        bar.progress(1.0, text=f"{fname} ✓")
        bar.empty()
