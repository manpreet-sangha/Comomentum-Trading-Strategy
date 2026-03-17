# streamlit_app/app.py
# =====================================================================
# Streamlit Front-End for Comomentum Trading Strategy
# =====================================================================
# Launch from the project root:
#     streamlit run streamlit_app/app.py
# =====================================================================

import sys
import os

# ── Make the project root importable so we can reach data_loader, etc. ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STREAMLIT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, STREAMLIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from page_methodology import render as render_methodology
from page_upload_and_run import render as render_upload_and_run
from page_data_loading import render as render_data_loading
from page_raw_data_preview import render as render_raw_data_preview

# ── Detect mode from query string BEFORE any rendering ───────────────
_is_raw_preview = st.query_params.get("view") == "raw_data"

st.set_page_config(
    page_title="Raw Data Preview" if _is_raw_preview else "Comomentum Trading Strategy",
    page_icon="👁" if _is_raw_preview else "📈",
    layout="wide",
    initial_sidebar_state="collapsed" if _is_raw_preview else "auto",
)

# =====================================================================
# RAW DATA PREVIEW (standalone, no sidebar, no buttons)
# =====================================================================
if _is_raw_preview:
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )
    render_raw_data_preview(PROJECT_ROOT)
    st.stop()

# =====================================================================
# MAIN APP
# =====================================================================
st.title("📈 Comomentum Trading Strategy")

# ─────────────────────────────────────────────────────────────────────
# Sidebar – Navigation  (synced with ?page= query param)
# ─────────────────────────────────────────────────────────────────────
_PAGES = ["Methodology", "Compute Comomentum", "Out-of-Sample Testing"]
_PAGE_KEYS = {p.lower().replace(" ", "-"): p for p in _PAGES}

# Read page from URL if present
_url_key = st.query_params.get("page", "").lower()
_default_index = list(_PAGE_KEYS.values()).index(_PAGE_KEYS[_url_key]) if _url_key in _PAGE_KEYS else 0

page = st.sidebar.radio(
    "Navigation",
    _PAGES,
    index=_default_index,
)

# Write page back to URL
st.query_params["page"] = page.lower().replace(" ", "-")

# =====================================================================
# Page routing
# =====================================================================
if page == "Methodology":
    render_methodology()
elif page == "Compute Comomentum":
    render_data_loading(PROJECT_ROOT)
elif page == "Out-of-Sample Testing":
    render_upload_and_run(PROJECT_ROOT)
