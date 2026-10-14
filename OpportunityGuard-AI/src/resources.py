"""
src/resources.py — Shared cached resource loaders for Streamlit pages.

Use st.cache_resource so the model and index are loaded only once per
server process regardless of which page is active.
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

_ROOT = Path(__file__).parent.parent


@st.cache_resource(show_spinner="Loading ML model…")
def get_model_and_config():
    from src.model_service import load_model, load_config
    config = load_config()
    model = load_model()
    return model, config


@st.cache_resource(show_spinner="Loading duplicate index…")
def get_duplicate_index():
    from src.duplicate_detector import load_duplicate_index
    return load_duplicate_index()
