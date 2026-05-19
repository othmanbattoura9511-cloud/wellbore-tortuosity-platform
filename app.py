"""Drilling engineering comparison platform — V/C/L sections, BHA & system KPIs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import run_analytics_pipeline
from bha_extraction import BHA_COLUMNS, extract_bha_runs_from_uploads, normalize_bha_intervals
from column_mapping import (
    SurveyFileType,
    analyze_survey_columns,
    missing_columns_message,
    suggest_column_index,
)
from loaders import SurveyDataLoader
from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV
from plots import WellPlots
from sections import filter_by_section_codes
from survey_merge import load_and_merge_surveys
from upload_utils import save_uploaded_file, uploaded_file_key


def _clear_survey_mapping_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("survey_mapped_"):
            del st.session_state[key]


def resolve_survey_columns(df: pd.DataFrame, file_key: str) -> pd.DataFrame | None:
    mapped_key = f"survey_mapped_{file_key}"
    if mapped_key in st.session_state:
        return st.session_state[mapped_key]

    analysis = analyze_survey_columns(df)
    if analysis.file_type not in (SurveyFileType.SURVEY_STATION, SurveyFileType.INTERVAL_SUMMARY):
        st.warning(analysis.message)
        return None

    if not analysis.missing:
        st.session_state[mapped_key] = analysis.survey
        return analysis.survey

    st.error("Could not map survey columns.")
    st.markdown(missing_columns_message(analysis.missing, analysis.survey.columns))
    col_options = list(df.columns)
    c1, c2, c3 = st.columns(3)
    md_pick = c1.selectbox("MD", col_options, index=suggest_column_index(col_options, "MD"), key=f"md_{file_key}")
    inc_pick = c2.selectbox(
        "Inclination", col_options, index=suggest_column_index(col_options, "Inclination"), key=f"inc_{file_key}"
    )
    azi_pick = c3.selectbox(
        "Azimuth", col_options, index=suggest_column_index(col_options, "Azimuth"), key=f"azi_{file_key}"
    )
    if st.button("Apply mapping", key=f"apply_{file_key}"):
        remapped = analyze_survey_columns(df, manual_map={"MD": md_pick, "Inclination": inc_pick, "Azimuth": azi_pick})
        if not remapped.missing:
            st.session_state[mapped_key] = remapped.survey
            st.rerun()
    return None


def _load_surveys(uploaded_files, use_sample: bool) -> pd.DataFrame | None:
    loader = SurveyDataLoader()
    if use_sample and SAMPLE_SURVEY_CSV.is_file():
        return resolve_survey_columns(loader.load_raw(SAMPLE_SURVEY_CSV), "sample:synthetic_survey")

    if not uploaded_files:
        return None

    paths = [(save_uploaded_file(f), f.name) for f in uploaded_files]
    file_key = "|".join(p[1] for p in paths)
    if st.session_state.get("survey_merge_key") == file_key and "survey_merged" in st.session_state:
        return st.session_state["survey_merged"]

    try:
        if len(paths) == 1:
            merged = resolve_survey_columns(loader.load_raw(paths[0][0]), file_key)
        else:
            merged = load_and_merge_surveys(loader, paths)
        if merged is not None:
            st.session_state["survey_merge_key"] = file_key
            st.session_state["survey_merged"] = merged
        return merged
    except ValueError as exc:
        st.error(str(exc))
        return None


def _bha_key(files) -> str:
    return "|".join(uploaded_file_key(f) for f in files)


st.set_page_config(page_title="Drilling Comparison Platform", layout="wide")
st.title("Drilling Engineering Comparison Platform")
st.markdown(
    "Compare **Motor vs RSS**, hole size, and **BHA performance** by well section "
    "**(V vertical · C curve/build · L lateral)** using survey inclination, DLS, and azimuth."
)

with st.sidebar:
    st.header("Well data")
    use_sample = st.button("Load sample survey", use_container_width=True)
    survey_files = st.file_uploader(
        "Survey file(s) — multiple allowed per well",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="survey_files",
    )
    bha_files = st.file_uploader(
        "BHA PDF(s) — one file per assembly/run",
        type=["pdf"],
        accept_multiple_files=True,
        key="bha_files",
    )

    st.header("Section display")
    filt_v = st.checkbox("V — Vertical", True)
    filt_c = st.checkbox("C — Curve / build", True)
    filt_l = st.checkbox("L — Lateral", True)
    section_codes = [c for c, on in zip(["V", "C", "L"], [filt_v, filt_c, filt_l]) if on]

    if SAMPLE_SURVEY_CSV.is_file():
        st.caption(f"Sample: `{SAMPLE_SURVEY_CSV.name}`")

# --- Surveys ---
if use_sample:
    st.session_state.pop("survey_merged", None)
    st.session_state.pop("survey_merge_key", None)

df_survey = _load_surveys(survey_files if survey_files else None, use_sample)
if df_survey is None:
    st.info("Upload one or more survey files, or load the sample well.")
    st.stop()

# --- BHA ---
bha_intervals = pd.DataFrame()
if bha_files:
    bk = _bha_key(bha_files)
    if st.session_state.get("bha_key") != bk:
        st.session_state["bha_key"] = bk
        paths = [(save_uploaded_file(f), f.name) for f in bha_files]
        st.session_state["bha_intervals_raw"] = extract_bha_runs_from_uploads(paths)
        st.session_state.pop("bha_intervals", None)

if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
    with st.expander("BHA runs — edit MD intervals & apply", expanded=True):
        edited = st.data_editor(
            st.session_state["bha_intervals_raw"][[c for c in BHA_COLUMNS if c in st.session_state["bha_intervals_raw"].columns]],
            use_container_width=True,
            num_rows="dynamic",
        )
        if st.button("Apply BHA MD mapping", type="primary"):
            st.session_state["bha_intervals"] = normalize_bha_intervals(edited)
            st.rerun()
    bha_intervals = st.session_state.get(
        "bha_intervals",
        normalize_bha_intervals(st.session_state["bha_intervals_raw"]),
    )
else:
    bha_intervals = st.session_state.get("bha_intervals", pd.DataFrame())

# --- Analytics ---
result = run_analytics_pipeline(df_survey, bha_intervals if not bha_intervals.empty else None)
df_full = result.survey
df = filter_by_section_codes(df_full, section_codes)
plots = WellPlots()

if df.empty:
    st.warning("No stations match the selected sections (V / C / L).")
    st.stop()

# --- Metrics strip ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Survey stations", len(df_full))
m2.metric("MD end (m)", f"{df_full['MD'].max():.0f}")
m3.metric("BHA runs", len(bha_intervals) if not bha_intervals.empty else 0)
m4.metric("Avg tortuosity", f"{result.kpi_summary.get('mean_tortuosity', 0):.2f}")
m5.metric("Avg stability", f"{result.kpi_summary.get('mean_stability', 0):.2f}")

# --- Charts (minimal) ---
st.subheader("Trajectory")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plots.plot_inclination_vs_md(df), use_container_width=True)
with c2:
    tort_fig = plots.plot_tortuosity_vs_md(df)
    if tort_fig:
        st.plotly_chart(tort_fig, use_container_width=True)

# --- TABLE 1 ---
st.subheader("Table 1 — Section comparison")
st.caption(
    "Per **V / C / L** interval within each **BHA run**: DLS, tortuosity, wellbore smoothness, and drilling stability."
)
if not result.section_comparison.empty:
    st.dataframe(result.section_comparison, use_container_width=True, hide_index=True)
else:
    st.info("Upload BHA PDFs with MD intervals to populate BHA and system columns, or data will show section-only rows.")

# --- TABLE 2 ---
st.subheader("Table 2 — Drilling system comparison")
st.caption("**Motor vs RSS vs Rotary** — average KPIs and best-performing section per system.")
if not result.system_comparison.empty:
    st.dataframe(result.system_comparison, use_container_width=True, hide_index=True)
else:
    st.caption("Map BHA PDFs with drilling system labels to enable system comparison.")

# --- Supplementary comparisons (compact) ---
with st.expander("RSS push-the-bit vs point-the-bit · Hole size"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**RSS steering mode**")
        if not result.rss_steering_comparison.empty:
            st.dataframe(result.rss_steering_comparison, use_container_width=True, hide_index=True)
        else:
            st.caption("No RSS runs in BHA table.")
    with col_b:
        st.markdown("**Hole size**")
        if not result.hole_size_comparison.empty:
            st.dataframe(result.hole_size_comparison, use_container_width=True, hide_index=True)
        else:
            st.caption("No hole size data from BHA PDFs.")

with st.expander("Survey station data (debug)"):
    show_cols = [
        c
        for c in [
            "MD",
            "Inclination",
            "Azimuth",
            "DLS",
            "Section_Code",
            "Well_Section",
            "BHA",
            "Drilling_System",
            "Hole_Size",
            "RSS_Type",
            "Tortuosity_Index",
            "Wellbore_Smoothness",
            "Stability",
            "Survey_File",
        ]
        if c in df_full.columns
    ]
    st.dataframe(df_full[show_cols], use_container_width=True)
