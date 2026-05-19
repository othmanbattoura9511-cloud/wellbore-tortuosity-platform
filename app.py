"""Wellbore Tortuosity Analytics Platform — drilling engineering dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import run_analytics_pipeline
from bha_extraction import extract_bha_runs_from_uploads, normalize_bha_intervals
from column_mapping import (
    SurveyFileType,
    analyze_survey_columns,
    missing_columns_message,
    suggest_column_index,
)
from dashboard_views import (
    render_comparisons_tab,
    render_drilling_systems_tab,
    render_overview_tab,
    render_section_analysis_tab,
    render_survey_data_tab,
)
from loaders import SurveyDataLoader
from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV
from plots import WellPlots
from sections import filter_by_section_codes
from ui_theme import (
    bha_editor_column_config,
    inject_theme,
    prepare_bha_for_editor,
    prepare_bha_from_editor,
    render_header,
)
from survey_merge import load_and_merge_surveys
from survey_quality import SurveyQualityAnalyzer
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


def _load_surveys(uploaded_files, use_sample: bool) -> tuple[pd.DataFrame | None, list[str]]:
    loader = SurveyDataLoader()
    labels: list[str] = []

    if use_sample and SAMPLE_SURVEY_CSV.is_file():
        labels.append(SAMPLE_SURVEY_CSV.name)
        return resolve_survey_columns(loader.load_raw(SAMPLE_SURVEY_CSV), "sample:synthetic_survey"), labels

    if not uploaded_files:
        return None, labels

    paths = [(save_uploaded_file(f), f.name) for f in uploaded_files]
    labels = [p[1] for p in paths]
    file_key = "|".join(labels)
    if st.session_state.get("survey_merge_key") == file_key and "survey_merged" in st.session_state:
        return st.session_state["survey_merged"], labels

    try:
        if len(paths) == 1:
            merged = resolve_survey_columns(loader.load_raw(paths[0][0]), file_key)
        else:
            merged = load_and_merge_surveys(loader, paths)
        if merged is not None:
            st.session_state["survey_merge_key"] = file_key
            st.session_state["survey_merged"] = merged
        return merged, labels
    except ValueError as exc:
        st.error(str(exc))
        return None, labels


def _bha_key(files) -> str:
    return "|".join(uploaded_file_key(f) for f in files)


st.set_page_config(page_title="Wellbore Tortuosity Platform", layout="wide", initial_sidebar_state="expanded")
inject_theme()

with st.sidebar:
    st.header("Data inputs")
    use_sample = st.button("Load sample survey", use_container_width=True)
    survey_files = st.file_uploader(
        "Survey file(s)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="survey_files",
    )
    bha_files = st.file_uploader(
        "BHA PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="bha_files",
    )

    st.header("Section filter")
    filt_v = st.checkbox("V — Vertical", True)
    filt_c = st.checkbox("C — Curve / build", True)
    filt_l = st.checkbox("L — Lateral", True)
    section_codes = [c for c, on in zip(["V", "C", "L"], [filt_v, filt_c, filt_l]) if on]
    show_hp = st.checkbox("Show H&P branding", value=True)

    if SAMPLE_SURVEY_CSV.is_file():
        st.caption(f"Sample: `{SAMPLE_SURVEY_CSV.relative_to(PROJECT_ROOT).as_posix()}`")

if use_sample:
    st.session_state.pop("survey_merged", None)
    st.session_state.pop("survey_merge_key", None)

df_survey, upload_labels = _load_surveys(survey_files if survey_files else None, use_sample)
if df_survey is None:
    st.info("Upload one or more survey files, or load the sample well.")
    st.stop()

bha_intervals = pd.DataFrame()
if bha_files:
    bk = _bha_key(bha_files)
    if st.session_state.get("bha_key") != bk:
        st.session_state["bha_key"] = bk
        paths = [(save_uploaded_file(f), f.name) for f in bha_files]
        st.session_state["bha_intervals_raw"] = extract_bha_runs_from_uploads(paths)
        st.session_state.pop("bha_intervals", None)

if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
    bha_intervals = st.session_state.get(
        "bha_intervals",
        normalize_bha_intervals(st.session_state["bha_intervals_raw"]),
    )
else:
    bha_intervals = st.session_state.get("bha_intervals", pd.DataFrame())

result = run_analytics_pipeline(df_survey, bha_intervals if not bha_intervals.empty else None)
df_full = result.survey
df = filter_by_section_codes(df_full, section_codes)
quality = SurveyQualityAnalyzer().evaluate(df_full)
plots = WellPlots()

if df.empty and section_codes:
    st.warning("No stations match the selected section filters.")
    st.stop()

render_header(show_hp_branding=show_hp)

tab_overview, tab_sections, tab_systems, tab_survey, tab_compare = st.tabs(
    [
        "Overview",
        "Section Analysis",
        "Drilling Systems",
        "Survey Data",
        "Comparisons",
    ]
)

with tab_overview:
    render_overview_tab(df_full, df, result, quality, plots)

with tab_sections:
    render_section_analysis_tab(df_full, plots)

with tab_systems:
    if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
        st.markdown("##### Map BHA intervals to survey MD")
        editor_df = prepare_bha_for_editor(st.session_state["bha_intervals_raw"])
        edited = st.data_editor(
            editor_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config=bha_editor_column_config(),
            key="bha_intervals_editor",
        )
        if st.button("Apply BHA MD mapping", type="primary"):
            st.session_state["bha_intervals"] = normalize_bha_intervals(prepare_bha_from_editor(edited))
            st.rerun()
    render_drilling_systems_tab(df_full, result, bha_intervals, plots)

with tab_survey:
    render_survey_data_tab(df_survey, df_full, quality, upload_labels)

with tab_compare:
    render_comparisons_tab(df, df_full, result, plots)
