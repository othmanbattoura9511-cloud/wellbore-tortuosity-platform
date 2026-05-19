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
    EMPTY_SURVEY_MESSAGE,
    NO_SECTION_MESSAGE,
    NO_STATIONS_MESSAGE,
    render_comparisons_tab,
    render_drilling_systems_tab,
    render_overview_tab,
    render_section_analysis_tab,
    render_survey_data_tab,
)
from display_labels import show_data_editor
from loaders import SurveyDataLoader
from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV
from plots import WellPlots
from sections import filter_by_section_codes
from survey_merge import load_and_merge_surveys
from survey_quality import SurveyQualityAnalyzer
from ui_theme import (
    bha_editor_column_config,
    inject_theme,
    prepare_bha_for_editor,
    prepare_bha_from_editor,
    render_header,
)
from upload_utils import save_uploaded_file, uploaded_file_key


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


def _align_mapped_survey(mapped: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    if mapped is None or mapped.empty:
        return pd.DataFrame()
    if filtered is None or filtered.empty:
        return mapped.iloc[0:0].copy()
    if "Section_Code" in mapped.columns:
        codes = filtered["Section_Code"].dropna().unique().tolist() if "Section_Code" in filtered.columns else []
        if codes:
            return filter_by_section_codes(mapped, codes)
    if "MD" in mapped.columns and "MD" in filtered.columns:
        return mapped[mapped["MD"].isin(filtered["MD"])].copy()
    return mapped.copy()


st.set_page_config(page_title="Wellbore Tortuosity Platform", layout="wide", initial_sidebar_state="expanded")
inject_theme()
render_header()

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
    filt_c = st.checkbox("C — Curve", True)
    filt_l = st.checkbox("L — Lateral", True)
    section_codes = [c for c, on in zip(["V", "C", "L"], [filt_v, filt_c, filt_l]) if on]
    if SAMPLE_SURVEY_CSV.is_file():
        st.caption(f"Sample: `{SAMPLE_SURVEY_CSV.relative_to(PROJECT_ROOT).as_posix()}`")

if use_sample:
    st.session_state.pop("survey_merged", None)
    st.session_state.pop("survey_merge_key", None)

df_survey, upload_labels = _load_surveys(survey_files if survey_files else None, use_sample)
has_survey = df_survey is not None
has_section_selection = bool(section_codes)

bha_intervals = pd.DataFrame()
if has_survey and bha_files:
    bk = _bha_key(bha_files)
    if st.session_state.get("bha_key") != bk:
        st.session_state["bha_key"] = bk
        paths = [(save_uploaded_file(f), f.name) for f in bha_files]
        st.session_state["bha_intervals_raw"] = extract_bha_runs_from_uploads(paths)
        st.session_state.pop("bha_intervals", None)

if has_survey and "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
    bha_intervals = st.session_state.get(
        "bha_intervals",
        normalize_bha_intervals(st.session_state["bha_intervals_raw"]),
    )
elif has_survey:
    bha_intervals = st.session_state.get("bha_intervals", pd.DataFrame())

result = None
df_full = pd.DataFrame()
df = pd.DataFrame()
quality: dict = {}
plots = WellPlots()

if has_survey:
    result = run_analytics_pipeline(df_survey, bha_intervals if not bha_intervals.empty else None)
    df_full = result.survey
    if has_section_selection:
        df = filter_by_section_codes(df_full, section_codes)
    quality = SurveyQualityAnalyzer().evaluate(df if not df.empty else df_full)

df_mapped = _align_mapped_survey(df_survey, df) if has_survey and has_section_selection else (df_survey if has_survey else None)

tab_overview, tab_sections, tab_systems, tab_survey, tab_compare = st.tabs(
    [
        "Overview",
        "Section Analysis",
        "Drilling Systems",
        "Survey Data",
        "Comparisons",
    ]
)


def _tab_guard() -> bool:
    if not has_survey:
        st.info(EMPTY_SURVEY_MESSAGE)
        return False
    if not has_section_selection:
        st.warning(NO_SECTION_MESSAGE)
        return False
    if df.empty:
        st.warning(NO_STATIONS_MESSAGE)
        return False
    return True


with tab_overview:
    if _tab_guard() and result is not None:
        render_overview_tab(df, result, quality, plots, section_codes, df_full, bha_intervals)

with tab_sections:
    if _tab_guard():
        render_section_analysis_tab(df, plots, section_codes)

with tab_systems:
    if not has_survey:
        st.info(EMPTY_SURVEY_MESSAGE)
    elif not has_section_selection:
        st.warning(NO_SECTION_MESSAGE)
    else:
        if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
            st.markdown("##### Map BHA intervals to survey MD")
            editor_df = prepare_bha_for_editor(st.session_state["bha_intervals_raw"])
            edited = show_data_editor(
                editor_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config=bha_editor_column_config(),
                key="bha_intervals_editor",
                csv_filename="bha_intervals.csv",
            )
            if st.button("Apply BHA MD mapping", type="primary"):
                st.session_state["bha_intervals"] = normalize_bha_intervals(prepare_bha_from_editor(edited))
                st.rerun()
        if result is not None and not df.empty:
            render_drilling_systems_tab(df, result, bha_intervals, plots, section_codes)
        elif has_section_selection:
            st.warning(NO_STATIONS_MESSAGE)

with tab_survey:
    if not has_survey:
        st.info(EMPTY_SURVEY_MESSAGE)
    elif not has_section_selection:
        st.warning(NO_SECTION_MESSAGE)
    else:
        mapped_view = df_mapped if df_mapped is not None else pd.DataFrame()
        processed_view = df if not df.empty else pd.DataFrame()
        render_survey_data_tab(mapped_view, processed_view, quality, upload_labels)

with tab_compare:
    if _tab_guard() and result is not None:
        render_comparisons_tab(df, result, plots, section_codes)
