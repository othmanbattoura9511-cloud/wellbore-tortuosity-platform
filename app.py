"""Wellbore Tortuosity Analytics Platform — Streamlit dashboard (well-agnostic)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import comparison_table, run_analytics_pipeline
from bha_analytics import bha_vs_section_comparison, compare_bha_groups
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
from sections import SECTION_CODES, filter_by_section_codes
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
    st.caption(analysis.message)

    if analysis.file_type not in (SurveyFileType.SURVEY_STATION, SurveyFileType.INTERVAL_SUMMARY):
        st.warning(analysis.message)
        st.markdown(
            "**Required survey columns:** MD (or MD Start/End for intervals), Inclination, Azimuth. "
            "Vendor names from Halliburton, Schlumberger, Baker Hughes, Landmark, Compass, and generic Excel exports are supported."
        )
        st.dataframe(df.head(15), use_container_width=True)
        return None

    if not analysis.missing:
        st.session_state[mapped_key] = analysis.survey
        if analysis.mapped_columns:
            with st.expander("Auto-detected column mapping", expanded=False):
                st.json(analysis.mapped_columns)
                if analysis.fuzzy_scores:
                    st.caption({k: round(v, 2) for k, v in analysis.fuzzy_scores.items()})
        return analysis.survey

    st.error("Could not automatically map all required survey columns.")
    st.markdown(missing_columns_message(analysis.missing, analysis.survey.columns))
    st.dataframe(df.head(15), use_container_width=True)

    col_options = list(df.columns)
    st.subheader("Manual column mapping")
    c1, c2, c3 = st.columns(3)
    md_pick = c1.selectbox(
        "MD column", col_options, index=suggest_column_index(col_options, "MD"), key=f"map_md_{file_key}"
    )
    inc_pick = c2.selectbox(
        "Inclination column",
        col_options,
        index=suggest_column_index(col_options, "Inclination"),
        key=f"map_inc_{file_key}",
    )
    azi_pick = c3.selectbox(
        "Azimuth column",
        col_options,
        index=suggest_column_index(col_options, "Azimuth"),
        key=f"map_azi_{file_key}",
    )
    if st.button("Apply column mapping", type="primary", key=f"apply_map_{file_key}"):
        remapped = analyze_survey_columns(df, manual_map={"MD": md_pick, "Inclination": inc_pick, "Azimuth": azi_pick})
        if remapped.missing:
            st.error(missing_columns_message(remapped.missing, remapped.survey.columns))
        else:
            st.session_state[mapped_key] = remapped.survey
            st.success("Column mapping applied.")
            st.rerun()
    return None


def _bha_upload_key(bha_files) -> str:
    return "|".join(uploaded_file_key(f) for f in bha_files)


def _load_bha_from_uploads(bha_files) -> pd.DataFrame:
    paths = [(save_uploaded_file(f), f.name) for f in bha_files]
    extracted = extract_bha_runs_from_uploads(paths)
    return normalize_bha_intervals(extracted)


def _show_comparison_table(df: pd.DataFrame, group_cols: list[str], title: str) -> None:
    st.subheader(title)
    table = comparison_table(df, group_cols)
    if table.empty:
        st.caption("No data for this comparison.")
    else:
        st.dataframe(table, use_container_width=True)


st.set_page_config(page_title="Wellbore Tortuosity Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Wellbore Tortuosity Analytics Platform")
st.caption("V / C / L sections · multi-BHA PDFs · patterns · RSS · performance ranking")

with st.sidebar:
    st.header("Data inputs")
    if st.button("Load sample survey", use_container_width=True):
        st.session_state["survey_source"] = "sample"
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("survey_file_key", None)
        _clear_survey_mapping_state()
        st.rerun()

    uploaded_file = st.file_uploader("Survey file", type=["xlsx", "xls", "csv"], key="survey_file_uploader")
    if uploaded_file is not None:
        file_key = uploaded_file_key(uploaded_file)
        if st.session_state.get("survey_file_key") != file_key:
            st.session_state["survey_source"] = "upload"
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["survey_file_key"] = file_key
            _clear_survey_mapping_state()

    bha_files = st.file_uploader(
        "BHA PDFs (optional, multiple runs)",
        type=["pdf"],
        accept_multiple_files=True,
        key="bha_upload",
    )
    if bha_files:
        upload_key = _bha_upload_key(bha_files)
        if st.session_state.get("bha_upload_key") != upload_key:
            st.session_state["bha_upload_key"] = upload_key
            st.session_state["bha_intervals_raw"] = _load_bha_from_uploads(bha_files)

    st.header("Section filter (V / C / L)")
    show_v = st.checkbox("V — Vertical", value=True, key="filter_v")
    show_c = st.checkbox("C — Curve", value=True, key="filter_c")
    show_l = st.checkbox("L — Lateral", value=True, key="filter_l")
    section_filter_codes = []
    if show_v:
        section_filter_codes.append("V")
    if show_c:
        section_filter_codes.append("C")
    if show_l:
        section_filter_codes.append("L")

    if SAMPLE_SURVEY_CSV.is_file():
        st.caption(f"Sample: `{SAMPLE_SURVEY_CSV.relative_to(PROJECT_ROOT).as_posix()}`")

# --- Load survey ---
survey_source = st.session_state.get("survey_source")
df_survey = None
if survey_source == "sample":
    if not SAMPLE_SURVEY_CSV.is_file():
        st.error("Sample not found. Run `python generate_sample_data.py`.")
        st.stop()
    file_key = "sample:synthetic_survey"
    df_survey = resolve_survey_columns(SurveyDataLoader().load_raw(SAMPLE_SURVEY_CSV), file_key)
elif survey_source == "upload":
    uploaded = st.session_state.get("uploaded_file")
    if uploaded:
        file_key = st.session_state.get("survey_file_key", uploaded.name)
        df_survey = resolve_survey_columns(SurveyDataLoader().load_raw(save_uploaded_file(uploaded)), file_key)

if df_survey is None:
    st.info("Upload a survey or load the sample dataset from the sidebar.")
    st.stop()

# --- BHA intervals (editable) ---
bha_intervals = pd.DataFrame()
if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
    st.subheader("BHA runs — review & edit MD intervals")
    st.caption("Enter MD_In / MD_Out when missing from PDF, then apply to remap survey stations.")
    editor_cols = [c for c in BHA_COLUMNS if c in st.session_state["bha_intervals_raw"].columns]
    edited = st.data_editor(
        st.session_state["bha_intervals_raw"][editor_cols],
        use_container_width=True,
        num_rows="dynamic",
        key="bha_intervals_editor",
    )
    if st.button("Apply BHA intervals & remap survey", type="primary", key="apply_bha_intervals"):
        st.session_state["bha_intervals"] = normalize_bha_intervals(edited)
        st.success("BHA intervals applied.")
        st.rerun()
    bha_intervals = st.session_state.get(
        "bha_intervals",
        normalize_bha_intervals(st.session_state["bha_intervals_raw"]),
    )
else:
    bha_intervals = st.session_state.get("bha_intervals", pd.DataFrame())

result = run_analytics_pipeline(df_survey, bha_intervals if not bha_intervals.empty else None)
df_full = result.survey
df = filter_by_section_codes(df_full, section_filter_codes)
quality = SurveyQualityAnalyzer().evaluate(df_full)
plots = WellPlots()

if df.empty and section_filter_codes:
    st.warning("No survey stations match the selected section filters. Enable V, C, or L in the sidebar.")
    st.stop()

# --- Tabs ---
tab_overview, tab_sections, tab_compare, tab_patterns, tab_bha, tab_survey = st.tabs(
    ["Overview", "Section analysis", "Comparisons", "Trajectory & tortuosity", "BHA & RSS", "Survey data"]
)

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stations", len(df))
    c2.metric("MD end (m)", f"{quality.get('md_end', df_full['MD'].max()):.0f}")
    c3.metric("Max DLS", f"{quality.get('max_dls', df_full['DLS'].max()):.2f}")
    c4.metric("Section confidence", f"{result.section_summary.get('mean_confidence', 0):.2f}")
    c5.metric("Dominant section", str(result.section_summary.get("dominant_section", "—")))

    st.plotly_chart(plots.plot_inclination(df, color_col="Well_Section"), use_container_width=True)
    st.plotly_chart(plots.plot_dls(df, color_col="Well_Section"), use_container_width=True)
    st.plotly_chart(plots.plot_section_distribution(df), use_container_width=True)

with tab_sections:
    st.subheader("Section classification (V / C / L)")
    st.json(result.section_summary)
    st.dataframe(
        df_full[
            ["MD", "Inclination", "Azimuth", "DLS", "Build_Rate", "Well_Section", "Section_Code", "Section_Confidence"]
        ].head(200),
        use_container_width=True,
    )
    fig = plots.plot_tortuosity_by_section(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    _show_comparison_table(df, ["Section_Code", "Well_Section"], "V vs C vs L")
    _show_comparison_table(df, ["Drilling_System"], "Motor vs RSS vs Rotary")
    _show_comparison_table(df, ["RSS_Type"], "Push-the-bit vs Point-the-bit")
    _show_comparison_table(df, ["Hole_Size"], "Hole size")
    _show_comparison_table(df, ["Trajectory_Severity"], "Trajectory severity (Stable → Critical)")
    _show_comparison_table(df, ["Primary_Concern"], "Primary engineering concern")
    _show_comparison_table(df, ["Section_Code", "Drilling_System"], "Section × drilling system")
    _show_comparison_table(df, ["Section_Code", "RSS_Type"], "Section × RSS type")
    _show_comparison_table(df, ["Section_Code", "Hole_Size"], "Section × hole size")
    _show_comparison_table(df, ["Section_Code", "Trajectory_Severity"], "Section × trajectory severity")
    if "BHA_Run" in df.columns and df["BHA_Run"].nunique() > 1:
        _show_comparison_table(df, ["BHA_Run"], "BHA run comparison (survey-mapped)")

with tab_patterns:
    st.subheader("Engineering trajectory quality")
    defs = result.pattern_summary.get("severity_definitions", {})
    if defs:
        st.markdown(
            "**Severity levels:** "
            + " · ".join(f"**{k}** — {v}" for k, v in defs.items())
        )
    c1, c2, c3, c4 = st.columns(4)
    sev_counts = result.pattern_summary.get("severity_counts", {})
    c1.metric("Stable", sev_counts.get("Stable", 0))
    c2.metric("Moderate", sev_counts.get("Moderate", 0))
    c3.metric("Aggressive", sev_counts.get("Aggressive", 0))
    c4.metric("Critical", sev_counts.get("Critical", 0))

    st.plotly_chart(plots.plot_severity_along_md(df), use_container_width=True)
    st.plotly_chart(plots.plot_severity_by_section(df), use_container_width=True)
    st.plotly_chart(plots.plot_kpi_timeline(df, "Oscillation_Score"), use_container_width=True)
    st.plotly_chart(plots.plot_kpi_timeline(df, "Steering_Smoothness_Score"), use_container_width=True)
    st.plotly_chart(plots.plot_tortuosity_map(df), use_container_width=True)

    st.subheader("Interval KPIs (by V / C / L section)")
    interval_rows = result.pattern_summary.get("interval_kpis", [])
    if interval_rows:
        st.dataframe(pd.DataFrame(interval_rows), use_container_width=True)
    else:
        st.caption("Section intervals will appear after survey classification.")

    st.subheader("Station diagnostics (engineering language)")
    diag_cols = [
        c
        for c in [
            "MD",
            "Well_Section",
            "Section_Code",
            "Trajectory_Severity",
            "Primary_Concern",
            "Engineering_Diagnostics",
            "Mean_DLS_Local",
            "DLS_Variance",
            "Oscillation_Score",
            "Steering_Smoothness_Score",
            "Tortuosity_Index",
            "Slide_Rotate_Tendency",
        ]
        if c in df_full.columns
    ]
    st.dataframe(df_full[diag_cols].head(150), use_container_width=True)

    with st.expander("Advanced pattern diagnostics (optional — expert review only)"):
        st.caption("Legacy sinusoidal / helical / micro-tortuosity proxies — not used for primary operational labels.")
        adv_cols = [c for c in df_full.columns if str(c).startswith("Diagnostic_Advanced_")]
        if adv_cols:
            st.dataframe(df_full[["MD"] + adv_cols].head(100), use_container_width=True)

with tab_bha:
    if not result.bha_runs.empty:
        st.subheader("BHA runs table")
        st.dataframe(result.bha_runs, use_container_width=True)

        tl = plots.plot_bha_timeline(result.bha_runs, float(df_full["MD"].max()))
        if tl:
            st.plotly_chart(tl, use_container_width=True)

        sec_cmp = bha_vs_section_comparison(df_full, bha_intervals)
        if not sec_cmp.empty:
            st.subheader("BHA vs section (V / C / L)")
            st.dataframe(sec_cmp, use_container_width=True)

        st.subheader("Motor vs RSS")
        st.dataframe(compare_bha_groups(result.bha_runs, "Drilling_System"), use_container_width=True)

        st.subheader("Push-the-bit vs Point-the-bit")
        st.dataframe(compare_bha_groups(result.bha_runs, "RSS_Type"), use_container_width=True)

        st.subheader("Hole size comparison")
        st.dataframe(compare_bha_groups(result.bha_runs, "Hole_Size"), use_container_width=True)

        if not result.bha_ranking.empty:
            st.subheader("BHA performance ranking")
            st.dataframe(result.bha_ranking, use_container_width=True)
            rank_fig = plots.plot_bha_ranking(result.bha_ranking)
            if rank_fig:
                st.plotly_chart(rank_fig, use_container_width=True)
    else:
        st.caption("Upload one or more BHA PDFs in the sidebar. Edit MD_In / MD_Out if not detected, then apply.")

    st.subheader("RSS summary (survey)")
    st.json(result.rss_summary)
    st.plotly_chart(plots.plot_rss_distribution(df_full), use_container_width=True)

with tab_survey:
    st.write(quality)
    st.dataframe(df, use_container_width=True)
