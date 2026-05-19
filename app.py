"""Wellbore Tortuosity Analytics Platform — drilling engineering dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import comparison_table, run_analytics_pipeline
from bha_analytics import incomplete_bha_runs
from bha_extraction import BHA_COLUMNS, extract_bha_runs_from_uploads, normalize_bha_intervals
from column_mapping import (
    SurveyFileType,
    analyze_survey_columns,
    missing_columns_message,
    suggest_column_index,
)
from loaders import SurveyDataLoader
from display_labels import rename_for_display
from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV
from plots import WellPlots
from sections import filter_by_section_codes
from ui_theme import (
    bha_editor_column_config,
    inject_theme,
    prepare_bha_for_editor,
    prepare_bha_from_editor,
    render_header,
    render_wellpath_hero,
    soft_warning,
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


def _show_table(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if df is None or df.empty:
        st.caption("No data available for this view.")
    else:
        st.dataframe(rename_for_display(df), use_container_width=True, hide_index=True)


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

df_survey = _load_surveys(survey_files if survey_files else None, use_sample)
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

tab_overview, tab_sections, tab_compare, tab_patterns, tab_bha, tab_survey = st.tabs(
    [
        "Overview",
        "Section analysis",
        "Comparisons",
        "Patterns & tortuosity",
        "BHA & RSS",
        "Survey data",
    ]
)

with tab_overview:
    render_wellpath_hero(df_full)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stations", len(df_full))
    c2.metric("MD end (m)", f"{quality.get('md_end', df_full['MD'].max()):.0f}")
    c3.metric("Max DLS", f"{quality.get('max_dls', df_full['DLS'].max()):.2f}")
    c4.metric("Avg tortuosity", f"{result.kpi_summary.get('mean_tortuosity', 0):.2f}")
    c5.metric("Dominant section", str(result.section_summary.get("dominant_section", "—")))

    st.plotly_chart(plots.plot_inclination_vs_md(df), use_container_width=True)
    st.plotly_chart(plots.plot_dls(df), use_container_width=True)
    tort_fig = plots.plot_tortuosity_vs_md(df)
    if tort_fig:
        st.plotly_chart(tort_fig, use_container_width=True)
    dist = plots.plot_section_distribution(df)
    if dist:
        st.plotly_chart(dist, use_container_width=True)

with tab_sections:
    st.markdown("Automatic **V / C / L** classification from inclination, DLS, and azimuth.")
    st.json(result.section_summary)
    sec_cols = [
        c
        for c in [
            "MD",
            "MD_In",
            "MD_Out",
            "Inclination",
            "Azimuth",
            "DLS",
            "Build_Rate",
            "Turn_Rate",
            "Well_Section",
            "Section_Code",
            "Section_Confidence",
        ]
        if c in df_full.columns
    ]
    st.dataframe(rename_for_display(df_full[sec_cols].head(250)), use_container_width=True)
    box = plots.plot_tortuosity_by_section(df)
    if box:
        st.plotly_chart(box, use_container_width=True)

with tab_compare:
    _show_table(result.section_comparison, "Section comparison (V / C / L × BHA)")
    _show_table(result.system_comparison, "Drilling system comparison (Motor vs RSS vs Rotary)")

    st.subheader("Detailed survey comparisons")
    _show_table(comparison_table(df, ["Section_Code", "Well_Section"]), "V vs C vs L")
    _show_table(comparison_table(df, ["Drilling_System"]), "Motor vs RSS vs Rotary")
    _show_table(comparison_table(df, ["RSS_Type"]), "Push-the-bit vs Point-the-bit")
    _show_table(comparison_table(df, ["Hole_Size"]), "Hole size")
    _show_table(comparison_table(df, ["Section_Code", "Drilling_System"]), "Section × drilling system")
    _show_table(comparison_table(df, ["Section_Code", "RSS_Type"]), "Section × RSS steering")
    _show_table(comparison_table(df, ["Section_Code", "Hole_Size"]), "Section × hole size")
    if "BHA_Run" in df.columns and df["BHA_Run"].nunique() > 1:
        _show_table(comparison_table(df, ["BHA_Run"]), "BHA run comparison")

with tab_patterns:
    st.markdown("Engineering trajectory KPIs — tortuosity, smoothness, stability, and control.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean DLS", f"{result.kpi_summary.get('mean_dls', 0):.2f}")
    k2.metric("Mean tortuosity", f"{result.kpi_summary.get('mean_tortuosity', 0):.2f}")
    k3.metric("Mean stability", f"{result.kpi_summary.get('mean_stability', 0):.2f}")
    k4.metric("Mean smoothness", f"{result.kpi_summary.get('mean_smoothness', 0):.2f}")

    st.plotly_chart(plots.plot_tortuosity_vs_md(df_full), use_container_width=True)
    smooth_fig = plots.plot_kpi_timeline(df_full, "Wellbore_Smoothness")
    if smooth_fig:
        st.plotly_chart(smooth_fig, use_container_width=True)
    stab_fig = plots.plot_kpi_timeline(df_full, "Stability")
    if stab_fig:
        st.plotly_chart(stab_fig, use_container_width=True)
    box = plots.plot_tortuosity_by_section(df)
    if box:
        st.plotly_chart(box, use_container_width=True)

with tab_bha:
    if not bha_intervals.empty:
        for _, row in incomplete_bha_runs(bha_intervals).iterrows():
            label = row.get("BHA_Run", row.get("Source_File", "BHA"))
            soft_warning(f"Please enter <b>MD In</b> and <b>MD Out</b> to map <b>{label}</b> to the survey.")

    st.subheader("BHA runs — edit MD intervals")
    if "bha_intervals_raw" in st.session_state and not st.session_state["bha_intervals_raw"].empty:
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
    else:
        st.caption("Upload BHA PDFs in the sidebar.")

    if not result.bha_runs.empty:
        st.subheader("BHA runs")
        st.dataframe(rename_for_display(result.bha_runs), use_container_width=True, hide_index=True)

        tl = plots.plot_bha_timeline(result.bha_runs, float(df_full["MD"].max()))
        if tl:
            st.plotly_chart(tl, use_container_width=True)

        ranked = result.bha_ranking
        if not ranked.empty and ranked["Rank"].notna().any():
            st.subheader("BHA performance ranking (complete intervals)")
            st.dataframe(
                ranked.sort_values("Rank", na_position="last"),
                use_container_width=True,
                hide_index=True,
            )
            rank_fig = plots.plot_bha_ranking(ranked)
            if rank_fig:
                st.plotly_chart(rank_fig, use_container_width=True)
    else:
        st.caption("Upload BHA PDFs to map assemblies to survey MD intervals.")

    st.subheader("Motor vs RSS")
    if not result.system_comparison.empty:
        st.dataframe(result.system_comparison, use_container_width=True, hide_index=True)
        fig = plots.plot_drilling_system_bar(result.system_comparison, "Avg Tortuosity")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        _show_table(comparison_table(df, ["Drilling_System"]), "Motor vs RSS (from survey)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Push-the-bit vs Point-the-bit")
        if not result.rss_steering_comparison.empty:
            st.dataframe(result.rss_steering_comparison, use_container_width=True, hide_index=True)
        else:
            st.caption("No RSS steering rows in section table.")
    with c2:
        st.subheader("Hole size comparison")
        if not result.hole_size_comparison.empty:
            st.dataframe(result.hole_size_comparison, use_container_width=True, hide_index=True)
        else:
            st.caption("No hole size data from BHA PDFs.")

    st.subheader("RSS summary")
    st.json(result.rss_summary)
    rss_pie = plots.plot_rss_distribution(df_full)
    if rss_pie:
        st.plotly_chart(rss_pie, use_container_width=True)

with tab_survey:
    st.write(quality)
    st.dataframe(rename_for_display(df_full), use_container_width=True)
