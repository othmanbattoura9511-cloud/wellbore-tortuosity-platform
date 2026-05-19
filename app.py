"""Wellbore Tortuosity Analytics Platform — Streamlit dashboard (well-agnostic)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import pdfplumber
import streamlit as st

from analytics import comparison_table, run_analytics_pipeline
from column_mapping import (
    apply_manual_column_map,
    missing_columns_message,
    standardize_survey_columns,
    suggest_column_index,
)
from loaders import SurveyDataLoader
from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV
from plots import WellPlots
from survey_quality import SurveyQualityAnalyzer


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".dat"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return Path(tmp.name)


def _clear_survey_mapping_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("survey_mapped_"):
            del st.session_state[key]


def resolve_survey_columns(df: pd.DataFrame, file_key: str) -> pd.DataFrame | None:
    mapped_key = f"survey_mapped_{file_key}"
    if mapped_key in st.session_state:
        return st.session_state[mapped_key]

    df_mapped, missing = standardize_survey_columns(df)
    if not missing:
        st.session_state[mapped_key] = df_mapped
        return df_mapped

    st.error("Could not automatically identify all required survey columns.")
    st.markdown(missing_columns_message(missing, df.columns))
    st.dataframe(df.head(10), use_container_width=True)

    col_options = list(df.columns)
    st.subheader("Manual column mapping")
    c1, c2, c3 = st.columns(3)
    md_pick = c1.selectbox("MD column", col_options, index=suggest_column_index(col_options, "MD"), key=f"map_md_{file_key}")
    inc_pick = c2.selectbox(
        "Inclination column", col_options, index=suggest_column_index(col_options, "Inclination"), key=f"map_inc_{file_key}"
    )
    azi_pick = c3.selectbox(
        "Azimuth column", col_options, index=suggest_column_index(col_options, "Azimuth"), key=f"map_azi_{file_key}"
    )
    if st.button("Apply column mapping", type="primary", key=f"apply_map_{file_key}"):
        df_manual = apply_manual_column_map(df, md_pick, inc_pick, azi_pick)
        _, still_missing = standardize_survey_columns(df_manual)
        if still_missing:
            st.error(missing_columns_message(still_missing, df_manual.columns))
        else:
            st.session_state[mapped_key] = df_manual
            st.rerun()
    return None


def extract_bha_intervals_from_pdfs(bha_files) -> pd.DataFrame:
    rows = []
    for bha_file in bha_files:
        with pdfplumber.open(save_uploaded_file(bha_file)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        text_upper = text.upper() if text else ""
        md_numbers = re.findall(r"\b\d+\.\d+|\b\d+\b", text)
        md_in = float(md_numbers[0]) if len(md_numbers) >= 1 else None
        md_out = float(md_numbers[1]) if len(md_numbers) >= 2 else None

        system = "Unknown"
        if "MOTOR" in text_upper:
            system = "Motor"
        elif "RSS" in text_upper or "ROTARY STEER" in text_upper:
            system = "RSS"
        elif "ROTARY" in text_upper:
            system = "Rotary"

        rss_type = "Unknown RSS Type"
        if any(k in text_upper for k in ("PUSH", "PAD", "RIB", "LUCIDA")):
            rss_type = "Push-the-bit"
        elif any(k in text_upper for k in ("POINT", "TILT", "BIT TILT")):
            rss_type = "Point-the-bit"

        rows.append(
            {
                "BHA": f"BHA {len(rows) + 1}",
                "MD_In": md_in,
                "MD_Out": md_out,
                "Drilling_System": system,
                "Hole_Size": "Unknown",
                "Bit_Type": "Unknown",
                "RSS_Type": rss_type,
            }
        )
    return pd.DataFrame(rows)


def _show_comparison_table(df: pd.DataFrame, group_cols: list[str], title: str) -> None:
    st.subheader(title)
    table = comparison_table(df, group_cols)
    if table.empty:
        st.caption("No data for this comparison.")
    else:
        st.dataframe(table, use_container_width=True)


st.set_page_config(page_title="Wellbore Tortuosity Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Wellbore Tortuosity Analytics Platform")
st.caption("Adaptive section classification · patterns · RSS · multi-well comparisons")

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
        file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        if st.session_state.get("survey_file_key") != file_key:
            st.session_state["survey_source"] = "upload"
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["survey_file_key"] = file_key
            _clear_survey_mapping_state()

    bha_files = st.file_uploader("BHA PDF (optional)", type=["pdf"], accept_multiple_files=True, key="bha_upload")

    if SAMPLE_SURVEY_CSV.is_file():
        st.caption(f"Sample: `{SAMPLE_SURVEY_CSV.relative_to(PROJECT_ROOT).as_posix()}`")

# --- Load survey ---
survey_source = st.session_state.get("survey_source")
df_survey = None
if survey_source == "sample":
    if not SAMPLE_SURVEY_CSV.is_file():
        st.error(f"Sample not found. Run `python generate_sample_data.py`.")
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

bha_intervals = extract_bha_intervals_from_pdfs(bha_files) if bha_files else pd.DataFrame()
result = run_analytics_pipeline(df_survey, bha_intervals if not bha_intervals.empty else None)
df = result.survey
quality = SurveyQualityAnalyzer().evaluate(df)
plots = WellPlots()

# --- Tabs ---
tab_overview, tab_sections, tab_compare, tab_patterns, tab_bha, tab_survey = st.tabs(
    ["Overview", "Section analysis", "Comparisons", "Patterns & tortuosity", "BHA & RSS", "Survey data"]
)

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stations", quality.get("n_surveys", len(df)))
    c2.metric("MD end (m)", f"{quality.get('md_end', df['MD'].max()):.0f}")
    c3.metric("Max DLS", f"{quality.get('max_dls', df['DLS'].max()):.2f}")
    c4.metric("Section confidence", f"{result.section_summary.get('mean_confidence', 0):.2f}")
    c5.metric("Dominant section", str(result.section_summary.get("dominant_section", "—")))

    st.plotly_chart(plots.plot_inclination(df, color_col="Well_Section"), use_container_width=True)
    st.plotly_chart(plots.plot_dls(df, color_col="Well_Section"), use_container_width=True)
    st.plotly_chart(plots.plot_section_distribution(df), use_container_width=True)

with tab_sections:
    st.subheader("Automatic section classification")
    st.json(result.section_summary)
    st.dataframe(
        df[["MD", "Inclination", "Azimuth", "DLS", "Build_Rate", "Turn_Rate", "Well_Section", "Section_Confidence"]].head(200),
        use_container_width=True,
    )
    fig = plots.plot_tortuosity_by_section(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    _show_comparison_table(df, ["Well_Section"], "Vertical / Curve / Horizontal (V·C·H)")
    _show_comparison_table(df, ["Well_Section", "Drilling_System"], "Drilling system vs section")
    _show_comparison_table(df, ["RSS_Type"], "RSS push vs point")
    _show_comparison_table(df, ["Hole_Size"], "Hole size comparison")
    _show_comparison_table(df, ["BHA"], "BHA comparison")
    _show_comparison_table(df, ["Pattern_Type"], "Pattern type comparison")
    _show_comparison_table(df, ["Well_Section", "Pattern_Type"], "Pattern by section")

with tab_patterns:
    st.json(result.pattern_summary)
    st.plotly_chart(plots.plot_pattern_by_section(df), use_container_width=True)
    st.plotly_chart(plots.plot_tortuosity_map(df), use_container_width=True)

with tab_bha:
    st.json(result.rss_summary)
    st.plotly_chart(plots.plot_rss_distribution(df), use_container_width=True)
    if not bha_intervals.empty:
        st.subheader("Extracted BHA intervals (from PDF)")
        st.dataframe(bha_intervals, use_container_width=True)
    else:
        st.caption("Upload BHA PDFs in the sidebar to map drilling system / RSS by MD interval.")

with tab_survey:
    st.write(quality)
    st.dataframe(df, use_container_width=True)
