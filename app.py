import re
import tempfile
from pathlib import Path

import pandas as pd
import pdfplumber
import streamlit as st
from loaders import SurveyDataLoader
from preprocess import SurveyPreprocessor
from sections import WellSectionClassifier
from survey_quality import SurveyQualityAnalyzer
from tortuosity import TortuosityAnalyzer
from plots import WellPlots


def save_uploaded_file(uploaded_file):
    """Persist Streamlit UploadedFile to a temp path for pathlib-based loaders."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


st.set_page_config(page_title="Wellbore Tortuosity Platform", layout="wide")
st.title("Wellbore Tortuosity Analytics Platform")
st.caption("Engineering-oriented survey, DLS, tortuosity and T&D&B impact analysis")
bha_info = {}

bha_runs = pd.DataFrame([
    {
        "BHA": "BHA 1",
        "MD_In": 593,
        "MD_Out": 2080,
        "Drilling_System": "Motor",
        "Hole_Size": "8.75",
        "Bit_Type": "P506",
    },
    {
        "BHA": "BHA 2",
        "MD_In": 2080,
        "MD_Out": 6172,
        "Drilling_System": "Unknown",
        "Hole_Size": "8.75",
        "Bit_Type": "Unknown",
    }
])
uploaded = st.file_uploader(
    "Upload survey file",
    type=["xlsx", "xls", "csv"]
)
bha_files = st.file_uploader(
    "Upload BHA Report PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if bha_files:
    for bha_file in bha_files:
        st.write(bha_file.name)
    text = ""

    with pdfplumber.open(save_uploaded_file(bha_file)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    bha_info = {}

    text_upper = text.upper()

    if "MOTOR" in text_upper:
        bha_info["Drilling_System"] = "Motor"

    if "RSS" in text_upper or "NAVIGAMMA" in text_upper:
        bha_info["RSS"] = "Yes"

    if "8.75" in text:
        bha_info["Hole_Size"] = "8.75"

    if "P506" in text_upper:
        bha_info["Bit_Type"] = "P506"

    st.subheader("Extracted BHA Information")

    st.json(bha_info)
    st.subheader("BHA Runs / MD Intervals")

    bha_runs = pd.DataFrame([
        {
            "BHA": "BHA 1",
            "MD_In": 593,
            "MD_Out": 2080,
            "Drilling_System": bha_info.get("Drilling_System", "Motor"),
            "Hole_Size": bha_info.get("Hole_Size", "8.75"),
            "Bit_Type": bha_info.get("Bit_Type", "P506"),
        },
        {
            "BHA": "BHA 2",
            "MD_In": 2080,
            "MD_Out": 6172,
            "Drilling_System": "Unknown",
            "Hole_Size": "8.75",
            "Bit_Type": "Unknown",
        }
    ])

    bha_runs = st.data_editor(
        bha_runs,
        num_rows="dynamic",
        use_container_width=True
    )
    st.subheader("Extracted BHA Text")

    st.text_area(
        "BHA Content",
        text[:5000],
        height=300
    )
bha_pdf_rows = []
if bha_files:
    for bha_file in bha_files:
        st.success(f"BHA PDF uploaded: {bha_file.name}")
        text = ""

        with pdfplumber.open(save_uploaded_file(bha_file)) as pdf:
            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

        text_upper = text.upper()
        md_in = None
        md_out = None

        md_numbers = re.findall(r"\b\d+\.\d+|\b\d+\b", text)

        if len(md_numbers) >= 2:
            md_in = float(md_numbers[0])
            md_out = float(md_numbers[1])
        system = "Unknown"
        bit_type = "Unknown"
        hole_size = "Unknown"

        if "MOTOR" in text_upper:
            system = "Motor"

        if "RSS" in text_upper or "LUCIDA" in text_upper:
            system = "RSS"

        if "P506" in text_upper:
            bit_type = "P506"

        if "8.75" in text:
            hole_size = "8.75"

    bha_pdf_rows.append({
        "PDF": bha_file.name,
        "BHA": f"BHA {len(bha_pdf_rows) + 1}",
        "MD_In": md_in,
        "MD_Out": md_out,
        "Drilling_System": system,
        "Hole_Size": hole_size,
        "Bit_Type": bit_type
    })
if bha_pdf_rows:
    st.subheader("Extracted BHA Summary from PDFs")
    st.dataframe(pd.DataFrame(bha_pdf_rows), use_container_width=True)
if uploaded is not None:
    st.session_state["uploaded_file"] = uploaded

uploaded = st.session_state.get("uploaded_file", None)
if uploaded:
    survey_path = save_uploaded_file(uploaded)
    df = SurveyDataLoader().load(survey_path)
    pre = SurveyPreprocessor()
    df = pre.clean(df)
    df = pre.interpolate_missing(df)
    df = WellSectionClassifier().classify(df)
    if "DLS" not in df.columns:
        df = TortuosityAnalyzer().calculate_dls(df)
        df["DLS"] = df["DLS_Calc"]
    df = TortuosityAnalyzer().add_indicators(df)
    df["BHA"] = "Unknown"
    df["Drilling_System"] = "Unknown"
    df["Hole_Size"] = "Unknown"
    df["Bit_Type"] = "Unknown"

    for _, row in bha_runs.iterrows():
        mask = (df["MD"] >= row["MD_In"]) & (df["MD"] <= row["MD_Out"])
        df.loc[mask, "BHA"] = row["BHA"]
        df.loc[mask, "Drilling_System"] = row["Drilling_System"]
        df.loc[mask, "Hole_Size"] = row["Hole_Size"]
        df.loc[mask, "Bit_Type"] = row["Bit_Type"]
    quality = SurveyQualityAnalyzer().evaluate(df)
    st.write(quality)
    st.dataframe(df, use_container_width=True)
    st.subheader("Analysis by Drilling System")
    st.dataframe(
        df.groupby("Drilling_System")["DLS"].describe(),
        use_container_width=True
    )

    st.subheader("Analysis by Well Section")
    st.dataframe(
    df.groupby("Well_Section")["DLS"].describe(),
    use_container_width=True
    )

    st.subheader("Analysis by BHA")
    st.subheader("Comparison by Hole Size")

    hole_summary = (
        df.groupby("Hole_Size")["DLS"]
        .agg(["count", "mean", "max", "std"])
        .reset_index()
    )

    st.dataframe(hole_summary, use_container_width=True)


    st.subheader("Comparison by Drilling System")

    system_summary = (
        df.groupby("Drilling_System")["DLS"]
        .agg(["count", "mean", "max", "std"])
        .reset_index()
    )

    st.dataframe(system_summary, use_container_width=True)


    st.subheader("Comparison by BHA")

    bha_summary = (
        df.groupby("BHA")["DLS"]
        .agg(["count", "mean", "max", "std"])
        .reset_index()
    )

    st.dataframe(bha_summary, use_container_width=True)
    st.dataframe(
        df.groupby("BHA")["DLS"].describe(),
        use_container_width=True
    )
    st.subheader("Comparison by Hole Size")
    st.dataframe(
        df.groupby("Hole_Size")[["DLS", "Tortuosity_Index_Local", "Delta_MD"]].agg(
            ["count", "mean", "max", "std"]
        ),
        use_container_width=True
    )

    st.subheader("Comparison by Drilling System")
    st.dataframe(
        df.groupby("Drilling_System")[["DLS", "Tortuosity_Index_Local", "Delta_MD"]].agg(
            ["count", "mean", "max", "std"]
        ),
        use_container_width=True
    )

    st.subheader("Comparison by Well Section and Drilling System")
    st.dataframe(
        df.groupby(["Well_Section", "Drilling_System"])[["DLS", "Tortuosity_Index_Local"]].agg(
            ["count", "mean", "max", "std"]
        ),
        use_container_width=True
    )
    st.subheader("KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Surveys", quality["n_surveys"])

    c2.metric(
        "Mean Spacing",
        f"{quality['mean_spacing']:.2f}"
    )

    c3.metric(
        "Max DLS",
        f"{quality.get('max_dls', 0):.2f}"
    )

    c4.metric(
        "Mean DLS",
        f"{quality.get('mean_dls', 0):.2f}"
    )

    c5.metric(
        "Poor Spacing %",
        f"{quality['poor_spacing_pct']:.1f}%"
    )
    section_option = st.selectbox(
        "Select Well Section",
        ["All", "vertical", "curve", "lateral"]
    )
    if section_option == "All":
            filtered_df = df
    else:
            filtered_df = df[df["Well_Section"] == section_option]

            plots = WellPlots()

            st.plotly_chart(plots.plot_inclination(filtered_df, color_col="Well_Section"), use_container_width=True)
            st.plotly_chart(plots.plot_dls(filtered_df, color_col="Well_Section"), use_container_width=True)
            st.plotly_chart(plots.plot_azimuth(filtered_df, color_col="Well_Section"), use_container_width=True)
            st.plotly_chart(plots.plot_tortuosity_map(filtered_df), use_container_width=True)

            st.subheader("Cleaned Data Preview")
            st.dataframe(filtered_df, use_container_width=True)

