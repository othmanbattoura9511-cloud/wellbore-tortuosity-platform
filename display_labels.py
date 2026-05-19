"""User-facing column labels and Streamlit-safe dataframe helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

# Internal column name -> dashboard label
SURVEY_DISPLAY_NAMES = {
    "MD": "MD (m)",
    "MD_In": "MD In",
    "MD_Out": "MD Out",
    "Inclination": "Inclination (°)",
    "Azimuth": "Azimuth (°)",
    "DLS": "DLS (°/30m)",
    "Well_Section": "Well section",
    "Section_Code": "Section",
    "Section_Confidence": "Trajectory stability",
    "Build_Rate": "Build rate (°/30m)",
    "Turn_Rate": "Turn rate (°/30m)",
    "Tortuosity_Index": "Tortuosity index",
    "Tortuosity_Index_Local": "Tortuosity index",
    "Wellbore_Smoothness": "Wellbore smoothness",
    "Stability": "Drilling stability",
    "DLS_Stability": "DLS stability",
    "Inclination_Control": "Inclination control",
    "Azimuth_Control": "Azimuth control",
    "Build_Efficiency": "Build efficiency",
    "Lateral_Smoothness": "Lateral smoothness",
    "Drilling_System": "Drilling system",
    "Hole_Size": "Hole size",
    "RSS_Type": "RSS steering",
    "BHA": "BHA",
    "BHA_Run": "BHA run",
    "Survey_File": "Survey file",
    "Primary_Concern": "Trajectory behavior",
    "Trajectory_Severity": "Trajectory quality",
    "Pattern_Type": "Trajectory behavior",
    "Pattern_Confidence": "Trajectory stability",
    "RSS_Confidence": "RSS confidence",
    "RSS_Severity": "RSS severity",
    "Steering_Stability": "Steering stability",
}

BHA_DISPLAY_NAMES = {
    "BHA_Run": "BHA run",
    "Source_File": "Source file",
    "MD_In": "MD In",
    "MD_Out": "MD Out",
    "Drilling_System": "Drilling system",
    "RSS_Type": "RSS steering",
    "Hole_Size": "Hole size",
    "Bit_Type": "Bit type",
    "BHA_Config": "BHA configuration",
    "Interval_Complete": "Interval complete",
    "Mean_DLS": "Mean DLS (°/30m)",
    "Max_DLS": "Max DLS (°/30m)",
    "Mean_Tortuosity": "Mean tortuosity",
    "Performance_Score": "Performance score",
    "Rank": "Rank",
    "Stations": "Survey stations",
    "Steering_Stability": "Steering stability",
    "RSS_Severity": "RSS severity",
}


def rename_for_display(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    mapping = mapping or SURVEY_DISPLAY_NAMES
    cols = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=cols)


def _serialize_cell(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            return str(value)
    return value


def make_streamlit_safe_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    """Convert a dataframe to types compatible with Streamlit/pyarrow display."""
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" | ".join(str(p) for p in col if str(p) != "") for col in out.columns]

    for col in out.columns:
        series = out[col]
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            out[col] = series.astype("boolean")
            continue

        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            continue

        if pd.api.types.is_datetime64_any_dtype(dtype):
            out[col] = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            out[col] = out[col].fillna("")
            continue

        if isinstance(dtype, pd.CategoricalDtype):
            out[col] = series.astype(str).replace({"nan": "", "None": ""})
            continue

        converted = series.map(_serialize_cell)
        if pd.api.types.is_numeric_dtype(converted.dtype):
            out[col] = converted
        else:
            out[col] = converted.astype(str).replace({"nan": "", "None": "", "<NA>": ""})

    return out


def show_dataframe(
    df: pd.DataFrame | None,
    *,
    use_display_names: bool = True,
    max_rows: int | None = None,
    **kwargs: Any,
) -> None:
    """Display a dataframe in Streamlit after sanitization for pyarrow."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.caption("No data available for this view.")
        return
    work = df.head(max_rows) if max_rows else df
    if use_display_names:
        work = rename_for_display(work)
    opts = {"use_container_width": True, "hide_index": True}
    opts.update(kwargs)
    st.dataframe(make_streamlit_safe_dataframe(work), **opts)
