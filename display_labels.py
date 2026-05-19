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


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
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
    return str(value)


def _coerce_to_dataframe(df: Any) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        return df.to_frame().copy()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    try:
        return pd.DataFrame(df)
    except (TypeError, ValueError):
        return pd.DataFrame({"value": [str(df)]})


def _dedupe_column_names(columns: pd.Index) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flat = [" | ".join(str(p) for p in col if str(p) not in ("", "nan")) for col in columns]
    else:
        flat = [str(c) for c in columns]

    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in flat:
        base = name if name else "column"
        if base not in seen:
            seen[base] = 0
            unique.append(base)
        else:
            seen[base] += 1
            unique.append(f"{base}_{seen[base]}")
    return unique


def _column_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """Return a single Series for a column (handles duplicate column names)."""
    selected = frame[col]
    if isinstance(selected, pd.DataFrame):
        parts = []
        for sub_col in selected.columns:
            parts.append(selected[sub_col].map(_serialize_cell))
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + " | " + part
        return combined
    if not isinstance(selected, pd.Series):
        return pd.Series([_serialize_cell(selected)] * len(frame), index=frame.index)
    return selected


def _sanitize_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(series).map(_serialize_cell)

    dtype = getattr(series, "dtype", None)

    if pd.api.types.is_bool_dtype(dtype):
        return series.astype("boolean")

    if pd.api.types.is_integer_dtype(dtype):
        return series

    if pd.api.types.is_float_dtype(dtype):
        return series

    if pd.api.types.is_datetime64_any_dtype(dtype):
        text = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        return text.fillna("")

    if isinstance(dtype, pd.CategoricalDtype):
        return series.astype(str).replace({"nan": "", "None": "", "<NA>": ""}).fillna("")

    converted = series.map(_serialize_cell)
    return converted.fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})


def make_streamlit_safe_dataframe(df: Any) -> pd.DataFrame:
    """Convert input to a pyarrow-friendly DataFrame for Streamlit display."""
    safe = _coerce_to_dataframe(df)
    if safe.empty:
        return safe

    safe.columns = _dedupe_column_names(safe.columns)

    out = pd.DataFrame(index=safe.index)
    for col in safe.columns:
        out[col] = _sanitize_series(_column_series(safe, col))

    return out.reset_index(drop=True)


def _csv_download_button(df: pd.DataFrame, label: str, filename: str, key: str | None = None) -> None:
    try:
        payload = df.to_csv(index=False).encode("utf-8")
    except Exception:
        payload = make_streamlit_safe_dataframe(df).to_csv(index=False).encode("utf-8")
    st.download_button(label, payload, filename, "text/csv", key=key)


def show_dataframe(
    df: pd.DataFrame | None,
    *,
    use_display_names: bool = True,
    max_rows: int | None = None,
    csv_filename: str = "table_export.csv",
    **kwargs: Any,
) -> None:
    """Display a dataframe in Streamlit; fall back to CSV if rendering fails."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.caption("No data available for this view.")
        return

    work = df.head(max_rows) if max_rows else df.copy()
    if use_display_names:
        work = rename_for_display(work)

    opts = {"use_container_width": True, "hide_index": True}
    opts.update(kwargs)

    try:
        st.dataframe(make_streamlit_safe_dataframe(work), **opts)
    except Exception:
        st.warning("This table could not be rendered in the browser. Download the CSV to view the data.")
        _csv_download_button(work, "Download table as CSV", csv_filename, key=kwargs.get("key"))


def show_data_editor(
    df: pd.DataFrame | None,
    *,
    csv_filename: str = "table_export.csv",
    **kwargs: Any,
) -> pd.DataFrame:
    """Editable dataframe with sanitization and CSV fallback on render failure."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.caption("No data available to edit.")
        return pd.DataFrame()

    work = df.copy()
    try:
        safe = make_streamlit_safe_dataframe(work)
        return st.data_editor(safe, **kwargs)
    except Exception:
        st.warning("This editor could not be rendered in the browser. Download the CSV, edit offline, and re-upload.")
        _csv_download_button(work, "Download table as CSV", csv_filename, key=kwargs.get("key"))
        return work
