"""End-to-end survey analytics pipeline (well-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from pattern_recognition import PatternRecognitionEngine
from preprocess import SurveyPreprocessor
from rss_analysis import RSSSteeringAnalyzer
from sections import WellSectionClassifier
from tortuosity import TortuosityAnalyzer


@dataclass
class AnalyticsResult:
    survey: pd.DataFrame
    section_summary: Dict[str, Any]
    pattern_summary: Dict[str, Any]
    rss_summary: Dict[str, Any]


def ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "BHA": "Unknown",
        "Drilling_System": "Unknown",
        "Hole_Size": "Unknown",
        "Bit_Type": "Unknown",
        "Survey_Type": "Unknown",
        "RSS_Type": "Unknown RSS Type",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default).replace("", default)
    return out


def apply_bha_intervals(df: pd.DataFrame, bha_intervals: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Map BHA / drilling metadata from extracted PDF intervals by MD."""
    out = ensure_metadata_columns(df)
    if bha_intervals is None or bha_intervals.empty:
        return out

    for _, row in bha_intervals.iterrows():
        md_in = row.get("MD_In")
        md_out = row.get("MD_Out")
        if pd.isna(md_in) or pd.isna(md_out):
            continue
        mask = (out["MD"] >= float(md_in)) & (out["MD"] <= float(md_out))
        for col in ("BHA", "Drilling_System", "Hole_Size", "Bit_Type", "RSS_Type"):
            if col in row.index and pd.notna(row.get(col)):
                out.loc[mask, col] = row[col]
    return out


def run_analytics_pipeline(df: pd.DataFrame, bha_intervals: Optional[pd.DataFrame] = None) -> AnalyticsResult:
    pre = SurveyPreprocessor()
    work = pre.interpolate_missing(pre.clean(df))

    tort = TortuosityAnalyzer()
    if "DLS" not in work.columns or work["DLS"].isna().all():
        work = tort.calculate_dls(work)
        work["DLS"] = work["DLS_Calc"]
    work = tort.add_indicators(work)

    section_result = WellSectionClassifier().classify_with_confidence(work)
    work = section_result.survey

    pattern_result = PatternRecognitionEngine().detect_patterns(work)
    work = pattern_result.survey

    rss_result = RSSSteeringAnalyzer().analyze(work)
    work = rss_result.survey

    work = apply_bha_intervals(work, bha_intervals)

    return AnalyticsResult(
        survey=work,
        section_summary=section_result.summary,
        pattern_summary=pattern_result.summary,
        rss_summary=rss_result.summary,
    )


def comparison_table(df: pd.DataFrame, group_cols: List[str], value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    value_cols = value_cols or ["DLS", "Tortuosity_Index_Local"]
    present = [c for c in value_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg = {c: ["count", "mean", "max", "std"] for c in present}
    return df.groupby(group_cols, dropna=False).agg(agg)
