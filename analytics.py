"""End-to-end survey analytics pipeline (well-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bha_analytics import build_bha_runs_table, rank_bha_performance
from bha_extraction import normalize_bha_intervals
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
    bha_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    bha_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)


def ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "BHA": "Unknown",
        "BHA_Run": "Unknown",
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
    """Map BHA / drilling metadata from PDF intervals onto survey stations by MD."""
    out = ensure_metadata_columns(df)
    intervals = normalize_bha_intervals(bha_intervals) if bha_intervals is not None else pd.DataFrame()
    if intervals.empty:
        return out

    for _, row in intervals.iterrows():
        md_in = row.get("MD_In")
        md_out = row.get("MD_Out")
        if pd.isna(md_in) or pd.isna(md_out):
            continue
        mask = (out["MD"] >= float(md_in)) & (out["MD"] <= float(md_out))
        mapping = {
            "BHA": row.get("BHA", row.get("BHA_Run")),
            "BHA_Run": row.get("BHA_Run"),
            "Drilling_System": row.get("Drilling_System"),
            "Hole_Size": row.get("Hole_Size"),
            "Bit_Type": row.get("Bit_Type"),
            "RSS_Type": row.get("RSS_Type"),
        }
        for col, val in mapping.items():
            if pd.notna(val):
                out.loc[mask, col] = val
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
    pattern_summary = pattern_result.summary

    rss_result = RSSSteeringAnalyzer().analyze(work)
    work = rss_result.survey

    intervals = normalize_bha_intervals(bha_intervals)
    work = apply_bha_intervals(work, intervals)

    bha_runs = build_bha_runs_table(work, intervals)
    bha_ranking = rank_bha_performance(bha_runs) if not bha_runs.empty else pd.DataFrame()

    return AnalyticsResult(
        survey=work,
        section_summary=section_result.summary,
        pattern_summary=pattern_summary,
        rss_summary=rss_result.summary,
        bha_runs=bha_runs,
        bha_ranking=bha_ranking,
    )


def comparison_table(df: pd.DataFrame, group_cols: List[str], value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    value_cols = value_cols or [
        "DLS",
        "Tortuosity_Index",
        "Mean_DLS_Local",
        "Oscillation_Score",
        "Steering_Smoothness_Score",
        "Composite_Risk",
        "RSS_Severity",
        "Steering_Stability",
    ]
    present = [c for c in value_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg = {c: ["count", "mean", "max", "std"] for c in present}
    return df.groupby(group_cols, dropna=False).agg(agg)
