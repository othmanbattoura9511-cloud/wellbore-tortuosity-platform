"""Drilling engineering analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bha_analytics import build_bha_runs_table, rank_bha_performance
from bha_extraction import normalize_bha_intervals
from comparison_tables import (
    build_drilling_system_comparison_table,
    build_hole_size_comparison,
    build_rss_steering_comparison,
    build_section_comparison_table,
)
from engineering_kpis import EngineeringKpiEngine
from preprocess import SurveyPreprocessor
from rss_analysis import RSSSteeringAnalyzer
from sections import WellSectionClassifier
from tortuosity import TortuosityAnalyzer


@dataclass
class AnalyticsResult:
    survey: pd.DataFrame
    section_summary: Dict[str, Any]
    kpi_summary: Dict[str, Any]
    section_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    system_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    rss_steering_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    hole_size_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    bha_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    bha_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    rss_summary: Dict[str, Any] = field(default_factory=dict)


def ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "BHA": "Not mapped",
        "BHA_Run": "Not mapped",
        "Drilling_System": "Unknown",
        "Hole_Size": "Unknown",
        "Bit_Type": "Unknown",
        "RSS_Type": "Unknown RSS Type",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default).replace("", default)
    return out


def apply_bha_intervals(df: pd.DataFrame, bha_intervals: Optional[pd.DataFrame]) -> pd.DataFrame:
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
        for col, key in (
            ("BHA", "BHA"),
            ("BHA_Run", "BHA_Run"),
            ("Drilling_System", "Drilling_System"),
            ("Hole_Size", "Hole_Size"),
            ("Bit_Type", "Bit_Type"),
            ("RSS_Type", "RSS_Type"),
        ):
            val = row.get(key, row.get("BHA_Run"))
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

    kpi_result = EngineeringKpiEngine().compute(work)
    work = kpi_result.survey

    rss_result = RSSSteeringAnalyzer().analyze(work)
    work = rss_result.survey
    work = apply_bha_intervals(work, bha_intervals)

    intervals = normalize_bha_intervals(bha_intervals)
    section_table = build_section_comparison_table(work, intervals)
    system_table = build_drilling_system_comparison_table(section_table)
    bha_runs = build_bha_runs_table(work, intervals)
    bha_ranking = rank_bha_performance(bha_runs) if not bha_runs.empty else pd.DataFrame()

    return AnalyticsResult(
        survey=work,
        section_summary=section_result.summary,
        kpi_summary=kpi_result.summary,
        section_comparison=section_table,
        system_comparison=system_table,
        rss_steering_comparison=build_rss_steering_comparison(section_table),
        hole_size_comparison=build_hole_size_comparison(section_table),
        bha_runs=bha_runs,
        bha_ranking=bha_ranking,
        rss_summary=rss_result.summary,
    )


def comparison_table(df: pd.DataFrame, group_cols: List[str], value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    value_cols = value_cols or [
        "DLS",
        "Tortuosity_Index",
        "Wellbore_Smoothness",
        "Stability",
        "Inclination_Control",
        "Azimuth_Control",
        "Build_Efficiency",
        "RSS_Severity",
        "Steering_Stability",
    ]
    present = [c for c in value_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg = {c: ["count", "mean", "max", "std"] for c in present}
    return df.groupby(group_cols, dropna=False).agg(agg)
