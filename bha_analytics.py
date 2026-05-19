"""BHA run summaries, comparisons, and performance ranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sections import SECTION_CODES, SECTION_LABELS


def _safe_mean(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def _safe_max(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.max()) if len(v) else float("nan")


def summarize_bha_run(survey: pd.DataFrame, row: pd.Series) -> Dict[str, Any]:
    """Aggregate survey metrics for one BHA MD interval."""
    md_in, md_out = row.get("MD_In"), row.get("MD_Out")
    if pd.isna(md_in) or pd.isna(md_out):
        subset = survey.iloc[0:0]
    else:
        subset = survey[(survey["MD"] >= float(md_in)) & (survey["MD"] <= float(md_out))]

    section_mix = {}
    if not subset.empty and "Section_Code" in subset.columns:
        section_mix = subset["Section_Code"].value_counts(normalize=True).round(3).to_dict()

    pattern_mode = "Unknown"
    if not subset.empty and "Pattern_Type" in subset.columns:
        pattern_mode = str(subset["Pattern_Type"].mode().iloc[0])

    return {
        "BHA_Run": row.get("BHA_Run", row.get("BHA", "Unknown")),
        "Source_File": row.get("Source_File", ""),
        "MD_In": md_in,
        "MD_Out": md_out,
        "Drilling_System": row.get("Drilling_System", "Unknown"),
        "RSS_Type": row.get("RSS_Type", "Unknown RSS Type"),
        "Hole_Size": row.get("Hole_Size", "Unknown"),
        "Bit_Type": row.get("Bit_Type", "Unknown"),
        "BHA_Config": row.get("BHA_Config", "Unknown"),
        "Stations": int(len(subset)),
        "Mean_DLS": _safe_mean(subset.get("DLS", pd.Series(dtype=float))),
        "Max_DLS": _safe_max(subset.get("DLS", pd.Series(dtype=float))),
        "Mean_Tortuosity": _safe_mean(subset.get("Tortuosity_Index_Local", pd.Series(dtype=float))),
        "Max_Tortuosity": _safe_max(subset.get("Tortuosity_Index_Local", pd.Series(dtype=float))),
        "Dominant_Pattern": pattern_mode,
        "RSS_Severity": _safe_mean(subset.get("RSS_Severity", pd.Series(dtype=float))),
        "Steering_Stability": _safe_mean(subset.get("Steering_Stability", pd.Series(dtype=float))),
        "Section_V_pct": section_mix.get("V", 0.0),
        "Section_C_pct": section_mix.get("C", 0.0),
        "Section_L_pct": section_mix.get("L", 0.0),
    }


def build_bha_runs_table(survey: pd.DataFrame, bha_intervals: pd.DataFrame) -> pd.DataFrame:
    if bha_intervals is None or bha_intervals.empty:
        return pd.DataFrame()
    rows = [summarize_bha_run(survey, row) for _, row in bha_intervals.iterrows()]
    return pd.DataFrame(rows)


def rank_bha_performance(runs_table: pd.DataFrame) -> pd.DataFrame:
    """Lower tortuosity / DLS / RSS severity ranks better."""
    if runs_table.empty:
        return runs_table
    out = runs_table.copy()
    score = (
        out["Mean_DLS"].fillna(out["Mean_DLS"].max())
        + out["Mean_Tortuosity"].fillna(out["Mean_Tortuosity"].max()) * 2
        + out["RSS_Severity"].fillna(0) * 0.5
        - out["Steering_Stability"].fillna(0) * 0.3
    )
    out["Performance_Score"] = score
    out["Rank"] = out["Performance_Score"].rank(method="min").astype(int)
    return out.sort_values("Rank")


def bha_vs_section_comparison(survey: pd.DataFrame, bha_intervals: pd.DataFrame) -> pd.DataFrame:
    """Mean DLS / tortuosity by BHA run and V/C/L section code."""
    if bha_intervals is None or bha_intervals.empty or "Section_Code" not in survey.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, bha in bha_intervals.iterrows():
        md_in, md_out = bha.get("MD_In"), bha.get("MD_Out")
        if pd.isna(md_in) or pd.isna(md_out):
            continue
        subset = survey[(survey["MD"] >= float(md_in)) & (survey["MD"] <= float(md_out))]
        for code in SECTION_CODES.values():
            sec = subset[subset["Section_Code"] == code]
            if sec.empty:
                continue
            rows.append(
                {
                    "BHA_Run": bha.get("BHA_Run", bha.get("BHA")),
                    "Section_Code": code,
                    "Section": [k for k, v in SECTION_CODES.items() if v == code][0],
                    "Stations": len(sec),
                    "Mean_DLS": _safe_mean(sec["DLS"]),
                    "Max_DLS": _safe_max(sec["DLS"]),
                    "Mean_Tortuosity": _safe_mean(sec.get("Tortuosity_Index_Local", pd.Series(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def compare_bha_groups(runs_table: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if runs_table.empty or group_col not in runs_table.columns:
        return pd.DataFrame()
    agg_cols = [
        "Mean_DLS",
        "Max_DLS",
        "Mean_Tortuosity",
        "RSS_Severity",
        "Steering_Stability",
        "Stations",
    ]
    present = [c for c in agg_cols if c in runs_table.columns]
    if not present:
        return pd.DataFrame()
    return runs_table.groupby(group_col, dropna=False)[present].agg(["mean", "max", "count"])
