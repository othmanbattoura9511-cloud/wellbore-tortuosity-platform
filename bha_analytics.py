"""BHA run summaries, comparisons, and performance ranking."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sections import SECTION_CODES


def _safe_mean(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def _safe_max(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.max()) if len(v) else float("nan")


def interval_is_complete(row: pd.Series) -> bool:
    md_in, md_out = row.get("MD_In"), row.get("MD_Out")
    return pd.notna(md_in) and pd.notna(md_out)


def incomplete_bha_runs(bha_intervals: pd.DataFrame) -> pd.DataFrame:
    """Rows missing MD_In or MD_Out (not eligible for ranking or survey mapping)."""
    if bha_intervals is None or bha_intervals.empty:
        return pd.DataFrame()
    mask = bha_intervals["MD_In"].isna() | bha_intervals["MD_Out"].isna()
    return bha_intervals.loc[mask].copy()


def summarize_bha_run(survey: pd.DataFrame, row: pd.Series) -> Dict[str, Any]:
    """Aggregate survey metrics for one BHA MD interval."""
    md_in, md_out = row.get("MD_In"), row.get("MD_Out")
    complete = interval_is_complete(row)
    if not complete:
        subset = survey.iloc[0:0]
    else:
        subset = survey[(survey["MD"] >= float(md_in)) & (survey["MD"] <= float(md_out))]

    section_mix = {}
    if not subset.empty and "Section_Code" in subset.columns:
        section_mix = subset["Section_Code"].value_counts(normalize=True).round(3).to_dict()

    tort_col = "Tortuosity_Index" if "Tortuosity_Index" in subset.columns else "Tortuosity_Index_Local"

    return {
        "BHA_Run": row.get("BHA_Run", row.get("BHA", "Unknown")),
        "Source_File": row.get("Source_File", ""),
        "MD_In": md_in,
        "MD_Out": md_out,
        "Interval_Complete": complete,
        "Drilling_System": row.get("Drilling_System", "Unknown"),
        "RSS_Type": row.get("RSS_Type", "Unknown RSS Type"),
        "Hole_Size": row.get("Hole_Size", "Unknown"),
        "Bit_Type": row.get("Bit_Type", "Unknown"),
        "BHA_Config": row.get("BHA_Config", "Unknown"),
        "Stations": int(len(subset)),
        "Mean_DLS": _safe_mean(subset.get("DLS", pd.Series(dtype=float))),
        "Max_DLS": _safe_max(subset.get("DLS", pd.Series(dtype=float))),
        "Mean_Tortuosity": _safe_mean(subset.get(tort_col, pd.Series(dtype=float))),
        "Max_Tortuosity": _safe_max(subset.get(tort_col, pd.Series(dtype=float))),
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


def _compute_performance_score(frame: pd.DataFrame) -> pd.Series:
    dls = pd.to_numeric(frame["Mean_DLS"], errors="coerce")
    tort = pd.to_numeric(frame["Mean_Tortuosity"], errors="coerce")
    rss = pd.to_numeric(frame.get("RSS_Severity", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    stab = pd.to_numeric(frame.get("Steering_Stability", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)

    dls_fill = dls.fillna(dls.max() if dls.notna().any() else np.nan)
    tort_fill = tort.fillna(tort.max() if tort.notna().any() else np.nan)
    return dls_fill + tort_fill * 2 + rss * 0.5 - stab * 0.3


def rank_bha_performance(runs_table: pd.DataFrame) -> pd.DataFrame:
    """Rank complete BHA intervals only (lower score = better). Incomplete intervals keep null Rank."""
    if runs_table.empty:
        return runs_table

    out = runs_table.copy()
    if "Interval_Complete" not in out.columns:
        out["Interval_Complete"] = out.apply(
            lambda r: pd.notna(r.get("MD_In")) and pd.notna(r.get("MD_Out")), axis=1
        )

    out["Performance_Score"] = np.nan
    out["Rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    complete_mask = out["Interval_Complete"].astype(bool)
    if not complete_mask.any():
        return out

    ranked_idx = out.index[complete_mask]
    ranked = out.loc[ranked_idx].copy()
    ranked["Performance_Score"] = _compute_performance_score(ranked)
    ranked["Rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    score_ok = ranked["Performance_Score"].notna()
    if score_ok.any():
        ranked.loc[score_ok, "Rank"] = (
            ranked.loc[score_ok, "Performance_Score"].rank(method="min").astype("Int64")
        )

    out.loc[ranked_idx, "Performance_Score"] = ranked["Performance_Score"]
    out.loc[ranked_idx, "Rank"] = ranked["Rank"]
    return out.sort_values(["Rank", "BHA_Run"], na_position="last", kind="mergesort")


def bha_vs_section_comparison(survey: pd.DataFrame, bha_intervals: pd.DataFrame) -> pd.DataFrame:
    if bha_intervals is None or bha_intervals.empty or "Section_Code" not in survey.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, bha in bha_intervals.iterrows():
        if not interval_is_complete(bha):
            continue
        md_in, md_out = float(bha["MD_In"]), float(bha["MD_Out"])
        subset = survey[(survey["MD"] >= md_in) & (survey["MD"] <= md_out)]
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
                    "Mean_Tortuosity": _safe_mean(
                        sec.get("Tortuosity_Index", sec.get("Tortuosity_Index_Local", pd.Series(dtype=float)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_bha_groups(runs_table: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if runs_table.empty or group_col not in runs_table.columns:
        return pd.DataFrame()
    ranked_only = runs_table
    if "Interval_Complete" in runs_table.columns:
        ranked_only = runs_table[runs_table["Interval_Complete"].astype(bool)]
    if ranked_only.empty:
        return pd.DataFrame()
    agg_cols = [
        "Mean_DLS",
        "Max_DLS",
        "Mean_Tortuosity",
        "RSS_Severity",
        "Steering_Stability",
        "Stations",
    ]
    present = [c for c in agg_cols if c in ranked_only.columns]
    if not present:
        return pd.DataFrame()
    return ranked_only.groupby(group_col, dropna=False)[present].agg(["mean", "max", "count"])
