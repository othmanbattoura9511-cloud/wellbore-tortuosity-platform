"""Core drilling comparison tables for the dashboard."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sections import SECTION_CODES


def _mean(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(v.mean()), 3) if len(v) else np.nan


def _md_interval_str(md_in: float, md_out: float) -> str:
    if pd.isna(md_in) or pd.isna(md_out):
        return "—"
    return f"{md_in:.0f} – {md_out:.0f} m"


def _section_rows_from_subset(
    subset: pd.DataFrame,
    section_code: str,
    bha: str,
    system: str,
    hole_size: str,
    rss_type: str,
) -> Dict[str, Any]:
    sec_name = next((k for k, v in SECTION_CODES.items() if v == section_code), section_code)
    block = subset[subset["Section_Code"] == section_code]
    if block.empty:
        return {}
    md_in = float(block["MD"].min())
    md_out = float(block["MD"].max())
    smoothness = block.get("Lateral_Smoothness", block.get("Wellbore_Smoothness", pd.Series(dtype=float)))
    smooth_val = _mean(smoothness.dropna()) if smoothness.notna().any() else _mean(block.get("Wellbore_Smoothness", pd.Series(dtype=float)))
    return {
        "Section": f"{section_code} — {sec_name}",
        "BHA": bha,
        "System": system,
        "RSS Type": rss_type if system == "RSS" else "—",
        "Hole Size": hole_size,
        "MD Interval": _md_interval_str(md_in, md_out),
        "Avg DLS (°/30m)": _mean(block["DLS"]),
        "Tortuosity Index": _mean(block.get("Tortuosity_Index", block.get("Tortuosity_Index_Local", pd.Series(dtype=float)))),
        "Smoothness": smooth_val,
        "Stability": _mean(block.get("Stability", pd.Series(dtype=float))),
        "Inclination Control": _mean(block.get("Inclination_Control", pd.Series(dtype=float))),
        "Azimuth Control": _mean(block.get("Azimuth_Control", pd.Series(dtype=float))),
        "Build Efficiency": _mean(block.get("Build_Efficiency", pd.Series(dtype=float))),
        "Stations": len(block),
    }


def build_section_comparison_table(
    survey: pd.DataFrame,
    bha_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    TABLE 1 — Section comparison: V/C/L × BHA with engineering KPIs.
    """
    if survey.empty or "Section_Code" not in survey.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    if bha_intervals is not None and not bha_intervals.empty:
        for _, bha in bha_intervals.iterrows():
            md_in, md_out = bha.get("MD_In"), bha.get("MD_Out")
            if pd.isna(md_in) or pd.isna(md_out):
                continue
            subset = survey[(survey["MD"] >= float(md_in)) & (survey["MD"] <= float(md_out))]
            bha_name = str(bha.get("BHA", bha.get("BHA_Run", "Unknown")))
            system = str(bha.get("Drilling_System", "Unknown"))
            hole = str(bha.get("Hole_Size", "Unknown"))
            rss = str(bha.get("RSS_Type", "—"))
            for code in SECTION_CODES.values():
                row = _section_rows_from_subset(subset, code, bha_name, system, hole, rss)
                if row:
                    rows.append(row)
    else:
        for code in SECTION_CODES.values():
            sys_mode = survey["Drilling_System"].mode().iloc[0] if "Drilling_System" in survey.columns else "Unknown"
            hole_mode = survey["Hole_Size"].mode().iloc[0] if "Hole_Size" in survey.columns else "Unknown"
            row = _section_rows_from_subset(survey, code, "Not mapped", str(sys_mode), str(hole_mode), "—")
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    cols = [
        "Section",
        "BHA",
        "System",
        "RSS Type",
        "Hole Size",
        "MD Interval",
        "Avg DLS (°/30m)",
        "Tortuosity Index",
        "Smoothness",
        "Stability",
        "Inclination Control",
        "Azimuth Control",
        "Build Efficiency",
        "Stations",
    ]
    return pd.DataFrame(rows)[cols]


def build_drilling_system_comparison_table(section_table: pd.DataFrame) -> pd.DataFrame:
    """
    TABLE 2 — Drilling system comparison (Motor vs RSS, etc.).
    """
    if section_table.empty or "System" not in section_table.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for system, grp in section_table.groupby("System", dropna=False):
        if str(system) in ("Unknown", "—", ""):
            continue
        tort = pd.to_numeric(grp["Tortuosity Index"], errors="coerce")
        dls = pd.to_numeric(grp["Avg DLS (°/30m)"], errors="coerce")
        stab = pd.to_numeric(grp["Stability"], errors="coerce")
        score = tort.fillna(tort.max()) + dls.fillna(dls.max()) - stab.fillna(0)
        best_idx = score.idxmin() if len(score) else None
        best_section = grp.loc[best_idx, "Section"] if best_idx is not None else "—"
        rows.append(
            {
                "System": system,
                "Avg Tortuosity": round(float(tort.mean()), 3),
                "Avg DLS (°/30m)": round(float(dls.mean()), 3),
                "Avg Stability": round(float(stab.mean()), 3),
                "Avg Smoothness": round(float(pd.to_numeric(grp["Smoothness"], errors="coerce").mean()), 3),
                "Best Performing Section": best_section,
                "Interval Count": len(grp),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Avg Tortuosity")


def build_rss_steering_comparison(section_table: pd.DataFrame) -> pd.DataFrame:
    """Push-the-bit vs Point-the-bit (RSS runs only)."""
    if section_table.empty:
        return pd.DataFrame()
    rss = section_table[section_table["System"] == "RSS"].copy()
    if rss.empty or "RSS Type" not in rss.columns:
        return pd.DataFrame()
    rows = []
    for rss_type, grp in rss.groupby("RSS Type", dropna=False):
        if str(rss_type) in ("—", "Unknown RSS Type", ""):
            continue
        rows.append(
            {
                "RSS Steering Mode": rss_type,
                "Avg Tortuosity": round(float(pd.to_numeric(grp["Tortuosity Index"], errors="coerce").mean()), 3),
                "Avg DLS (°/30m)": round(float(pd.to_numeric(grp["Avg DLS (°/30m)"], errors="coerce").mean()), 3),
                "Avg Stability": round(float(pd.to_numeric(grp["Stability"], errors="coerce").mean()), 3),
                "Intervals": len(grp),
            }
        )
    return pd.DataFrame(rows)


def build_hole_size_comparison(section_table: pd.DataFrame) -> pd.DataFrame:
    if section_table.empty or "Hole Size" not in section_table.columns:
        return pd.DataFrame()
    rows = []
    for hole, grp in section_table.groupby("Hole Size", dropna=False):
        if str(hole) in ("Unknown", "—", ""):
            continue
        rows.append(
            {
                "Hole Size": hole,
                "Avg Tortuosity": round(float(pd.to_numeric(grp["Tortuosity Index"], errors="coerce").mean()), 3),
                "Avg DLS (°/30m)": round(float(pd.to_numeric(grp["Avg DLS (°/30m)"], errors="coerce").mean()), 3),
                "Avg Stability": round(float(pd.to_numeric(grp["Stability"], errors="coerce").mean()), 3),
                "Intervals": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("Avg Tortuosity") if rows else pd.DataFrame()
