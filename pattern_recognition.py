"""
Engineering-oriented trajectory quality analysis (well-agnostic).

Primary outputs: KPIs, Trajectory_Severity (Stable / Moderate / Aggressive / Critical),
and human-readable drilling diagnostics. Legacy AI pattern proxies are optional diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEVERITY_LEVELS = ("Stable", "Moderate", "Aggressive", "Critical")
SEVERITY_ORDER = list(SEVERITY_LEVELS)

# Optional advanced diagnostics (not used for primary dashboard labels)
LEGACY_PATTERN_TYPES = ("Sinusoidal", "Helical", "Micro-tortuosity")


@dataclass
class PatternRecognitionResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str | list | dict]
    interval_kpis: pd.DataFrame


class PatternRecognitionEngine:
    """Compute drilling engineering KPIs, severity, and operational diagnostics."""

    def __init__(self, window: int = 15, md_col: str = "MD"):
        self.window = max(5, window)
        self.md_col = md_col

    def detect_patterns(self, df: pd.DataFrame) -> PatternRecognitionResult:
        work = df.copy().sort_values(self.md_col).reset_index(drop=True)
        work = self._compute_kpis(work)
        work = self._assign_severity_and_concerns(work)
        work = self._attach_legacy_diagnostics(work)
        interval_kpis = self._summarize_section_intervals(work)
        summary = self._build_summary(work, interval_kpis)
        return PatternRecognitionResult(survey=work, summary=summary, interval_kpis=interval_kpis)

    def _compute_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.window
        dls = pd.to_numeric(df.get("DLS", df.get("DLS_Calc", 0)), errors="coerce").fillna(0)
        inc = df["Inclination"].astype(float)
        azi = df["Azimuth"].astype(float)
        build = df.get("Build_Rate", inc.diff()).astype(float).fillna(0)
        turn = df.get("Turn_Rate", pd.Series(0, index=df.index)).astype(float).abs()

        inc_detrend = inc - inc.rolling(w, center=True, min_periods=3).mean()
        azi_wrap = azi.diff().abs()
        azi_wrap = np.minimum(azi_wrap, 360 - azi_wrap)

        df["Mean_DLS_Local"] = dls.rolling(w, center=True, min_periods=3).mean()
        df["DLS_Variance"] = dls.rolling(w, center=True, min_periods=3).var().fillna(0)
        df["Inclination_Variance"] = inc_detrend.rolling(w, center=True, min_periods=3).var().fillna(0)
        df["Azimuth_Variance"] = azi_wrap.rolling(w, center=True, min_periods=3).var().fillna(0)

        dls_p90 = dls.quantile(0.9) + 1e-6
        turn_p90 = turn.quantile(0.9) + 1e-6
        build_p90 = build.abs().quantile(0.9) + 1e-6

        df["Oscillation_Score"] = (
            (df["DLS_Variance"] / (df["DLS_Variance"].quantile(0.9) + 1e-6)).clip(0, 2) * 0.5
            + (df["Inclination_Variance"] / (df["Inclination_Variance"].quantile(0.9) + 1e-6)).clip(0, 2) * 0.25
            + (df["Azimuth_Variance"] / (df["Azimuth_Variance"].quantile(0.9) + 1e-6)).clip(0, 2) * 0.25
        ).clip(0, 1)

        df["Steering_Smoothness_Score"] = (
            1.0
            - (turn / turn_p90).clip(0, 2) * 0.4
            - df["Oscillation_Score"] * 0.6
        ).clip(0, 1)

        if "Tortuosity_Index_Local" not in df.columns:
            df["Tortuosity_Index_Local"] = df["Mean_DLS_Local"].fillna(0) + np.sqrt(df["DLS_Variance"])
        df["Tortuosity_Index"] = df["Tortuosity_Index_Local"]

        rotate_tend = (turn / turn_p90).clip(0, 2)
        slide_tend = (build.abs() / build_p90).clip(0, 2)
        df["Slide_Rotate_Tendency"] = np.where(
            rotate_tend > slide_tend + 0.15,
            "Rotate-dominated",
            np.where(slide_tend > rotate_tend + 0.15, "Slide / build-dominated", "Mixed"),
        )

        df["Oversteering_Score"] = (
            (dls / dls_p90).clip(0, 2) * 0.4
            + (build.abs() / build_p90).clip(0, 2) * 0.3
            + df["Oscillation_Score"] * 0.3
        ).clip(0, 1)

        df["Inclination_Instability"] = (
            df["Inclination_Variance"] / (df["Inclination_Variance"].quantile(0.9) + 1e-6)
        ).clip(0, 1)
        df["Azimuth_Instability"] = (
            df["Azimuth_Variance"] / (df["Azimuth_Variance"].quantile(0.9) + 1e-6)
        ).clip(0, 1)

        df["RSS_Aggressiveness"] = (
            (dls / dls_p90).clip(0, 2) * 0.35
            + df["Oscillation_Score"] * 0.35
            + (1.0 - df["Steering_Smoothness_Score"]) * 0.3
        ).clip(0, 1)

        is_curve = df.get("Section_Code", pd.Series("", index=df.index)) == "C"
        is_lateral = df.get("Section_Code", pd.Series("", index=df.index)) == "L"
        build_smooth = 1.0 - (build.rolling(w, center=True, min_periods=3).std().fillna(0) / (build_p90 + 1e-6)).clip(0, 1)
        df["Build_Efficiency"] = np.where(is_curve, build_smooth, np.nan)
        lat_smooth = 1.0 - df["Inclination_Instability"]
        df["Lateral_Smoothness"] = np.where(is_lateral, lat_smooth, np.nan)

        return df

    def _composite_risk(self, df: pd.DataFrame) -> pd.Series:
        dls_n = (df["Mean_DLS_Local"] / (df["Mean_DLS_Local"].quantile(0.9) + 1e-6)).clip(0, 2)
        tort_n = (df["Tortuosity_Index"] / (df["Tortuosity_Index"].quantile(0.9) + 1e-6)).clip(0, 2)
        risk = (
            dls_n * 0.2
            + tort_n * 0.2
            + df["Oscillation_Score"] * 0.2
            + df["Oversteering_Score"] * 0.15
            + df["Inclination_Instability"] * 0.1
            + df["Azimuth_Instability"] * 0.1
            + (1.0 - df["Steering_Smoothness_Score"]) * 0.05
        ) / 1.0
        return risk.clip(0, 1)

    def _assign_severity_and_concerns(self, df: pd.DataFrame) -> pd.DataFrame:
        risk = self._composite_risk(df)
        df["Composite_Risk"] = risk
        q50, q75, q90 = risk.quantile([0.5, 0.75, 0.9])
        severity = np.where(
            risk <= q50,
            "Stable",
            np.where(risk <= q75, "Moderate", np.where(risk <= q90, "Aggressive", "Critical")),
        )
        df["Trajectory_Severity"] = severity
        df["Primary_Concern"] = df.apply(self._primary_concern, axis=1)
        df["Engineering_Diagnostics"] = df.apply(self._diagnostics_text, axis=1)
        # Backward-compatible alias for grouping tables still referencing Pattern_Type
        df["Pattern_Type"] = df["Primary_Concern"]
        return df

    @staticmethod
    def _primary_concern(row: pd.Series) -> str:
        scores: Dict[str, float] = {
            "High steering instability": float(row.get("Azimuth_Instability", 0))
            + (1.0 - float(row.get("Steering_Smoothness_Score", 1))) * 0.5,
            "Aggressive DLS oscillation": float(row.get("Oscillation_Score", 0)),
            "Inclination instability": float(row.get("Inclination_Instability", 0)),
            "Oversteering / aggressive tool response": float(row.get("Oversteering_Score", 0)),
            "Elevated tortuosity": float(row.get("Tortuosity_Index", 0))
            / (float(row.get("Tortuosity_Index", 0)) + 1.0),
        }
        if pd.notna(row.get("Lateral_Smoothness")):
            scores["Poor lateral smoothness"] = 1.0 - float(row.get("Lateral_Smoothness", 1))
        best = max(scores, key=scores.get)
        if scores[best] < 0.35:
            return "Stable trajectory — within expected drilling limits"
        return best

    @staticmethod
    def _diagnostics_text(row: pd.Series) -> str:
        msgs: List[str] = []
        if row.get("Azimuth_Instability", 0) > 0.6:
            msgs.append("High steering instability (azimuth scatter)")
        if row.get("Oscillation_Score", 0) > 0.6:
            msgs.append("Aggressive DLS oscillation")
        if pd.notna(row.get("Lateral_Smoothness")) and row.get("Lateral_Smoothness", 1) < 0.5:
            msgs.append("Poor lateral smoothness")
        if row.get("Inclination_Instability", 0) > 0.6:
            msgs.append("Inclination instability in build/drop")
        if row.get("Oversteering_Score", 0) > 0.65:
            msgs.append("Oversteering / aggressive tool response")
        if row.get("RSS_Aggressiveness", 0) > 0.65:
            msgs.append("High RSS aggressiveness proxy")
        if row.get("Build_Efficiency", 1) is not None and pd.notna(row.get("Build_Efficiency")):
            if row.get("Build_Efficiency", 1) < 0.45:
                msgs.append("Low build efficiency in curve")
        if not msgs:
            sev = row.get("Trajectory_Severity", "Stable")
            if sev == "Stable":
                return "Stable drilling response — DLS and steering within expected limits"
            return f"{sev} trajectory — monitor DLS and steering trends"
        return "; ".join(msgs)

    def _attach_legacy_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional advanced pattern proxies for expert review only."""
        w = self.window
        inc = df["Inclination"].astype(float)
        azi = df["Azimuth"].astype(float)
        dls = pd.to_numeric(df.get("DLS", 0), errors="coerce").fillna(0)
        inc_osc = (inc - inc.rolling(w, center=True, min_periods=3).mean()).rolling(9, center=True, min_periods=3).std().fillna(0)
        azi_osc = (azi - azi.rolling(w, center=True, min_periods=3).mean()).rolling(9, center=True, min_periods=3).std().fillna(0)
        dls_std = dls.rolling(w, center=True, min_periods=3).std().fillna(0)
        df["Diagnostic_Advanced_Sinusoidal"] = (inc_osc + azi_osc) / ((inc_osc + azi_osc).max() + 1e-6)
        df["Diagnostic_Advanced_Helical"] = (inc_osc / (inc_osc.quantile(0.9) + 1e-6)) * (
            azi_osc / (azi_osc.quantile(0.9) + 1e-6)
        )
        df["Diagnostic_Advanced_MicroTortuosity"] = dls_std / (dls_std.quantile(0.9) + 1e-6)
        return df

    def _summarize_section_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Well_Section" not in df.columns:
            return pd.DataFrame()
        rows: List[Dict[str, object]] = []
        labels = df["Well_Section"].to_numpy()
        md = df[self.md_col].to_numpy()
        i, n = 0, len(labels)
        while i < n:
            j = i + 1
            while j < n and labels[j] == labels[i]:
                j += 1
            block = df.iloc[i:j]
            rows.append(self._interval_row(block, labels[i], float(md[i]), float(md[j - 1])))
            i = j
        return pd.DataFrame(rows)

    def _interval_row(self, block: pd.DataFrame, section: str, md_in: float, md_out: float) -> Dict[str, object]:
        return {
            "Well_Section": section,
            "Section_Code": block["Section_Code"].iloc[0] if "Section_Code" in block.columns else "",
            "MD_In": md_in,
            "MD_Out": md_out,
            "Stations": len(block),
            "Mean_DLS": float(block["Mean_DLS_Local"].mean()),
            "DLS_Variance": float(block["DLS_Variance"].mean()),
            "Inclination_Variance": float(block["Inclination_Variance"].mean()),
            "Azimuth_Variance": float(block["Azimuth_Variance"].mean()),
            "Steering_Smoothness_Score": float(block["Steering_Smoothness_Score"].mean()),
            "Oscillation_Score": float(block["Oscillation_Score"].mean()),
            "Tortuosity_Index": float(block["Tortuosity_Index"].mean()),
            "Trajectory_Severity": block["Trajectory_Severity"].mode().iloc[0],
            "Primary_Concern": block["Primary_Concern"].mode().iloc[0],
            "Composite_Risk": float(block["Composite_Risk"].mean()),
        }

    def _build_summary(self, df: pd.DataFrame, intervals: pd.DataFrame) -> Dict[str, object]:
        return {
            "severity_counts": df["Trajectory_Severity"].value_counts().to_dict(),
            "dominant_severity": df["Trajectory_Severity"].mode().iloc[0] if len(df) else "Stable",
            "mean_composite_risk": float(df["Composite_Risk"].mean()),
            "concern_counts": df["Primary_Concern"].value_counts().head(8).to_dict(),
            "interval_kpis": intervals.to_dict(orient="records") if not intervals.empty else [],
            "severity_definitions": {
                "Stable": "DLS and steering variation within expected operational limits.",
                "Moderate": "Noticeable oscillation or steering variation — monitor trends.",
                "Aggressive": "Sustained high DLS, oscillation, or steering instability — review parameters.",
                "Critical": "Severe tortuosity or steering instability — mitigation recommended.",
            },
        }


def severity_color(severity: str) -> str:
    return {
        "Stable": "#2e7d32",
        "Moderate": "#fbc02d",
        "Aggressive": "#ef6c00",
        "Critical": "#c62828",
    }.get(severity, "#757575")
