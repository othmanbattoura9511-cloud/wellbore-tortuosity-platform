"""Drilling engineering KPIs (no AI pattern labels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EngineeringKpiResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class EngineeringKpiEngine:
    """Compute operational KPIs from inclination, azimuth, and DLS."""

    def __init__(self, window: int = 15, md_col: str = "MD"):
        self.window = max(5, window)
        self.md_col = md_col

    def compute(self, df: pd.DataFrame) -> EngineeringKpiResult:
        work = df.copy().sort_values(self.md_col).reset_index(drop=True)
        w = self.window

        dls = pd.to_numeric(work.get("DLS", work.get("DLS_Calc", 0)), errors="coerce").fillna(0)
        inc = work["Inclination"].astype(float)
        azi = work["Azimuth"].astype(float)
        build = work.get("Build_Rate", inc.diff()).astype(float).fillna(0)
        turn = work.get("Turn_Rate", pd.Series(0, index=work.index)).astype(float).abs()

        inc_detrend = inc - inc.rolling(w, center=True, min_periods=3).mean()
        azi_step = azi.diff().abs()
        azi_step = np.minimum(azi_step, 360 - azi_step)

        work["Mean_DLS_Local"] = dls.rolling(w, center=True, min_periods=3).mean()
        work["DLS_Variance"] = dls.rolling(w, center=True, min_periods=3).var().fillna(0)

        dls_var_q = work["DLS_Variance"].quantile(0.9) + 1e-6
        work["DLS_Stability"] = (1.0 - (work["DLS_Variance"] / dls_var_q).clip(0, 2)).clip(0, 1)

        inc_var_q = inc_detrend.rolling(w, center=True, min_periods=3).var().fillna(0).quantile(0.9) + 1e-6
        azi_var_q = azi_step.rolling(w, center=True, min_periods=3).var().fillna(0).quantile(0.9) + 1e-6

        work["Inclination_Control"] = (
            1.0 - (inc_detrend.rolling(w, center=True, min_periods=3).var().fillna(0) / inc_var_q).clip(0, 2)
        ).clip(0, 1)
        work["Azimuth_Control"] = (
            1.0 - (azi_step.rolling(w, center=True, min_periods=3).var().fillna(0) / azi_var_q).clip(0, 2)
        ).clip(0, 1)

        turn_p90 = turn.quantile(0.9) + 1e-6
        work["Wellbore_Smoothness"] = (
            1.0 - (turn / turn_p90).clip(0, 2) * 0.5 - (work["DLS_Variance"] / dls_var_q).clip(0, 2) * 0.5
        ).clip(0, 1)

        if "Tortuosity_Index_Local" not in work.columns:
            work["Tortuosity_Index_Local"] = work["Mean_DLS_Local"].fillna(0) + np.sqrt(work["DLS_Variance"])
        work["Tortuosity_Index"] = work["Tortuosity_Index_Local"]

        is_curve = work.get("Section_Code", "") == "C"
        build_std = build.rolling(w, center=True, min_periods=3).std().fillna(0)
        build_p90 = build.abs().quantile(0.9) + 1e-6
        work["Build_Efficiency"] = np.where(
            is_curve, (1.0 - (build_std / build_p90).clip(0, 2)).clip(0, 1), np.nan
        )

        is_lateral = work.get("Section_Code", "") == "L"
        work["Lateral_Smoothness"] = np.where(is_lateral, work["Inclination_Control"], np.nan)

        work["Stability"] = (
            work["DLS_Stability"] * 0.5 + work["Wellbore_Smoothness"] * 0.3 + work["Azimuth_Control"] * 0.2
        ).clip(0, 1)

        summary = {
            "mean_tortuosity": float(work["Tortuosity_Index"].mean()),
            "mean_dls": float(dls.mean()),
            "mean_stability": float(work["Stability"].mean()),
            "mean_smoothness": float(work["Wellbore_Smoothness"].mean()),
        }
        return EngineeringKpiResult(survey=work, summary=summary)
