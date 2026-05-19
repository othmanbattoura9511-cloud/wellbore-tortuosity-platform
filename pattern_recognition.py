"""Heuristic trajectory pattern recognition (well-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

PATTERN_TYPES = (
    "Normal",
    "Sinusoidal",
    "Helical",
    "Micro-tortuosity",
    "Chaotic",
)


@dataclass
class PatternRecognitionResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class PatternRecognitionEngine:
    def detect_patterns(self, df: pd.DataFrame) -> PatternRecognitionResult:
        work = df.copy().sort_values("MD").reset_index(drop=True)
        dls = work["DLS"] if "DLS" in work.columns else work.get("DLS_Calc", pd.Series(0, index=work.index))
        dls = pd.to_numeric(dls, errors="coerce").fillna(0)

        inc = work["Inclination"].astype(float)
        azi = work["Azimuth"].astype(float)
        md = work["MD"].astype(float)

        inc_detrend = inc - inc.rolling(15, min_periods=5, center=True).mean()
        azi_detrend = azi - azi.rolling(15, min_periods=5, center=True).mean()
        inc_osc = inc_detrend.rolling(9, min_periods=3, center=True).std().fillna(0)
        azi_osc = azi_detrend.rolling(9, min_periods=3, center=True).std().fillna(0)
        dls_spike = (dls - dls.rolling(9, min_periods=3, center=True).mean()).abs().fillna(0)
        dls_std = dls.rolling(15, min_periods=5, center=True).std().fillna(0)

        inc_p75, dls_p75, osc_p75 = inc_osc.quantile(0.75), dls.quantile(0.75), dls_std.quantile(0.75)

        helical_proxy = (inc_osc / (inc_p75 + 1e-6)) * (azi_osc / (azi_osc.quantile(0.75) + 1e-6))
        sinusoidal_proxy = (inc_osc + azi_osc) / 2.0
        micro_proxy = (dls_spike / (dls_p75 + 1e-6)) + (dls_std / (dls_std.quantile(0.75) + 1e-6))
        chaotic_proxy = dls_std / (dls_std.quantile(0.75) + 1e-6)
        normal_proxy = 1.0 / (1.0 + sinusoidal_proxy + micro_proxy + chaotic_proxy)

        scores = pd.DataFrame(
            {
                "Normal": normal_proxy.clip(0, 1),
                "Sinusoidal": (sinusoidal_proxy / (sinusoidal_proxy.max() + 1e-6)).clip(0, 1),
                "Helical": (helical_proxy / (helical_proxy.max() + 1e-6)).clip(0, 1),
                "Micro-tortuosity": (micro_proxy / (micro_proxy.max() + 1e-6)).clip(0, 1),
                "Chaotic": (chaotic_proxy / (chaotic_proxy.max() + 1e-6)).clip(0, 1),
            }
        )
        work["Pattern_Type"] = scores.idxmax(axis=1)
        work["Pattern_Confidence"] = scores.max(axis=1) / (scores.sum(axis=1) + 1e-6)

        summary = {
            "mean_confidence": float(work["Pattern_Confidence"].mean()),
            "pattern_counts": work["Pattern_Type"].value_counts().to_dict(),
        }
        return PatternRecognitionResult(survey=work, summary=summary)
