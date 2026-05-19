"""RSS steering mode proxies derived from survey + pattern indicators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

RSS_TYPES = (
    "Push-the-bit",
    "Point-the-bit",
    "Unknown RSS Type",
)


@dataclass
class RSSAnalysisResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class RSSSteeringAnalyzer:
    def analyze(self, df: pd.DataFrame) -> RSSAnalysisResult:
        work = df.copy()
        pattern = work.get("Pattern_Type", pd.Series("Normal", index=work.index))
        dls = pd.to_numeric(work.get("DLS", work.get("DLS_Calc", 0)), errors="coerce").fillna(0)
        build = work.get("Build_Rate", pd.Series(0, index=work.index)).abs()

        push_score = (
            pattern.isin(["Sinusoidal", "Helical"]).astype(float) * 0.5
            + (build / (build.quantile(0.75) + 1e-6)).clip(0, 2) * 0.3
            + (1.0 - (dls / (dls.quantile(0.95) + 1e-6)).clip(0, 2)) * 0.2
        )
        point_score = (
            pattern.isin(["Micro-tortuosity", "Chaotic"]).astype(float) * 0.6
            + (dls / (dls.quantile(0.75) + 1e-6)).clip(0, 3) * 0.4
        )

        rss_type = np.where(
            push_score > point_score + 0.1,
            "Push-the-bit",
            np.where(point_score > push_score + 0.1, "Point-the-bit", "Unknown RSS Type"),
        )
        work["RSS_Type"] = rss_type
        work["RSS_Confidence"] = (np.maximum(push_score, point_score) / (push_score + point_score + 1e-6)).clip(0, 1)

        summary = {
            "rss_counts": pd.Series(rss_type).value_counts().to_dict(),
            "mean_confidence": float(work["RSS_Confidence"].mean()),
        }
        return RSSAnalysisResult(survey=work, summary=summary)
