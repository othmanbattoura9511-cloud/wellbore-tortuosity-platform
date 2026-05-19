"""
Adaptive well-section classification from survey kinematics (any well profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SECTION_LABELS = (
    "Vertical",
    "Curve",
    "Horizontal",
    "Tangent",
    "Transition",
)


@dataclass
class SectionClassificationResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class WellSectionClassifier:
    """Classify survey stations using inclination, azimuth, DLS, and derived rates."""

    def __init__(
        self,
        md_col: str = "MD",
        inc_col: str = "Inclination",
        azi_col: str = "Azimuth",
        dls_col: str = "DLS",
    ):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.classify_with_confidence(df)
        return result.survey

    def classify_with_confidence(self, df: pd.DataFrame) -> SectionClassificationResult:
        work = self._prepare_kinematics(df)
        scores = self._section_scores(work)
        labels, confidence = self._assign_labels(scores, work)

        out = work.copy()
        out["Well_Section"] = labels
        out["Section_Confidence"] = confidence
        # Legacy lowercase alias used in some plots/filters
        out["Well_Section_Legacy"] = out["Well_Section"].str.lower().replace(
            {"horizontal": "lateral", "tangent": "hold"}
        )

        summary = self._build_summary(out)
        return SectionClassificationResult(survey=out, summary=summary)

    def _prepare_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values(self.md_col).reset_index(drop=True)
        delta_md = out[self.md_col].diff().replace(0, np.nan)

        out["Build_Rate"] = out[self.inc_col].diff() / delta_md * 30.0
        azi_prev = out[self.azi_col].shift(1)
        azi_diff = np.abs((out[self.azi_col] - azi_prev + 180) % 360 - 180)
        out["Turn_Rate"] = azi_diff / delta_md * 30.0
        out["Inclination_Gradient"] = out["Build_Rate"]
        out["Azimuth_Gradient"] = out["Turn_Rate"]

        if self.dls_col not in out.columns and "DLS_Calc" in out.columns:
            out[self.dls_col] = out["DLS_Calc"]

        for col in ("Build_Rate", "Turn_Rate", "Inclination_Gradient", "Azimuth_Gradient"):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].interpolate(limit_direction="both").fillna(0.0)

        return out

    def _section_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        inc = df[self.inc_col].astype(float)
        build = df["Build_Rate"].abs()
        turn = df["Turn_Rate"].abs()
        dls = df[self.dls_col].astype(float) if self.dls_col in df.columns else pd.Series(0.0, index=df.index)

        inc_p25, inc_p50, inc_p75 = inc.quantile([0.25, 0.5, 0.75])
        build_p50, build_p75 = build.quantile([0.5, 0.75])
        turn_p50 = turn.quantile(0.5)
        dls_p75 = dls.quantile(0.75) if dls.notna().any() else 0.0

        # Adaptive gates — derived from the uploaded well, not fixed MD/inc limits.
        vertical = (
            (1.0 - (inc / (inc_p50 + 1e-6)).clip(0, 1.5))
            + (1.0 - (build / (build_p75 + 1e-6)).clip(0, 2))
            + (1.0 - (turn / (turn_p50 + 1e-6)).clip(0, 2))
        ) / 3.0

        horizontal = (
            ((inc - inc_p75).clip(lower=0) / (inc.max() - inc_p75 + 1e-6))
            + (1.0 - (build / (build_p50 + 1e-6)).clip(0, 2))
            + ((inc >= inc_p75).astype(float))
        ) / 3.0

        curve = (
            (build / (build_p75 + 1e-6)).clip(0, 3)
            + (df["Inclination_Gradient"].abs() / (build_p75 + 1e-6)).clip(0, 3)
            + (dls / (dls_p75 + 1e-6)).clip(0, 3)
        ) / 3.0

        tangent = (
            ((inc - inc_p25).clip(lower=0) / (inc_p75 - inc_p25 + 1e-6)).clip(0, 1)
            * (1.0 - (build / (build_p50 + 1e-6)).clip(0, 2))
            * (1.0 - (turn / (turn_p50 + 1e-6)).clip(0, 2))
        )

        # Transition: ambiguous boundaries (mid inclination with mixed signals).
        transition = (
            (1.0 - (inc - inc_p50).abs() / (inc.std() + 1e-6)).clip(0, 1)
            * (build / (build_p50 + 1e-6)).clip(0, 2)
            * 0.5
        )

        scores = pd.DataFrame(
            {
                "Vertical": vertical.clip(0, 1),
                "Curve": curve.clip(0, 1),
                "Horizontal": horizontal.clip(0, 1),
                "Tangent": tangent.clip(0, 1),
                "Transition": transition.clip(0, 1),
            },
            index=df.index,
        )
        return scores

    def _assign_labels(self, scores: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        inc = df[self.inc_col].astype(float)
        build = df["Build_Rate"].abs()
        inc_p20, inc_p80 = inc.quantile([0.2, 0.8])
        build_p50, build_p75 = build.quantile([0.5, 0.75])

        ranked = scores.values.argsort(axis=1)
        top = scores.columns[ranked[:, -1]]
        top_score = scores.max(axis=1)
        second_score = scores.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
        margin = (top_score - second_score).clip(lower=0)

        labels = pd.Series(top, index=scores.index, dtype=object)
        inc_span = float(inc.max() - inc.min())
        inc_p90 = float(inc.quantile(0.9))

        # Engineering overrides (adaptive percentiles from the uploaded well).
        if inc_p90 < 15.0:
            build_p90 = build.quantile(0.9)
            labels.loc[build <= build_p90] = "Vertical"
            labels.loc[build > build_p90] = "Curve"
        else:
            labels.loc[(inc <= inc_p20) & (build <= build_p75)] = "Vertical"
            labels.loc[(inc >= inc_p80) & (build <= build_p50) & (inc_span > 20)] = "Horizontal"
            labels.loc[build >= build_p75] = "Curve"

        ambiguous = (margin < 0.1) & (top_score < 0.4)
        labels.loc[ambiguous] = "Transition"

        confidence = (top_score / (scores.sum(axis=1) + 1e-6)).clip(0, 1)
        confidence.loc[ambiguous] = confidence.loc[ambiguous] * 0.75
        return labels, confidence

    def _build_summary(self, df: pd.DataFrame) -> Dict[str, float | int | str]:
        counts = df["Well_Section"].value_counts().to_dict()
        return {
            "n_stations": int(len(df)),
            "mean_confidence": float(df["Section_Confidence"].mean()),
            "section_counts": counts,
            "dominant_section": df["Well_Section"].mode().iloc[0] if len(df) else "Unknown",
        }
