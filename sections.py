"""
Engineering-oriented well-section classification from survey kinematics.

Sections: Vertical, Build, Drop, Horizontal / Lateral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SECTION_LABELS = (
    "Vertical",
    "Build",
    "Drop",
    "Horizontal / Lateral",
)

# Display order for plots and comparison tables
SECTION_ORDER = list(SECTION_LABELS)


@dataclass
class SectionClassificationResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class WellSectionClassifier:
    """Classify survey stations using smoothed inclination, build rate, and DLS."""

    def __init__(
        self,
        md_col: str = "MD",
        inc_col: str = "Inclination",
        azi_col: str = "Azimuth",
        dls_col: str = "DLS",
        smooth_window: int = 7,
        min_interval_stations: int = 5,
        hysteresis_margin: float = 0.12,
        horizontal_inc_deg: float = 75.0,
    ):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col
        self.smooth_window = max(3, smooth_window if smooth_window % 2 == 1 else smooth_window + 1)
        self.min_interval_stations = max(2, min_interval_stations)
        self.hysteresis_margin = hysteresis_margin
        self.horizontal_inc_deg = horizontal_inc_deg

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.classify_with_confidence(df).survey

    def classify_with_confidence(self, df: pd.DataFrame) -> SectionClassificationResult:
        work = self._prepare_kinematics(df)
        scores = self._section_scores(work)
        raw_labels, raw_confidence = self._score_to_labels(scores)
        labels = self._apply_hysteresis(raw_labels, scores)
        labels = self._merge_short_intervals(labels, scores, work[self.md_col])
        confidence = self._interval_confidence(labels, scores, raw_confidence)

        out = work.copy()
        out["Well_Section"] = labels.values
        out["Section_Confidence"] = confidence.values

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

        w = self.smooth_window
        out["Inclination_Smooth"] = (
            out[self.inc_col].rolling(w, center=True, min_periods=1).median()
        )
        out["Build_Rate_Smooth"] = (
            out["Build_Rate"].rolling(w, center=True, min_periods=1).median()
        )
        if self.dls_col in out.columns:
            out["DLS_Smooth"] = (
                pd.to_numeric(out[self.dls_col], errors="coerce")
                .rolling(w, center=True, min_periods=1)
                .median()
                .fillna(0.0)
            )
        else:
            out["DLS_Smooth"] = 0.0

        return out

    def _section_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        inc = df["Inclination_Smooth"].astype(float)
        build = df["Build_Rate_Smooth"].astype(float)
        build_abs = build.abs()
        dls = df["DLS_Smooth"].astype(float)

        inc_p25, inc_p50, inc_p75 = inc.quantile([0.25, 0.5, 0.75])
        inc_p85 = float(inc.quantile(0.85))
        build_pos = build.clip(lower=0)
        build_neg = (-build.clip(upper=0))
        build_abs_p50 = build_abs.quantile(0.5)
        build_abs_p75 = build_abs.quantile(0.75)
        build_pos_p75 = build_pos.quantile(0.75) + 1e-6
        build_neg_p75 = build_neg.quantile(0.75) + 1e-6
        dls_p75 = dls.quantile(0.75) + 1e-6

        horizontal_gate = max(self.horizontal_inc_deg, inc_p75)

        # Vertical: low inclination, stable (low |build|), low DLS
        vertical = (
            (1.0 - (inc / (inc_p50 + 1e-6)).clip(0, 2))
            + (1.0 - (build_abs / (build_abs_p75 + 1e-6)).clip(0, 2))
            + (1.0 - (dls / dls_p75).clip(0, 2))
        ) / 3.0

        # Build: positive inclination gradient, sustained build
        build_score = (
            (build_pos / build_pos_p75).clip(0, 3)
            + (build_pos.rolling(self.smooth_window, center=True, min_periods=1).mean() / build_pos_p75).clip(0, 2)
        ) / 2.0
        build_score = build_score * (inc < horizontal_gate).astype(float)

        # Drop: negative inclination gradient, sustained drop
        drop_score = (
            (build_neg / build_neg_p75).clip(0, 3)
            + (build_neg.rolling(self.smooth_window, center=True, min_periods=1).mean() / build_neg_p75).clip(0, 2)
        ) / 2.0
        drop_score = drop_score * (inc < horizontal_gate).astype(float)

        # Horizontal / Lateral: high inclination, stable near horizontal
        horizontal = (
            ((inc - horizontal_gate).clip(lower=0) / (inc.max() - horizontal_gate + 1e-6))
            + (1.0 - (build_abs / (build_abs_p50 + 1e-6)).clip(0, 2))
            + ((inc >= horizontal_gate).astype(float))
        ) / 3.0

        scores = pd.DataFrame(
            {
                "Vertical": vertical.clip(0, 1),
                "Build": build_score.clip(0, 1),
                "Drop": drop_score.clip(0, 1),
                "Horizontal / Lateral": horizontal.clip(0, 1),
            },
            index=df.index,
        )

        # Suppress conflicting build/drop when nearly flat
        flat = build_abs < (build_abs_p50 * 0.35 + 1e-6)
        scores.loc[flat, "Build"] *= 0.25
        scores.loc[flat, "Drop"] *= 0.25
        scores.loc[inc >= horizontal_gate, "Vertical"] *= 0.2
        scores.loc[inc <= inc_p25, ["Build", "Drop"]] *= 0.5

        # Low-angle wells: gentle inclination drift stays vertical, not build/drop.
        if float(inc.max()) < 20.0:
            gentle = build_abs < max(0.2, build_abs_p75 * 0.6)
            scores.loc[gentle, "Vertical"] = (scores.loc[gentle, "Vertical"] + 0.35).clip(0, 1)
            scores.loc[gentle, ["Build", "Drop"]] *= 0.3

        return scores

    def _score_to_labels(self, scores: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        ranked = scores.values.argsort(axis=1)
        top = scores.columns[ranked[:, -1]]
        top_score = scores.max(axis=1)
        second_score = scores.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
        margin = (top_score - second_score).clip(lower=0)
        labels = pd.Series(top, index=scores.index, dtype=object)
        confidence = (margin + top_score) / (scores.sum(axis=1) + 1e-6)
        confidence = confidence.clip(0, 1)
        return labels, confidence

    def _apply_hysteresis(self, labels: pd.Series, scores: pd.DataFrame) -> pd.Series:
        if labels.empty:
            return labels
        out = labels.copy()
        current = out.iloc[0]
        for i in range(1, len(out)):
            candidate = out.iloc[i]
            if candidate == current:
                continue
            cur_score = scores.iloc[i][current]
            new_score = scores.iloc[i][candidate]
            if new_score >= cur_score + self.hysteresis_margin:
                current = candidate
            out.iloc[i] = current
        return out

    def _merge_short_intervals(
        self, labels: pd.Series, scores: pd.DataFrame, md: pd.Series
    ) -> pd.Series:
        if len(labels) < 2:
            return labels

        min_len = self.min_interval_stations
        n = len(labels)
        arr = labels.to_numpy(copy=True)
        i = 0
        while i < n:
            j = i + 1
            while j < n and arr[j] == arr[i]:
                j += 1
            run_len = j - i
            if run_len < min_len:
                left = arr[i - 1] if i > 0 else None
                right = arr[j] if j < n else None
                replacement = self._pick_merge_target(left, right, scores.iloc[i:j].mean())
                arr[i:j] = replacement
            i = j

        return pd.Series(arr, index=labels.index, dtype=object)

    @staticmethod
    def _pick_merge_target(
        left: str | None, right: str | None, run_means: pd.Series
    ) -> str:
        candidates = [c for c in (left, right) if c is not None]
        if not candidates:
            return str(run_means.idxmax())
        if len(candidates) == 1:
            return candidates[0]
        left_score = run_means.get(left, 0.0)
        right_score = run_means.get(right, 0.0)
        return left if left_score >= right_score else right

    def _interval_confidence(
        self,
        labels: pd.Series,
        scores: pd.DataFrame,
        raw_confidence: pd.Series,
    ) -> pd.Series:
        conf = raw_confidence.copy()
        for section in SECTION_LABELS:
            mask = labels == section
            if not mask.any():
                continue
            section_score = scores.loc[mask, section]
            conf.loc[mask] = (conf.loc[mask] * 0.4 + section_score * 0.6).clip(0, 1)
        return conf

    def _build_summary(self, df: pd.DataFrame) -> Dict[str, float | int | str]:
        counts = df["Well_Section"].value_counts().to_dict()
        interval_stats: List[Dict[str, object]] = []
        labels = df["Well_Section"].to_numpy()
        md = df[self.md_col].to_numpy()
        i = 0
        n = len(labels)
        while i < n:
            j = i + 1
            while j < n and labels[j] == labels[i]:
                j += 1
            interval_stats.append(
                {
                    "section": labels[i],
                    "md_in": float(md[i]),
                    "md_out": float(md[j - 1]),
                    "length_m": float(md[j - 1] - md[i]) if j > i else 0.0,
                    "stations": j - i,
                    "mean_confidence": float(df["Section_Confidence"].iloc[i:j].mean()),
                }
            )
            i = j

        return {
            "n_stations": int(len(df)),
            "mean_confidence": float(df["Section_Confidence"].mean()),
            "section_counts": counts,
            "dominant_section": df["Well_Section"].mode().iloc[0] if len(df) else "Unknown",
            "intervals": interval_stats,
        }
