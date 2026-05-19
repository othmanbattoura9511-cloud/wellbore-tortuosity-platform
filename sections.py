"""
Well-section classification: Vertical (V), Curve (C), Lateral (L).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SECTION_LABELS = ("Vertical", "Curve", "Lateral")
SECTION_CODES = {"Vertical": "V", "Curve": "C", "Lateral": "L"}
CODE_TO_LABEL = {"V": "Vertical", "C": "Curve", "L": "Lateral"}
LABEL_TO_CODE = {label: code for code, label in CODE_TO_LABEL.items()}
SECTION_ORDER = list(SECTION_LABELS)


@dataclass
class SectionClassificationResult:
    survey: pd.DataFrame
    summary: Dict[str, float | int | str]


class WellSectionClassifier:
    """Classify stations into V / C / L using smoothed inclination and build rate."""

    def __init__(
        self,
        md_col: str = "MD",
        inc_col: str = "Inclination",
        azi_col: str = "Azimuth",
        dls_col: str = "DLS",
        smooth_window: int = 7,
        min_interval_stations: int = 5,
        hysteresis_margin: float = 0.12,
        lateral_inc_deg: float = 75.0,
    ):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col
        self.smooth_window = max(3, smooth_window if smooth_window % 2 == 1 else smooth_window + 1)
        self.min_interval_stations = max(2, min_interval_stations)
        self.hysteresis_margin = hysteresis_margin
        self.lateral_inc_deg = lateral_inc_deg

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.classify_with_confidence(df).survey

    def classify_with_confidence(self, df: pd.DataFrame) -> SectionClassificationResult:
        work = self._prepare_kinematics(df)
        scores = self._section_scores(work)
        raw_labels, raw_confidence = self._score_to_labels(scores)
        labels = self._apply_hysteresis(raw_labels, scores)
        labels = self._merge_short_intervals(labels, scores)
        confidence = self._interval_confidence(labels, scores, raw_confidence)

        out = work.copy()
        out["Well_Section"] = labels.values
        out["Section_Code"] = out["Well_Section"].map(SECTION_CODES)
        out["Section_Confidence"] = confidence.values

        return SectionClassificationResult(survey=out, summary=self._build_summary(out))

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
        out["Inclination_Smooth"] = out[self.inc_col].rolling(w, center=True, min_periods=1).median()
        out["Build_Rate_Smooth"] = out["Build_Rate"].rolling(w, center=True, min_periods=1).median()
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
        build_abs = df["Build_Rate_Smooth"].astype(float).abs()
        dls = df["DLS_Smooth"].astype(float)

        inc_p25, inc_p50, inc_p75 = inc.quantile([0.25, 0.5, 0.75])
        build_p50 = build_abs.quantile(0.5)
        build_p75 = build_abs.quantile(0.75) + 1e-6
        dls_p75 = dls.quantile(0.75) + 1e-6
        lateral_gate = max(self.lateral_inc_deg, float(inc_p75))

        vertical = (
            (1.0 - (inc / (inc_p50 + 1e-6)).clip(0, 2))
            + (1.0 - (build_abs / build_p75).clip(0, 2))
            + (1.0 - (dls / dls_p75).clip(0, 2))
        ) / 3.0

        curve = (
            (build_abs / build_p75).clip(0, 3)
            + build_abs.rolling(self.smooth_window, center=True, min_periods=1).mean().clip(lower=0)
            / build_p75
        ) / 2.0
        curve = curve.clip(0, 1) * (inc < lateral_gate).astype(float)

        lateral = (
            ((inc - lateral_gate).clip(lower=0) / (inc.max() - lateral_gate + 1e-6))
            + (1.0 - (build_abs / (build_p50 + 1e-6)).clip(0, 2))
            + (inc >= lateral_gate).astype(float)
        ) / 3.0

        scores = pd.DataFrame(
            {"Vertical": vertical.clip(0, 1), "Curve": curve, "Lateral": lateral.clip(0, 1)},
            index=df.index,
        )

        flat = build_abs < build_p50 * 0.35
        scores.loc[flat, "Curve"] *= 0.25
        scores.loc[inc >= lateral_gate, "Vertical"] *= 0.2
        scores.loc[inc <= inc_p25, "Lateral"] *= 0.2

        if float(inc.max()) < 20.0:
            gentle = build_abs < max(0.2, build_p75 * 0.6)
            scores.loc[gentle, "Vertical"] = (scores.loc[gentle, "Vertical"] + 0.35).clip(0, 1)
            scores.loc[gentle, "Curve"] *= 0.3

        return scores

    def _score_to_labels(self, scores: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        ranked = scores.values.argsort(axis=1)
        top = scores.columns[ranked[:, -1]]
        top_score = scores.max(axis=1)
        second_score = scores.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
        margin = (top_score - second_score).clip(lower=0)
        labels = pd.Series(top, index=scores.index, dtype=object)
        confidence = ((margin + top_score) / (scores.sum(axis=1) + 1e-6)).clip(0, 1)
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
            if scores.iloc[i][candidate] >= scores.iloc[i][current] + self.hysteresis_margin:
                current = candidate
            out.iloc[i] = current
        return out

    def _merge_short_intervals(self, labels: pd.Series, scores: pd.DataFrame) -> pd.Series:
        if len(labels) < 2:
            return labels
        arr = labels.to_numpy(copy=True)
        n = len(arr)
        i = 0
        while i < n:
            j = i + 1
            while j < n and arr[j] == arr[i]:
                j += 1
            if j - i < self.min_interval_stations:
                left = arr[i - 1] if i > 0 else None
                right = arr[j] if j < n else None
                means = scores.iloc[i:j].mean()
                candidates = [c for c in (left, right) if c is not None]
                if len(candidates) == 1:
                    arr[i:j] = candidates[0]
                elif len(candidates) == 2:
                    arr[i:j] = left if means.get(left, 0) >= means.get(right, 0) else right
                else:
                    arr[i:j] = str(means.idxmax())
            i = j
        return pd.Series(arr, index=labels.index, dtype=object)

    def _interval_confidence(
        self, labels: pd.Series, scores: pd.DataFrame, raw_confidence: pd.Series
    ) -> pd.Series:
        conf = raw_confidence.copy()
        for section in SECTION_LABELS:
            mask = labels == section
            if mask.any():
                conf.loc[mask] = (conf.loc[mask] * 0.4 + scores.loc[mask, section] * 0.6).clip(0, 1)
        return conf

    def _build_summary(self, df: pd.DataFrame) -> Dict[str, float | int | str]:
        counts = df["Well_Section"].value_counts().to_dict()
        code_counts = df["Section_Code"].value_counts().to_dict()
        return {
            "n_stations": int(len(df)),
            "mean_confidence": float(df["Section_Confidence"].mean()),
            "section_counts": counts,
            "section_code_counts": code_counts,
            "dominant_section": df["Well_Section"].mode().iloc[0] if len(df) else "Unknown",
        }


def filter_by_section_codes(df: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    """Filter survey rows by V / C / L codes mapped to Vertical / Curve / Lateral."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    if not codes:
        return df.iloc[0:0].copy()

    allowed_codes = {str(c).upper() for c in codes}
    allowed_labels = {CODE_TO_LABEL[c] for c in allowed_codes if c in CODE_TO_LABEL}

    mask = pd.Series(False, index=df.index)
    if "Section_Code" in df.columns:
        mask = mask | df["Section_Code"].astype(str).str.upper().isin(allowed_codes)
    if "Well_Section" in df.columns:
        mask = mask | df["Well_Section"].isin(allowed_labels)
    if not mask.any() and "Section_Code" not in df.columns and "Well_Section" not in df.columns:
        return df.copy()
    return df.loc[mask].copy()
