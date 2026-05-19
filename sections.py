"""
Well-section classification: Vertical (V), Curve (C), Lateral (L).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
    section_intervals: pd.DataFrame


class WellSectionClassifier:
    """
    Classify stations into exactly one continuous V → C → L progression per well.

    Engineering rule: Vertical (build starts) → Curve/Build → Lateral. No overlaps,
    gaps, backward transitions, or alternate section names.
    """

    def __init__(
        self,
        md_col: str = "MD",
        inc_col: str = "Inclination",
        azi_col: str = "Azimuth",
        dls_col: str = "DLS",
        smooth_window: int = 9,
        min_interval_stations: int = 5,
        hysteresis_margin: float = 0.12,
        lateral_inc_deg: float = 75.0,
        vertical_inc_deg: float = 30.0,
    ):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col
        self.smooth_window = max(3, smooth_window if smooth_window % 2 == 1 else smooth_window + 1)
        self.min_interval_stations = max(2, min_interval_stations)
        self.hysteresis_margin = hysteresis_margin
        self.lateral_inc_deg = lateral_inc_deg
        self.vertical_inc_deg = vertical_inc_deg

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.classify_with_confidence(df).survey

    def classify_with_confidence(self, df: pd.DataFrame) -> SectionClassificationResult:
        work = self._prepare_kinematics(df)
        scores = self._section_scores(work)
        b1, b2 = self._detect_boundary_indices(work, scores)
        labels = self._assign_monotonic_vcl(scores, work, b1_seed=b1, b2_seed=b2)
        confidence = self._interval_confidence(labels, scores, pd.Series(0.5, index=labels.index))

        out = work.copy()
        out["Well_Section"] = labels.values
        out["Section_Code"] = out["Well_Section"].map(SECTION_CODES)
        out["Section_Confidence"] = confidence.values

        intervals = compute_well_section_intervals(out, self.md_col)
        out = apply_section_interval_columns(out, intervals)

        summary = self._build_summary(out)
        summary["section_boundaries"] = {
            "curve_onset_index": int(b1),
            "lateral_onset_index": int(b2),
        }
        return SectionClassificationResult(
            survey=out,
            summary=summary,
            section_intervals=intervals,
        )

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

    def _detect_boundary_indices(self, work: pd.DataFrame, scores: pd.DataFrame) -> tuple[int, int]:
        """Detect curve onset (V→C) and lateral stabilization (C→L) from smoothed survey."""
        n = len(work)
        if n <= 1:
            return n, n

        min_s = self.min_interval_stations
        inc = work["Inclination_Smooth"].astype(float).values
        build = np.abs(work["Build_Rate_Smooth"].astype(float).values)
        build_thr = float(np.percentile(build, 55))

        b1 = n
        for i in range(1, n - min_s):
            if inc[i] >= self.vertical_inc_deg and np.mean(build[i : i + min_s]) >= build_thr:
                b1 = i
                break

        b2 = n
        start_l = max(b1 + 1, min_s)
        for i in range(start_l, n - min_s + 1):
            if inc[i] >= self.lateral_inc_deg and np.mean(inc[i : i + min_s]) >= self.lateral_inc_deg * 0.95:
                b2 = i
                break

        if b2 <= b1 and b1 < n:
            b2 = min(n, b1 + min_s)
        return b1, b2

    def _labels_from_boundaries(self, n: int, b1: int, b2: int) -> np.ndarray:
        labels = np.array(["Vertical"] * n, dtype=object)
        if b1 < n:
            labels[b1:b2] = "Curve"
        if b2 < n:
            labels[b2:] = "Lateral"
        return labels

    def _assign_monotonic_vcl(
        self,
        scores: pd.DataFrame,
        work: pd.DataFrame,
        b1_seed: int | None = None,
        b2_seed: int | None = None,
    ) -> pd.Series:
        """
        Partition into contiguous V → C → L. Optimizes cut points; seeds from physics detection.
        """
        n = len(scores)
        if n == 0:
            return pd.Series(dtype=object)
        if n == 1:
            return pd.Series(["Vertical"], index=scores.index, dtype=object)

        min_s = self.min_interval_stations
        inc = work["Inclination_Smooth"].astype(float)
        build = work["Build_Rate_Smooth"].astype(float)

        best_total = -np.inf
        best_b1, best_b2 = n, n

        b1_range = range(1, n + 1)
        if b1_seed is not None and 0 < b1_seed < n:
            b1_range = range(max(1, b1_seed - 30), min(n + 1, b1_seed + 31))

        for b1 in b1_range:
            b2_start = max(b1, b2_seed - 30) if b2_seed is not None else b1
            b2_end = min(n + 1, (b2_seed + 31) if b2_seed is not None else n + 1)
            for b2 in range(max(b1, b2_start), b2_end):
                if b1 < min_s and b1 < n:
                    continue
                if b2 < n and (b2 - b1) < min_s and b1 < n:
                    continue
                if b2 < n and (n - b2) < min_s:
                    continue

                total = float(scores["Vertical"].iloc[:b1].sum())
                if b1 < b2:
                    total += float(scores["Curve"].iloc[b1:b2].sum())
                if b2 < n:
                    total += float(scores["Lateral"].iloc[b2:].sum())
                total += self._boundary_incentive(b1, b2, n, inc, build)
                if total > best_total:
                    best_total = total
                    best_b1, best_b2 = b1, b2

        return pd.Series(self._labels_from_boundaries(n, best_b1, best_b2), index=scores.index, dtype=object)

    def _boundary_incentive(self, b1: int, b2: int, n: int, inc: pd.Series, build: pd.Series) -> float:
        """Favor boundaries aligned with inclination / build physics."""
        bonus = 0.0
        if 0 < b1 < n:
            bonus += 0.15 if float(inc.iloc[b1]) > float(inc.iloc[max(0, b1 - 1)]) else 0.0
            bonus += 0.1 if float(build.iloc[b1]) > float(build.quantile(0.4)) else 0.0
        if 0 < b2 < n:
            bonus += 0.2 if float(inc.iloc[b2]) >= self.lateral_inc_deg * 0.9 else 0.0
        if b1 == n and float(inc.max()) < self.vertical_inc_deg + 5:
            bonus += 0.5
        return bonus

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


def compute_well_section_intervals(df: pd.DataFrame, md_col: str = "MD") -> pd.DataFrame:
    """
    Contiguous well-wide MD intervals: V → C → L with shared boundaries (no overlap).

    V ends where C starts; C ends where L starts.
    """
    if df is None or df.empty or md_col not in df.columns:
        return pd.DataFrame()

    work = df.sort_values(md_col).reset_index(drop=True)
    md = pd.to_numeric(work[md_col], errors="coerce")
    n = len(work)

    codes = work["Section_Code"].astype(str).tolist() if "Section_Code" in work.columns else []
    if not codes:
        return pd.DataFrame()

    b1 = next((i for i, c in enumerate(codes) if c == "C"), n)
    b2 = next((i for i, c in enumerate(codes) if c == "L"), n)
    if b1 == n and b2 < n:
        b1 = b2

    def _md_at(idx: int) -> float:
        return float(md.iloc[min(max(idx, 0), n - 1)])

    rows: list[dict[str, float | int | str]] = []

    if b1 >= n:
        rows.append(
            {
                "Section_Code": "V",
                "Well_Section": "Vertical",
                "Section": "V — Vertical",
                "MD_In": _md_at(0),
                "MD_Out": _md_at(n - 1),
                "Stations": int(n),
            }
        )
    else:
        if b1 > 0:
            rows.append(
                {
                    "Section_Code": "V",
                    "Well_Section": "Vertical",
                    "Section": "V — Vertical",
                    "MD_In": _md_at(0),
                    "MD_Out": _md_at(b1),
                    "Stations": int(b1),
                }
            )
        if b2 > b1:
            rows.append(
                {
                    "Section_Code": "C",
                    "Well_Section": "Curve",
                    "Section": "C — Curve",
                    "MD_In": _md_at(b1),
                    "MD_Out": _md_at(b2),
                    "Stations": int(b2 - b1),
                }
            )
        if b2 < n:
            rows.append(
                {
                    "Section_Code": "L",
                    "Well_Section": "Lateral",
                    "Section": "L — Lateral",
                    "MD_In": _md_at(b2),
                    "MD_Out": _md_at(n - 1),
                    "Stations": int(n - b2),
                }
            )

    return pd.DataFrame(rows)


def apply_section_interval_columns(survey: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical section MD_In/MD_Out to each survey station from well-wide intervals."""
    out = survey.copy()
    if intervals is None or intervals.empty:
        out["Section_MD_In"] = np.nan
        out["Section_MD_Out"] = np.nan
        return out

    out["Section_MD_In"] = np.nan
    out["Section_MD_Out"] = np.nan
    for _, row in intervals.iterrows():
        code = row["Section_Code"]
        mask = out["Section_Code"] == code
        out.loc[mask, "Section_MD_In"] = row["MD_In"]
        out.loc[mask, "Section_MD_Out"] = row["MD_Out"]
    return out


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
