import numpy as np
import pandas as pd

from column_mapping import standardize_survey_columns
from engineering_kpis import EngineeringKpiEngine
from sections import SECTION_LABELS, WellSectionClassifier


def _analyze(md, inc):
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": 120 + 0.02 * md})
    df, _ = standardize_survey_columns(df)
    df["DLS"] = df["Inclination"].diff().abs().fillna(0) * 2
    df = WellSectionClassifier(min_interval_stations=3).classify(df)
    return EngineeringKpiEngine().compute(df).survey


def test_engineering_kpis_present():
    md = np.arange(0, 2000, 30, dtype=float)
    inc = np.piecewise(md, [md < 500, md >= 500], [0, 70.0]).astype(float)
    out = _analyze(md, inc)
    for col in ("Tortuosity_Index", "Stability", "Wellbore_Smoothness", "DLS_Stability"):
        assert col in out.columns


def test_vcl_sections_only():
    assert set(SECTION_LABELS) == {"Vertical", "Curve", "Lateral"}
    md = np.arange(0, 1000, 25, dtype=float)
    out = _analyze(md, 10 + 5 * np.sin(md / 80))
    assert "Trajectory_Severity" not in out.columns
    forbidden = {"Chaotic", "Sinusoidal", "Helical", "Normal", "Transition", "Tangent"}
    if "Well_Section" in out.columns:
        assert not forbidden.intersection(set(out["Well_Section"].unique()))
