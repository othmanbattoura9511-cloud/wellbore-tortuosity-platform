import numpy as np
import pandas as pd

from column_mapping import standardize_survey_columns
from pattern_recognition import SEVERITY_LEVELS, PatternRecognitionEngine
from sections import WellSectionClassifier


def _analyze(md, inc):
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": 120 + 0.02 * md})
    df, _ = standardize_survey_columns(df)
    df["DLS"] = df["Inclination"].diff().abs().fillna(0) * 2
    df = WellSectionClassifier(min_interval_stations=3).classify(df)
    return PatternRecognitionEngine().detect_patterns(df).survey


def test_severity_levels_present():
    md = np.arange(0, 2000, 30, dtype=float)
    inc = np.piecewise(md, [md < 500, md >= 500], [0, 70.0]).astype(float)
    out = _analyze(md, inc)
    assert "Trajectory_Severity" in out.columns
    assert set(out["Trajectory_Severity"].unique()).issubset(set(SEVERITY_LEVELS))
    assert "Primary_Concern" in out.columns
    assert "Engineering_Diagnostics" in out.columns


def test_no_legacy_primary_labels():
    md = np.arange(0, 1000, 25, dtype=float)
    inc = 10 + 5 * np.sin(md / 80)
    out = _analyze(md, inc)
    legacy = {"Chaotic", "Normal", "Sinusoidal", "Helical", "Micro-tortuosity"}
    assert not legacy.intersection(set(out["Primary_Concern"].unique()))


def test_interval_kpi_columns():
    md = np.arange(0, 1500, 30, dtype=float)
    inc = np.linspace(0, 80, len(md))
    out = _analyze(md, inc)
    for col in ("Mean_DLS_Local", "DLS_Variance", "Oscillation_Score", "Tortuosity_Index"):
        assert col in out.columns
