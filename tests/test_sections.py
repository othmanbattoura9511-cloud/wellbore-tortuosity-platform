import numpy as np
import pandas as pd

from column_mapping import standardize_survey_columns
from sections import WellSectionClassifier


def _classify(md, inc, azi=None):
    azi = azi if azi is not None else np.zeros_like(md) + 120
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": azi})
    df, _ = standardize_survey_columns(df)
    return WellSectionClassifier().classify_with_confidence(df).survey


def test_vertical_only_well():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.linspace(0, 8, len(md))
    out = _classify(md, inc)
    assert (out["Well_Section"] == "Vertical").mean() > 0.6
    assert out["Section_Confidence"].mean() > 0.2


def test_horizontal_well():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.concatenate([np.linspace(0, 85, 40), np.full(len(md) - 40, 88.0)])
    out = _classify(md, inc)
    lateral = out["Well_Section"].isin(["Horizontal", "Tangent"]).mean()
    assert lateral > 0.35


def test_build_and_hold_has_curve():
    md = np.arange(0, 4000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2200), md >= 2200],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    out = _classify(md, inc)
    assert "Curve" in out["Well_Section"].values


def test_section_confidence_column_present():
    md = np.arange(0, 1000, 25, dtype=float)
    inc = 20 + 30 * np.sin(md / 200)
    out = _classify(md, inc)
    assert "Section_Confidence" in out.columns
    assert out["Section_Confidence"].between(0, 1).all()
