import numpy as np
import pandas as pd

from column_mapping import standardize_survey_columns
from sections import SECTION_LABELS, WellSectionClassifier


def _classify(md, inc, azi=None):
    azi = azi if azi is not None else np.zeros_like(md) + 120
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": azi})
    df, _ = standardize_survey_columns(df)
    return WellSectionClassifier(min_interval_stations=3).classify_with_confidence(df).survey


def _forbidden_sections(series: pd.Series) -> set:
    forbidden = {"Transition", "Tangent", "Curve"}
    return forbidden & set(series.unique())


def test_vertical_only_well():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.linspace(0, 8, len(md))
    out = _classify(md, inc)
    assert (out["Well_Section"] == "Vertical").mean() > 0.5
    assert out["Section_Confidence"].mean() > 0.2
    assert not _forbidden_sections(out["Well_Section"])


def test_horizontal_lateral_section():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.concatenate([np.linspace(0, 85, 40), np.full(len(md) - 40, 88.0)])
    out = _classify(md, inc)
    lateral = (out["Well_Section"] == "Horizontal / Lateral").mean()
    assert lateral > 0.25
    assert not _forbidden_sections(out["Well_Section"])


def test_build_section_present():
    md = np.arange(0, 4000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2200), md >= 2200],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    out = _classify(md, inc)
    assert "Build" in out["Well_Section"].values
    assert not _forbidden_sections(out["Well_Section"])


def test_drop_section_present():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.concatenate(
        [
            np.linspace(0, 60, 40),
            np.linspace(60, 25, 30),
            np.full(len(md) - 70, 25.0),
        ]
    )
    out = _classify(md, inc)
    assert "Drop" in out["Well_Section"].values
    assert not _forbidden_sections(out["Well_Section"])


def test_section_labels_enum():
    assert set(SECTION_LABELS) == {
        "Vertical",
        "Build",
        "Drop",
        "Horizontal / Lateral",
    }


def test_section_confidence_column_present():
    md = np.arange(0, 1000, 25, dtype=float)
    inc = 20 + 30 * np.sin(md / 200)
    out = _classify(md, inc)
    assert "Section_Confidence" in out.columns
    assert out["Section_Confidence"].between(0, 1).all()
