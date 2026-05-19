import numpy as np
import pandas as pd

from column_mapping import standardize_survey_columns
from sections import SECTION_CODES, SECTION_LABELS, WellSectionClassifier, filter_by_section_codes


def _classify(md, inc, azi=None):
    azi = azi if azi is not None else np.zeros_like(md) + 120
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": azi})
    df, _ = standardize_survey_columns(df)
    return WellSectionClassifier(min_interval_stations=3).classify_with_confidence(df).survey


def _forbidden_sections(series: pd.Series) -> set:
    return {"Build", "Drop", "Tangent", "Transition", "Horizontal / Lateral"} & set(series.unique())


def test_vertical_only_well():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.linspace(0, 8, len(md))
    out = _classify(md, inc)
    assert (out["Well_Section"] == "Vertical").mean() > 0.5
    assert (out["Section_Code"] == "V").mean() > 0.5
    assert not _forbidden_sections(out["Well_Section"])


def test_lateral_section():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.concatenate([np.linspace(0, 85, 40), np.full(len(md) - 40, 88.0)])
    out = _classify(md, inc)
    assert (out["Well_Section"] == "Lateral").mean() > 0.25
    assert "L" in out["Section_Code"].values


def test_curve_section_present():
    md = np.arange(0, 4000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2200), md >= 2200],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    out = _classify(md, inc)
    assert "Curve" in out["Well_Section"].values
    assert "C" in out["Section_Code"].values


def test_section_labels_vcl():
    assert set(SECTION_LABELS) == {"Vertical", "Curve", "Lateral"}
    assert SECTION_CODES == {"Vertical": "V", "Curve": "C", "Lateral": "L"}


def test_section_filter():
    md = np.arange(0, 900, 30, dtype=float)
    inc = np.piecewise(md, [md < 300, md >= 300], [0, 80.0]).astype(float)
    out = _classify(md, inc)
    only_v = filter_by_section_codes(out, ["V"])
    assert (only_v["Section_Code"] == "V").all()
    assert len(only_v) < len(out)

    only_l = filter_by_section_codes(out, ["L"])
    if not only_l.empty:
        assert set(only_l["Well_Section"].unique()) <= {"Lateral"}
        assert (only_l["Section_Code"] == "L").all()


def test_section_confidence_column_present():
    md = np.arange(0, 1000, 25, dtype=float)
    inc = 20 + 30 * np.sin(md / 200)
    out = _classify(md, inc)
    assert "Section_Confidence" in out.columns
    assert out["Section_Confidence"].between(0, 1).all()


def _section_rank(series: pd.Series) -> list[int]:
    order = {"Vertical": 0, "Curve": 1, "Lateral": 2}
    return [order[str(s)] for s in series]


def _contiguous_blocks(series: pd.Series, label: str) -> int:
    blocks = 0
    active = False
    for value in series:
        if value == label:
            if not active:
                blocks += 1
                active = True
        else:
            active = False
    return blocks


def test_monotonic_vcl_progression():
    md = np.arange(0, 4000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2200), md >= 2200],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    out = _classify(md, inc)
    ranks = _section_rank(out["Well_Section"])
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
    assert _contiguous_blocks(out["Well_Section"], "Vertical") <= 1
    assert _contiguous_blocks(out["Well_Section"], "Curve") <= 1
    assert _contiguous_blocks(out["Well_Section"], "Lateral") <= 1
    assert not _forbidden_sections(out["Well_Section"])


def test_no_lateral_to_vertical_regression():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.concatenate([np.linspace(0, 85, 40), np.full(len(md) - 40, 88.0)])
    out = _classify(md, inc)
    ranks = _section_rank(out["Well_Section"])
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))


def test_well_section_intervals_contiguous_non_overlapping():
    from sections import compute_well_section_intervals

    md = np.arange(0, 4000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2200), md >= 2200],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    out = _classify(md, inc)
    intervals = compute_well_section_intervals(out)
    assert not intervals.empty
    for i in range(len(intervals) - 1):
        assert float(intervals.iloc[i]["MD_Out"]) <= float(intervals.iloc[i + 1]["MD_In"]) + 1e-6
    codes = intervals["Section_Code"].tolist()
    assert codes == [c for c in ["V", "C", "L"] if c in codes]
