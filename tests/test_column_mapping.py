import pandas as pd

from column_mapping import (
    analyze_survey_columns,
    apply_manual_column_map,
    fuzzy_score,
    standardize_survey_columns,
)


def test_auto_maps_underscore_and_abbreviations():
    df = pd.DataFrame(
        {
            "Measured_Depth": [100.0, 110.0],
            "INC": [10.0, 11.0],
            "AZIM": [45.0, 46.0],
        }
    )
    mapped, missing = standardize_survey_columns(df)
    assert missing == []
    assert list(mapped.columns[:3]) == ["MD", "Inclination", "Azimuth"]


def test_auto_maps_units_in_parentheses():
    df = pd.DataFrame(
        {
            "MD (ft)": [100.0, 110.0],
            "Angle": [10.0, 11.0],
            "Bearing": [45.0, 46.0],
        }
    )
    mapped, missing = standardize_survey_columns(df)
    assert missing == []
    assert {"MD", "Inclination", "Azimuth"}.issubset(mapped.columns)


def test_manual_mapping_custom_headers():
    df = pd.DataFrame(
        {
            "Station": [1, 2],
            "HoleDepth": [100.0, 110.0],
            "DevAngle": [10.0, 11.0],
            "Dir": [45.0, 46.0],
        }
    )
    mapped = apply_manual_column_map(df, "HoleDepth", "DevAngle", "Dir")
    _, missing = standardize_survey_columns(mapped)
    assert missing == []


def test_md_start_not_mapped_as_md_on_interval_file():
    df = pd.DataFrame(
        {
            "MD Start": [10.0, 20.0],
            "MD End": [20.0, 30.0],
            "INC": [1.0, 2.0],
            "AZI": [3.0, 4.0],
        }
    )
    result = analyze_survey_columns(df)
    assert not result.missing
    assert result.survey["MD"].tolist() == [15.0, 25.0]


def test_fuzzy_azm_matches_azimuth():
    assert fuzzy_score("Azm", "Azimuth") >= 0.72
