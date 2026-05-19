import pandas as pd

from column_mapping import (
    apply_manual_column_map,
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
