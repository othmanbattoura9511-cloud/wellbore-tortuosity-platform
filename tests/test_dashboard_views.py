"""Dashboard view helpers."""

import pandas as pd

from dashboard_views import build_drilling_systems_table, build_section_engineering_table


def test_section_engineering_table_vcl():
    df = pd.DataFrame(
        {
            "MD": [100, 200, 300, 400],
            "Inclination": [5, 45, 80, 90],
            "DLS": [1.0, 3.0, 2.0, 1.5],
            "Section_Code": ["V", "C", "L", "L"],
            "Tortuosity_Index": [0.5, 1.2, 1.0, 0.9],
            "Stability": [0.9, 0.7, 0.8, 0.85],
            "Hole_Size": ["8.5 in", "8.5 in", "6.75 in", "6.75 in"],
        }
    )
    table = build_section_engineering_table(df)
    assert len(table) == 3
    assert "V — Vertical" in table["Section"].values


def test_drilling_systems_table_from_bha_runs():
    survey = pd.DataFrame(
        {
            "MD": [100, 150, 200],
            "DLS": [2.0, 3.0, 2.5],
            "Section_Code": ["V", "V", "C"],
            "Tortuosity_Index": [1.0, 1.1, 1.2],
            "Stability": [0.8, 0.75, 0.7],
            "Build_Efficiency": [0.5, 0.6, 0.7],
        }
    )
    runs = pd.DataFrame(
        [
            {
                "BHA_Run": "BHA-1",
                "MD_In": 100.0,
                "MD_Out": 200.0,
                "Drilling_System": "RSS",
                "RSS_Type": "Push-the-bit",
                "Hole_Size": "8.5 in",
                "Mean_DLS": 2.5,
                "Mean_Tortuosity": 1.1,
                "Interval_Complete": True,
            }
        ]
    )
    out = build_drilling_systems_table(survey, runs)
    assert len(out) == 1
    assert out.iloc[0]["Motor or RSS"] == "RSS"
    assert out.iloc[0]["RSS type"] == "Push-the-bit"
