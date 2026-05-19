"""Engineering summary table."""

import numpy as np
import pandas as pd

from comparison_tables import ENGINEERING_SUMMARY_COLUMNS, build_engineering_summary_table
from dashboard_views import _filter_section_comparison


def test_engineering_summary_columns_and_filter():
    md = np.arange(0, 900, 30, dtype=float)
    inc = np.piecewise(md, [md < 300, md >= 300], [0, 80.0]).astype(float)
    survey = pd.DataFrame(
        {
            "MD": md,
            "Inclination": inc,
            "Azimuth": 120.0,
            "DLS": 2.0,
            "Section_Code": ["V"] * 10 + ["L"] * (len(md) - 10),
            "Well_Section": ["Vertical"] * 10 + ["Lateral"] * (len(md) - 10),
            "Tortuosity_Index": 1.0,
            "Stability": 0.8,
            "Wellbore_Smoothness": 0.7,
            "Drilling_System": "Motor",
            "Hole_Size": "8.5 in",
        }
    )
    bha = pd.DataFrame(
        [
            {
                "BHA_Run": "Run-1",
                "BHA": "Run-1",
                "MD_In": 0.0,
                "MD_Out": 900.0,
                "Drilling_System": "Motor",
                "Hole_Size": "8.5 in",
                "RSS_Type": "Unknown RSS Type",
                "Source_File": "bha.pdf",
            }
        ]
    )
    table = build_engineering_summary_table(survey, bha)
    assert not table.empty
    assert set(ENGINEERING_SUMMARY_COLUMNS).issubset(set(table.columns))
    assert "MD In" in table.columns
    assert "MD Out" in table.columns

    only_l = _filter_section_comparison(table, ["L"])
    assert not only_l.empty
    assert only_l["Section"].str.startswith("L").all()
    assert only_l["Section"].str.contains("Lateral").all()
