import numpy as np
import pandas as pd

from analytics import run_analytics_pipeline
from column_mapping import standardize_survey_columns
from comparison_tables import build_drilling_system_comparison_table, build_section_comparison_table


def test_section_and_system_tables():
    md = np.arange(0, 3000, 30, dtype=float)
    inc = np.piecewise(
        md,
        [md < 800, (md >= 800) & (md < 2000), md >= 2000],
        [lambda x: x * 0.01, lambda x: 5 + (x - 800) * 0.04, lambda x: 88.0],
    ).astype(float)
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": 120.0})
    df, _ = standardize_survey_columns(df)

    bha = pd.DataFrame(
        [
            {
                "BHA_Run": "Run 1",
                "BHA": "Run 1",
                "MD_In": 0,
                "MD_Out": 1500,
                "Drilling_System": "Motor",
                "Hole_Size": "8.5 in",
                "RSS_Type": "Unknown RSS Type",
                "Source_File": "a.pdf",
                "Bit_Type": "PDC",
                "BHA_Config": "Unknown",
            },
            {
                "BHA_Run": "Run 2",
                "BHA": "Run 2",
                "MD_In": 1500,
                "MD_Out": 3000,
                "Drilling_System": "RSS",
                "Hole_Size": "8.5 in",
                "RSS_Type": "Push-the-bit",
                "Source_File": "b.pdf",
                "Bit_Type": "PDC",
                "BHA_Config": "Unknown",
            },
        ]
    )

    result = run_analytics_pipeline(df, bha)
    t1 = result.section_comparison
    t2 = result.system_comparison
    assert not t1.empty
    assert "Section" in t1.columns
    assert "Avg DLS (°/30m)" in t1.columns
    assert not t2.empty
    assert "Motor" in t2["System"].values or "RSS" in t2["System"].values
