import numpy as np
import pandas as pd

from analytics import run_analytics_pipeline
from column_mapping import standardize_survey_columns


def test_pipeline_runs_on_synthetic_well():
    md = np.arange(0, 2000, 30, dtype=float)
    inc = np.piecewise(md, [md < 400, md >= 400], [lambda x: x * 0.01, lambda x: 70.0]).astype(float)
    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": 120 + 0.01 * md})
    df, missing = standardize_survey_columns(df)
    assert not missing

    result = run_analytics_pipeline(df)
    survey = result.survey
    for col in (
        "Well_Section",
        "Section_Code",
        "Section_Confidence",
        "Tortuosity_Index",
        "Stability",
        "Wellbore_Smoothness",
        "DLS",
    ):
        assert col in survey.columns
    assert not result.section_comparison.empty
    assert set(survey["Section_Code"].unique()).issubset({"V", "C", "L"})
