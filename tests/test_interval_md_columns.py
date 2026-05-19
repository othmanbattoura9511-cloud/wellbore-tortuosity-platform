import pandas as pd

from column_mapping import analyze_survey_columns, expand_interval_survey


def test_md_start_end_maps_to_md_in_out():
    df = pd.DataFrame(
        {
            "MD Start": [100.0, 200.0],
            "MD End": [150.0, 250.0],
            "Inc": [10.0, 40.0],
            "Azi": [90.0, 91.0],
        }
    )
    out = expand_interval_survey(df)
    assert "MD_In" in out.columns
    assert "MD_Out" in out.columns
    assert out["MD_In"].tolist() == [100.0, 200.0]
    assert out["MD_Out"].tolist() == [150.0, 250.0]
    assert out["MD"].tolist() == [125.0, 225.0]

    result = analyze_survey_columns(df)
    assert not result.missing
    assert "MD_In" in result.survey.columns
    assert "MD_Out" in result.survey.columns
