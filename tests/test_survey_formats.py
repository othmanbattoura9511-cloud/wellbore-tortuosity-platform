import pandas as pd

from column_mapping import (
    SurveyFileType,
    analyze_survey_columns,
    drop_unnamed_columns,
    expand_interval_survey,
    fuzzy_score,
    preprocess_survey_frame,
)


def test_halliburton_style_columns():
    df = pd.DataFrame(
        {
            "Meas. Depth": [100.0, 200.0],
            "Inc": [5.0, 15.0],
            "Azi": [45.0, 46.0],
        }
    )
    result = analyze_survey_columns(df)
    assert result.file_type == SurveyFileType.SURVEY_STATION
    assert not result.missing
    assert "MD" in result.survey.columns


def test_landmark_interval_export():
    df = pd.DataFrame(
        {
            "MD Start": [100.0, 200.0],
            "MD End": [150.0, 250.0],
            "Inclination Deg": [10.0, 40.0],
            "Azimuth Deg": [90.0, 91.0],
        }
    )
    result = analyze_survey_columns(df)
    assert result.file_type == SurveyFileType.INTERVAL_SUMMARY
    assert not result.missing
    assert result.survey["MD"].tolist() == [125.0, 225.0]


def test_schlumberger_abbreviations():
    df = pd.DataFrame(
        {
            "MD": [1.0, 2.0],
            "INC": [3.0, 4.0],
            "AZIM": [5.0, 6.0],
        }
    )
    result = analyze_survey_columns(df)
    assert not result.missing


def test_compass_survey_depth():
    df = pd.DataFrame(
        {
            "Survey Depth": [500.0, 510.0],
            "Incl": [20.0, 21.0],
            "Azm": [100.0, 101.0],
        }
    )
    result = analyze_survey_columns(df)
    assert not result.missing


def test_detects_bha_table():
    df = pd.DataFrame(
        {
            "Component": ["Motor", "Bit"],
            "OD": [6.5, 8.5],
            "Description": ["Mud motor", "PDC"],
        }
    )
    result = analyze_survey_columns(df)
    assert result.file_type == SurveyFileType.BHA_TABLE
    assert result.missing


def test_detects_rss_report():
    df = pd.DataFrame(
        {
            "RSS Mode": ["Push", "Point"],
            "Toolface": [45, 90],
            "Steering Mode": ["A", "B"],
        }
    )
    result = analyze_survey_columns(df)
    assert result.file_type == SurveyFileType.RSS_REPORT


def test_drop_unnamed_columns():
    df = pd.DataFrame(
        {
            "MD": [1, 2],
            "Unnamed: 2": [None, None],
            "Inc": [3, 4],
            "Azimuth": [5, 6],
        }
    )
    cleaned = drop_unnamed_columns(df)
    assert "Unnamed: 2" not in cleaned.columns


def test_preprocess_strips_bom_and_spaces():
    df = pd.DataFrame({"\ufeffMD ": [1], " Inc": [2], "Azimuth ": [3]})
    result = analyze_survey_columns(df)
    assert not result.missing


def test_fuzzy_inclination_deg():
    assert fuzzy_score("Inclination Deg", "Inclination") >= 0.85


def test_manual_map_via_analyze():
    df = pd.DataFrame({"HoleDepth": [1], "Dev": [2], "Dir": [3]})
    result = analyze_survey_columns(
        df,
        manual_map={"MD": "HoleDepth", "Inclination": "Dev", "Azimuth": "Dir"},
    )
    assert not result.missing


def test_expand_interval_midpoint():
    df = pd.DataFrame({"MD Start": [0.0], "MD End": [100.0], "Inc": [10.0], "Azi": [20.0]})
    out = expand_interval_survey(df)
    assert out["MD"].iloc[0] == 50.0
