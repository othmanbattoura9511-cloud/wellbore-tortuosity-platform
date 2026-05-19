import numpy as np
import pandas as pd

from bha_analytics import build_bha_runs_table, rank_bha_performance
from bha_extraction import _normalize_hole_size, extract_bha_from_text, normalize_bha_intervals


def test_normalize_hole_size_rejects_unit_only():
    assert _normalize_hole_size("in") == "Unknown"
    assert _normalize_hole_size("IN") == "Unknown"
    assert _normalize_hole_size("8 1/2 IN") != "Unknown"


def test_extract_missing_md_no_crash():
    row = extract_bha_from_text("MUD MOTOR BHA HOLE SIZE: IN", source_file="no_md.pdf")
    assert pd.isna(row["MD_In"])
    assert pd.isna(row["MD_Out"])
    assert row["Hole_Size"] == "Unknown"


def test_rank_skips_incomplete_intervals():
    runs = pd.DataFrame(
        [
            {
                "BHA_Run": "Run 1",
                "MD_In": np.nan,
                "MD_Out": np.nan,
                "Interval_Complete": False,
                "Mean_DLS": 5.0,
                "Mean_Tortuosity": 3.0,
            },
            {
                "BHA_Run": "Run 2",
                "MD_In": 0.0,
                "MD_Out": 1000.0,
                "Interval_Complete": True,
                "Mean_DLS": 2.0,
                "Mean_Tortuosity": 1.0,
            },
        ]
    )
    ranked = rank_bha_performance(runs)
    incomplete = ranked[~ranked["Interval_Complete"].astype(bool)]
    complete = ranked[ranked["Interval_Complete"].astype(bool)]
    assert incomplete["Rank"].isna().all()
    assert complete["Rank"].iloc[0] == 1
    assert complete["Rank"].dtype.name == "Int64"


def test_rank_with_nan_scores_leaves_rank_null():
    runs = pd.DataFrame(
        [
            {
                "BHA_Run": "Run 1",
                "MD_In": 0.0,
                "MD_Out": 500.0,
                "Interval_Complete": True,
                "Mean_DLS": np.nan,
                "Mean_Tortuosity": np.nan,
            },
        ]
    )
    ranked = rank_bha_performance(runs)
    assert "Rank" in ranked.columns
    assert ranked["Rank"].dtype.name == "Int64"
    assert ranked["Rank"].isna().all()


def test_build_runs_marks_incomplete():
    survey = pd.DataFrame({"MD": [100, 200], "DLS": [1.0, 2.0]})
    bha = normalize_bha_intervals(
        pd.DataFrame([{"BHA_Run": "Run 1", "MD_In": None, "MD_Out": None, "Drilling_System": "Motor"}])
    )
    table = build_bha_runs_table(survey, bha)
    assert table.loc[0, "Interval_Complete"] is False or table.loc[0, "Interval_Complete"] == False
    assert table.loc[0, "Stations"] == 0
