"""User-facing column labels (hide internal engineering names)."""

from __future__ import annotations

import pandas as pd

# Internal column name -> dashboard label
SURVEY_DISPLAY_NAMES = {
    "MD": "MD (m)",
    "MD_In": "MD In",
    "MD_Out": "MD Out",
    "Inclination": "Inclination (°)",
    "Azimuth": "Azimuth (°)",
    "DLS": "DLS (°/30m)",
    "Well_Section": "Well section",
    "Section_Code": "Section",
    "Section_Confidence": "Trajectory stability",
    "Build_Rate": "Build rate (°/30m)",
    "Turn_Rate": "Turn rate (°/30m)",
    "Tortuosity_Index": "Tortuosity index",
    "Tortuosity_Index_Local": "Tortuosity index",
    "Wellbore_Smoothness": "Wellbore smoothness",
    "Stability": "Drilling stability",
    "DLS_Stability": "DLS stability",
    "Inclination_Control": "Inclination control",
    "Azimuth_Control": "Azimuth control",
    "Build_Efficiency": "Build efficiency",
    "Lateral_Smoothness": "Lateral smoothness",
    "Drilling_System": "Drilling system",
    "Hole_Size": "Hole size",
    "RSS_Type": "RSS steering",
    "BHA": "BHA",
    "BHA_Run": "BHA run",
    "Survey_File": "Survey file",
    "Primary_Concern": "Trajectory behavior",
    "Trajectory_Severity": "Trajectory quality",
    "Pattern_Type": "Trajectory behavior",
    "Pattern_Confidence": "Trajectory stability",
    "RSS_Confidence": "RSS confidence",
    "RSS_Severity": "RSS severity",
    "Steering_Stability": "Steering stability",
}

BHA_DISPLAY_NAMES = {
    "BHA_Run": "BHA run",
    "Source_File": "Source file",
    "MD_In": "MD In",
    "MD_Out": "MD Out",
    "Drilling_System": "Drilling system",
    "RSS_Type": "RSS steering",
    "Hole_Size": "Hole size",
    "Bit_Type": "Bit type",
    "BHA_Config": "BHA configuration",
    "Interval_Complete": "Interval complete",
    "Mean_DLS": "Mean DLS (°/30m)",
    "Max_DLS": "Max DLS (°/30m)",
    "Mean_Tortuosity": "Mean tortuosity",
    "Performance_Score": "Performance score",
    "Rank": "Rank",
    "Stations": "Survey stations",
    "Interval_Complete": "Interval complete",
    "Mean_Tortuosity": "Mean tortuosity",
    "Max_DLS": "Max DLS",
    "Steering_Stability": "Steering stability",
    "RSS_Severity": "RSS severity",
}


def rename_for_display(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    mapping = mapping or SURVEY_DISPLAY_NAMES
    cols = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=cols)
