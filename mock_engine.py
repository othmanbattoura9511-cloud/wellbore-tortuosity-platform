import numpy as np
import pandas as pd


class TortuosityAnalyzer:
    """Engineering-oriented tortuosity indicators based on DLS and local variation patterns."""

    def __init__(self, md_col="MD", inc_col="Inclination", azi_col="Azimuth", dls_col="DLS"):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col

    @staticmethod
    def dogleg_angle_rad(i1, i2, a1, a2):
        i1, i2, a1, a2 = map(np.radians, [i1, i2, a1, a2])
        cos_dl = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1)
        return np.arccos(np.clip(cos_dl, -1.0, 1.0))

    def calculate_dls(self, df: pd.DataFrame, unit_length=30) -> pd.DataFrame:
        df = df.copy()
        theta = [np.nan]
        for idx in range(1, len(df)):
            theta.append(self.dogleg_angle_rad(
                df.loc[idx-1, self.inc_col], df.loc[idx, self.inc_col],
                df.loc[idx-1, self.azi_col], df.loc[idx, self.azi_col]
            ))
        df["Dogleg_Angle_rad"] = theta
        df["DLS_Calc"] = np.degrees(df["Dogleg_Angle_rad"]) / df[self.md_col].diff() * unit_length
        return df

    def add_indicators(self, df: pd.DataFrame, rolling_window=10) -> pd.DataFrame:
        df = df.copy()
        dls = df[self.dls_col] if self.dls_col in df.columns else df.get("DLS_Calc")
        if dls is None:
            raise ValueError("DLS column not found. Run calculate_dls or provide DLS.")
        df["DLS_Rolling_Mean"] = dls.rolling(rolling_window, min_periods=3).mean()
        df["DLS_Rolling_Std"] = dls.rolling(rolling_window, min_periods=3).std()
        df["DLS_Abs_Change"] = dls.diff().abs()
        df["Tortuosity_Index_Local"] = df["DLS_Rolling_Mean"].fillna(0) + df["DLS_Rolling_Std"].fillna(0)
        df["High_Frequency_Indicator"] = df["DLS_Abs_Change"].rolling(rolling_window, min_periods=3).mean()
        return df

    def summarize_by_group(self, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        dls_col = self.dls_col if self.dls_col in df.columns else "DLS_Calc"
        return df.groupby(group_cols).agg(
            n_surveys=(self.md_col, "count"),
            md_start=(self.md_col, "min"),
            md_end=(self.md_col, "max"),
            mean_dls=(dls_col, "mean"),
            max_dls=(dls_col, "max"),
            std_dls=(dls_col, "std"),
            mean_tortuosity=("Tortuosity_Index_Local", "mean"),
            max_tortuosity=("Tortuosity_Index_Local", "max"),
        ).reset_index()
