import numpy as np
import pandas as pd

from column_mapping import missing_columns_message, standardize_survey_columns


class SurveyPreprocessor:
    """Clean survey data and standardize columns for engineering analysis."""

    def __init__(self, md_col="MD", inc_col="Inclination", azi_col="Azimuth", dls_col="DLS"):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df, missing = standardize_survey_columns(df)
        if missing:
            raise ValueError(missing_columns_message(missing, df.columns))

        df = df.copy()
        required = [self.md_col, self.inc_col, self.azi_col]
        for col in required + ([self.dls_col] if self.dls_col in df.columns else []):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)
        df = df.sort_values(self.md_col).drop_duplicates(subset=[self.md_col])
        df["Delta_MD"] = df[self.md_col].diff()
        df.loc[df["Delta_MD"] <= 0, "Delta_MD"] = np.nan
        return df.reset_index(drop=True)

    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
        return df
