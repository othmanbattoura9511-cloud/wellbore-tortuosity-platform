import pandas as pd


class SurveyQualityAnalyzer:
    def __init__(self, md_col="MD", inc_col="Inclination", azi_col="Azimuth", dls_col="DLS"):
        self.md_col = md_col
        self.inc_col = inc_col
        self.azi_col = azi_col
        self.dls_col = dls_col

    def evaluate(self, df: pd.DataFrame) -> dict:
        spacing = df[self.md_col].diff().dropna()
        metrics = {
            "n_surveys": int(len(df)),
            "md_start": float(df[self.md_col].min()),
            "md_end": float(df[self.md_col].max()),
            "mean_spacing": float(spacing.mean()) if len(spacing) else None,
            "min_spacing": float(spacing.min()) if len(spacing) else None,
            "max_spacing": float(spacing.max()) if len(spacing) else None,
            "poor_spacing_pct": float((spacing > 30).mean() * 100),
            "missing_values": df.isna().sum().to_dict(),
        }
        if self.dls_col in df.columns:
            metrics.update({
                "mean_dls": float(df[self.dls_col].mean()),
                "max_dls": float(df[self.dls_col].max()),
                "std_dls": float(df[self.dls_col].std()),
            })
        return metrics

    def classify_spacing(self, df: pd.DataFrame, dense_threshold=5, coarse_threshold=30) -> pd.DataFrame:
        df = df.copy()
        spacing = df[self.md_col].diff()
        df["Survey_Density_Class"] = "medium"
        df.loc[spacing <= dense_threshold, "Survey_Density_Class"] = "dense"
        df.loc[spacing >= coarse_threshold, "Survey_Density_Class"] = "coarse"
        return df
