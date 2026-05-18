import pandas as pd


class WellSectionClassifier:
    def __init__(self, inc_col="Inclination", vertical_max=10, lateral_min=80):
        self.inc_col = inc_col
        self.vertical_max = vertical_max
        self.lateral_min = lateral_min

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Well_Section"] = "curve"
        df.loc[df[self.inc_col] <= self.vertical_max, "Well_Section"] = "vertical"
        df.loc[df[self.inc_col] >= self.lateral_min, "Well_Section"] = "lateral"
        return df
