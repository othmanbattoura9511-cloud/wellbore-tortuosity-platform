import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class PatternRecognition:
    def cluster_intervals(self, df: pd.DataFrame, features=None, n_clusters=3) -> pd.DataFrame:
        features = features or ["DLS_Rolling_Mean", "DLS_Rolling_Std", "High_Frequency_Indicator"]
        work = df.dropna(subset=features).copy()
        if work.empty:
            df["Pattern_Cluster"] = None
            return df
        x = StandardScaler().fit_transform(work[features])
        work["Pattern_Cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(x)
        df = df.copy()
        df["Pattern_Cluster"] = None
        df.loc[work.index, "Pattern_Cluster"] = work["Pattern_Cluster"]
        return df
