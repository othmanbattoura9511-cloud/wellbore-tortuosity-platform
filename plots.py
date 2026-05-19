import pandas as pd
import plotly.express as px

from sections import SECTION_ORDER


SECTION_COLORS = {
    "Vertical": "#2e7d32",
    "Build": "#ef6c00",
    "Drop": "#c62828",
    "Horizontal / Lateral": "#1565c0",
    "Horizontal": "#1565c0",
}

PATTERN_COLORS = {
    "Normal": "#546e7a",
    "Sinusoidal": "#1565c0",
    "Helical": "#6a1b9a",
    "Micro-tortuosity": "#c62828",
    "Chaotic": "#ef6c00",
}


def _section_color_map(df: pd.DataFrame) -> dict:
    present = [s for s in SECTION_ORDER if s in df.get("Well_Section", pd.Series(dtype=str)).unique()]
    legacy = [s for s in df["Well_Section"].unique() if s not in present]
    ordered = present + list(legacy)
    return {s: SECTION_COLORS.get(s, "#757575") for s in ordered}


class WellPlots:
    def plot_dls(self, df, md_col="MD", dls_col="DLS", color_col=None):
        return px.line(df, x=md_col, y=dls_col, color=color_col, title="DLS vs Measured Depth")

    def plot_inclination(self, df, md_col="MD", inc_col="Inclination", color_col=None):
        return px.line(df, x=md_col, y=inc_col, color=color_col, title="Inclination vs Measured Depth")

    def plot_azimuth(self, df, md_col="MD", azi_col="Azimuth", color_col=None):
        return px.line(df, x=md_col, y=azi_col, color=color_col, title="Azimuth vs Measured Depth")

    def plot_tortuosity_map(self, df, x_col="Local_North", y_col="Local_East", color_col="Tortuosity_Index_Local"):
        if x_col in df.columns and y_col in df.columns:
            return px.scatter(df, x=x_col, y=y_col, color=color_col, title="Tortuosity Map")
        return px.scatter(df, x="MD", y=color_col, color=color_col, title="Tortuosity Index vs MD")

    def plot_section_distribution(self, df):
        if "Well_Section" not in df.columns:
            return None
        counts = df["Well_Section"].value_counts().reset_index()
        counts.columns = ["Well_Section", "count"]
        color_map = _section_color_map(df)
        counts["Well_Section"] = pd.Categorical(
            counts["Well_Section"], categories=list(color_map.keys()), ordered=True
        )
        return px.bar(
            counts,
            x="Well_Section",
            y="count",
            color="Well_Section",
            color_discrete_map=color_map,
            title="Well section distribution",
            category_orders={"Well_Section": list(color_map.keys())},
        )

    def plot_pattern_by_section(self, df):
        if "Pattern_Type" not in df.columns or "Well_Section" not in df.columns:
            return None
        cross = df.groupby(["Well_Section", "Pattern_Type"], observed=False).size().reset_index(name="count")
        color_map = _section_color_map(df)
        return px.bar(
            cross,
            x="Well_Section",
            y="count",
            color="Pattern_Type",
            color_discrete_map=PATTERN_COLORS,
            title="Pattern distribution by section",
            barmode="group",
            category_orders={"Well_Section": list(color_map.keys())},
        )

    def plot_tortuosity_by_section(self, df):
        if "Well_Section" not in df.columns or "Tortuosity_Index_Local" not in df.columns:
            return None
        color_map = _section_color_map(df)
        return px.box(
            df,
            x="Well_Section",
            y="Tortuosity_Index_Local",
            color="Well_Section",
            color_discrete_map=color_map,
            title="Tortuosity by well section",
            category_orders={"Well_Section": list(color_map.keys())},
        )

    def plot_rss_distribution(self, df):
        if "RSS_Type" not in df.columns:
            return None
        counts = df["RSS_Type"].value_counts().reset_index()
        counts.columns = ["RSS_Type", "count"]
        return px.pie(counts, names="RSS_Type", values="count", title="RSS type distribution")
