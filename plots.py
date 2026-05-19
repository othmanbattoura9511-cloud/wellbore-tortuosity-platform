import pandas as pd
import plotly.express as px

from plot_theme import CODE_COLORS, SECTION_COLORS, apply_chart_style
from sections import SECTION_ORDER


class WellPlots:
    def _color_map(self, df, color_col):
        if color_col == "Well_Section":
            return {s: SECTION_COLORS.get(s, "#757575") for s in SECTION_ORDER if s in df.get("Well_Section", [])}
        return CODE_COLORS

    def plot_inclination(self, df, md_col="MD", inc_col="Inclination", color_col="Section_Code"):
        color = color_col if color_col in df.columns else "Well_Section"
        fig = px.line(df, x=md_col, y=inc_col, color=color, color_discrete_map=self._color_map(df, color))
        return apply_chart_style(fig, "Inclination vs measured depth", "Inclination (deg)")

    def plot_inclination_vs_md(self, df, color_col="Section_Code"):
        return self.plot_inclination(df, color_col=color_col)

    def plot_dls(self, df, md_col="MD", dls_col="DLS", color_col="Section_Code"):
        color = color_col if color_col in df.columns else "Well_Section"
        fig = px.line(df, x=md_col, y=dls_col, color=color, color_discrete_map=self._color_map(df, color))
        return apply_chart_style(fig, "DLS vs measured depth", "DLS (deg/30m)")

    def plot_tortuosity_vs_md(self, df: pd.DataFrame):
        col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
        if col not in df.columns:
            return None
        color = "Section_Code" if "Section_Code" in df.columns else "Well_Section"
        fig = px.line(df, x="MD", y=col, color=color, color_discrete_map=self._color_map(df, color))
        return apply_chart_style(fig, "Tortuosity index vs measured depth", "Tortuosity index")

    def plot_section_distribution(self, df):
        col = "Well_Section" if "Well_Section" in df.columns else "Section_Code"
        if col not in df.columns:
            return None
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", color=col, color_discrete_map=self._color_map(df, col))
        return apply_chart_style(fig, "Well section distribution (V / C / L)", "Station count", rangeslider=False)

    def plot_tortuosity_by_section(self, df):
        col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
        section_col = "Well_Section" if "Well_Section" in df.columns else "Section_Code"
        if col not in df.columns or section_col not in df.columns:
            return None
        fig = px.box(df, x=section_col, y=col, color=section_col, color_discrete_map=self._color_map(df, section_col))
        return apply_chart_style(fig, "Tortuosity by well section", "Tortuosity index", rangeslider=False)

    def plot_kpi_timeline(self, df, kpi_col: str = "Wellbore_Smoothness"):
        if kpi_col not in df.columns:
            return None
        color = "Section_Code" if "Section_Code" in df.columns else None
        fig = px.line(df, x="MD", y=kpi_col, color=color, color_discrete_map=CODE_COLORS if color else None)
        title = kpi_col.replace("_", " ").title() + " vs MD"
        return apply_chart_style(fig, title, title)

    def plot_bha_timeline(self, bha_runs: pd.DataFrame, md_max: float):
        if bha_runs is None or bha_runs.empty:
            return None
        rows = []
        for _, r in bha_runs.iterrows():
            if pd.isna(r.get("MD_In")) or pd.isna(r.get("MD_Out")):
                continue
            rows.append({"BHA_Run": r.get("BHA_Run", r.get("BHA")), "MD_In": float(r["MD_In"]), "Length": float(r["MD_Out"]) - float(r["MD_In"]), "Drilling_System": r.get("Drilling_System", "Unknown")})
        if not rows:
            return None
        fig = px.bar(pd.DataFrame(rows), x="Length", y="BHA_Run", base="MD_In", orientation="h", color="Drilling_System", title="BHA run timeline")
        fig.update_layout(yaxis_title="BHA run")
        return apply_chart_style(fig, "BHA run timeline (MD intervals)", rangeslider=False)

    def plot_bha_ranking(self, ranking: pd.DataFrame):
        ranked = ranking[ranking["Rank"].notna()] if "Rank" in ranking.columns else ranking
        if ranked.empty or "Performance_Score" not in ranked.columns:
            return None
        fig = px.bar(ranked, x="BHA_Run", y="Performance_Score", color="Drilling_System", title="BHA performance")
        return apply_chart_style(fig, "BHA performance (lower is better)", "Score", rangeslider=False)

    def plot_rss_distribution(self, df):
        if "RSS_Type" not in df.columns:
            return None
        counts = df["RSS_Type"].value_counts().reset_index()
        counts.columns = ["RSS_Type", "count"]
        fig = px.pie(counts, names="RSS_Type", values="count", title="RSS steering mode")
        return apply_chart_style(fig, "RSS steering mode", rangeslider=False)

    def plot_drilling_system_bar(self, system_table: pd.DataFrame, metric: str = "Avg Tortuosity"):
        if system_table is None or system_table.empty or metric not in system_table.columns:
            return None
        fig = px.bar(system_table, x="System", y=metric, color="System", title=metric)
        return apply_chart_style(fig, f"{metric} by drilling system", metric, rangeslider=False)
