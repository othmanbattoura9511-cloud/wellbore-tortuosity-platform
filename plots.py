import pandas as pd
import plotly.express as px

from sections import SECTION_ORDER

SEVERITY_ORDER = ["Stable", "Moderate", "Aggressive", "Critical"]
SEVERITY_COLORS = {
    "Stable": "#2e7d32",
    "Moderate": "#fbc02d",
    "Aggressive": "#ef6c00",
    "Critical": "#c62828",
}


SECTION_COLORS = {
    "Vertical": "#2e7d32",
    "Curve": "#fb8c00",
    "Lateral": "#1565c0",
}

CODE_COLORS = {
    "V": SECTION_COLORS["Vertical"],
    "C": SECTION_COLORS["Curve"],
    "L": SECTION_COLORS["Lateral"],
}


def _section_color_map(df: pd.DataFrame) -> dict:
    if "Well_Section" in df.columns:
        present = [s for s in SECTION_ORDER if s in df["Well_Section"].unique()]
        return {s: SECTION_COLORS.get(s, "#757575") for s in present}
    if "Section_Code" in df.columns:
        return {c: CODE_COLORS.get(c, "#757575") for c in df["Section_Code"].unique()}
    return SECTION_COLORS


class WellPlots:
    def plot_inclination(self, df, md_col="MD", inc_col="Inclination", color_col="Section_Code"):
        color = color_col if color_col in df.columns else "Well_Section"
        return px.line(
            df,
            x=md_col,
            y=inc_col,
            color=color,
            title="Inclination vs measured depth",
            color_discrete_map=_section_color_map(df) if color == "Well_Section" else CODE_COLORS,
        )

    def plot_inclination_vs_md(self, df, color_col="Section_Code"):
        return self.plot_inclination(df, color_col=color_col)

    def plot_dls(self, df, md_col="MD", dls_col="DLS", color_col="Section_Code"):
        color = color_col if color_col in df.columns else "Well_Section"
        return px.line(
            df,
            x=md_col,
            y=dls_col,
            color=color,
            title="DLS vs measured depth",
            labels={dls_col: "DLS (°/30m)"},
            color_discrete_map=_section_color_map(df) if color == "Well_Section" else CODE_COLORS,
        )

    def plot_tortuosity_vs_md(self, df: pd.DataFrame):
        col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
        if col not in df.columns:
            return None
        color = "Section_Code" if "Section_Code" in df.columns else "Well_Section"
        return px.line(
            df,
            x="MD",
            y=col,
            color=color,
            title="Tortuosity index vs measured depth",
            color_discrete_map=CODE_COLORS if color == "Section_Code" else _section_color_map(df),
        )

    def plot_section_distribution(self, df):
        if "Well_Section" not in df.columns and "Section_Code" not in df.columns:
            return None
        col = "Well_Section" if "Well_Section" in df.columns else "Section_Code"
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        color_map = _section_color_map(df) if col == "Well_Section" else CODE_COLORS
        return px.bar(
            counts,
            x=col,
            y="count",
            color=col,
            color_discrete_map=color_map,
            title="Well section distribution (V / C / L)",
        )

    def plot_tortuosity_by_section(self, df):
        col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
        section_col = "Well_Section" if "Well_Section" in df.columns else "Section_Code"
        if col not in df.columns or section_col not in df.columns:
            return None
        color_map = _section_color_map(df) if section_col == "Well_Section" else CODE_COLORS
        return px.box(
            df,
            x=section_col,
            y=col,
            color=section_col,
            color_discrete_map=color_map,
            title="Tortuosity by well section",
        )

    def plot_severity_by_section(self, df):
        if "Trajectory_Severity" not in df.columns:
            return None
        section_col = "Well_Section" if "Well_Section" in df.columns else "Section_Code"
        cross = df.groupby([section_col, "Trajectory_Severity"], observed=False).size().reset_index(name="count")
        return px.bar(
            cross,
            x=section_col,
            y="count",
            color="Trajectory_Severity",
            color_discrete_map=SEVERITY_COLORS,
            title="Trajectory severity by section",
            barmode="group",
            category_orders={"Trajectory_Severity": SEVERITY_ORDER},
        )

    def plot_kpi_timeline(self, df, kpi_col: str = "Oscillation_Score"):
        if kpi_col not in df.columns:
            return None
        color = "Trajectory_Severity" if "Trajectory_Severity" in df.columns else "Section_Code"
        cmap = SEVERITY_COLORS if color == "Trajectory_Severity" else CODE_COLORS
        return px.line(
            df,
            x="MD",
            y=kpi_col,
            color=color if color in df.columns else None,
            color_discrete_map=cmap,
            title=kpi_col.replace("_", " ") + " vs MD",
        )

    def plot_bha_timeline(self, bha_runs: pd.DataFrame, md_max: float):
        if bha_runs is None or bha_runs.empty:
            return None
        rows = []
        for _, r in bha_runs.iterrows():
            if not r.get("Interval_Complete", True):
                continue
            if pd.isna(r.get("MD_In")) or pd.isna(r.get("MD_Out")):
                continue
            rows.append(
                {
                    "BHA_Run": r.get("BHA_Run", r.get("BHA")),
                    "MD_In": float(r["MD_In"]),
                    "Length": float(r["MD_Out"]) - float(r["MD_In"]),
                    "Drilling_System": r.get("Drilling_System", "Unknown"),
                }
            )
        if not rows:
            return None
        tl = pd.DataFrame(rows)
        return px.bar(
            tl,
            x="Length",
            y="BHA_Run",
            base="MD_In",
            orientation="h",
            color="Drilling_System",
            title="BHA run timeline (MD intervals)",
        )

    def plot_bha_ranking(self, ranking: pd.DataFrame):
        ranked = ranking[ranking["Rank"].notna()] if "Rank" in ranking.columns else ranking
        if ranked.empty or "Performance_Score" not in ranked.columns:
            return None
        return px.bar(
            ranked,
            x="BHA_Run",
            y="Performance_Score",
            color="Drilling_System",
            title="BHA performance score (lower = better; complete intervals only)",
        )

    def plot_rss_distribution(self, df):
        if "RSS_Type" not in df.columns:
            return None
        counts = df["RSS_Type"].value_counts().reset_index()
        counts.columns = ["RSS_Type", "count"]
        return px.pie(counts, names="RSS_Type", values="count", title="RSS steering mode distribution")

    def plot_drilling_system_bar(self, system_table: pd.DataFrame, metric: str = "Avg Tortuosity"):
        if system_table is None or system_table.empty or metric not in system_table.columns:
            return None
        return px.bar(
            system_table,
            x="System",
            y=metric,
            color="System",
            title=f"{metric} by drilling system",
        )
