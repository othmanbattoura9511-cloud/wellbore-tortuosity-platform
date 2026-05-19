import pandas as pd
import plotly.express as px

from sections import SECTION_ORDER


SECTION_COLORS = {
    "Vertical": "#2e7d32",
    "Curve": "#fb8c00",
    "Lateral": "#1565c0",
}


class WellPlots:
    """Minimal engineering charts for the comparison dashboard."""

    def plot_inclination_vs_md(self, df: pd.DataFrame, color_col: str = "Section_Code") -> object:
        return px.line(
            df,
            x="MD",
            y="Inclination",
            color=color_col,
            title="Inclination vs measured depth",
            labels={"MD": "MD (m)", "Inclination": "Inclination (°)", "Section_Code": "Section"},
            color_discrete_map={"V": SECTION_COLORS["Vertical"], "C": SECTION_COLORS["Curve"], "L": SECTION_COLORS["Lateral"]},
        )

    def plot_tortuosity_vs_md(self, df: pd.DataFrame) -> object | None:
        col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
        if col not in df.columns:
            return None
        return px.line(
            df,
            x="MD",
            y=col,
            color="Section_Code",
            title="Tortuosity index vs measured depth",
            labels={"MD": "MD (m)", col: "Tortuosity index", "Section_Code": "Section"},
            color_discrete_map={"V": SECTION_COLORS["Vertical"], "C": SECTION_COLORS["Curve"], "L": SECTION_COLORS["Lateral"]},
        )
