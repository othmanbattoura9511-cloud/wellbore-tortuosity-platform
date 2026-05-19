"""Streamlit UI theme, branding, and layout helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from display_labels import BHA_DISPLAY_NAMES
from plot_theme import PALETTE, apply_chart_style

HP_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="40" viewBox="0 0 120 40">'
    '<rect width="120" height="40" rx="6" fill="#1a2332" stroke="#d4a017" stroke-width="1"/>'
    '<text x="60" y="26" text-anchor="middle" font-family="Segoe UI, Arial" '
    'font-size="18" font-weight="700" fill="#d4a017">H&amp;P</text></svg>'
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
        .stApp { background: linear-gradient(165deg, #0a0e14 0%, #0f1419 40%, #121c28 100%); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #121c28 0%, #0f1419 100%);
            border-right: 1px solid #2d3a4d;
        }
        .dashboard-hero {
            background: linear-gradient(135deg, #1a2332 0%, #243447 50%, #1a2332 100%);
            border: 1px solid #2d3a4d; border-radius: 16px; padding: 1.25rem 1.5rem;
            margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        }
        .dashboard-hero h1 { color: #f5f7fa; font-size: 1.75rem; font-weight: 700; margin: 0 0 0.25rem 0; }
        .dashboard-hero p { color: #8fa3b8; margin: 0; font-size: 0.95rem; }
        .hp-logo { float: right; margin-top: -4px; }
        div[data-testid="metric-container"] {
            background: #1a2332; border: 1px solid #2d3a4d; border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="metric-container"] label { color: #8fa3b8 !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f5f7fa !important; }
        .soft-warning {
            background: rgba(212, 160, 23, 0.12); border-left: 4px solid #d4a017;
            padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; color: #e8eef4; margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(show_hp_branding: bool = True) -> None:
    logo = f'<div class="hp-logo">{HP_LOGO_SVG}</div>' if show_hp_branding else ""
    st.markdown(
        f'<div class="dashboard-hero">{logo}'
        "<h1>Wellbore Tortuosity Analytics</h1>"
        "<p>Drilling engineering · V / C / L · Motor vs RSS · multi-BHA</p></div>",
        unsafe_allow_html=True,
    )


def soft_warning(message: str) -> None:
    st.markdown(f'<div class="soft-warning">{message}</div>', unsafe_allow_html=True)


def render_wellpath_hero(df: pd.DataFrame) -> None:
    if df.empty or "MD" not in df.columns or "Inclination" not in df.columns:
        return
    work = df.sort_values("MD")
    fig = go.Figure()
    if "Section_Code" in work.columns:
        for code, label, c in [("V", "Vertical", PALETTE["vertical"]), ("C", "Curve", PALETTE["curve"]), ("L", "Lateral", PALETTE["lateral"])]:
            sub = work[work["Section_Code"] == code]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Inclination"], y=sub["MD"], mode="lines", name=label, line=dict(color=c, width=3)))
    else:
        fig.add_trace(go.Scatter(x=work["Inclination"], y=work["MD"], mode="lines", name="Wellpath", line=dict(color=PALETTE["accent"], width=3)))
    fig.update_layout(title="Well trajectory", xaxis_title="Inclination (deg)", yaxis_title="MD (m)", yaxis=dict(autorange="reversed"), height=300)
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def bha_editor_column_config() -> dict:
    from streamlit import column_config
    return {
        "MD In": column_config.NumberColumn("MD In", help="Start MD (m)", format="%.1f"),
        "MD Out": column_config.NumberColumn("MD Out", help="End MD (m)", format="%.1f"),
    }


def prepare_bha_for_editor(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in BHA_DISPLAY_NAMES.items() if k in df.columns})


def prepare_bha_from_editor(df: pd.DataFrame) -> pd.DataFrame:
    inv = {v: k for k, v in BHA_DISPLAY_NAMES.items()}
    return df.rename(columns={c: inv[c] for c in df.columns if c in inv})
