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
        :root {
            --dash-bg: #0f1419;
            --dash-surface: #1a2332;
            --dash-border: #2d3a4d;
            --dash-text: #f5f7fa;
            --dash-muted: #8fa3b8;
            --dash-accent: #d4a017;
        }
        [data-theme="light"] {
            --dash-bg: #f4f6f9;
            --dash-surface: #ffffff;
            --dash-border: #d0d7e2;
            --dash-text: #1a2332;
            --dash-muted: #5c6b7a;
            --dash-accent: #b8860b;
        }
        html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
        .stApp {
            background: linear-gradient(165deg, var(--dash-bg) 0%, var(--dash-surface) 55%, var(--dash-bg) 100%);
        }
        [data-testid="stSidebar"] {
            background: var(--dash-surface);
            border-right: 1px solid var(--dash-border);
        }
        .dashboard-hero {
            background: linear-gradient(135deg, var(--dash-surface) 0%, var(--dash-border) 120%, var(--dash-surface) 100%);
            border: 1px solid var(--dash-border);
            border-radius: 16px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.12);
        }
        .dashboard-hero h1 { color: var(--dash-text); font-size: 1.85rem; font-weight: 700; margin: 0 0 0.35rem 0; }
        .dashboard-hero .tagline { color: var(--dash-muted); margin: 0; font-size: 1rem; line-height: 1.45; }
        .dashboard-hero .chips { margin-top: 0.65rem; color: var(--dash-muted); font-size: 0.82rem; }
        .hp-logo { float: right; margin-top: -4px; }
        div[data-testid="metric-container"] {
            background: var(--dash-surface);
            border: 1px solid var(--dash-border);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        div[data-testid="metric-container"] label { color: var(--dash-muted) !important; font-size: 0.8rem !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--dash-text) !important;
            font-weight: 600 !important;
        }
        .kpi-strip { margin: 0.75rem 0 1.25rem 0; }
        .section-panel {
            background: var(--dash-surface);
            border: 1px solid var(--dash-border);
            border-left: 4px solid var(--dash-accent);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
        }
        .section-panel h4 { color: var(--dash-text); margin: 0 0 0.65rem 0; font-size: 1rem; }
        .section-panel .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 0.5rem 1rem;
        }
        .section-panel .item label {
            display: block; color: var(--dash-muted); font-size: 0.72rem;
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .section-panel .item span { color: var(--dash-text); font-size: 0.92rem; font-weight: 500; }
        .soft-warning {
            background: rgba(212, 160, 23, 0.12);
            border-left: 4px solid var(--dash-accent);
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            color: var(--dash-text);
            margin: 0.5rem 0;
        }
        [data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(show_hp_branding: bool = True) -> None:
    logo = f'<motion-placeholder class="hp-logo">{HP_LOGO_SVG}</motion-placeholder>' if show_hp_branding else ""
    html = (
        f'<motion-placeholder class="dashboard-hero">{logo}'
        "<h1>Wellbore Tortuosity Analytics</h1>"
        '<p class="tagline">Drilling engineering analytics and BHA performance platform</p>'
        '<p class="chips">V / C / L sections · Motor vs RSS · multi-BHA · multi-survey</p>'
        "</motion-placeholder>"
    )
    st.markdown(html.replace("motion-placeholder", "div"), unsafe_allow_html=True)


def soft_warning(message: str) -> None:
    st.markdown(
        f'<motion-placeholder class="soft-warning">{message}</motion-placeholder>'.replace("motion-placeholder", "div"),
        unsafe_allow_html=True,
    )


def render_kpi_row(metrics: list[tuple[str, str]], compact: bool = False) -> None:
    cols = st.columns(len(metrics))
    for slot, (label, value) in zip(cols, metrics):
        slot.metric(label, value)
    if not compact:
        st.markdown('<div class="kpi-strip"></div>', unsafe_allow_html=True)


def _panel_value(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return str(val)


def render_section_panel(title: str, fields: list[tuple[str, object]], accent: str = "vertical") -> None:
    border_colors = {
        "vertical": PALETTE["vertical"],
        "curve": PALETTE["curve"],
        "lateral": PALETTE["lateral"],
        "accent": PALETTE["accent"],
    }
    color = border_colors.get(accent, PALETTE["accent"])
    items = "".join(
        f"<motion-placeholder class='item'><label>{lbl}</label><span>{_panel_value(val)}</span></motion-placeholder>"
        for lbl, val in fields
    ).replace("motion-placeholder", "div")
    html = (
        f"<motion-placeholder class='section-panel' style='border-left-color:{color};'>"
        f"<h4>{title}</h4><div class='grid'>{items}</div></motion-placeholder>"
    ).replace("motion-placeholder", "div")
    st.markdown(html, unsafe_allow_html=True)


def render_wellpath_hero(df: pd.DataFrame) -> None:
    if df.empty or "MD" not in df.columns or "Inclination" not in df.columns:
        return
    work = df.sort_values("MD")
    fig = go.Figure()
    if "Section_Code" in work.columns:
        for code, label, c in [
            ("V", "Vertical", PALETTE["vertical"]),
            ("C", "Curve", PALETTE["curve"]),
            ("L", "Lateral", PALETTE["lateral"]),
        ]:
            sub = work[work["Section_Code"] == code]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(x=sub["Inclination"], y=sub["MD"], mode="lines", name=label, line=dict(color=c, width=3))
                )
    else:
        fig.add_trace(
            go.Scatter(x=work["Inclination"], y=work["MD"], mode="lines", name="Wellpath", line=dict(color=PALETTE["accent"], width=3))
        )
    fig.update_layout(title="Well trajectory", yaxis=dict(autorange="reversed"), height=320)
    apply_chart_style(fig, "Well trajectory (inclination vs MD)", "MD (m)", "Inclination (deg)", rangeslider=False)
    fig.update_xaxes(title_text="Inclination (deg)")
    fig.update_yaxes(title_text="MD (m)")
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
