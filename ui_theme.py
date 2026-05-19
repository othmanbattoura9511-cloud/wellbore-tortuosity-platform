"""Streamlit UI theme and layout helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from display_labels import BHA_DISPLAY_NAMES
from plot_theme import PALETTE, apply_chart_style


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
        .dashboard-hero {
            background: #ffffff;
            border: 1px solid #dde3ea;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(26, 35, 50, 0.06);
        }
        .dashboard-hero h1 { color: #1a2332; font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
        .dashboard-hero .tagline { color: #5c6b7a; margin: 0; font-size: 0.9rem; line-height: 1.5; }
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #dde3ea;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        .section-panel {
            background: #ffffff;
            border: 1px solid #dde3ea;
            border-left: 4px solid #1565c0;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .section-panel h4 { color: #1a2332; margin: 0 0 0.55rem 0; font-size: 0.98rem; }
        .section-panel .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 0.45rem 1rem;
        }
        .section-panel .item label {
            display: block; color: #5c6b7a; font-size: 0.72rem;
            text-transform: uppercase; letter-spacing: 0.03em;
        }
        .section-panel .item span { color: #1a2332; font-size: 0.9rem; font-weight: 500; }
        .soft-warning {
            background: #fff8e6;
            border-left: 4px solid #ef6c00;
            padding: 0.7rem 1rem;
            border-radius: 0 8px 8px 0;
            color: #1a2332;
            margin: 0.5rem 0;
        }
        [data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        '<div class="dashboard-hero">'
        "<h1>Wellbore Tortuosity Analytics Platform</h1>"
        '<p class="tagline">Survey · tortuosity · pattern recognition · RSS steering · BHA intelligence · ML features</p>'
        "</div>",
        unsafe_allow_html=True,
    )


def soft_warning(message: str) -> None:
    st.markdown(f'<div class="soft-warning">{message}</div>', unsafe_allow_html=True)


def render_kpi_row(metrics: list[tuple[str, str]], compact: bool = False) -> None:
    cols = st.columns(len(metrics))
    for slot, (label, value) in zip(cols, metrics):
        slot.metric(label, value)
    if not compact:
        st.markdown("")


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
        f"<div class='item'><label>{lbl}</label><span>{_panel_value(val)}</span></div>" for lbl, val in fields
    )
    html = (
        f"<div class='section-panel' style='border-left-color:{color};'>"
        f"<h4>{title}</h4><div class='grid'>{items}</div></div>"
    )
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
                    go.Scatter(x=sub["Inclination"], y=sub["MD"], mode="lines", name=label, line=dict(color=c, width=2.5))
                )
    else:
        fig.add_trace(
            go.Scatter(x=work["Inclination"], y=work["MD"], mode="lines", name="Wellpath", line=dict(color=PALETTE["accent"], width=2.5))
        )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=320)
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
