"""Plotly styling for drilling engineering charts (light, readable)."""

from __future__ import annotations

import plotly.graph_objects as go

PALETTE = {
    "bg": "#f8f9fb",
    "paper": "#ffffff",
    "grid": "#dde3ea",
    "text": "#1a2332",
    "muted": "#5c6b7a",
    "accent": "#1565c0",
    "vertical": "#2e7d32",
    "curve": "#ef6c00",
    "lateral": "#1565c0",
}

CODE_COLORS = {"V": PALETTE["vertical"], "C": PALETTE["curve"], "L": PALETTE["lateral"]}

SECTION_COLORS = {
    "Vertical": PALETTE["vertical"],
    "Curve": PALETTE["curve"],
    "Lateral": PALETTE["lateral"],
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13, color=PALETTE["text"]),
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["bg"],
        colorway=[PALETTE["vertical"], PALETTE["curve"], PALETTE["lateral"], PALETTE["accent"]],
        hoverlabel=dict(bgcolor=PALETTE["paper"], font_size=12, font_color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=PALETTE["grid"], borderwidth=1),
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"], title_font=dict(size=12)),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"], title_font=dict(size=12)),
        margin=dict(l=48, r=24, t=56, b=48),
    )
)


def apply_chart_style(
    fig,
    title: str | None = None,
    y_title: str | None = None,
    x_title: str = "MD (m)",
    rangeslider: bool = True,
):
    if fig is None:
        return None
    fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="x unified")
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)))
    if rangeslider:
        fig.update_xaxes(title_text=x_title, rangeslider=dict(visible=True, thickness=0.05))
    elif x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_traces(line=dict(width=2.5), selector=dict(type="scatter"))
    return fig
