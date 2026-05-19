"""Tab content builders for the drilling engineering dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from analytics import AnalyticsResult, comparison_table
from bha_analytics import incomplete_bha_runs
from display_labels import show_dataframe
from plots import WellPlots
from sections import SECTION_CODES
from ui_theme import render_kpi_row, render_section_panel, soft_warning


def _mean(series: pd.Series) -> float | None:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(v.mean()), 3) if len(v) else None


def _dominant_section(section_mix: dict[str, float]) -> str:
    if not section_mix:
        return "—"
    code = max(section_mix, key=section_mix.get)
    labels = {"Vertical": "Vertical", "Curve": "Curve", "Lateral": "Lateral"}
    name = next((k for k, v in SECTION_CODES.items() if v == code), code)
    return f"{code} — {labels.get(name, name)}"


def build_section_engineering_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per V / C / L engineering summary."""
    if df.empty or "Section_Code" not in df.columns:
        return pd.DataFrame()

    tort_col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
    rows: list[dict[str, Any]] = []
    section_labels = {"Vertical": "Vertical", "Curve": "Curve", "Lateral": "Lateral"}
    for label, code in SECTION_CODES.items():
        display_label = section_labels.get(label, label)
        block = df[df["Section_Code"] == code]
        if block.empty:
            continue
        hole = "—"
        if "Hole_Size" in block.columns:
            modes = block["Hole_Size"].replace("Unknown", pd.NA).dropna()
            hole = str(modes.mode().iloc[0]) if len(modes) else "—"
        perf = _mean(block.get("Stability", pd.Series(dtype=float)))
        rows.append(
            {
                "Section": f"{code} — {display_label}",
                "MD interval (m)": f"{block['MD'].min():.0f} – {block['MD'].max():.0f}",
                "Stations": len(block),
                "Avg inclination (°)": _mean(block["Inclination"]),
                "Avg DLS (°/30m)": _mean(block["DLS"]),
                "Tortuosity index": _mean(block.get(tort_col, pd.Series(dtype=float))),
                "Stability": _mean(block.get("Stability", pd.Series(dtype=float))),
                "Hole size": hole,
                "Section performance": perf,
            }
        )
    return pd.DataFrame(rows)


def build_drilling_systems_table(survey: pd.DataFrame, bha_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per BHA with drilling-performance fields."""
    if bha_runs is None or bha_runs.empty:
        return pd.DataFrame()

    tort_col = "Tortuosity_Index" if "Tortuosity_Index" in survey.columns else "Tortuosity_Index_Local"
    rows: list[dict[str, Any]] = []
    for _, r in bha_runs.iterrows():
        md_in, md_out = r.get("MD_In"), r.get("MD_Out")
        complete = pd.notna(md_in) and pd.notna(md_out)
        if complete:
            subset = survey[(survey["MD"] >= float(md_in)) & (survey["MD"] <= float(md_out))]
        else:
            subset = survey.iloc[0:0]

        section_mix: dict[str, float] = {}
        if not subset.empty and "Section_Code" in subset.columns:
            section_mix = subset["Section_Code"].value_counts(normalize=True).to_dict()

        steering = _mean(subset.get("Build_Efficiency", pd.Series(dtype=float)))
        if steering is None:
            steering = _mean(subset.get("Steering_Stability", pd.Series(dtype=float)))

        system = str(r.get("Drilling_System", "Unknown"))
        rss = str(r.get("RSS_Type", "—"))
        if system != "RSS":
            rss = "—"

        rows.append(
            {
                "BHA name": r.get("BHA_Run", r.get("BHA", "Unknown")),
                "Motor or RSS": system,
                "RSS type": rss,
                "Hole size": r.get("Hole_Size", "Unknown"),
                "MD In (m)": md_in,
                "MD Out (m)": md_out,
                "Section used in": _dominant_section(section_mix),
                "Avg DLS (°/30m)": r.get("Mean_DLS"),
                "Tortuosity index": r.get("Mean_Tortuosity"),
                "Stability": _mean(subset.get("Stability", pd.Series(dtype=float))),
                "Steering efficiency": steering,
                "Performance score": r.get("Performance_Score"),
                "Rank": r.get("Rank"),
                "Interval complete": bool(r.get("Interval_Complete", complete)),
            }
        )
    return pd.DataFrame(rows)


def render_overview_tab(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    result: AnalyticsResult,
    quality: dict,
    plots: WellPlots,
) -> None:
    from ui_theme import render_wellpath_hero

    render_wellpath_hero(df_full)
    md_end = quality.get("md_end", df_full["MD"].max() if "MD" in df_full.columns else 0)
    render_kpi_row(
        [
            ("MD end (m)", f"{md_end:.0f}"),
            ("Max DLS (°/30m)", f"{quality.get('max_dls', df_full['DLS'].max()):.2f}"),
            ("Avg tortuosity", f"{result.kpi_summary.get('mean_tortuosity', 0):.2f}"),
            ("Dominant section", str(result.section_summary.get("dominant_section", "—"))),
            ("Survey stations", str(len(df_full))),
        ]
    )

    st.markdown("##### Main KPIs")
    render_kpi_row(
        [
            ("Mean DLS", f"{result.kpi_summary.get('mean_dls', 0):.2f}"),
            ("Mean stability", f"{result.kpi_summary.get('mean_stability', 0):.2f}"),
            ("Mean smoothness", f"{result.kpi_summary.get('mean_smoothness', 0):.2f}"),
            ("Vertical (V)", str(result.section_summary.get("section_code_counts", {}).get("V", 0))),
            ("Lateral (L)", str(result.section_summary.get("section_code_counts", {}).get("L", 0))),
        ],
        compact=True,
    )

    st.markdown("##### Trajectory & drilling response")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_inclination_vs_md(df_filtered), use_container_width=True)
        tort_fig = plots.plot_tortuosity_vs_md(df_filtered)
        if tort_fig:
            st.plotly_chart(tort_fig, use_container_width=True)
    with c2:
        st.plotly_chart(plots.plot_dls(df_filtered), use_container_width=True)
        dist = plots.plot_section_distribution(df_filtered)
        if dist:
            st.plotly_chart(dist, use_container_width=True)

def render_section_analysis_tab(df_full: pd.DataFrame, plots: WellPlots) -> None:
    st.markdown("Engineering analysis by **V — Vertical**, **C — Curve**, and **L — Lateral**.")
    table = build_section_engineering_table(df_full)
    if table.empty:
        st.caption("No classified sections in the current survey.")
        return

    accent_map = {"V": "vertical", "C": "curve", "L": "lateral"}
    for _, row in table.iterrows():
        code = str(row["Section"]).split("—")[0].strip()
        render_section_panel(
            str(row["Section"]),
            [
                ("MD interval", row["MD interval (m)"]),
                ("Avg inclination", row["Avg inclination (°)"]),
                ("Avg DLS", row["Avg DLS (°/30m)"]),
                ("Tortuosity", row["Tortuosity index"]),
                ("Stability", row["Stability"]),
                ("Hole size", row["Hole size"]),
                ("Performance", row["Section performance"]),
            ],
            accent=accent_map.get(code, "accent"),
        )

    st.markdown("##### Section summary table")
    show_dataframe(table, use_display_names=False)

    box = plots.plot_tortuosity_by_section(df_full)
    if box:
        st.plotly_chart(box, use_container_width=True)


def render_drilling_systems_tab(
    survey: pd.DataFrame,
    result: AnalyticsResult,
    bha_intervals: pd.DataFrame,
    plots: WellPlots,
) -> None:
    st.markdown("BHA and drilling-system performance — map PDF assemblies to survey MD, then review KPIs.")

    if not bha_intervals.empty:
        for _, row in incomplete_bha_runs(bha_intervals).iterrows():
            label = row.get("BHA_Run", row.get("Source_File", "BHA"))
            soft_warning(f"Enter <b>MD In</b> and <b>MD Out</b> for <b>{label}</b> to include this run in performance metrics.")

    systems = build_drilling_systems_table(survey, result.bha_runs)
    if systems.empty:
        st.info("Upload BHA PDFs in the sidebar and set MD intervals to populate this dashboard.")
        return

    complete = systems[systems["Interval complete"] == True]  # noqa: E712
    if not complete.empty:
        st.markdown("##### BHA performance cards")
        for _, row in complete.iterrows():
            render_section_panel(
                str(row["BHA name"]),
                [
                    ("System", row["Motor or RSS"]),
                    ("RSS type", row["RSS type"]),
                    ("Hole size", row["Hole size"]),
                    ("MD In / Out", f"{row['MD In (m)']} – {row['MD Out (m)']} m"),
                    ("Section", row["Section used in"]),
                    ("Avg DLS", row["Avg DLS (°/30m)"]),
                    ("Tortuosity", row["Tortuosity index"]),
                    ("Stability", row["Stability"]),
                    ("Steering efficiency", row["Steering efficiency"]),
                ],
                accent="accent",
            )

    st.markdown("##### All BHA runs")
    show_dataframe(systems, use_display_names=False)

    tl = plots.plot_bha_timeline(result.bha_runs, float(survey["MD"].max()))
    if tl:
        st.plotly_chart(tl, use_container_width=True)

    ranked = result.bha_ranking
    ranked_complete = ranked[ranked["Rank"].notna()] if not ranked.empty and "Rank" in ranked.columns else ranked
    if not ranked_complete.empty:
        st.markdown("##### Performance ranking (complete MD intervals only)")
        show_dataframe(ranked_complete.sort_values("Rank", na_position="last"))
        rank_fig = plots.plot_bha_ranking(ranked_complete)
        if rank_fig:
            st.plotly_chart(rank_fig, use_container_width=True)


def render_survey_data_tab(
    df_mapped: pd.DataFrame,
    df_processed: pd.DataFrame,
    quality: dict,
    upload_labels: list[str],
) -> None:
    st.markdown("Uploaded surveys, column mapping, and processed tables for export.")

    st.markdown("##### Data sources")
    if upload_labels:
        for name in upload_labels:
            st.caption(f"• {name}")
    else:
        st.caption("No files loaded.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Mapped / merged survey")
        show_dataframe(df_mapped, max_rows=500)
        st.download_button(
            "Export mapped CSV",
            df_mapped.to_csv(index=False).encode("utf-8"),
            "survey_mapped.csv",
            "text/csv",
        )
    with c2:
        st.markdown("##### Processed survey (cleaned + KPIs)")
        show_dataframe(df_processed, max_rows=500)
        st.download_button(
            "Export processed CSV",
            df_processed.to_csv(index=False).encode("utf-8"),
            "survey_processed.csv",
            "text/csv",
            key="export_processed",
        )

    with st.expander("Survey quality metrics"):
        st.json(quality)


def render_comparisons_tab(
    df: pd.DataFrame,
    df_full: pd.DataFrame,
    result: AnalyticsResult,
    plots: WellPlots,
) -> None:
    st.markdown(
        "Compare drilling systems, RSS modes, hole sizes, BHAs, and well sections. "
        "**Trajectory patterns and engineering KPI timelines** appear only in this tab."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Section comparison (V / C / L × BHA)")
        if result.section_comparison.empty:
            st.caption("No section comparison data.")
        else:
            show_dataframe(result.section_comparison, use_display_names=False)
    with col_r:
        st.markdown("##### Drilling system comparison")
        if result.system_comparison.empty:
            st.caption("No system comparison data.")
        else:
            show_dataframe(result.system_comparison, use_display_names=False)
            fig = plots.plot_drilling_system_bar(result.system_comparison, "Avg Tortuosity")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Detailed comparisons")
    groups = [
        ("V vs C vs L", ["Section_Code", "Well_Section"]),
        ("Motor vs RSS vs Rotary", ["Drilling_System"]),
        ("RSS Push vs Point", ["RSS_Type"]),
        ("Hole size", ["Hole_Size"]),
        ("Section × drilling system", ["Section_Code", "Drilling_System"]),
        ("Section × RSS steering", ["Section_Code", "RSS_Type"]),
        ("Section × hole size", ["Section_Code", "Hole_Size"]),
    ]
    if "BHA_Run" in df.columns and df["BHA_Run"].nunique() > 1:
        groups.append(("BHA run", ["BHA_Run"]))
    if "Survey_File" in df.columns and df["Survey_File"].nunique() > 1:
        groups.append(("Survey run", ["Survey_File"]))

    for title, cols in groups:
        tbl = comparison_table(df, cols)
        if tbl.empty:
            continue
        with st.expander(title, expanded=title.startswith("Motor") or title.startswith("RSS")):
            show_dataframe(tbl.reset_index())

    st.markdown("---")
    st.markdown("##### Patterns & engineering KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean DLS", f"{result.kpi_summary.get('mean_dls', 0):.2f}")
    k2.metric("Mean tortuosity", f"{result.kpi_summary.get('mean_tortuosity', 0):.2f}")
    k3.metric("Mean stability", f"{result.kpi_summary.get('mean_stability', 0):.2f}")
    k4.metric("Mean smoothness", f"{result.kpi_summary.get('mean_smoothness', 0):.2f}")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(plots.plot_dls(df_full), use_container_width=True)
        tort_fig = plots.plot_tortuosity_vs_md(df_full)
        if tort_fig:
            st.plotly_chart(tort_fig, use_container_width=True)
    with chart_cols[1]:
        smooth_fig = plots.plot_kpi_timeline(df_full, "Wellbore_Smoothness")
        if smooth_fig:
            st.plotly_chart(smooth_fig, use_container_width=True)
        stab_fig = plots.plot_kpi_timeline(df_full, "Stability")
        if stab_fig:
            st.plotly_chart(stab_fig, use_container_width=True)

    box = plots.plot_tortuosity_by_section(df_full)
    if box:
        st.plotly_chart(box, use_container_width=True)

    st.markdown("##### RSS & hole size")
    c1, c2 = st.columns(2)
    with c1:
        if not result.rss_steering_comparison.empty:
            show_dataframe(result.rss_steering_comparison, use_display_names=False)
        else:
            st.caption("No RSS steering comparison.")
    with c2:
        if not result.hole_size_comparison.empty:
            show_dataframe(result.hole_size_comparison, use_display_names=False)
        else:
            st.caption("No hole size comparison.")

    if not result.bha_ranking.empty and result.bha_ranking["Rank"].notna().any():
        st.markdown("##### BHA performance ranking")
        rank_fig = plots.plot_bha_ranking(result.bha_ranking)
        if rank_fig:
            st.plotly_chart(rank_fig, use_container_width=True)

    rss_pie = plots.plot_rss_distribution(df_full)
    if rss_pie:
        st.plotly_chart(rss_pie, use_container_width=True)
