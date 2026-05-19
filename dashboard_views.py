"""Tab content builders for the drilling engineering dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from analytics import AnalyticsResult, comparison_table
from bha_analytics import incomplete_bha_runs
from comparison_tables import (
    build_drilling_system_comparison_table,
    build_hole_size_comparison,
    build_rss_steering_comparison,
)
from display_labels import show_dataframe
from plots import WellPlots
from sections import CODE_TO_LABEL, SECTION_CODES
from ui_theme import render_kpi_row, render_section_panel, render_wellpath_hero, soft_warning

EMPTY_SURVEY_MESSAGE = "Upload one or more survey files, or load the sample well."
NO_SECTION_MESSAGE = "Please select at least one section."
NO_STATIONS_MESSAGE = "No stations match the selected section filters."


def _plot_chart(fig, key: str) -> None:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _mean(series: pd.Series) -> float | None:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(v.mean()), 3) if len(v) else None


def _kpi_summary_from_survey(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"mean_dls": 0.0, "mean_tortuosity": 0.0, "mean_stability": 0.0, "mean_smoothness": 0.0}
    tort_col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
    return {
        "mean_dls": float(pd.to_numeric(df.get("DLS"), errors="coerce").mean() or 0),
        "mean_tortuosity": float(pd.to_numeric(df.get(tort_col), errors="coerce").mean() or 0),
        "mean_stability": float(pd.to_numeric(df.get("Stability"), errors="coerce").mean() or 0),
        "mean_smoothness": float(pd.to_numeric(df.get("Wellbore_Smoothness"), errors="coerce").mean() or 0),
    }


def _dominant_section(section_mix: dict[str, float]) -> str:
    if not section_mix:
        return "—"
    code = max(section_mix, key=section_mix.get)
    name = CODE_TO_LABEL.get(code, code)
    return f"{code} — {name}"


def _filter_section_comparison(table: pd.DataFrame, section_codes: list[str]) -> pd.DataFrame:
    if table.empty or not section_codes:
        return table.iloc[0:0].copy()
    allowed = set(section_codes)
    prefixes = table["Section"].astype(str).str.split("—").str[0].str.strip()
    return table[prefixes.isin(allowed)].copy()


def build_section_engineering_table(df: pd.DataFrame, section_codes: list[str]) -> pd.DataFrame:
    """Per selected V / C / L engineering summary."""
    if df.empty or "Section_Code" not in df.columns:
        return pd.DataFrame()

    tort_col = "Tortuosity_Index" if "Tortuosity_Index" in df.columns else "Tortuosity_Index_Local"
    rows: list[dict[str, Any]] = []
    for label, code in SECTION_CODES.items():
        if code not in section_codes:
            continue
        display_label = CODE_TO_LABEL.get(code, label)
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
    """One row per BHA with drilling-performance fields (survey already section-filtered)."""
    if bha_runs is None or bha_runs.empty:
        return pd.DataFrame()

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

        mean_dls = _mean(subset.get("DLS", pd.Series(dtype=float)))
        tort_col = "Tortuosity_Index" if "Tortuosity_Index" in subset.columns else "Tortuosity_Index_Local"
        mean_tort = _mean(subset.get(tort_col, pd.Series(dtype=float)))

        rows.append(
            {
                "BHA name": r.get("BHA_Run", r.get("BHA", "Unknown")),
                "Motor or RSS": system,
                "RSS type": rss,
                "Hole size": r.get("Hole_Size", "Unknown"),
                "MD In (m)": md_in,
                "MD Out (m)": md_out,
                "Section used in": _dominant_section(section_mix),
                "Avg DLS (°/30m)": mean_dls,
                "Tortuosity index": mean_tort,
                "Stability": _mean(subset.get("Stability", pd.Series(dtype=float))),
                "Steering efficiency": steering,
                "Performance score": r.get("Performance_Score"),
                "Rank": r.get("Rank"),
                "Interval complete": bool(r.get("Interval_Complete", complete)),
            }
        )
    return pd.DataFrame(rows)


def render_overview_tab(
    df: pd.DataFrame,
    result: AnalyticsResult,
    quality: dict,
    plots: WellPlots,
    section_codes: list[str],
) -> None:
    render_wellpath_hero(df, chart_key="overview_wellpath")
    kpis = _kpi_summary_from_survey(df)
    md_end = quality.get("md_end", df["MD"].max() if "MD" in df.columns and not df.empty else 0)
    dominant = df["Well_Section"].mode().iloc[0] if "Well_Section" in df.columns and not df.empty else "—"
    counts = df["Section_Code"].value_counts().to_dict() if "Section_Code" in df.columns else {}

    render_kpi_row(
        [
            ("MD end (m)", f"{md_end:.0f}"),
            ("Max DLS (°/30m)", f"{quality.get('max_dls', pd.to_numeric(df.get('DLS'), errors='coerce').max()):.2f}"),
            ("Avg tortuosity", f"{kpis['mean_tortuosity']:.2f}"),
            ("Dominant section", str(dominant)),
            ("Survey stations", str(len(df))),
        ]
    )

    st.markdown("##### Main KPIs")
    render_kpi_row(
        [
            ("Mean DLS", f"{kpis['mean_dls']:.2f}"),
            ("Mean stability", f"{kpis['mean_stability']:.2f}"),
            ("Mean smoothness", f"{kpis['mean_smoothness']:.2f}"),
            ("Vertical (V)", str(counts.get("V", 0))),
            ("Lateral (L)", str(counts.get("L", 0))),
        ],
        compact=True,
    )

    st.markdown("##### Trajectory & drilling response")
    c1, c2 = st.columns(2)
    with c1:
        _plot_chart(plots.plot_inclination_vs_md(df), "overview_inclination_md")
        _plot_chart(plots.plot_tortuosity_vs_md(df), "overview_tortuosity_md")
    with c2:
        _plot_chart(plots.plot_dls(df), "overview_dls_md")
        _plot_chart(plots.plot_section_distribution(df), "overview_section_distribution")


def render_section_analysis_tab(df: pd.DataFrame, plots: WellPlots, section_codes: list[str]) -> None:
    st.markdown("Engineering analysis by **V — Vertical**, **C — Curve**, and **L — Lateral**.")
    table = build_section_engineering_table(df, section_codes)
    if table.empty:
        st.caption("No classified sections in the current filter.")
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
    _plot_chart(plots.plot_tortuosity_by_section(df), "section_tortuosity_box")


def render_drilling_systems_tab(
    survey: pd.DataFrame,
    result: AnalyticsResult,
    bha_intervals: pd.DataFrame,
    plots: WellPlots,
    section_codes: list[str],
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

    md_max = float(survey["MD"].max()) if "MD" in survey.columns and not survey.empty else 0.0
    _plot_chart(plots.plot_bha_timeline(result.bha_runs, md_max), "drilling_bha_timeline")

    ranked = result.bha_ranking
    ranked_complete = ranked[ranked["Rank"].notna()] if not ranked.empty and "Rank" in ranked.columns else ranked
    if not ranked_complete.empty:
        st.markdown("##### Performance ranking (complete MD intervals only)")
        show_dataframe(ranked_complete.sort_values("Rank", na_position="last"))
        _plot_chart(plots.plot_bha_ranking(ranked_complete), "drilling_bha_ranking")


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
        st.markdown("##### Mapped / merged survey (filtered)")
        show_dataframe(df_mapped, max_rows=500)
        st.download_button(
            "Export mapped CSV",
            df_mapped.to_csv(index=False).encode("utf-8"),
            "survey_mapped.csv",
            "text/csv",
            key="download_mapped_csv",
        )
    with c2:
        st.markdown("##### Processed survey (filtered, cleaned + KPIs)")
        show_dataframe(df_processed, max_rows=500)
        st.download_button(
            "Export processed CSV",
            df_processed.to_csv(index=False).encode("utf-8"),
            "survey_processed.csv",
            "text/csv",
            key="download_processed_csv",
        )

    with st.expander("Survey quality metrics"):
        st.json(quality)


def render_comparisons_tab(
    df: pd.DataFrame,
    result: AnalyticsResult,
    plots: WellPlots,
    section_codes: list[str],
) -> None:
    st.markdown(
        "Compare drilling systems, RSS modes, hole sizes, BHAs, and well sections. "
        "**Trajectory patterns and engineering KPI timelines** appear only in this tab."
    )

    section_tbl = _filter_section_comparison(result.section_comparison, section_codes)
    system_tbl = build_drilling_system_comparison_table(section_tbl)
    rss_tbl = build_rss_steering_comparison(section_tbl)
    hole_tbl = build_hole_size_comparison(section_tbl)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Section comparison (V / C / L × BHA)")
        if section_tbl.empty:
            st.caption("No section comparison data for the selected filter.")
        else:
            show_dataframe(section_tbl, use_display_names=False)
    with col_r:
        st.markdown("##### Drilling system comparison")
        if system_tbl.empty:
            st.caption("No system comparison data for the selected filter.")
        else:
            show_dataframe(system_tbl, use_display_names=False)
            _plot_chart(plots.plot_drilling_system_bar(system_tbl, "Avg Tortuosity"), "comparison_system_bar")

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

    for idx, (title, cols) in enumerate(groups):
        tbl = comparison_table(df, cols)
        if tbl.empty:
            continue
        with st.expander(title, expanded=title.startswith("Motor") or title.startswith("RSS")):
            show_dataframe(tbl.reset_index(), csv_filename=f"comparison_{idx}.csv")

    st.markdown("---")
    st.markdown("##### Patterns & engineering KPIs")
    kpis = _kpi_summary_from_survey(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean DLS", f"{kpis['mean_dls']:.2f}")
    k2.metric("Mean tortuosity", f"{kpis['mean_tortuosity']:.2f}")
    k3.metric("Mean stability", f"{kpis['mean_stability']:.2f}")
    k4.metric("Mean smoothness", f"{kpis['mean_smoothness']:.2f}")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        _plot_chart(plots.plot_dls(df), "comparison_dls")
        _plot_chart(plots.plot_tortuosity_vs_md(df), "comparison_tortuosity")
    with chart_cols[1]:
        _plot_chart(plots.plot_kpi_timeline(df, "Wellbore_Smoothness"), "comparison_smoothness")
        _plot_chart(plots.plot_kpi_timeline(df, "Stability"), "comparison_stability")

    _plot_chart(plots.plot_tortuosity_by_section(df), "comparison_tortuosity_box")

    st.markdown("##### RSS & hole size")
    c1, c2 = st.columns(2)
    with c1:
        if not rss_tbl.empty:
            show_dataframe(rss_tbl, use_display_names=False)
        else:
            st.caption("No RSS steering comparison for the selected filter.")
    with c2:
        if not hole_tbl.empty:
            show_dataframe(hole_tbl, use_display_names=False)
        else:
            st.caption("No hole size comparison for the selected filter.")

    ranked = result.bha_ranking
    if not ranked.empty and ranked["Rank"].notna().any():
        st.markdown("##### BHA performance ranking")
        _plot_chart(plots.plot_bha_ranking(ranked[ranked["Rank"].notna()]), "comparison_bha_ranking")

    _plot_chart(plots.plot_rss_distribution(df), "comparison_rss_pie")
