"""Survey column normalization and manual mapping helpers."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

REQUIRED_SURVEY_COLUMNS = ("MD", "Inclination", "Azimuth")

MD_ALIASES = (
    "MD",
    "Measured Depth",
    "Measured_Depth",
    "Depth",
    "DEPTH",
    "MD (m)",
    "MD(m)",
    "MD(ft)",
    "MD (ft)",
    "Definitive MD",
)

INCLINATION_ALIASES = (
    "Inclination",
    "INC",
    "Inc",
    "INCL",
    "Incl",
    "Angle",
    "Inclination (deg)",
    "Inclination (°)",
    "Definitive Inc",
)

AZIMUTH_ALIASES = (
    "Azimuth",
    "AZI",
    "Azi",
    "AZIM",
    "Azim",
    "Bearing",
    "Azimuth (deg)",
    "Azimuth (°)",
    "Definitive Azi",
)

OPTIONAL_ALIASES = {
    "DLS": ("DLS", "Dogleg Rate", "Dogleg\nRate", "DLS (deg/30m)"),
    "TVD": ("TVD", "Vertical Depth", "TVD (m)"),
}


def normalize_column_name(name: object) -> str:
    text = str(name).strip().replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def column_match_key(name: object) -> str:
    """Normalized key for alias lookup (case/unit insensitive)."""
    text = normalize_column_name(name).lower().replace("_", " ")
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    groups = (
        ("MD", MD_ALIASES),
        ("Inclination", INCLINATION_ALIASES),
        ("Azimuth", AZIMUTH_ALIASES),
    )
    for canonical, aliases in groups:
        for alias in aliases:
            lookup[column_match_key(alias)] = canonical
        lookup[column_match_key(canonical)] = canonical
    for canonical, aliases in OPTIONAL_ALIASES.items():
        for alias in aliases:
            lookup[column_match_key(alias)] = canonical
    return lookup


ALIAS_KEY_TO_CANONICAL = _build_alias_lookup()


def detect_canonical_column(column_name: object) -> Optional[str]:
    return ALIAS_KEY_TO_CANONICAL.get(column_match_key(column_name))


def suggest_column_index(columns: Iterable[object], canonical: str) -> int:
    cols = list(columns)
    for idx, col in enumerate(cols):
        if detect_canonical_column(col) == canonical:
            return idx
    return 0


def apply_manual_column_map(
    df: pd.DataFrame,
    md_column: str,
    inclination_column: str,
    azimuth_column: str,
) -> pd.DataFrame:
    manual = {
        "MD": md_column,
        "Inclination": inclination_column,
        "Azimuth": azimuth_column,
    }
    mapped, _ = standardize_survey_columns(df, manual_map=manual)
    return mapped


def standardize_survey_columns(
    df: pd.DataFrame,
    manual_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rename survey columns to MD / Inclination / Azimuth.

    manual_map: {canonical_name: source_column_name_in_df}
    """
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    out = out.loc[:, ~out.columns.astype(str).str.contains("^Unnamed", case=False, na=False)]

    if manual_map:
        for canonical, source in manual_map.items():
            if not source or source not in out.columns:
                continue
            if source != canonical:
                out = out.rename(columns={source: canonical})

    rename_pairs: Dict[str, str] = {}
    for col in list(out.columns):
        if col in REQUIRED_SURVEY_COLUMNS:
            continue
        canonical = detect_canonical_column(col)
        if canonical and canonical in REQUIRED_SURVEY_COLUMNS and col != canonical:
            rename_pairs[col] = canonical
    if rename_pairs:
        out = out.rename(columns=rename_pairs)

    missing = [c for c in REQUIRED_SURVEY_COLUMNS if c not in out.columns]
    return out, missing


def missing_columns_message(missing: List[str], available: Iterable[object]) -> str:
    cols = ", ".join(f"`{c}`" for c in available)
    need = ", ".join(f"**{c}**" for c in missing)
    return (
        f"Missing required survey columns: {need}. "
        f"Available columns in your file: {cols}. "
        "Use the column mapping section below to assign MD, Inclination, and Azimuth."
    )
