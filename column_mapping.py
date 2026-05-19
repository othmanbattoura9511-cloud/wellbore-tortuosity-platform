"""Survey column normalization, vendor aliases, fuzzy matching, and file-type detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

REQUIRED_SURVEY_COLUMNS = ("MD", "Inclination", "Azimuth")

# --- Vendor alias pools (merged for matching) ---
_GENERIC_MD = (
    "MD",
    "Measured Depth",
    "Measured_Depth",
    "Survey Depth",
    "Survey MD",
    "Depth",
    "DEPTH",
    "Hole Depth",
    "MD (m)",
    "MD(m)",
    "MD [m]",
    "MD (ft)",
    "MD(ft)",
    "MD [ft]",
    "MD m",
    "MD ft",
    "Definitive MD",
    "Svy MD",
    "Station MD",
)
_INTERVAL_MD = ("MD Start", "MD End", "MD From", "MD To", "Start MD", "End MD", "From MD", "To MD")
_HALLIBURTON_MD = ("Meas Depth", "Meas. Depth", "MDepth")
_SLB_MD = ("MDMSL", "MDKB", "MD RT", "MDRT", "SURVEY MD")
_BAKER_MD = ("MD bdf", "Bit MD")
_LANDMARK_MD = ("MD (ftUS)", "MD(usft)", "TVDSS", "TVD SS")
_COMPASS_MD = ("Proj MD", "Project MD", "Plan MD")
_WELLPLAN_MD = ("Wellpath MD", "WP MD")

_GENERIC_INC = (
    "Inclination",
    "INC",
    "Inc",
    "INCL",
    "Incl",
    "Inclination Deg",
    "Inclination (deg)",
    "Inclination (°)",
    "Incl Deg",
    "Inc (deg)",
    "Angle",
    "Dev Angle",
    "Deviation",
    "Hole Incl",
    "Definitive Inc",
    "Inclination Degrees",
)
_SLB_INC = ("INCL(deg)", "Inc RT", "INC RT", "Hole Deviation")
_BAKER_INC = ("Incl.", "Deviation Angle")
_LANDMARK_INC = ("Incl (°)", "DEVI")
_COMPASS_INC = ("Inc Deg", "Survey Inc")

_GENERIC_AZI = (
    "Azimuth",
    "AZI",
    "Azi",
    "AZIM",
    "Azim",
    "Azm",
    "Azimuth Deg",
    "Azimuth (deg)",
    "Azimuth (°)",
    "Azi Deg",
    "AZI (deg)",
    "Bearing",
    "Grid Azimuth",
    "Definitive Azi",
    "Azimuth Degrees",
)
_SLB_AZI = ("AZI RT", "AZIM RT", "GTF", "Toolface")  # GTF only if no better azi col
_BAKER_AZI = ("Azi.", "Brg")
_LANDMARK_AZI = ("Azi (°)", "AZI Grid")

MD_ALIASES = _GENERIC_MD + _INTERVAL_MD + _HALLIBURTON_MD + _SLB_MD + _BAKER_MD + _LANDMARK_MD + _COMPASS_MD + _WELLPLAN_MD
INCLINATION_ALIASES = _GENERIC_INC + _SLB_INC + _BAKER_INC + _LANDMARK_INC + _COMPASS_INC
AZIMUTH_ALIASES = tuple(
    a for a in _GENERIC_AZI + _SLB_AZI + _BAKER_AZI + _LANDMARK_AZI if a not in ("Toolface", "GTF")
)

OPTIONAL_ALIASES = {
    "DLS": ("DLS", "Dogleg Rate", "Dogleg Severity", "DLS (deg/30m)", "DLS (deg/100ft)", "DL"),
    "TVD": ("TVD", "Vertical Depth", "TVD (m)", "TVDSS", "True Vertical Depth"),
    "MD_Start": _INTERVAL_MD[:4],
    "MD_End": _INTERVAL_MD[4:8],
}

FUZZY_MATCH_THRESHOLD = 0.72

# Columns that indicate non-survey content
_BHA_TOKENS = (
    "bha", "component", "description", "serial", "od", "id", "weight", "grade",
    "motor", "stabilizer", "mwd", "lwd", "jar", "bit type",
)
_RSS_TOKENS = (
    "rss", "rotary steer", "push the bit", "point the bit", "steering mode",
    "toolface", "pad", "rib", "navigamma", "autotrack",
)


class SurveyFileType(str, Enum):
    SURVEY_STATION = "survey_station_table"
    INTERVAL_SUMMARY = "interval_survey_export"
    BHA_TABLE = "bha_table"
    RSS_REPORT = "rss_report"
    INVALID = "invalid_survey_file"


@dataclass
class ColumnDetectionResult:
    file_type: SurveyFileType
    survey: pd.DataFrame = field(default_factory=pd.DataFrame)
    mapped_columns: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    fuzzy_scores: Dict[str, float] = field(default_factory=dict)
    message: str = ""


def normalize_column_name(name: object) -> str:
    text = str(name).replace("\ufeff", "").replace("\xa0", " ")
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[\x00-\x1f]", "", text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def column_match_key(name: object) -> str:
    text = normalize_column_name(name).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
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
        if canonical in REQUIRED_SURVEY_COLUMNS:
            continue
        for alias in aliases:
            key = column_match_key(alias)
            if key not in lookup:
                lookup[key] = canonical
    return lookup


ALIAS_KEY_TO_CANONICAL = _build_alias_lookup()

_CANONICAL_ALIAS_KEYS: Dict[str, List[str]] = {
    "MD": [column_match_key(a) for a in MD_ALIASES],
    "Inclination": [column_match_key(a) for a in INCLINATION_ALIASES],
    "Azimuth": [column_match_key(a) for a in AZIMUTH_ALIASES],
}


def fuzzy_score(column_name: object, canonical: str) -> float:
    """Similarity score [0,1] between a column header and a canonical survey field."""
    key = column_match_key(column_name)
    if not key:
        return 0.0
    if key in ALIAS_KEY_TO_CANONICAL and ALIAS_KEY_TO_CANONICAL[key] == canonical:
        return 1.0
    best = 0.0
    for alias_key in _CANONICAL_ALIAS_KEYS.get(canonical, []):
        best = max(best, SequenceMatcher(None, key, alias_key).ratio())
        if key in alias_key or alias_key in key:
            best = max(best, 0.85)
    return best


def detect_canonical_column(column_name: object, threshold: float = FUZZY_MATCH_THRESHOLD) -> Optional[str]:
    exact = ALIAS_KEY_TO_CANONICAL.get(column_match_key(column_name))
    if exact in REQUIRED_SURVEY_COLUMNS:
        return exact
    scores = {c: fuzzy_score(column_name, c) for c in REQUIRED_SURVEY_COLUMNS}
    best_canonical = max(scores, key=scores.get)
    if scores[best_canonical] >= threshold:
        return best_canonical
    return None


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        name = str(col).strip()
        if re.match(r"^Unnamed\s*:?\s*\d*$", name, re.I):
            continue
        if name == "" or name.lower() == "nan":
            continue
        keep.append(col)
    return df.loc[:, keep]


def preprocess_survey_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Trim headers, remove hidden characters, drop empty unnamed columns."""
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    out = drop_unnamed_columns(out)
    # Drop wholly empty columns
    empty_cols = [c for c in out.columns if out[c].isna().all() or (out[c].astype(str).str.strip() == "").all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
    return out


def _column_keys(df: pd.DataFrame) -> List[str]:
    return [column_match_key(c) for c in df.columns]


def _has_tokens(keys: Iterable[str], tokens: Tuple[str, ...]) -> bool:
    """Match whole words or multi-word phrases, not arbitrary substrings (e.g. 'od' in 'mode')."""
    blob = " ".join(keys)
    words = set(blob.split())
    for token in tokens:
        if " " in token:
            if token in blob:
                return True
        elif token in words:
            return True
    return False


def _is_interval_export(keys: List[str]) -> bool:
    has_start = any("md start" in k or "start md" in k or "md from" in k for k in keys)
    has_end = any("md end" in k or "end md" in k or "md to" in k for k in keys)
    return has_start and has_end


def detect_survey_file_type(df: pd.DataFrame) -> SurveyFileType:
    keys = _column_keys(df)
    if not keys:
        return SurveyFileType.INVALID

    if _has_tokens(keys, _BHA_TOKENS) and not _has_tokens(keys, ("inclination", "inc", "azi", "azimuth")):
        return SurveyFileType.BHA_TABLE
    if _has_tokens(keys, _RSS_TOKENS) and not _has_tokens(keys, ("inclination", "inc")):
        return SurveyFileType.RSS_REPORT

    if _is_interval_export(keys):
        inc_ok = any(fuzzy_score(c, "Inclination") >= FUZZY_MATCH_THRESHOLD for c in df.columns)
        azi_ok = any(fuzzy_score(c, "Azimuth") >= FUZZY_MATCH_THRESHOLD for c in df.columns)
        if inc_ok and azi_ok:
            return SurveyFileType.INTERVAL_SUMMARY

    md_ok = sum(1 for c in df.columns if fuzzy_score(c, "MD") >= FUZZY_MATCH_THRESHOLD)
    inc_ok = sum(1 for c in df.columns if fuzzy_score(c, "Inclination") >= FUZZY_MATCH_THRESHOLD)
    azi_ok = sum(1 for c in df.columns if fuzzy_score(c, "Azimuth") >= FUZZY_MATCH_THRESHOLD)
    if md_ok >= 1 and inc_ok >= 1 and azi_ok >= 1:
        return SurveyFileType.SURVEY_STATION

    return SurveyFileType.INVALID


def _find_best_column(df: pd.DataFrame, canonical: str, used: set) -> Tuple[Optional[str], float]:
    best_col = None
    best_score = 0.0
    for col in df.columns:
        if col in used:
            continue
        score = fuzzy_score(col, canonical)
        if score > best_score:
            best_score = score
            best_col = col
    if best_col and best_score >= FUZZY_MATCH_THRESHOLD:
        return best_col, best_score
    return None, best_score


def detect_column_mapping(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Auto-map canonical names to source columns with fuzzy matching."""
    used: set = set()
    mapping: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    for canonical in REQUIRED_SURVEY_COLUMNS:
        if canonical in df.columns:
            mapping[canonical] = canonical
            scores[canonical] = 1.0
            used.add(canonical)
            continue
        col, score = _find_best_column(df, canonical, used)
        if col:
            mapping[canonical] = col
            scores[canonical] = score
            used.add(col)
    return mapping, scores


def _resolve_md_start_end_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    start_col = end_col = None
    for col in df.columns:
        key = column_match_key(col)
        if "md start" in key or "start md" in key or key in {"md from", "from md"}:
            start_col = col
        if "md end" in key or "end md" in key or key in {"md to", "to md"}:
            end_col = col
    return start_col, end_col


def expand_interval_survey(df: pd.DataFrame) -> pd.DataFrame:
    """Map MD Start/End to MD_In/MD_Out and compute station MD (midpoint)."""
    out = df.copy()
    start_col, end_col = _resolve_md_start_end_columns(out)
    if start_col and end_col:
        start = pd.to_numeric(out[start_col], errors="coerce")
        end = pd.to_numeric(out[end_col], errors="coerce")
        out["MD_In"] = start
        out["MD_Out"] = end
        out["MD"] = ((start + end) / 2.0).where(start.notna() & end.notna(), end.fillna(start))
    elif end_col:
        out["MD_Out"] = pd.to_numeric(out[end_col], errors="coerce")
        out["MD"] = out["MD_Out"]
    elif start_col:
        out["MD_In"] = pd.to_numeric(out[start_col], errors="coerce")
        out["MD"] = out["MD_In"]
    return out


def file_type_user_message(file_type: SurveyFileType, columns: Iterable[object]) -> str:
    cols = ", ".join(f"`{c}`" for c in columns)
    if file_type == SurveyFileType.SURVEY_STATION:
        return "Detected a **survey station table**."
    if file_type == SurveyFileType.INTERVAL_SUMMARY:
        return (
            "Detected an **interval survey export** (MD Start / MD End). "
            "Mapped to **MD In** / **MD Out**; station MD uses the interval midpoint."
        )
    if file_type == SurveyFileType.BHA_TABLE:
        return (
            "This file looks like a **BHA component table**, not a directional survey. "
            f"Detected columns: {cols}. "
            "Upload a survey with measured depth, inclination, and azimuth."
        )
    if file_type == SurveyFileType.RSS_REPORT:
        return (
            "This file looks like an **RSS / steering report**, not a survey station table. "
            f"Detected columns: {cols}."
        )
    return (
        "Could not validate this as a **survey station table**. "
        f"Detected columns: {cols}. "
        "Required fields: **MD**, **Inclination**, **Azimuth** (vendor-specific names are supported)."
    )


def suggest_column_index(columns: Iterable[object], canonical: str) -> int:
    cols = list(columns)
    mapping, _ = detect_column_mapping(pd.DataFrame(columns={c: [] for c in cols}))
    if canonical in mapping:
        return cols.index(mapping[canonical])
    for idx, col in enumerate(cols):
        if detect_canonical_column(col) == canonical:
            return idx
    best_idx, best_score = 0, 0.0
    for idx, col in enumerate(cols):
        score = fuzzy_score(col, canonical)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


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
    auto_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    out = preprocess_survey_frame(df)

    if manual_map:
        for canonical, source in manual_map.items():
            if source and source in out.columns and source != canonical:
                out = out.rename(columns={source: canonical})
    elif auto_map:
        for canonical, source in auto_map.items():
            if source in out.columns and source != canonical:
                out = out.rename(columns={source: canonical})

    rename_pairs: Dict[str, str] = {}
    used_targets = set(out.columns) & set(REQUIRED_SURVEY_COLUMNS)
    for col in list(out.columns):
        if col in REQUIRED_SURVEY_COLUMNS:
            continue
        canonical = detect_canonical_column(col)
        if canonical and canonical in REQUIRED_SURVEY_COLUMNS and canonical not in used_targets:
            rename_pairs[col] = canonical
            used_targets.add(canonical)
    if rename_pairs:
        out = out.rename(columns=rename_pairs)

    missing = [c for c in REQUIRED_SURVEY_COLUMNS if c not in out.columns]
    return out, missing


def analyze_survey_columns(
    df: pd.DataFrame,
    manual_map: Optional[Dict[str, str]] = None,
) -> ColumnDetectionResult:
    work = preprocess_survey_frame(df)
    file_type = detect_survey_file_type(work)

    if manual_map:
        if file_type == SurveyFileType.INTERVAL_SUMMARY:
            work = expand_interval_survey(work)
        mapped, missing = standardize_survey_columns(work, manual_map=manual_map)
        if not missing:
            file_type = SurveyFileType.SURVEY_STATION
        return ColumnDetectionResult(
            file_type=file_type,
            survey=mapped,
            mapped_columns=manual_map,
            missing=missing,
            message=file_type_user_message(file_type, work.columns)
            if missing
            else "Applied manual column mapping.",
        )

    if file_type not in (SurveyFileType.SURVEY_STATION, SurveyFileType.INTERVAL_SUMMARY):
        return ColumnDetectionResult(
            file_type=file_type,
            survey=work,
            missing=list(REQUIRED_SURVEY_COLUMNS),
            message=file_type_user_message(file_type, work.columns),
        )

    if file_type == SurveyFileType.INTERVAL_SUMMARY:
        work = expand_interval_survey(work)

    auto_map, fuzzy_scores = detect_column_mapping(work)
    mapped, missing = standardize_survey_columns(work, auto_map=auto_map)

    return ColumnDetectionResult(
        file_type=file_type,
        survey=mapped,
        mapped_columns=auto_map,
        missing=missing,
        fuzzy_scores=fuzzy_scores,
        message=file_type_user_message(file_type, work.columns),
    )


def missing_columns_message(missing: List[str], available: Iterable[object]) -> str:
    cols = ", ".join(f"`{c}`" for c in available)
    need = ", ".join(f"**{c}**" for c in missing)
    return (
        f"Missing required survey columns: {need}. "
        f"Available columns in your file: {cols}. "
        "Map MD, Inclination, and Azimuth below (vendor aliases and fuzzy matching are applied automatically)."
    )
