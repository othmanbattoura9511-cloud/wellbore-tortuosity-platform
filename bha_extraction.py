"""Extract BHA run metadata from one or more PDF reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber

DRILLING_SYSTEMS = ("Motor", "RSS", "Rotary", "Unknown")
RSS_TYPES = ("Push-the-bit", "Point-the-bit", "Unknown RSS Type")

BHA_COLUMNS = [
    "BHA_Run",
    "Source_File",
    "MD_In",
    "MD_Out",
    "Drilling_System",
    "RSS_Type",
    "Hole_Size",
    "Bit_Type",
    "BHA_Config",
]


def _first_match(patterns: List[str], text: str, flags: int = re.I) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None


def _parse_md_pair(text: str) -> tuple[Optional[float], Optional[float]]:
    patterns = [
        r"MD\s*IN\s*[:=]?\s*([\d.,]+)\s*(?:M|FT|FT\.|M\.)?\s*(?:MD\s*OUT|TO)\s*[:=]?\s*([\d.,]+)",
        r"FROM\s*([\d.,]+)\s*(?:M|FT)?\s*TO\s*([\d.,]+)",
        r"DEPTH\s*IN\s*[:=]?\s*([\d.,]+).*?DEPTH\s*OUT\s*[:=]?\s*([\d.,]+)",
        r"([\d.,]+)\s*[-–]\s*([\d.,]+)\s*(?:M|FT|MD)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I | re.S)
        if m:
            try:
                a = float(m.group(1).replace(",", ""))
                b = float(m.group(2).replace(",", ""))
                return (min(a, b), max(a, b))
            except ValueError:
                continue

    singles = re.findall(
        r"(?:MD\s*(?:IN|OUT|FROM|TO)?|DEPTH)\s*[:=]?\s*([\d.,]+)\s*(?:M|FT|MD)?",
        text,
        re.I,
    )
    nums = []
    for s in singles[:4]:
        try:
            nums.append(float(s.replace(",", "")))
        except ValueError:
            pass
    if len(nums) >= 2:
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    if len(nums) == 1:
        return nums[0], None
    return None, None


def _classify_drilling_system(text_upper: str) -> str:
    if re.search(r"\bRSS\b|ROTARY\s*STEER|ROTARY\s*STEERING|AUTO\s*TRACK", text_upper):
        return "RSS"
    if re.search(r"\bMOTOR\b|MUD\s*MOTOR|PDM\b", text_upper):
        return "Motor"
    if re.search(r"\bROTARY\b|ROTARY\s*BHA|TOP\s*DRIVE", text_upper):
        return "Rotary"
    return "Unknown"


def _classify_rss_type(text_upper: str, drilling_system: str) -> str:
    if drilling_system != "RSS":
        return "Unknown RSS Type"
    if re.search(r"PUSH|PAD\s*STEER|RIB\s*STEER|LUCIDA|AUTOTRACK\s*PUSH", text_upper):
        return "Push-the-bit"
    if re.search(r"POINT|BIT\s*TILT|TILT\s*UNIT|GEO\s*PILOT", text_upper):
        return "Point-the-bit"
    return "Unknown RSS Type"


def extract_bha_from_text(text: str, source_file: str = "", run_index: int = 1) -> Dict[str, Any]:
    text_upper = (text or "").upper()
    md_in, md_out = _parse_md_pair(text or "")

    hole = _first_match(
        [
            r"HOLE\s*SIZE\s*[:=]?\s*([\d./\s]+(?:IN|MM|INCH)?)",
            r"BIT\s*SIZE\s*[:=]?\s*([\d./\s]+(?:IN|MM)?)",
            r"([\d]+\s*/\s*[\d]+\s*IN(?:CH)?)",
        ],
        text or "",
    )
    bit_type = _first_match(
        [
            r"BIT\s*TYPE\s*[:=]?\s*([A-Z0-9\-\s]+)",
            r"(PDC|ROLLER\s*CONE|TRI[- ]?CONE|HYBRID)\s*BIT",
        ],
        text_upper,
    )
    config = _first_match(
        [
            r"(?:BHA|ASSEMBLY)\s*(?:CONFIG|DESCRIPTION)\s*[:=]?\s*(.{10,120})",
            r"(STABILIZER|MWD|LWD|NMDC|MOTOR|RSS).{0,80}(?:STABILIZER|BIT)",
        ],
        text or "",
    )

    system = _classify_drilling_system(text_upper)
    return {
        "BHA_Run": f"Run {run_index}",
        "Source_File": source_file or f"BHA_{run_index}",
        "MD_In": md_in,
        "MD_Out": md_out,
        "Drilling_System": system,
        "RSS_Type": _classify_rss_type(text_upper, system),
        "Hole_Size": hole or "Unknown",
        "Bit_Type": bit_type or "Unknown",
        "BHA_Config": (config or "Unknown")[:200],
    }


def extract_bha_from_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    path = Path(pdf_path)
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return extract_bha_from_text(text, source_file=path.name, run_index=1)


def extract_bha_runs_from_uploads(file_paths: List[tuple[str, str]]) -> pd.DataFrame:
    """
    Extract one row per PDF.
    file_paths: list of (saved_path, original_filename)
    """
    rows: List[Dict[str, Any]] = []
    for idx, (path, name) in enumerate(file_paths, start=1):
        row = extract_bha_from_pdf(path)
        row["BHA_Run"] = f"Run {idx}"
        row["Source_File"] = name
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=BHA_COLUMNS)
    return pd.DataFrame(rows)[BHA_COLUMNS]


def normalize_bha_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns and numeric MD fields."""
    if df is None or df.empty:
        return pd.DataFrame(columns=BHA_COLUMNS)
    out = df.copy()
    for col in BHA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out["MD_In"] = pd.to_numeric(out["MD_In"], errors="coerce")
    out["MD_Out"] = pd.to_numeric(out["MD_Out"], errors="coerce")
    for col in ("Drilling_System", "RSS_Type", "Hole_Size", "Bit_Type", "BHA_Config"):
        out[col] = out[col].fillna("Unknown").replace("", "Unknown")
    out["BHA"] = out.get("BHA", out["BHA_Run"]).fillna(out["BHA_Run"])
    return out
