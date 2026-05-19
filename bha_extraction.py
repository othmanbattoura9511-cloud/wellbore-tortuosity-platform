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

# RSS tools / assemblies (checked before generic motor keywords).
RSS_COMPONENT_PATTERNS = (
    r"\bRSS\b",
    r"POWER\s*DRIVE",
    r"AUTOTRAK",
    r"AUTO\s*TRAK",
    r"GEOPILOT",
    r"GEO\s*PILOT",
    r"ROTARY\s*STEERABLE",
    r"ROTARY\s*STEER(?:ING)?",
)

# Motor drilling systems (Steerable Motor before ambiguous "steerable" RSS wording).
MOTOR_COMPONENT_PATTERNS = (
    r"STEERABLE\s+MOTOR",
    r"MUD\s+MOTOR",
    r"\bPDM\b",
    r"(?<![A-Z])MOTOR\b",
)

PUSH_RSS_PATTERNS = (
    r"PUSH\s*THE\s*BIT",
    r"PUSH-THE-BIT",
    r"PUSH\s*BIT",
    r"\bPUSH\b",
    r"PAD\s*STEER",
    r"RIB\s*STEER",
)

POINT_RSS_PATTERNS = (
    r"POINT\s*THE\s*BIT",
    r"POINT-THE-BIT",
    r"POINT\s*BIT",
    r"\bPOINT\b",
    r"BIT\s*TILT",
    r"TILT\s*UNIT",
)


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


def _extract_component_blob(text: str, config: Optional[str] = None) -> str:
    """Build searchable component list text from PDF body and parsed config snippet."""
    parts: list[str] = []
    if config:
        parts.append(str(config))

    section = re.search(
        r"(?:COMPONENT|COMPONENTS|BHA\s*COMPONENT|TOOL\s*STRING|ASSEMBLY)\s*(?:LIST|DESCRIPTION)?\s*[:\-]?\s*(.+?)(?:\n\s*\n|MD\s*IN|DEPTH|RUN\s*SUMMARY|$)",
        text or "",
        re.I | re.S,
    )
    if section:
        parts.append(section.group(1))

    component_lines = []
    for line in (text or "").splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(
            r"MOTOR|RSS|MWD|LWD|STABIL|NMDC|BIT|PDM|AGITATOR|POWER\s*DRIVE|GEOPILOT|AUTOTRAK|STEER",
            line_stripped,
            re.I,
        ):
            component_lines.append(line_stripped)
    if component_lines:
        parts.append("\n".join(component_lines))

    parts.append(text or "")
    return "\n".join(parts).upper()


def _matches_any(patterns: tuple[str, ...], blob: str) -> bool:
    return any(re.search(pat, blob) for pat in patterns)


def _classify_drilling_system(component_blob: str) -> str:
    """
    Classify drilling system from BHA component list / report text.

    Motor: Steerable Motor, Mud Motor, Motor.
    RSS: RSS, PowerDrive, AutoTrak, GeoPilot, Rotary Steerable.
    """
    blob = (component_blob or "").upper()
    if not blob.strip():
        return "Unknown"

    rss_strong = (
        r"\bRSS\b",
        r"POWER\s*DRIVE",
        r"AUTOTRAK",
        r"AUTO\s*TRAK",
        r"GEOPILOT",
        r"GEO\s*PILOT",
    )
    motor_strong = (
        r"STEERABLE\s+MOTOR",
        r"MUD\s+MOTOR",
        r"\bPDM\b",
    )

    if _matches_any(motor_strong, blob):
        return "Motor"
    if _matches_any(rss_strong, blob):
        return "RSS"
    if _matches_any(
        (r"ROTARY\s*STEERABLE", r"ROTARY\s*STEER(?:ING)?"),
        blob,
    ):
        return "RSS"
    if _matches_any((r"(?<![A-Z])MOTOR\b",), blob):
        return "Motor"
    if re.search(r"\bROTARY\b|TOP\s*DRIVE", blob):
        return "Rotary"
    return "Unknown"


def _classify_rss_type(component_blob: str, drilling_system: str) -> str:
    """RSS steering mode from component list / text (Push vs Point)."""
    if drilling_system != "RSS":
        return "Unknown RSS Type"

    blob = (component_blob or "").upper()
    point = _matches_any(POINT_RSS_PATTERNS, blob)
    push = _matches_any(PUSH_RSS_PATTERNS, blob)

    if point and not push:
        return "Point-the-bit"
    if push and not point:
        return "Push-the-bit"
    if point and push:
        point_pos = min(
            (m.start() for pat in POINT_RSS_PATTERNS for m in [re.search(pat, blob)] if m),
            default=10**9,
        )
        push_pos = min(
            (m.start() for pat in PUSH_RSS_PATTERNS for m in [re.search(pat, blob)] if m),
            default=10**9,
        )
        return "Point-the-bit" if point_pos <= push_pos else "Push-the-bit"
    return "Unknown RSS Type"


_SIZE_NUM = r"(\d+(?:\s+\d+/\d+|\.\d+)?)"
_SIZE_UNIT = r"(?:\s*(?:IN(?:CH(?:ES)?)?|MM|\"))?"

PDF_HOLE_SIZE_PATTERNS = [
    r"HOLE\s*SIZE\s*[:=]?\s*" + _SIZE_NUM + _SIZE_UNIT,
    r"HOLE\s*DIAMETER\s*[:=]?\s*" + _SIZE_NUM + _SIZE_UNIT,
    r"WELLBORE\s*SIZE\s*[:=]?\s*" + _SIZE_NUM + _SIZE_UNIT,
    r"BIT\s*SIZE\s*[:=]?\s*" + _SIZE_NUM + _SIZE_UNIT,
    r"SIZE\s*[:=]?\s*" + _SIZE_NUM + r"\s*IN(?:CH)?",
]

FILENAME_HOLE_SIZE_PATTERNS = [
    r"(?:HOLE[_\s\-]?SIZE|HOLESIZE|HS)[_\s\-]*(\d+(?:[._/-]\d+)?)",
    r"(?:^|[_\-\s])(\d+\.\d{1,2})(?:in|inch|inches)?(?:[_\-\s.]|$)",
    r"(?:^|[_\-\s])(\d+)[_\-\.](\d+)[_\-\.](\d+)(?:in|inch)?(?:[_\-\s.]|$)",
    r"(?:^|[_\-\s])(\d+(?:\s*/\s*\d+)?)\s*IN(?:CH(?:ES)?)?(?:[_\-\s.]|$)",
]

_BIT_SIZE_ON_LINE = re.compile(
    r"(\d+(?:\s+\d+/\d+|\.\d+)?)\s*(?:IN(?:CH(?:ES)?)?|MM|\"|IN\.)?",
    re.I,
)


def _format_size_value(raw: str) -> str:
    """Normalize a captured size token to a display string."""
    text = re.sub(r"\s+", " ", str(raw).strip())
    if not text:
        return ""
    # 8_1_2 or 8-1-2 style from filenames
    parts = re.fullmatch(r"(\d+)[._/-](\d+)[._/-](\d+)", text)
    if parts:
        return f"{parts.group(1)} {parts.group(2)}/{parts.group(3)} in"
    if re.fullmatch(r"\d+\.\d+", text):
        return f"{text} in"
    if re.search(r"/", text):
        return f"{text} in" if not re.search(r"\bIN\b", text, re.I) else text
    if re.search(r"\d", text) and not re.search(r"\b(IN|MM|INCH)\b", text, re.I):
        return f"{text} in"
    return text


def _normalize_hole_size(value: Optional[str]) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    if re.fullmatch(r"(in|inch|inches|mm|cm|m|ft|feet|\"|)?", text, re.I):
        return "Unknown"
    if not re.search(r"\d", text):
        return "Unknown"
    formatted = _format_size_value(text)
    return formatted[:80] if formatted else "Unknown"


def _extract_hole_size_from_pdf_text(text: str) -> Optional[str]:
    for pat in PDF_HOLE_SIZE_PATTERNS:
        m = re.search(pat, text or "", re.I)
        if m:
            return _format_size_value(m.group(1))
    return None


def _extract_hole_size_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    stem = Path(filename).stem
    for pat in FILENAME_HOLE_SIZE_PATTERNS:
        m = re.search(pat, stem, re.I)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 3:
            return _format_size_value(f"{m.group(1)} {m.group(2)}/{m.group(3)}")
        return _format_size_value(m.group(1))
    return None


def _extract_hole_size_from_bit_row(
    text: str,
    tables: Optional[List[List[List[Optional[str]]]]] = None,
) -> Optional[str]:
    """Parse bit-size from BHA tally lines and PDF tables."""
    for line in (text or "").splitlines():
        if not re.search(r"\bBIT\b", line, re.I):
            continue
        best: tuple[float, str] | None = None
        for m in _BIT_SIZE_ON_LINE.finditer(line):
            raw = m.group(1)
            score = m.start() / 1000.0
            if re.search(r"/|\.", raw):
                score += 10.0
            lead = re.match(r"\d+", raw)
            if lead and float(lead.group()) >= 4:
                score += 5.0
            elif lead and float(lead.group()) <= 3 and not re.search(r"/|\.", raw):
                score -= 5.0
            if best is None or score > best[0]:
                best = (score, raw)
        if best:
            return _format_size_value(best[1])

    if not tables:
        return None

    for table in tables:
        if not table or len(table) < 2:
            continue
        header_idx = None
        headers: list[str] = []
        for i, row in enumerate(table[:6]):
            cells = [str(c or "").strip() for c in row]
            if not any(cells):
                continue
            upper = [c.upper() for c in cells]
            if any("BIT" in c or "DESCRIPTION" in c or "COMPONENT" in c for c in upper):
                header_idx = i
                headers = upper
                break
        if header_idx is None:
            continue

        size_col = None
        desc_col = None
        for j, h in enumerate(headers):
            if re.search(r"BIT\s*SIZE|^SIZE$|OD|DIAM", h):
                size_col = j
            if re.search(r"DESCRIPTION|COMPONENT|ITEM|TOOL", h):
                desc_col = j

        for row in table[header_idx + 1 :]:
            cells = [str(c or "").strip() for c in row]
            if not any(cells):
                continue
            desc = cells[desc_col] if desc_col is not None and desc_col < len(cells) else ""
            row_text = " ".join(cells)
            is_bit = bool(re.search(r"\bBIT\b", desc or row_text, re.I))
            if not is_bit:
                continue
            if size_col is not None and size_col < len(cells):
                val = cells[size_col]
                if val and re.search(r"\d", val):
                    return _format_size_value(val)
            m = re.search(r"(\d+(?:\s+\d+/\d+|\.\d+)?)\s*(?:IN|MM|IN\.)?", row_text, re.I)
            if m:
                return _format_size_value(m.group(1))
    return None


def _resolve_hole_size(
    *,
    pdf_text: Optional[str] = None,
    bit_row: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Merge hole size from PDF labels, bit tally row, then filename."""
    for candidate in (pdf_text, bit_row, filename):
        normalized = _normalize_hole_size(candidate)
        if normalized != "Unknown":
            return normalized
    return "Unknown"


def extract_bha_from_text(
    text: str,
    source_file: str = "",
    run_index: int = 1,
    tables: Optional[List[List[List[Optional[str]]]]] = None,
) -> Dict[str, Any]:
    text_upper = (text or "").upper()

    hole_pdf = _extract_hole_size_from_pdf_text(text or "")
    hole_bit = _extract_hole_size_from_bit_row(text or "", tables)
    hole_file = _extract_hole_size_from_filename(source_file)
    hole = _resolve_hole_size(pdf_text=hole_pdf, bit_row=hole_bit, filename=hole_file)
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

    md_in, md_out = _parse_md_pair(text or "")
    component_blob = _extract_component_blob(text, config)
    system = _classify_drilling_system(component_blob)

    return {
        "BHA_Run": f"Run {run_index}",
        "Source_File": source_file or f"BHA_{run_index}",
        "MD_In": md_in,
        "MD_Out": md_out,
        "Drilling_System": system,
        "RSS_Type": _classify_rss_type(component_blob, system),
        "Hole_Size": hole,
        "Bit_Type": bit_type or "Unknown",
        "BHA_Config": (config or "Unknown")[:200],
    }


def extract_bha_from_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    path = Path(pdf_path)
    text_parts: list[str] = []
    tables: list[list[list[Optional[str]]]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
            for table in page.extract_tables() or []:
                if table:
                    tables.append(table)
    text = "\n".join(text_parts)
    return extract_bha_from_text(
        text,
        source_file=path.name,
        run_index=1,
        tables=tables,
    )


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
        if _normalize_hole_size(row.get("Hole_Size")) == "Unknown":
            from_name = _extract_hole_size_from_filename(name)
            if from_name:
                row["Hole_Size"] = _normalize_hole_size(from_name)
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
    for col in ("Drilling_System", "RSS_Type", "Bit_Type", "BHA_Config"):
        out[col] = out[col].fillna("Unknown").replace("", "Unknown")
    out["Hole_Size"] = out["Hole_Size"].apply(_normalize_hole_size)
    out["BHA"] = out.get("BHA", out["BHA_Run"]).fillna(out["BHA_Run"])
    return out
