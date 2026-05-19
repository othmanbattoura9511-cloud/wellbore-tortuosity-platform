"""Project-root paths for local runs and Streamlit Cloud."""

from __future__ import annotations

import tempfile
from pathlib import Path

# Repository root (directory containing app.py and paths.py).
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_SURVEY_CSV = SAMPLE_DIR / "synthetic_survey.csv"
SAMPLE_BHA_PDF = SAMPLE_DIR / "sample_bha.pdf"

_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


def is_allowed_read_path(path: Path) -> bool:
    """True for project files or system temp upload files (never arbitrary absolute paths)."""
    resolved = path.resolve()
    if not resolved.is_file():
        return False
    try:
        resolved.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        pass
    try:
        resolved.relative_to(_TEMP_ROOT)
        return True
    except ValueError:
        return False


def resolve_project_path(path: str | Path) -> Path:
    """Resolve paths relative to the project root only."""
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if is_allowed_read_path(resolved):
            return resolved
        raise FileNotFoundError(
            f"Path is outside the project and not a valid upload temp file: {candidate}. "
            f"Project root: {PROJECT_ROOT}"
        )
    resolved = (PROJECT_ROOT / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved
