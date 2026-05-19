"""Project-root paths for local runs and Streamlit Cloud."""

from __future__ import annotations

from pathlib import Path

# Repository root (directory containing app.py).
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_SURVEY_CSV = SAMPLE_DIR / "synthetic_survey.csv"
SAMPLE_BHA_PDF = SAMPLE_DIR / "sample_bha.pdf"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the project root; leave absolute paths unchanged."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()
