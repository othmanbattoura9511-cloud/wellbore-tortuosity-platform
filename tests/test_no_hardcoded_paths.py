"""Ensure repository sources do not embed machine-specific paths."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("C:\\Users", "C:/Users", "\\Downloads\\", "/Downloads/", "othma\\Downloads")


@pytest.mark.parametrize("relative", [
    "app.py",
    "loaders.py",
    "paths.py",
    "upload_utils.py",
    "analytics.py",
    "sections.py",
])
def test_source_files_have_no_windows_paths(relative: str):
    path = PROJECT_ROOT / relative
    text = path.read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in text, f"{relative} contains forbidden path fragment: {token}"
