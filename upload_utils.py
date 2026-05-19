"""Streamlit upload helpers — always use in-memory bytes + system temp files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st


def save_uploaded_file(uploaded_file) -> Path:
    """
    Persist a Streamlit UploadedFile to a temp file.

    Never uses uploaded_file.name as a filesystem path (on Windows it may be a full path).
    """
    original_name = Path(str(getattr(uploaded_file, "name", "upload.dat"))).name
    suffix = Path(original_name).suffix or ".dat"
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.getbuffer().tobytes()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)


def uploaded_file_key(uploaded_file) -> str:
    """Stable session key from basename + size (no local directories)."""
    name = Path(str(getattr(uploaded_file, "name", "upload"))).name
    size = getattr(uploaded_file, "size", None)
    return f"{name}:{size if size is not None else 'na'}"
