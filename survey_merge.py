"""Merge multiple survey files for one well."""

from __future__ import annotations

import pandas as pd

from column_mapping import analyze_survey_columns


def merge_survey_files(frames: list[pd.DataFrame], source_names: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    parts = []
    for df, name in zip(frames, source_names):
        work = df.copy()
        work["Survey_File"] = name
        parts.append(work)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values("MD").drop_duplicates(subset=["MD"], keep="last")
    return merged.reset_index(drop=True)


def load_and_merge_surveys(loader, file_paths: list[tuple[str, str]]) -> pd.DataFrame:
    """Load raw files from (path, display_name) pairs and merge."""
    frames, names = [], []
    for path, name in file_paths:
        raw = loader.load_raw(path)
        result = analyze_survey_columns(raw)
        if result.missing:
            raise ValueError(f"Could not map columns in {name}: {result.missing}")
        frames.append(result.survey)
        names.append(name)
    return merge_survey_files(frames, names)
