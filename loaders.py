import pandas as pd
from pathlib import Path

from column_mapping import (
    REQUIRED_SURVEY_COLUMNS,
    column_match_key,
    missing_columns_message,
    normalize_column_name,
    standardize_survey_columns,
)
from paths import resolve_project_path


class SurveyDataLoader:
    def _resolve_path(self, path) -> Path:
        return resolve_project_path(path)

    def load_raw(self, path) -> pd.DataFrame:
        """Load survey file without requiring standard column names."""
        path = self._resolve_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Survey file not found: {path}")

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        raw = pd.read_excel(path, header=None)
        header_row = None
        for i in range(len(raw)):
            row_values = [column_match_key(v) for v in raw.iloc[i].tolist()]
            if any("measured" in v and "depth" in v for v in row_values) or any(
                v in {"md", "depth"} or v.startswith("md") for v in row_values
            ):
                header_row = i
                break

        if header_row is None:
            raise ValueError("Could not find survey header row in Excel file.")

        return pd.read_excel(path, header=header_row)

    def load(self, path, manual_map=None) -> pd.DataFrame:
        df = self.load_raw(path)
        df, missing = standardize_survey_columns(df, manual_map=manual_map)
        if missing:
            raise ValueError(missing_columns_message(missing, df.columns))

        df = df.dropna(subset=list(REQUIRED_SURVEY_COLUMNS))
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")
        for col in ("Inclination", "Azimuth", "DLS", "TVD"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[df["MD"] > 0]
