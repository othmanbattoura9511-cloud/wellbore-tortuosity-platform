import pandas as pd
from pathlib import Path

from column_mapping import (
    REQUIRED_SURVEY_COLUMNS,
    analyze_survey_columns,
    column_match_key,
    fuzzy_score,
    missing_columns_message,
    preprocess_survey_frame,
)
from paths import is_allowed_read_path, resolve_project_path


class SurveyDataLoader:
    def _resolve_path(self, path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if is_allowed_read_path(resolved):
                return resolved
            raise FileNotFoundError(
                f"Cannot open absolute path (not in project or temp uploads): {candidate}"
            )
        return resolve_project_path(candidate)

    def _find_excel_header_row(self, raw: pd.DataFrame) -> int:
        best_row, best_score = 0, 0.0
        for i in range(min(len(raw), 40)):
            row_values = [column_match_key(v) for v in raw.iloc[i].tolist() if str(v).strip()]
            if not row_values:
                continue
            md_hit = max((fuzzy_score(v, "MD") for v in raw.iloc[i].tolist()), default=0.0)
            inc_hit = max((fuzzy_score(v, "Inclination") for v in raw.iloc[i].tolist()), default=0.0)
            azi_hit = max((fuzzy_score(v, "Azimuth") for v in raw.iloc[i].tolist()), default=0.0)
            score = md_hit + inc_hit + azi_hit
            if score > best_score:
                best_score, best_row = score, i
        if best_score < 1.5:
            raise ValueError("Could not find survey header row in Excel file.")
        return best_row

    def load_raw(self, path) -> pd.DataFrame:
        """Load survey file without requiring standard column names."""
        path = self._resolve_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Survey file not found: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raw = pd.read_excel(path, header=None)
            header_row = self._find_excel_header_row(raw)
            df = pd.read_excel(path, header=header_row)

        return preprocess_survey_frame(df)

    def load(self, path, manual_map=None) -> pd.DataFrame:
        df = self.load_raw(path)
        analysis = analyze_survey_columns(df, manual_map=manual_map)
        if analysis.missing:
            raise ValueError(analysis.message or missing_columns_message(analysis.missing, analysis.survey.columns))

        survey = analysis.survey.dropna(subset=list(REQUIRED_SURVEY_COLUMNS))
        survey["MD"] = pd.to_numeric(survey["MD"], errors="coerce")
        for col in ("Inclination", "Azimuth", "DLS", "TVD"):
            if col in survey.columns:
                survey[col] = pd.to_numeric(survey[col], errors="coerce")
        return survey[survey["MD"] > 0]
