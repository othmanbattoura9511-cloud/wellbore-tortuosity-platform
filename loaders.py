from pathlib import Path

import pandas as pd


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]

    rename_map = {
        "Measured Depth": "MD",
        "Measured depth": "MD",
        "MEASURED DEPTH": "MD",
        "MD (m)": "MD",
        "MD(m)": "MD",
        "Depth": "MD",
        "DEPTH": "MD",
        "Definitive Inc": "Inclination",
        "Definitive INC": "Inclination",
        "INC": "Inclination",
        "Inc": "Inclination",
        "Incl": "Inclination",
        "Inclination (deg)": "Inclination",
        "Inclination (°)": "Inclination",
        "Definitive Azi": "Azimuth",
        "Definitive AZI": "Azimuth",
        "AZI": "Azimuth",
        "Azi": "Azimuth",
        "Azimuth (deg)": "Azimuth",
        "Azimuth (°)": "Azimuth",
        "Dogleg Rate": "DLS",
        "Dogleg rate": "DLS",
        "DLS (deg/30m)": "DLS",
        "DLS (deg/100ft)": "DLS",
        "Vertical Depth": "TVD",
        "TVD (m)": "TVD",
    }
    df = df.rename(columns=rename_map)

    lower_to_canonical = {
        "md": "MD",
        "measured depth": "MD",
        "inclination": "Inclination",
        "inc": "Inclination",
        "azi": "Azimuth",
        "azimuth": "Azimuth",
        "dls": "DLS",
        "dogleg rate": "DLS",
        "tvd": "TVD",
        "vertical depth": "TVD",
    }
    for col in list(df.columns):
        key = _normalize_column_name(col).lower()
        if key in lower_to_canonical and col != lower_to_canonical[key]:
            df = df.rename(columns={col: lower_to_canonical[key]})

    return df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, na=False)]


class SurveyDataLoader:
    def load(self, path):
        path = Path(path)

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raw = pd.read_excel(path, header=None)

            header_row = None
            for i in range(len(raw)):
                row_values = [_normalize_column_name(v).lower() for v in raw.iloc[i].tolist()]
                if any("measured" in v and "depth" in v for v in row_values) or any(
                    v in {"md", "depth"} or v.startswith("md ") for v in row_values
                ):
                    header_row = i
                    break

            if header_row is None:
                raise ValueError("Could not find survey header row in Excel file.")

            df = pd.read_excel(path, header=header_row)

        df = _standardize_columns(df)
        required = ["MD", "Inclination", "Azimuth"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required survey columns after mapping: {missing}")

        df = df.dropna(subset=required)
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")
        for col in ("Inclination", "Azimuth", "DLS", "TVD"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["MD"] > 0]
        return df
