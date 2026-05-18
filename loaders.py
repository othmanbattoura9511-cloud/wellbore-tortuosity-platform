from pathlib import Path
import pandas as pd


class SurveyDataLoader:
    def load(self, path):
        path = Path(path)

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raw = pd.read_excel(path, header=None)

            header_row = None
            for i in range(len(raw)):
                row_values = [str(v).lower() for v in raw.iloc[i].tolist()]
                if any("measured" in v and "depth" in v for v in row_values) or "md" in row_values:
                    header_row = i
                    break

            if header_row is None:
                raise ValueError("Could not find survey header row in Excel file.")

            df = pd.read_excel(path, header=header_row)

        df.columns = [str(c).strip() for c in df.columns]

        rename_map = {
            "Measured Depth": "MD",
            "Measured\nDepth": "MD",
            "MD": "MD",
            "Definitive Inc": "Inclination",
            "Inclination": "Inclination",
            "Definitive Azi": "Azimuth",
            "Azimuth": "Azimuth",
            "Dogleg\nRate": "DLS",
            "DLS": "DLS",
            "Vertical Depth": "TVD",
            "TVD": "TVD",
        }

        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        required = ["MD", "Inclination", "Azimuth"]
        df = df.dropna(subset=[c for c in required if c in df.columns])
        df["MD"] = pd.to_numeric(df["MD"], errors="coerce")
        df = df[df["MD"] > 0]
        return df