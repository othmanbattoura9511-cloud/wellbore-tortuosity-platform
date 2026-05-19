"""Generate synthetic survey sample data under data/sample/."""

import numpy as np
import pandas as pd

from paths import PROJECT_ROOT, SAMPLE_DIR, SAMPLE_SURVEY_CSV


def main() -> None:
    np.random.seed(42)
    md = np.r_[np.arange(0, 600, 30), np.arange(600, 2500, 30), np.arange(2500, 6200, 5)]
    inc = np.piecewise(
        md,
        [md < 600, (md >= 600) & (md < 3000), md >= 3000],
        [lambda x: x * 0.005, lambda x: 3 + (x - 600) * 0.035, lambda x: 88 + 0.5 * np.sin(x / 80)],
    ).astype(float)
    azi = 120 + 8 * np.sin(md / 600) + np.random.normal(0, 0.2, len(md))
    inc += np.where(md > 3500, 0.3 * np.sin(md / 12), 0) + np.random.normal(0, 0.05, len(md))

    df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": azi})
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_SURVEY_CSV, index=False)
    print(f"Wrote {SAMPLE_SURVEY_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
