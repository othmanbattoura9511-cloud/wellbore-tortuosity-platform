import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
md = np.r_[np.arange(0, 600, 30), np.arange(600, 2500, 30), np.arange(2500, 6200, 5)]
inc = np.piecewise(md, [md < 600, (md >= 600) & (md < 3000), md >= 3000], [lambda x: x*0.005, lambda x: 3 + (x-600)*0.035, lambda x: 88 + 0.5*np.sin(x/80)])
azi = 120 + 8*np.sin(md/600) + np.random.normal(0, 0.2, len(md))
inc += np.where(md > 3500, 0.3*np.sin(md/12), 0) + np.random.normal(0, 0.05, len(md))

df = pd.DataFrame({"MD": md, "Inclination": inc, "Azimuth": azi})
Path("data/sample").mkdir(parents=True, exist_ok=True)
df.to_csv("data/sample/synthetic_survey.csv", index=False)
print("Wrote data/sample/synthetic_survey.csv")
