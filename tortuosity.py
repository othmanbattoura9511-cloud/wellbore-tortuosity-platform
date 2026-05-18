# Wellbore Tortuosity Analytics Platform

Engineering-oriented Python platform for internship/thesis work on wellbore trajectory tortuosity and its impact on Torque, Drag, Buckling and Contact Forces, designed to support WellScan T&D&B workflows.

## Core workflow
1. Understand and frame trajectory tortuosity concepts.
2. Load and clean directional survey data.
3. Characterize tortuosity using DLS, curvature, survey density and pattern indicators.
4. Compare smooth vs tortuous trajectories using a mock T&D&B impact engine or exported WellScan results.
5. Generate technical reports, executive summaries and dashboard-ready visualizations.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/run_pipeline.py --config configs/default.yaml
streamlit run src/wellbore_tortuosity/dashboard/app.py
```

## Folder structure
```text
configs/                 YAML configuration files
data/sample/             Synthetic survey datasets
notebooks/               Jupyter notebooks by phase
src/wellbore_tortuosity/ Main Python package
tests/                   Unit tests
reports/                 Generated reports
presentations/           PowerPoint-ready outputs
```

## Engineering focus
This is not only a data analysis tool. The platform is structured around drilling engineering interpretation: survey quality, trajectory behavior, BHA influence, drilling system response, mechanical impact and practical recommendations.
