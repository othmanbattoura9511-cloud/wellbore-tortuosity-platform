import argparse
from pathlib import Path
from wellbore_tortuosity.utils.config import load_config
from wellbore_tortuosity.io.loaders import SurveyDataLoader
from wellbore_tortuosity.io.preprocess import SurveyPreprocessor
from wellbore_tortuosity.analysis.sections import WellSectionClassifier
from wellbore_tortuosity.analysis.survey_quality import SurveyQualityAnalyzer
from wellbore_tortuosity.analysis.tortuosity import TortuosityAnalyzer
from wellbore_tortuosity.analysis.patterns import PatternRecognition
from wellbore_tortuosity.simulation.mock_engine import MockTDBEngine
from wellbore_tortuosity.reporting.report_generator import ReportGenerator
from wellbore_tortuosity.reporting.recommendations import RecommendationEngine

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--input", default="data/sample/synthetic_survey.csv")
args = parser.parse_args()

cfg = load_config(args.config)
df = SurveyDataLoader().load(args.input)
pre = SurveyPreprocessor()
df = pre.interpolate_missing(pre.clean(df))
df = WellSectionClassifier().classify(df)
tort = TortuosityAnalyzer()
df = tort.calculate_dls(df)
df["DLS"] = df.get("DLS", df["DLS_Calc"])
df = tort.add_indicators(df)
df = PatternRecognition().cluster_intervals(df)
quality = SurveyQualityAnalyzer().evaluate(df)
sim = MockTDBEngine(**cfg["simulation"]).run(df)
recs = RecommendationEngine().generate(quality)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/processed_survey.csv", index=False)
sim.to_csv("reports/mock_simulation_results.csv", index=False)
ReportGenerator().render_markdown(
    "reports/phase_report.md",
    title="Phase Report - Survey and Tortuosity Analysis",
    executive_summary="Initial survey quality, DLS and tortuosity indicators were evaluated. Results should be interpreted with survey spacing, BHA and drilling system context.",
    quality=quality,
    observations=["Survey spacing influences DLS variability.", "High-frequency DLS changes may indicate tortuosity or noise.", "Mechanical impact should be checked with WellScan or mock sensitivity analysis."],
    recommendations=recs,
)
print("Pipeline completed. Outputs written to reports/.")
