from pathlib import Path

from paths import PROJECT_ROOT, SAMPLE_SURVEY_CSV, resolve_project_path


def test_project_root_is_repo_directory():
    assert (PROJECT_ROOT / "app.py").is_file()


def test_resolve_relative_from_project_root():
    resolved = resolve_project_path("data/sample/synthetic_survey.csv")
    assert resolved == SAMPLE_SURVEY_CSV.resolve()


def test_resolve_absolute_path_unchanged(tmp_path: Path):
    target = tmp_path / "survey.csv"
    target.write_text("MD,Inclination,Azimuth\n1,2,3\n", encoding="utf-8")
    assert resolve_project_path(target) == target.resolve()
