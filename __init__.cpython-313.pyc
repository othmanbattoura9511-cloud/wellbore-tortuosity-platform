from dataclasses import dataclass


@dataclass
class WellMetadata:
    well_name: str
    operator: str | None = None
    field: str | None = None
    unit_system: str = "metric"


@dataclass
class SurveyInterval:
    name: str
    md_start: float
    md_end: float
    survey_type: str
    drilling_system: str | None = None
    hole_size: float | None = None
    bha_id: str | None = None
