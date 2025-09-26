from pydantic import BaseModel, field_validator
from datetime import datetime
from .validators import format_date  # utilitaire commun


class HistoriqueSupportSchema(BaseModel):
    id: int
    modif_quand: str | None = None
    source: str | None = None
    id_source: int | None = None
    date: str | None = None
    id_support: str | None = None
    nbuc: float | None = None
    vl: float | None = None
    prmp: float | None = None
    valo: float | None = None

    class Config:
        from_attributes = True

    @field_validator("modif_quand", "date", mode="before")
    @classmethod
    def parse_dates(cls, v):
        return format_date(v)

    @field_validator("nbuc", "vl", "prmp", "valo", mode="before")
    @classmethod
    def parse_float_fields(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip().replace(",", ".")
            if v.startswith("00."):
                v = v[1:]
        try:
            return float(v)
        except Exception:
            return None


# Création (API → DB)
class HistoriqueSupportCreateSchema(BaseModel):
    modif_quand: datetime | None = None
    source: str | None = None
    id_source: int | None = None
    date: datetime | None = None
    id_support: str | None = None
    nbuc: float | None = None
    vl: float | None = None
    prmp: float | None = None
    valo: float | None = None
