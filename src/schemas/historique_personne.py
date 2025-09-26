from pydantic import BaseModel, field_validator
from datetime import datetime
from .validators import format_date   # utilitaire commun

class HistoriquePersonneSchema(BaseModel):
    id: int
    date: str | None = None
    valo: float | None = None
    mouvement: float | None = None
    volat: float | None = None
    annee: int | None = None

    class Config:
        from_attributes = True

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        return format_date(v)

    @field_validator("valo", "mouvement", "volat", mode="before")
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
class HistoriquePersonneCreateSchema(BaseModel):
    date: datetime | None = None
    valo: float | None = None
    mouvement: float | None = None
    volat: float | None = None
    annee: int | None = None
