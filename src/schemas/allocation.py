from pydantic import BaseModel, field_validator
from typing import Optional
from .validators import format_date


class AllocationBase(BaseModel):
    date: Optional[str] = None
    valo: Optional[float] = None
    mouvement: Optional[float] = None
    sicav: Optional[float] = None
    perf_sicav_hebdo: Optional[float] = None
    perf_sicav_52: Optional[float] = None
    volat: Optional[float] = None
    annee: Optional[int] = None
    nom: Optional[str] = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        return format_date(v)

    @field_validator(
        "valo",
        "mouvement",
        "sicav",
        "perf_sicav_hebdo",
        "perf_sicav_52",
        "volat",
        mode="before",
    )
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


class AllocationCreateSchema(AllocationBase):
    date: str  # obligatoire en création


class AllocationSchema(AllocationBase):
    id: int

    class Config:
        from_attributes = True
