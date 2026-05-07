from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, field_validator
from typing import Any


class SocieteIdentifiantFournisseurBase(BaseModel):
    societe_id: int
    identifiant_externe: str
    actif: int = 1

    @field_validator("identifiant_externe", mode="before")
    @classmethod
    def _strip_id(cls, v: Any) -> str:
        return str(v).strip()


class SocieteIdentifiantFournisseurCreate(SocieteIdentifiantFournisseurBase):
    pass


class SocieteIdentifiantFournisseurUpdate(BaseModel):
    societe_id: int | None = None
    identifiant_externe: str | None = None
    actif: int | None = None

    @field_validator("identifiant_externe", mode="before")
    @classmethod
    def _strip_id(cls, v: Any) -> str | None:
        return str(v).strip() if v is not None else None


class SocieteIdentifiantFournisseurSchema(SocieteIdentifiantFournisseurBase):
    id: int
    date_creation: datetime

    model_config = {"from_attributes": True}
