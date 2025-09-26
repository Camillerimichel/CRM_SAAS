from datetime import datetime
from pydantic import BaseModel, Field, AliasChoices, field_validator

from .validators import format_date  # réutilisable


# Schéma pour lecture
class AffaireSchema(BaseModel):
    id: int
    id_personne: int
    ref: str | None = None
    date_debut: str | None = None
    date_cle: str | None = None
    SRRI: int | None = None
    frais_courtier: float | None = None

    class Config:
        from_attributes = True

    @field_validator("date_debut", "date_cle", mode="before")
    @classmethod
    def parse_dates(cls, v):
        return format_date(v)


# Schéma pour création
class AffaireCreateSchema(BaseModel):
    id_personne: int
    ref: str
    srri: int | None = Field(default=None, validation_alias=AliasChoices("srri", "SRRI"))
    date_debut: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("date_debut", "dateDebut"),
    )
    date_cle: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("date_cle", "dateCle"),
    )
    frais_courtier: float | None = Field(
        default=None,
        validation_alias=AliasChoices("frais_courtier", "Frais_courtier"),
    )

    class Config:
        populate_by_name = True

    @field_validator("date_debut", "date_cle", mode="before")
    @classmethod
    def parse_optional_dates(cls, v):
        if v in ("", None):
            return None
        return v

    @field_validator("frais_courtier", mode="before")
    @classmethod
    def parse_decimal(cls, v):
        if v in ("", None):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            normalized = str(v).replace(" ", "").replace(",", ".")
            return float(normalized)
        except Exception:
            return v
