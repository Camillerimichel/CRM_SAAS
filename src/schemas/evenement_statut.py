from pydantic import BaseModel, field_validator
from datetime import datetime
from .validators import format_date


class EvenementStatutSchema(BaseModel):
    id: int
    evenement_id: int
    statut_id: int
    date_statut: str
    utilisateur_responsable: str | None = None
    commentaire: str | None = None

    class Config:
        from_attributes = True

    @field_validator("date_statut", mode="before")
    @classmethod
    def parse_date(cls, v):
        return format_date(v)


class EvenementStatutCreateSchema(BaseModel):
    statut_id: int
    date_statut: datetime | None = None
    utilisateur_responsable: str | None = None
    commentaire: str | None = None

