from pydantic import BaseModel, field_validator
from datetime import datetime
from .validators import format_date


class EvenementEnvoiSchema(BaseModel):
    id: int
    evenement_id: int
    canal: str
    destinataire: str
    objet: str | None = None
    contenu: str | None = None
    date_envoi: str
    statut: str | None = None
    modele_id: int | None = None

    class Config:
        from_attributes = True

    @field_validator("date_envoi", mode="before")
    @classmethod
    def parse_date(cls, v):
        return format_date(v)


class EvenementEnvoiCreateSchema(BaseModel):
    canal: str
    destinataire: str
    objet: str | None = None
    contenu: str | None = None
    date_envoi: datetime | None = None
    statut: str | None = None
    modele_id: int | None = None
    placeholders: dict[str, str] | None = None
