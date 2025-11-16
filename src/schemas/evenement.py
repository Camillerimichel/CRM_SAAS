from pydantic import BaseModel, field_validator
from datetime import datetime
from .validators import format_date


class EvenementSchema(BaseModel):
    id: int
    type_id: int
    client_id: int | None = None
    affaire_id: int | None = None
    support_id: int | None = None
    date_evenement: str
    statut: str
    commentaire: str | None = None
    utilisateur_responsable: str | None = None
    rh_id: int | None = None
    statut_reclamation_id: int | None = None

    class Config:
        from_attributes = True

    @field_validator("date_evenement", mode="before")
    @classmethod
    def parse_date(cls, v):
        return format_date(v)


class EvenementCreateSchema(BaseModel):
    type_id: int
    date_evenement: datetime
    client_id: int | None = None
    affaire_id: int | None = None
    support_id: int | None = None
    statut: str | None = None
    commentaire: str | None = None
    utilisateur_responsable: str | None = None
    rh_id: int | None = None
    statut_reclamation_id: int | None = None


class EvenementUpdateSchema(BaseModel):
    type_id: int | None = None
    date_evenement: datetime | None = None
    client_id: int | None = None
    affaire_id: int | None = None
    support_id: int | None = None
    statut: str | None = None
    commentaire: str | None = None
    utilisateur_responsable: str | None = None
    rh_id: int | None = None
    statut_reclamation_id: int | None = None


class TacheCreateSchema(BaseModel):
    # Crée une tâche en garantissant l'existence du type
    type_libelle: str = "tâche"
    categorie: str | None = "tache"
    date_evenement: datetime | None = None
    client_id: int | None = None
    affaire_id: int | None = None
    support_id: int | None = None
    statut: str | None = None
    commentaire: str | None = None
    utilisateur_responsable: str | None = None
    rh_id: int | None = None
    statut_reclamation_id: int | None = None
