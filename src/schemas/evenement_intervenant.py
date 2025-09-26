from pydantic import BaseModel


class EvenementIntervenantSchema(BaseModel):
    id: int
    evenement_id: int
    role: str
    nom_intervenant: str
    contact: str | None = None

    class Config:
        from_attributes = True


class EvenementIntervenantCreateSchema(BaseModel):
    role: str
    nom_intervenant: str
    contact: str | None = None

