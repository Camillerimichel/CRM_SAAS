from pydantic import BaseModel

# Schéma pour la lecture (DB → API → JSON)
class ClientSchema(BaseModel):
    id: int
    nom: str | None = None
    prenom: str | None = None
    SRRI: int | None = None
    telephone: str | None = None
    adresse_postale: str | None = None
    email: str | None = None
    commercial_id: int | None = None
    total_valo: float | None = None
    perf_52_sem: float | None = None
    volatilite: float | None = None

    class Config:
        from_attributes = True  # indispensable avec SQLAlchemy


# Schéma pour la création (API → DB)
class ClientCreateSchema(BaseModel):
    nom: str
    prenom: str
    SRRI: int | None = None
    telephone: str | None = None
    adresse_postale: str | None = None
    email: str | None = None
    commercial_id: int | None = None


# Schéma pour la mise à jour (API → DB)
class ClientUpdateSchema(BaseModel):
    nom: str | None = None
    prenom: str | None = None
    SRRI: int | None = None
    telephone: str | None = None
    adresse_postale: str | None = None
    email: str | None = None
    commercial_id: int | None = None
