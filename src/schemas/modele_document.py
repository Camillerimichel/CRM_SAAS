from pydantic import BaseModel


class ModeleDocumentSchema(BaseModel):
    id: int
    nom: str
    canal: str
    objet: str | None = None
    contenu: str
    actif: int | None = 1

    class Config:
        from_attributes = True


class ModeleDocumentCreateSchema(BaseModel):
    nom: str
    canal: str
    objet: str | None = None
    contenu: str
    actif: int | None = 1

