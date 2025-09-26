from pydantic import BaseModel


class TypeEvenementSchema(BaseModel):
    id: int
    libelle: str
    categorie: str

    class Config:
        from_attributes = True


class TypeEvenementCreateSchema(BaseModel):
    libelle: str
    categorie: str

