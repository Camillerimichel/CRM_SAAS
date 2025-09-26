from pydantic import BaseModel


class StatutEvenementSchema(BaseModel):
    id: int
    libelle: str

    class Config:
        from_attributes = True


class StatutEvenementCreateSchema(BaseModel):
    libelle: str

