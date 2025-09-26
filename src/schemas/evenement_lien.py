from pydantic import BaseModel


class EvenementLienSchema(BaseModel):
    id: int
    evenement_source_id: int
    evenement_cible_id: int
    type_lien: str

    class Config:
        from_attributes = True


class EvenementLienCreateSchema(BaseModel):
    evenement_cible_id: int
    type_lien: str

