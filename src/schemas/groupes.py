from pydantic import BaseModel
from typing import Optional, Literal
from datetime import date


class GroupeDetailSchema(BaseModel):
    id: int
    type_groupe: Optional[str] = None
    nom: str
    date_creation: Optional[date] = None
    date_fin: Optional[date] = None
    responsable_id: int
    motif: Optional[str] = None
    actif: Optional[int] = 1

    class Config:
        from_attributes = True


class GroupeLinkSchema(BaseModel):
    id: Optional[int] = None
    groupe_id: int
    client_id: Optional[int] = None
    affaire_id: Optional[int] = None
    intervenant_id: Optional[int] = None
    date_ajout: Optional[str] = None
    date_retrait: Optional[str] = None
    actif: Optional[int] = 1

    class Config:
        from_attributes = True


class GroupeLinkCreateSchema(BaseModel):
    groupe_id: int
    client_id: Optional[int] = None
    affaire_id: Optional[int] = None
    intervenant_id: Optional[int] = None

    @classmethod
    def validate_target(cls, v: "GroupeLinkCreateSchema"):
        return v


class GroupeMembershipBatchSchema(BaseModel):
    action: Literal["add", "remove"]
    groupe_id: int
    client_ids: list[int] = []
    affaire_ids: list[int] = []


class GroupeMembershipBatchResultSchema(BaseModel):
    action: str
    groupe_id: int
    requested: int
    affected: int
