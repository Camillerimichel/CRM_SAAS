from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime
from .validators import format_date

class DocumentClientSchema(BaseModel):
    id: int
    id_client: int
    nom_client: Optional[str] = None
    id_document_base: int
    nom_document: Optional[str] = None
    date_creation: Optional[datetime] = None
    date_obsolescence: Optional[datetime] = None
    obsolescence: Optional[str] = None

    class Config:
        from_attributes = True

    @field_validator("date_creation", "date_obsolescence", mode="before")
    @classmethod
    def parse_dates(cls, v):
        return format_date(v)


class DocumentClientCreateSchema(BaseModel):
    id_client: int
    id_document_base: int
    nom_document: Optional[str] = None
    date_creation: Optional[datetime] = None
    date_obsolescence: Optional[datetime] = None
    obsolescence: Optional[str] = None

    @field_validator("date_creation", "date_obsolescence", mode="before")
    @classmethod
    def parse_dates(cls, v):
        return format_date(v)
