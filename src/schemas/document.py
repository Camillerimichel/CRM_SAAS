from pydantic import BaseModel
from typing import Optional

# Lecture (DB → API)
class DocumentSchema(BaseModel):
    id_document_base: int
    documents: Optional[str] = None
    niveau: Optional[str] = None
    obsolescence_annees: Optional[int] = None
    risque: Optional[str] = None

    class Config:
        from_attributes = True


# Création (API → DB)
class DocumentCreateSchema(BaseModel):
    documents: str
    niveau: str
    obsolescence_annees: Optional[int] = None
    risque: Optional[str] = None
