from sqlalchemy import Column, Integer, String
from src.database import Base

class Document(Base):
    __tablename__ = "Documents"

    id_document_base = Column("Id document base", Integer, primary_key=True, autoincrement=True)
    documents = Column("Documents", String, nullable=True)
    niveau = Column("Niveau", String, nullable=True)
    obsolescence_annees = Column("Obsolescence (années)", Integer, nullable=True)
    risque = Column("Risque", String, nullable=True)
