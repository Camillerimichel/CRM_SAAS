from sqlalchemy import Column, Integer, String
from src.database import Base

class Document(Base):
    __tablename__ = "Documents"

    id_document_base = Column("id_document_base", Integer, primary_key=True, autoincrement=True)
    documents = Column("documents", String, nullable=True)
    niveau = Column("niveau", String, nullable=True)
    obsolescence_annees = Column("obsolescence__annes", Integer, nullable=True)
    risque = Column("risque", String, nullable=True)
