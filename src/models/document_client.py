from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from src.database import Base

class DocumentClient(Base):
    __tablename__ = "Documents_client"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_client = Column("id_client", Integer, ForeignKey("mariadb_clients.id"), nullable=False)
    nom_client = Column("nom_client", String, nullable=True)
    id_document_base = Column(
        "id_document_base",
        Integer,
        ForeignKey("Documents.id_document_base"),
        nullable=False,
    )
    nom_document = Column("nom_document", String, nullable=True)
    date_creation = Column("date_creation", DateTime, nullable=True)
    date_obsolescence = Column("date_obsolescence", DateTime, nullable=True)
    obsolescence = Column("obsolescence", String, nullable=True)
