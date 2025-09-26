from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from src.database import Base

class DocumentClient(Base):
    __tablename__ = "Documents_client"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_client = Column("Id_client", Integer, ForeignKey("mariadb_clients.id"), nullable=False)
    nom_client = Column("Nom_client", String, nullable=True)
    # Référence la colonne physique "Id document base" de la table Documents
    id_document_base = Column(
        "Id_document_base",
        Integer,
        ForeignKey("Documents.Id document base"),
        nullable=False,
    )
    nom_document = Column("Nom_Document", String, nullable=True)
    date_creation = Column("Date_creation", DateTime, nullable=True)
    date_obsolescence = Column("Date_obsolescence", DateTime, nullable=True)
    obsolescence = Column("Obsolescence", String, nullable=True)
