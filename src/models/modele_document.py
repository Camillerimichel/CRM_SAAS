from sqlalchemy import Column, Integer, String
from src.database import Base


class ModeleDocument(Base):
    __tablename__ = "mariadb_modele_document"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    nom = Column(String, nullable=False)
    canal = Column(String, nullable=False)
    objet = Column(String, nullable=True)
    contenu = Column(String, nullable=False)
    actif = Column(Integer, nullable=True, default=1)

