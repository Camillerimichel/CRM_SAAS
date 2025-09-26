from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from src.database import Base


class EvenementEnvoi(Base):
    __tablename__ = "mariadb_evenement_envoi"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    evenement_id = Column(Integer, ForeignKey("mariadb_evenement.id"), nullable=False)
    canal = Column(String, nullable=False)
    destinataire = Column(String, nullable=False)
    objet = Column(String, nullable=True)
    contenu = Column(String, nullable=True)
    date_envoi = Column(DateTime, nullable=False)
    statut = Column(String, nullable=True)
    modele_id = Column(Integer, ForeignKey("mariadb_modele_document.id"), nullable=True)

