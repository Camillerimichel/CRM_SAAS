from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from src.database import Base


class EvenementStatut(Base):
    __tablename__ = "mariadb_evenement_statut"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    evenement_id = Column(Integer, ForeignKey("mariadb_evenement.id"), nullable=False)
    statut_id = Column(Integer, ForeignKey("mariadb_statut_evenement.id"), nullable=False)
    date_statut = Column(DateTime, nullable=False)
    utilisateur_responsable = Column(String, nullable=True)
    commentaire = Column(String, nullable=True)

