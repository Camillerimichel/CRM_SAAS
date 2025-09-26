from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from src.database import Base


class Evenement(Base):
    __tablename__ = "mariadb_evenement"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    type_id = Column(Integer, ForeignKey("mariadb_type_evenement.id"), nullable=False)
    client_id = Column(Integer, ForeignKey("mariadb_clients.id"), nullable=True)
    affaire_id = Column(Integer, ForeignKey("mariadb_affaires.id"), nullable=True)
    support_id = Column(Integer, ForeignKey("mariadb_support.id"), nullable=True)
    date_evenement = Column(DateTime, nullable=False)
    statut = Column(String, nullable=False, default="à faire")
    commentaire = Column(String, nullable=True)
    utilisateur_responsable = Column(String, nullable=True)

