from sqlalchemy import Column, Integer, String, Text, Date, ForeignKey
from src.database import Base


class AffaireSociete(Base):
    __tablename__ = "mariadb_affaire_societe"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    affaire_id = Column(Integer, ForeignKey("mariadb_affaires.id"), nullable=False)
    societe_id = Column(Integer, ForeignKey("mariadb_societe_gestion.id"), nullable=False)
    role = Column(String, nullable=False)  # courtier / gestion
    date_debut = Column(Date, nullable=True)
    date_fin = Column(Date, nullable=True)
    commentaire = Column(Text, nullable=True)
