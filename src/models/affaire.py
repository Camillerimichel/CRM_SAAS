from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Numeric
from src.database import Base  # Base est bien dans src/database.py

class Affaire(Base):
    __tablename__ = "mariadb_affaires"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_personne = Column(Integer, ForeignKey("mariadb_clients.id"))
    ref = Column(String, nullable=True)
    date_debut = Column(DateTime, nullable=True)
    date_cle = Column(DateTime, nullable=True)
    SRRI = Column("srri", Integer, nullable=True)
    frais_negocies = Column("frais_negocies", Numeric(15, 2), nullable=True)
    id_affaire_generique = Column(Integer, ForeignKey("mariadb_affaires_generique.id"), nullable=True)
