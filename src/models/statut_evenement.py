from sqlalchemy import Column, Integer, String
from src.database import Base


class StatutEvenement(Base):
    __tablename__ = "mariadb_statut_evenement"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    libelle = Column(String, nullable=False)

