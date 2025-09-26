from sqlalchemy import Column, Integer, String
from src.database import Base


class TypeEvenement(Base):
    __tablename__ = "mariadb_type_evenement"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    libelle = Column(String, nullable=False)
    categorie = Column(String, nullable=False)

