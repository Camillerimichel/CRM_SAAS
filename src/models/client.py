from sqlalchemy import Column, Integer, String, Text
from src.database import Base

class Client(Base):
    __tablename__ = "mariadb_clients"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=True)
    prenom = Column(String, nullable=True)
    SRRI = Column(Integer, nullable=True)
    telephone = Column(String, nullable=True)
    adresse_postale = Column(Text, nullable=True)
    email = Column(String, nullable=True)
    # Nouveau champ: commercial_id (FK logique vers administration_RH.id)
    commercial_id = Column(Integer, nullable=True)
