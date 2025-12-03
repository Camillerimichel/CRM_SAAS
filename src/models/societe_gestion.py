from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, func
from src.database import Base


class SocieteGestion(Base):
    __tablename__ = "mariadb_societe_gestion"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    nom = Column(String, nullable=False)
    nature = Column(String, nullable=False)  # courtier / wealth management / banque privée / courtier CIF / co-courtier
    siret = Column(String, nullable=True)
    rcs = Column(String, nullable=True)
    contact = Column(String, nullable=True)
    telephone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    adresse = Column(Text, nullable=True)
    actif = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
