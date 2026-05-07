from sqlalchemy import Column, Integer, String, DateTime, SmallInteger, ForeignKey
from sqlalchemy.sql import func
from src.database import Base


class ClientIdentifiantFournisseur(Base):
    __tablename__ = "mariadb_client_identifiants_fournisseur"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    client_id           = Column(Integer, ForeignKey("mariadb_clients.id", ondelete="CASCADE"), nullable=False, index=True)
    fournisseur         = Column(String(100), nullable=False, index=True)
    identifiant_externe = Column(String(255), nullable=False)
    date_creation       = Column(DateTime, nullable=False, server_default=func.now())
    actif               = Column(SmallInteger, nullable=False, default=1)
