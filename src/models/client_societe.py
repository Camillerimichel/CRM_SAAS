from sqlalchemy import Column, Integer, String, Text, Date, ForeignKey
from src.database import Base


class ClientSociete(Base):
    __tablename__ = "mariadb_client_societe"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    client_id = Column(Integer, ForeignKey("mariadb_clients.id"), nullable=False)
    societe_id = Column(Integer, ForeignKey("mariadb_societe_gestion.id"), nullable=False)
    role = Column(String, nullable=False)  # courtier / gestion / apporteur
    date_debut = Column(Date, nullable=True)
    date_fin = Column(Date, nullable=True)
    commentaire = Column(Text, nullable=True)
