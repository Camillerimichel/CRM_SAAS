from sqlalchemy import Column, Integer, DateTime, ForeignKey
from src.database import Base


class AdministrationGroupe(Base):
    __tablename__ = "administration_groupe"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    groupe_id = Column(Integer, ForeignKey("administration_groupe_detail.id"), nullable=False)
    client_id = Column(Integer, nullable=True)
    affaire_id = Column(Integer, nullable=True)
    intervenant_id = Column(Integer, nullable=True)
    date_ajout = Column(DateTime, nullable=True)
    date_retrait = Column(DateTime, nullable=True)
    actif = Column(Integer, nullable=True, default=1)

