from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer
from src.database import Base

class HistoriqueAffaire(Base):
    __tablename__ = "mariadb_historique_affaire_w"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_affaire = Column(Integer, ForeignKey("mariadb_affaires.id"), index=True, nullable=True)
    date = Column(DateTime, nullable=True)
    valo = Column(Float, nullable=True)
    mouvement = Column(Float, nullable=True)
    sicav = Column(Float, nullable=True)
    perf_sicav_hebdo = Column(Float, nullable=True)
    perf_sicav_52 = Column(Float, nullable=True)
    volat = Column(Float, nullable=True)
    annee = Column("anne", Integer, nullable=True)
    id_societe_gestion = Column(Integer, ForeignKey("mariadb_societe_gestion.id"), nullable=True)
