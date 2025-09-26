from sqlalchemy import Column, Integer, Float, DateTime, String
from src.database import Base

class HistoriquePersonne(Base):
    __tablename__ = "mariadb_historique_personne_w"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    date = Column(DateTime, nullable=True)
    valo = Column(Float, nullable=True)
    mouvement = Column(Float, nullable=True)
    sicav = Column(Float, nullable=True)
    perf_sicav_hebdo = Column(Float, nullable=True)
    perf_sicav_52 = Column(Float, nullable=True)
    volat = Column(Float, nullable=True)
    SRRI = Column(Integer, nullable=True)
    annee = Column("Année", Integer, nullable=True)
