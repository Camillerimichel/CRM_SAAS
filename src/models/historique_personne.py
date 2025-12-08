from sqlalchemy import Column, Integer, Float, DateTime, String
from src.database import Base

class HistoriquePersonne(Base):
    __tablename__ = "mariadb_historique_personne_w"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Certains déploiements n'ont pas la colonne id_personne; utiliser id comme identifiant client si absent.
    id_personne = Column(Integer, index=True, nullable=True)
    date = Column(DateTime, nullable=True)
    valo = Column(Float, nullable=True)
    mouvement = Column(Float, nullable=True)
    sicav = Column(Float, nullable=True)
    perf_sicav_hebdo = Column(Float, nullable=True)
    perf_sicav_52 = Column(Float, nullable=True)
    volat = Column(Float, nullable=True)
    SRRI = Column("srri", Integer, nullable=True)
    annee = Column("anne", Integer, nullable=True)
