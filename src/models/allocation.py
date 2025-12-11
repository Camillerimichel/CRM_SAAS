from sqlalchemy import Column, Integer, String, DateTime, Float
from src.database import Base

class Allocation(Base):
    __tablename__ = "allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column("date", DateTime, nullable=True)
    valo = Column("valo", Integer, nullable=True)
    mouvement = Column("mouvement", Integer, nullable=True)
    sicav = Column("sicav", Integer, nullable=True)
    perf_sicav_hebdo = Column("perf_sicav_hebdo", Float, nullable=True)
    perf_sicav_52 = Column("perf_sicav_52", Float, nullable=True)
    volat = Column("volat", Float, nullable=True)
    srri = Column("srri", Integer, nullable=True)
    sri = Column("SRI", Integer, nullable=True)
    annee = Column("anne", Integer, nullable=True)
    nom = Column("nom", String, nullable=True)
    isin = Column("isin", String, nullable=True)
