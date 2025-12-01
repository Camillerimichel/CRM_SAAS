from sqlalchemy import Column, Integer, String, DateTime
from src.database import Base

class Allocation(Base):
    __tablename__ = "allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column("date", DateTime, nullable=True)
    valo = Column("valo", Integer, nullable=True)
    mouvement = Column("mouvement", Integer, nullable=True)
    sicav = Column("sicav", Integer, nullable=True)
    perf_sicav_hebdo = Column("perf_sicav_hebdo", Integer, nullable=True)
    perf_sicav_52 = Column("perf_sicav_52", Integer, nullable=True)
    volat = Column("volat", Integer, nullable=True)
    annee = Column("anne", Integer, nullable=True)
    nom = Column("nom", String, nullable=True)
    isin = Column("isin", String, nullable=True)
