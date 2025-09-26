from sqlalchemy import Column, Integer, String, Float, DateTime
from src.database import Base

class HistoriqueSupport(Base):
    __tablename__ = "mariadb_historique_support_w"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    modif_quand = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    id_source = Column(Integer, nullable=True)
    date = Column(DateTime, nullable=True)
    id_support = Column(String, nullable=True)
    nbuc = Column(Float, nullable=True)
    vl = Column(Float, nullable=True)
    prmp = Column(Float, nullable=True)
    valo = Column(Float, nullable=True)
