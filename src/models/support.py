from sqlalchemy import Column, Integer, String, Float
from src.database import Base

class Support(Base):
    __tablename__ = "mariadb_support"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code_isin = Column(String, nullable=True)
    nom = Column(String, nullable=True)
    cat_gene = Column(String, nullable=True)
    cat_principale = Column(String, nullable=True)
    cat_det = Column(String, nullable=True)
    cat_geo = Column(String, nullable=True)
    promoteur = Column(String, nullable=True)
    taux_retro = Column("Taux rétro", Float, nullable=True)
    SRRI = Column(Integer, nullable=True)
