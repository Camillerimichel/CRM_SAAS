from sqlalchemy import Column, Integer, String, Date, Text
from src.database import Base


class AdministrationGroupeDetail(Base):
    __tablename__ = "administration_groupe_detail"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type_groupe = Column(String, nullable=True)  # 'client' ou 'affaire'
    nom = Column(String, nullable=False)
    date_creation = Column(Date, nullable=True)
    date_fin = Column(Date, nullable=True)
    responsable_id = Column(Integer, nullable=False)
    motif = Column(Text, nullable=True)
    actif = Column(Integer, nullable=True, default=1)

