from sqlalchemy import Column, Integer, String, ForeignKey
from src.database import Base


class EvenementLien(Base):
    __tablename__ = "mariadb_evenement_lien"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    evenement_source_id = Column(Integer, ForeignKey("mariadb_evenement.id"), nullable=False)
    evenement_cible_id = Column(Integer, ForeignKey("mariadb_evenement.id"), nullable=False)
    type_lien = Column(String, nullable=False)

