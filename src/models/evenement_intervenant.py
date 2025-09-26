from sqlalchemy import Column, Integer, String, ForeignKey
from src.database import Base


class EvenementIntervenant(Base):
    __tablename__ = "mariadb_evenement_intervenant"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    evenement_id = Column(Integer, ForeignKey("mariadb_evenement.id"), nullable=False)
    role = Column(String, nullable=False)
    nom_intervenant = Column(String, nullable=False)
    contact = Column(String, nullable=True)

