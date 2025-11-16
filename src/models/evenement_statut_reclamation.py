from sqlalchemy import Column, Integer, String

from src.database import Base


class EvenementStatutReclamation(Base):
    __tablename__ = "mariadb_evenement_statut_reclamation"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    code = Column(String, nullable=False)
    libelle = Column(String, nullable=False)
    is_cloture = Column(Integer, nullable=False, default=0)

