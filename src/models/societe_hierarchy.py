from sqlalchemy import Column, Integer, DateTime, ForeignKey, func

from src.database import Base


class SocieteHierarchy(Base):
    __tablename__ = "mariadb_societe_hierarchy"

    ancestor_societe_id = Column(Integer, ForeignKey("mariadb_societe_gestion.id"), primary_key=True)
    descendant_societe_id = Column(Integer, ForeignKey("mariadb_societe_gestion.id"), primary_key=True)
    depth = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
