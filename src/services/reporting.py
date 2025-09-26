from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.models.client import Client
from src.models.affaire import Affaire
from src.models.support import Support
from src.models.allocation import Allocation

# Récupérer tous les clients
def get_all_clients(db: Session):
    clients = db.query(Client).all()
    return [c for c in clients if c is not None]

# Récupérer les top clients (par SRRI ou autre critère)
def get_top_clients(db: Session, limit: int = 5):
    clients = (
        db.query(Client)
        .order_by(desc(Client.SRRI))
        .limit(limit)
        .all()
    )
    return [c for c in clients if c is not None]

# Récupérer toutes les affaires
def get_all_affaires(db: Session):
    affaires = db.query(Affaire).all()
    return [a for a in affaires if a is not None]

# Récupérer toutes les allocations
def get_all_allocations(db: Session):
    allocations = db.query(Allocation).all()
    return [a for a in allocations if a is not None]

# Récupérer tous les supports
def get_all_supports(db: Session):
    supports = db.query(Support).all()
    return [s for s in supports if s is not None]
