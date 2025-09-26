from sqlalchemy.orm import Session
from src.models.allocation import Allocation

# ---------------- CREATE ----------------
def create_allocation(db: Session, date=None, valo=None, mouvement=None,
                      sicav=None, perf_sicav_hebdo=None, perf_sicav_52=None,
                      volat=None, annee=None, nom=None):
    allocation = Allocation(
        date=date,
        valo=valo,
        mouvement=mouvement,
        sicav=sicav,
        perf_sicav_hebdo=perf_sicav_hebdo,
        perf_sicav_52=perf_sicav_52,
        volat=volat,
        annee=annee,    # ← correspond au modèle
        nom=nom,
    )
    db.add(allocation)
    db.commit()
    db.refresh(allocation)
    return allocation

# ---------------- READ ----------------
def get_allocations(db: Session):
    return db.query(Allocation).all()

def get_allocation(db: Session, allocation_id: int):
    return db.query(Allocation).filter(Allocation.id == allocation_id).first()

# ---------------- DELETE ----------------
def delete_allocation(db: Session, allocation_id: int):
    allocation = db.query(Allocation).filter(Allocation.id == allocation_id).first()
    if not allocation:
        return None
    db.delete(allocation)
    db.commit()
    return allocation
