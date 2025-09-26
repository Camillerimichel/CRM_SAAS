from sqlalchemy.orm import Session
from src.models.historique_affaire import HistoriqueAffaire
from src.models.historique_personne import HistoriquePersonne
from src.models.historique_support import HistoriqueSupport


# ---------------- Personne ----------------
def get_historiques_personne(db: Session):
    data = db.query(HistoriquePersonne).all()
    return [h for h in data if h is not None]


def get_historique_personne(db: Session, hist_id: int):
    return db.query(HistoriquePersonne).filter(HistoriquePersonne.id == hist_id).first()

def create_historique_personne(
    db: Session,
    date=None,
    valo: float | None = None,
    mouvement: float | None = None,
    volat: float | None = None,
    annee: int | None = None,
):
    hist = HistoriquePersonne(
        date=date,
        valo=valo,
        mouvement=mouvement,
        volat=volat,
        annee=annee,
    )
    db.add(hist)
    db.commit()
    db.refresh(hist)
    return hist


# ---------------- Affaire ----------------
def get_historiques_affaire(db: Session):
    data = db.query(HistoriqueAffaire).all()
    return [h for h in data if h is not None]


def get_historique_affaire(db: Session, hist_id: int):
    return db.query(HistoriqueAffaire).filter(HistoriqueAffaire.id == hist_id).first()


def create_historique_affaire(
    db: Session,
    date=None,
    valo: float | None = None,
    mouvement: float | None = None,
    sicav: float | None = None,
    perf_sicav_hebdo: float | None = None,
    perf_sicav_52: float | None = None,
    volat: float | None = None,
    annee: int | None = None,
):
    hist = HistoriqueAffaire(
        date=date,
        valo=valo,
        mouvement=mouvement,
        sicav=sicav,
        perf_sicav_hebdo=perf_sicav_hebdo,
        perf_sicav_52=perf_sicav_52,
        volat=volat,
        annee=annee,
    )
    db.add(hist)
    db.commit()
    db.refresh(hist)
    return hist


# ---------------- Support ----------------
def get_historiques_support(db: Session):
    data = db.query(HistoriqueSupport).all()
    return [h for h in data if h is not None]


def get_historique_support(db: Session, hist_id: int):
    return db.query(HistoriqueSupport).filter(HistoriqueSupport.id == hist_id).first()
def create_historique_support(
    db: Session,
    modif_quand=None,
    source: str | None = None,
    id_source: int | None = None,
    date=None,
    id_support: str | None = None,
    nbuc: float | None = None,
    vl: float | None = None,
    prmp: float | None = None,
    valo: float | None = None,
):
    hist = HistoriqueSupport(
        modif_quand=modif_quand,
        source=source,
        id_source=id_source,
        date=date,
        id_support=id_support,
        nbuc=nbuc,
        vl=vl,
        prmp=prmp,
        valo=valo,
    )
    db.add(hist)
    db.commit()
    db.refresh(hist)
    return hist
