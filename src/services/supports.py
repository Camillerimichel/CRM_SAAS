from sqlalchemy.orm import Session
from src.models.support import Support


# Récupérer tous les supports
def get_supports(db: Session):
    supports = db.query(Support).all()
    return [s for s in supports if s is not None]


# Récupérer un support par id
def get_support(db: Session, support_id: int):
    return db.query(Support).filter(Support.id == support_id).first()


# Ajouter un support
def create_support(
    db: Session,
    code_isin: str,
    nom: str,
    cat_gene: str | None = None,
    cat_principale: str | None = None,
    cat_det: str | None = None,
    cat_geo: str | None = None,
    promoteur: str | None = None,
    taux_retro: float | None = None,
    srri: int | None = None,
):
    support = Support(
        code_isin=code_isin,
        nom=nom,
        cat_gene=cat_gene,
        cat_principale=cat_principale,
        cat_det=cat_det,
        cat_geo=cat_geo,
        promoteur=promoteur,
        taux_retro=taux_retro,
        SRRI=srri,
    )
    db.add(support)
    db.commit()
    db.refresh(support)
    return support
