from sqlalchemy.orm import Session
from src.models.affaire import Affaire


# Récupérer toutes les affaires
def get_affaires(db: Session):
    affaires = db.query(Affaire).all()
    # filtre de sécurité : on ne renvoie pas de None
    return [a for a in affaires if a is not None]


# Récupérer une affaire par id
def get_affaire(db: Session, affaire_id: int):
    return db.query(Affaire).filter(Affaire.id == affaire_id).first()


# Ajouter une affaire
def create_affaire(
    db: Session,
    id_personne: int,
    ref: str,
    srri: int | None = None,
    date_debut=None,
    date_cle=None,
    frais_courtier=None,
):
    affaire = Affaire(
        id_personne=id_personne,
        ref=ref,
        SRRI=srri,
        date_debut=date_debut,
        date_cle=date_cle,
        frais_courtier=frais_courtier,
    )
    db.add(affaire)
    db.commit()
    db.refresh(affaire)  # recharge l'objet pour éviter l'erreur de sérialisation
    return affaire
