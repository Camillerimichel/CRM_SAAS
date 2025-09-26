from sqlalchemy.orm import Session
from sqlalchemy import func, select
from src.models.client import Client
from src.models.historique_personne import HistoriquePersonne
from src.schemas.client import ClientCreateSchema, ClientUpdateSchema, ClientSchema

def get_clients(db: Session):
    """
    Retourne la liste des clients enrichie avec :
    - valorisation totale
    - performance 52 semaines
    - volatilité
    sur la dernière date connue par client
    """
    # sous-requête : dernière date d'historique par client
    subquery = (
        select(
            HistoriquePersonne.id.label("client_id"),
            func.max(HistoriquePersonne.date).label("last_date")
        )
        .group_by(HistoriquePersonne.id)
        .subquery()
    )

    # jointure clients + historique dernière date
    query = (
        db.query(
            Client.id,
            Client.nom,
            Client.prenom,
            Client.SRRI,
            HistoriquePersonne.SRRI.label("srri_hist"),
            HistoriquePersonne.valo.label("total_valo"),
            HistoriquePersonne.perf_sicav_52.label("perf_52_sem"),
            HistoriquePersonne.volat.label("volatilite")
        )
        .join(subquery, subquery.c.client_id == Client.id)
        .join(
            HistoriquePersonne,
            (HistoriquePersonne.id == subquery.c.client_id) &
            (HistoriquePersonne.date == subquery.c.last_date)
        )
        .order_by(Client.nom)
    )

    return query.all()


# Récupérer un client par ID
def get_client(db: Session, client_id: int):
    return db.query(Client).filter(Client.id == client_id).first()


# Créer un client
def create_client(db: Session, client: ClientCreateSchema):
    db_client = Client(
        nom=client.nom,
        prenom=client.prenom,
        SRRI=client.SRRI,
        telephone=client.telephone,
        adresse_postale=client.adresse_postale,
        email=client.email,
    )
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client


# Mettre à jour un client
def update_client(db: Session, client_id: int, client: ClientUpdateSchema):
    db_client = db.query(Client).filter(Client.id == client_id).first()
    if not db_client:
        return None
    for key, value in client.dict(exclude_unset=True).items():
        setattr(db_client, key, value)
    db.commit()
    db.refresh(db_client)
    return db_client


# Supprimer un client
def delete_client(db: Session, client_id: int):
    db_client = db.query(Client).filter(Client.id == client_id).first()
    if not db_client:
        return None
    db.delete(db_client)
    db.commit()
    return db_client
