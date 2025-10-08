from typing import Optional, Literal
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src.models.administration_groupe_detail import AdministrationGroupeDetail
from src.models.administration_groupe import AdministrationGroupe
from datetime import datetime


def list_group_details(db: Session, type_groupe: Optional[Literal['client','affaire']] = None, actifs_only: bool = False):
    q = db.query(AdministrationGroupeDetail)
    if type_groupe:
        # Robust filter on normalized type (trim + lower), supporting synonyms
        tnorm = func.lower(func.trim(AdministrationGroupeDetail.type_groupe))
        if str(type_groupe).lower() == 'client':
            q = q.filter(tnorm.in_(['client', 'clients', 'personne', 'personnes']))
        elif str(type_groupe).lower() == 'affaire':
            q = q.filter(tnorm.in_(['affaire', 'affaires', 'contrat', 'contrats']))
    if actifs_only:
        # Consider any non-zero/truthy as active to be robust across legacy imports
        q = q.filter((AdministrationGroupeDetail.actif.is_(None)) | (AdministrationGroupeDetail.actif != 0))
    return q.order_by(AdministrationGroupeDetail.nom.asc()).all()


def list_memberships_for_client(db: Session, client_id: int):
    q = (
        db.query(AdministrationGroupe)
        .filter(AdministrationGroupe.client_id == client_id)
        # SQLite compatible: place NULLs last then newest first
        .order_by((AdministrationGroupe.date_ajout.is_(None)).asc(), AdministrationGroupe.date_ajout.desc())
    )
    return q.all()


def list_memberships_for_affaire(db: Session, affaire_id: int):
    q = (
        db.query(AdministrationGroupe)
        .filter(AdministrationGroupe.affaire_id == affaire_id)
        .order_by((AdministrationGroupe.date_ajout.is_(None)).asc(), AdministrationGroupe.date_ajout.desc())
    )
    return q.all()


def add_membership(db: Session, *, groupe_id: int, client_id: Optional[int] = None, affaire_id: Optional[int] = None, intervenant_id: Optional[int] = None):
    # Idempotent: if a link exists for (groupe_id, client_id) or (groupe_id, affaire_id),
    #   - if inactive, reactivate it (actif=1, date_retrait=NULL, set date_ajout if empty)
    #   - if already active, return as-is
    if client_id is not None:
        existing = (
            db.query(AdministrationGroupe)
            .filter(
                AdministrationGroupe.groupe_id == groupe_id,
                AdministrationGroupe.client_id == client_id,
            )
        ).first()
        if existing:
            if (existing.actif is None or existing.actif != 0) and (getattr(existing, 'date_retrait', None) is None):
                return existing
            existing.actif = 1
            existing.date_retrait = None
            if not getattr(existing, 'date_ajout', None):
                try:
                    existing.date_ajout = datetime.utcnow()
                except Exception:
                    pass
            db.commit()
            db.refresh(existing)
            return existing
    if affaire_id is not None:
        existing = (
            db.query(AdministrationGroupe)
            .filter(
                AdministrationGroupe.groupe_id == groupe_id,
                AdministrationGroupe.affaire_id == affaire_id,
            )
        ).first()
        if existing:
            if (existing.actif is None or existing.actif != 0) and (getattr(existing, 'date_retrait', None) is None):
                return existing
            existing.actif = 1
            existing.date_retrait = None
            if not getattr(existing, 'date_ajout', None):
                try:
                    existing.date_ajout = datetime.utcnow()
                except Exception:
                    pass
            db.commit()
            db.refresh(existing)
            return existing
    # Compute next id to ensure non-NULL IDs even on legacy schemas
    try:
        next_id = (db.query(func.max(AdministrationGroupe.id)).scalar() or 0) + 1
    except Exception:
        next_id = None
    link = AdministrationGroupe(
        id=next_id,
        groupe_id=groupe_id,
        client_id=client_id,
        affaire_id=affaire_id,
        intervenant_id=intervenant_id,
    )
    db.add(link)
    db.commit()
    try:
        db.refresh(link)
    except Exception:
        pass
    return link


def soft_delete_membership(db: Session, link_id: int):
    link = db.query(AdministrationGroupe).filter(AdministrationGroupe.id == link_id).first()
    if not link:
        return None
    # Rely on SQLite trigger to perform logical delete on physical delete
    db.delete(link)
    db.commit()
    return link
