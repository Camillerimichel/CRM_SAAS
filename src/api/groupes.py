from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, Literal
from sqlalchemy import text as _text

from src.database import get_db
from src.schemas.groupes import GroupeDetailSchema, GroupeLinkSchema, GroupeLinkCreateSchema
from src.services.groupes import (
    list_group_details,
    list_memberships_for_client,
    list_memberships_for_affaire,
    add_membership,
    soft_delete_membership,
)


router = APIRouter(prefix="/api/groupes", tags=["Groupes"])


@router.get("/details", response_model=list[GroupeDetailSchema])
def api_list_group_details(
    type: Optional[Literal['client','affaire']] = Query(default=None, alias="type"),
    actifs_only: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    return list_group_details(db, type_groupe=type, actifs_only=actifs_only)


@router.get("/memberships", response_model=list[GroupeLinkSchema])
def api_list_memberships(
    client_id: Optional[int] = None,
    affaire_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    if not client_id and not affaire_id:
        raise HTTPException(status_code=400, detail="Fournir client_id ou affaire_id")
    try:
        links = list_memberships_for_client(db, client_id) if client_id else list_memberships_for_affaire(db, int(affaire_id))  # type: ignore[arg-type]
        out: list[dict] = []
        for l in links or []:
            def _to_str_dt(v):
                try:
                    from datetime import datetime, date
                    if isinstance(v, (datetime, date)):
                        return v.isoformat(sep=' ', timespec='seconds') if isinstance(v, datetime) else v.isoformat()
                    return str(v) if v is not None else None
                except Exception:
                    return None
            out.append({
                'id': getattr(l, 'id', None),
                'groupe_id': getattr(l, 'groupe_id', None),
                'client_id': getattr(l, 'client_id', None),
                'affaire_id': getattr(l, 'affaire_id', None),
                'intervenant_id': getattr(l, 'intervenant_id', None),
                'date_ajout': _to_str_dt(getattr(l, 'date_ajout', None)),
                'date_retrait': _to_str_dt(getattr(l, 'date_retrait', None)),
                'actif': getattr(l, 'actif', None),
            })
        return out
    except HTTPException:
        raise
    except Exception as e:
        # Surface error in API for easier debugging
        raise HTTPException(status_code=500, detail=f"Erreur memberships: {e}")


@router.post("/memberships", response_model=GroupeLinkSchema)
def api_add_membership(payload: GroupeLinkCreateSchema, db: Session = Depends(get_db)):
    if not payload.client_id and not payload.affaire_id:
        raise HTTPException(status_code=400, detail="Fournir client_id ou affaire_id")
    return add_membership(
        db,
        groupe_id=payload.groupe_id,
        client_id=payload.client_id,
        affaire_id=payload.affaire_id,
        intervenant_id=payload.intervenant_id,
    )


@router.delete("/memberships/{link_id}", response_model=GroupeLinkSchema)
def api_delete_membership(link_id: int, db: Session = Depends(get_db)):
    link = soft_delete_membership(db, link_id)
    if not link:
        raise HTTPException(status_code=404, detail="Lien introuvable")
    return link


@router.get("/overview/{groupe_key}")
def api_group_overview(
    groupe_key: int,
    by: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Retourne les métadonnées du groupe + la liste de ses membres (clients/affaires).
    Supporte by=rowid si certaines lignes ont id NULL.
    """
    # Détail groupe (par id ou rowid)
    try:
        if by == 'rowid':
            row = db.execute(_text(
                """
                SELECT rowid AS __rid, id, type_groupe, nom, date_creation, date_fin, responsable_id, motif, actif
                FROM administration_groupe_detail
                WHERE rowid = :gid
                LIMIT 1
                """
            ), {"gid": groupe_key}).fetchone()
        else:
            row = db.execute(_text(
                """
                SELECT id, type_groupe, nom, date_creation, date_fin, responsable_id, motif, actif
                FROM administration_groupe_detail
                WHERE id = :gid
                LIMIT 1
                """
            ), {"gid": groupe_key}).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Groupe introuvable")
        m = row._mapping
        detail = {k: m.get(k) for k in m.keys()}
        # Responsable nom complet
        resp_nom = None
        if detail.get('responsable_id') is not None:
            rh = db.execute(_text("SELECT prenom, nom FROM administration_RH WHERE id = :id"), {"id": detail['responsable_id']}).fetchone()
            if rh:
                rm = rh._mapping
                resp_nom = f"{rm.get('prenom') or ''} {rm.get('nom') or ''}".strip()
        detail['responsable_nom'] = resp_nom
        # Label type display
        tg = str(detail.get('type_groupe') or '').lower()
        if tg in ('client','clients','personne','personnes'):
            detail['type_display'] = 'Personnes'
        elif tg in ('affaire','affaires','contrat','contrats'):
            detail['type_display'] = 'Affaires'
        else:
            detail['type_display'] = detail.get('type_groupe')
        # Members: résoudre id groupe réel si by=rowid
        real_id = detail.get('id')
        members = []
        if real_id is not None:
            rows = db.execute(_text(
                """
                SELECT g.id,
                       g.client_id,
                       g.affaire_id,
                       g.intervenant_id,
                       g.date_ajout,
                       g.date_retrait,
                       g.actif,
                       c.prenom AS client_prenom,
                       c.nom AS client_nom,
                       a.ref AS affaire_ref
                FROM administration_groupe g
                LEFT JOIN mariadb_clients c ON c.id = g.client_id
                LEFT JOIN mariadb_affaires a ON a.id = g.affaire_id
                WHERE g.groupe_id = :gid
                ORDER BY g.date_ajout DESC NULLS LAST, g.id DESC
                """
            ), {"gid": real_id}).fetchall() or []
            for r in rows:
                rm = r._mapping
                etype = 'client' if rm.get('client_id') is not None else ('affaire' if rm.get('affaire_id') is not None else 'autre')
                if etype == 'client':
                    label = f"{rm.get('client_prenom') or ''} {rm.get('client_nom') or ''}".strip() or f"Client #{rm.get('client_id')}"
                    entity_id = rm.get('client_id')
                elif etype == 'affaire':
                    label = rm.get('affaire_ref') or f"Affaire #{rm.get('affaire_id')}"
                    entity_id = rm.get('affaire_id')
                else:
                    label = '—'
                    entity_id = None
                members.append({
                    'id': rm.get('id'),
                    'type': etype,
                    'entity_id': entity_id,
                    'label': label,
                    'date_ajout': rm.get('date_ajout'),
                    'date_retrait': rm.get('date_retrait'),
                    'actif': rm.get('actif'),
                })
        return {"detail": detail, "members": members}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement groupe: {e}")
