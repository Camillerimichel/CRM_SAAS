from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import Optional, Literal
from sqlalchemy import text as _text

from src.database import get_db
from src.security.rbac import load_access, require_permission, extract_user_context, pick_scope
from src.schemas.groupes import (
    GroupeDetailSchema,
    GroupeLinkSchema,
    GroupeLinkCreateSchema,
    GroupeMembershipBatchSchema,
    GroupeMembershipBatchResultSchema,
)
from src.services.groupes import (
    list_group_details,
    list_memberships_for_client,
    list_memberships_for_affaire,
    add_membership,
    soft_delete_membership,
)
from src.models.administration_groupe import AdministrationGroupe
from src.models.administration_groupe_detail import AdministrationGroupeDetail
from src.models.document_client import DocumentClient

from pydantic import BaseModel
from datetime import date as _date
from sqlalchemy import func as _func


router = APIRouter(prefix="/api/groupes", tags=["Groupes"])


def _get_db_dialect(db: Session) -> str:
    bind = None
    try:
        bind = db.get_bind()
    except Exception:
        bind = getattr(db, "bind", None)
    if not bind:
        return ""
    dialect = getattr(getattr(bind, "dialect", None), "name", "") or ""
    return str(dialect).lower()


def _ensure_group_ids(db: Session):
    if not _get_db_dialect(db).startswith("sqlite"):
        return
    try:
        missing = db.execute(_text("SELECT rowid FROM administration_groupe_detail WHERE id IS NULL")).fetchall()
    except Exception:
        return
    if not missing:
        return
    try:
        next_id_row = db.execute(_text("SELECT MAX(id) AS max_id FROM administration_groupe_detail")).fetchone()
        next_id = ((next_id_row[0] if next_id_row else 0) or 0) + 1
    except Exception:
        next_id = 1
    try:
        for row in missing:
            rid = row[0]
            db.execute(
                _text("UPDATE administration_groupe_detail SET id = :new_id WHERE rowid = :rid"),
                {"new_id": next_id, "rid": rid},
            )
            next_id += 1
        db.commit()
    except Exception:
        db.rollback()


def _assert_group_permission(request: Request, db: Session):
    """Vérifie que l'utilisateur peut gérer les groupes."""
    user_type, user_id, req_scope = extract_user_context(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Non authentifié")
    access = load_access(db, user_type=user_type, user_id=user_id)
    scope = pick_scope(access, req_scope)
    require_permission(access, "groups", "manage", societe_id=scope)
    return access, scope


@router.get("/details", response_model=list[GroupeDetailSchema])
def api_list_group_details(
    type: Optional[Literal['client','affaire']] = Query(default=None, alias="type"),
    actifs_only: bool = Query(default=True),
    request: Request = None,
    db: Session = Depends(get_db),
):
    _assert_group_permission(request, db)
    _ensure_group_ids(db)
    return list_group_details(db, type_groupe=type, actifs_only=actifs_only)


@router.get("/memberships", response_model=list[GroupeLinkSchema])
def api_list_memberships(
    client_id: Optional[int] = None,
    affaire_id: Optional[int] = None,
    request: Request = None,
    db: Session = Depends(get_db),
):
    _assert_group_permission(request, db)
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
def api_add_membership(payload: GroupeLinkCreateSchema, request: Request = None, db: Session = Depends(get_db)):
    _assert_group_permission(request, db)
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
def api_delete_membership(link_id: int, request: Request = None, db: Session = Depends(get_db)):
    _assert_group_permission(request, db)
    link = soft_delete_membership(db, link_id)
    if not link:
        raise HTTPException(status_code=404, detail="Lien introuvable")
    return link


@router.post("/memberships/batch", response_model=GroupeMembershipBatchResultSchema)
def api_batch_memberships(payload: GroupeMembershipBatchSchema, request: Request = None, db: Session = Depends(get_db)):
    _assert_group_permission(request, db)
    raw_ids = None
    target = None
    if payload.client_ids:
        raw_ids = payload.client_ids
        target = "client"
    elif payload.affaire_ids:
        raw_ids = payload.affaire_ids
        target = "affaire"
    else:
        raise HTTPException(status_code=400, detail="client_ids ou affaire_ids requis")

    ids = []
    for cid in raw_ids or []:
        try:
            n = int(cid)
        except Exception:
            continue
        if n > 0:
            ids.append(n)
    # dédoublonnage en conservant l'ordre
    seen = set()
    uniq = []
    for n in ids:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)
    if not uniq:
        raise HTTPException(status_code=400, detail="client_ids invalides")
    if len(uniq) > 2000:
        raise HTTPException(status_code=400, detail="Trop d'éléments (max 2000).")

    if payload.action == "add":
        affected = 0
        for cid in uniq:
            if target == "client":
                add_membership(db, groupe_id=payload.groupe_id, client_id=cid)
            else:
                add_membership(db, groupe_id=payload.groupe_id, affaire_id=cid)
            affected += 1
        return {
            "action": "add",
            "groupe_id": payload.groupe_id,
            "requested": len(uniq),
            "affected": affected,
        }

    # remove
    try:
        q = db.query(AdministrationGroupe).filter(AdministrationGroupe.groupe_id == payload.groupe_id)
        if target == "client":
            q = q.filter(AdministrationGroupe.client_id.in_(uniq))
        else:
            q = q.filter(AdministrationGroupe.affaire_id.in_(uniq))
        affected = int(q.delete(synchronize_session=False) or 0)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur suppression: {e}")
    return {
        "action": "remove",
        "groupe_id": payload.groupe_id,
        "requested": len(uniq),
        "affected": affected,
    }


class GroupeCreateFromDocumentsPayload(BaseModel):
    nom: str
    responsable_id: int
    document_client_ids: list[int]


@router.post("/create_from_documents")
def api_create_group_from_documents(payload: GroupeCreateFromDocumentsPayload, request: Request = None, db: Session = Depends(get_db)):
    """Crée un groupe de personnes (type client) à partir d'une sélection de documents."""
    _assert_group_permission(request, db)
    _ensure_group_ids(db)

    name = (payload.nom or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nom du groupe requis.")
    try:
        responsable_id = int(payload.responsable_id)
    except Exception:
        responsable_id = None
    if not responsable_id or responsable_id <= 0:
        raise HTTPException(status_code=400, detail="Responsable requis.")

    raw_ids = payload.document_client_ids or []
    doc_ids: list[int] = []
    for did in raw_ids:
        try:
            n = int(did)
        except Exception:
            continue
        if n > 0:
            doc_ids.append(n)
    # dédoublonnage en conservant l'ordre
    seen = set()
    uniq_doc_ids: list[int] = []
    for n in doc_ids:
        if n in seen:
            continue
        seen.add(n)
        uniq_doc_ids.append(n)
    if not uniq_doc_ids:
        raise HTTPException(status_code=400, detail="Aucun document sélectionné.")
    if len(uniq_doc_ids) > 2000:
        raise HTTPException(status_code=400, detail="Trop de documents (max 2000).")

    rows = (
        db.query(DocumentClient.id_client)
        .filter(DocumentClient.id.in_(uniq_doc_ids))
        .all()
    )
    client_ids: list[int] = []
    seen_cli = set()
    for (cid,) in rows or []:
        try:
            c = int(cid) if cid is not None else None
        except Exception:
            c = None
        if not c or c <= 0:
            continue
        if c in seen_cli:
            continue
        seen_cli.add(c)
        client_ids.append(c)
    if not client_ids:
        raise HTTPException(status_code=400, detail="Aucun client lié aux documents sélectionnés.")

    try:
        next_gid: int | None
        try:
            max_id = db.query(_func.max(AdministrationGroupeDetail.id)).scalar()
            next_gid = (max_id or 0) + 1
        except Exception:
            next_gid = None
        motif = f"Créé depuis Documents ({len(uniq_doc_ids)} doc(s))"
        g_payload = {
            "type_groupe": "client",
            "nom": name,
            "date_creation": _date.today(),
            "responsable_id": responsable_id,
            "motif": motif,
            "actif": 1,
        }
        if next_gid is not None:
            g_payload["id"] = next_gid
        group = AdministrationGroupeDetail(**g_payload)
        db.add(group)
        db.flush()
        group_id = getattr(group, "id", None)
        if group_id is None:
            db.commit()
            db.refresh(group)
            group_id = getattr(group, "id", None)
        if group_id is None:
            raise HTTPException(status_code=500, detail="Création du groupe impossible (id manquant).")

        # Memberships (batch)
        try:
            next_link_id = (db.query(_func.max(AdministrationGroupe.id)).scalar() or 0) + 1
        except Exception:
            next_link_id = None
        links: list[AdministrationGroupe] = []
        for cid in client_ids:
            link_kwargs = {"groupe_id": int(group_id), "client_id": int(cid)}
            if next_link_id is not None:
                link_kwargs["id"] = int(next_link_id)
                next_link_id += 1
            links.append(AdministrationGroupe(**link_kwargs))
        db.add_all(links)
        db.commit()
        return {
            "status": "ok",
            "groupe_id": int(group_id),
            "groupe_nom": name,
            "documents_selected": len(uniq_doc_ids),
            "clients_added": len(client_ids),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Création impossible: {e}")


@router.get("/overview/{groupe_key}")
def api_group_overview(
    groupe_key: int,
    by: Optional[str] = Query(default=None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """Retourne les métadonnées du groupe + la liste de ses membres (clients/affaires).
    Supporte by=rowid sur SQLite si certaines lignes ont id NULL.
    """
    _assert_group_permission(request, db)
    _ensure_group_ids(db)
    # Détail groupe (par id ou rowid)
    try:
        dialect = _get_db_dialect(db)
        if by == 'rowid':
            if not dialect.startswith("sqlite"):
                raise HTTPException(status_code=400, detail="Recherche par rowid indisponible sur ce SGBD.")
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
                ORDER BY (g.date_ajout IS NULL), g.date_ajout DESC, g.id DESC
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
