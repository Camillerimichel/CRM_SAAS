"""
Endpoints CRUD pour la table mariadb_societe_identifiants_fournisseur.

Permet de gérer les correspondances entre identifiants externes fournisseurs
et sociétés de gestion CRM. Utilisé en amont de l'import de fichiers fournisseurs
pour résoudre automatiquement la société à partir du code transmis par l'assureur.

Endpoints :
  GET    /societes/{societe_id}/identifiants-fournisseur      – liste par société
  GET    /societes/identifiants-fournisseur/{id}              – détail
  POST   /societes/identifiants-fournisseur                   – création
  PUT    /societes/identifiants-fournisseur/{id}              – mise à jour
  DELETE /societes/identifiants-fournisseur/{id}              – suppression
  GET    /societes/identifiants-fournisseur/resolve           – résolution fournisseur → societe_id
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.societe_identifiant_fournisseur import (
    SocieteIdentifiantFournisseurCreate,
    SocieteIdentifiantFournisseurSchema,
    SocieteIdentifiantFournisseurUpdate,
)

router = APIRouter(tags=["societes-identifiants-fournisseur"])

TABLE = "mariadb_societe_identifiants_fournisseur"
COLS  = "id, societe_id, fournisseur, identifiant_externe, date_creation, actif"


def _get_or_404(db: Session, record_id: int):
    row = db.execute(
        text(f"SELECT {COLS} FROM {TABLE} WHERE id = :id"),
        {"id": record_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Identifiant fournisseur introuvable")
    return row


def _row_to_dict(row) -> dict:
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    keys = ["id", "societe_id", "fournisseur", "identifiant_externe", "date_creation", "actif"]
    return dict(zip(keys, row))


# ─── Liste par société ────────────────────────────────────────────────────────

@router.get(
    "/societes/{societe_id}/identifiants-fournisseur",
    response_model=list[SocieteIdentifiantFournisseurSchema],
    summary="Liste des identifiants fournisseurs d'une société de gestion",
)
def list_by_societe(societe_id: int, db: Session = Depends(get_db)):
    rows = db.execute(
        text(f"SELECT {COLS} FROM {TABLE} WHERE societe_id = :sid ORDER BY fournisseur"),
        {"sid": societe_id},
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ─── Résolution ───────────────────────────────────────────────────────────────

@router.get(
    "/societes/identifiants-fournisseur/resolve",
    summary="Résoudre un identifiant externe → societe_id CRM",
    response_model=dict,
)
def resolve(
    fournisseur: str = Query(..., description="Code fournisseur (ex: GENERALI)"),
    identifiant_externe: str = Query(..., description="Identifiant de la société chez ce fournisseur"),
    db: Session = Depends(get_db),
):
    row = db.execute(
        text(
            f"SELECT societe_id FROM {TABLE} "
            "WHERE fournisseur = :f AND identifiant_externe = :id AND actif = 1"
        ),
        {"f": fournisseur.strip().upper(), "id": identifiant_externe.strip()},
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Aucune société trouvée pour cet identifiant fournisseur",
        )
    sid = row[0] if not hasattr(row, "_mapping") else row._mapping["societe_id"]
    return {
        "societe_id": sid,
        "fournisseur": fournisseur.upper(),
        "identifiant_externe": identifiant_externe,
    }


# ─── Détail ───────────────────────────────────────────────────────────────────

@router.get(
    "/societes/identifiants-fournisseur/{record_id}",
    response_model=SocieteIdentifiantFournisseurSchema,
    summary="Détail d'un identifiant fournisseur",
)
def get_one(record_id: int, db: Session = Depends(get_db)):
    return _row_to_dict(_get_or_404(db, record_id))


# ─── Création ─────────────────────────────────────────────────────────────────

@router.post(
    "/societes/identifiants-fournisseur",
    response_model=SocieteIdentifiantFournisseurSchema,
    status_code=201,
    summary="Créer une correspondance société ↔ fournisseur",
)
def create(payload: SocieteIdentifiantFournisseurCreate, db: Session = Depends(get_db)):
    exists = db.execute(
        text("SELECT id FROM mariadb_societe_gestion WHERE id = :id"),
        {"id": payload.societe_id},
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Société {payload.societe_id} introuvable")

    conflict = db.execute(
        text(f"SELECT id, societe_id FROM {TABLE} WHERE fournisseur = :f AND identifiant_externe = :id"),
        {"f": payload.fournisseur, "id": payload.identifiant_externe},
    ).fetchone()
    if conflict:
        sid = conflict[1] if not hasattr(conflict, "_mapping") else conflict._mapping["societe_id"]
        raise HTTPException(
            status_code=409,
            detail=f"Cet identifiant externe est déjà attribué à la société {sid} chez {payload.fournisseur}",
        )

    db.execute(
        text(
            f"INSERT INTO {TABLE} (societe_id, fournisseur, identifiant_externe, actif) "
            "VALUES (:sid, :f, :id, :actif)"
        ),
        {
            "sid": payload.societe_id,
            "f": payload.fournisseur,
            "id": payload.identifiant_externe,
            "actif": payload.actif,
        },
    )
    db.commit()
    new_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()
    return _row_to_dict(_get_or_404(db, new_id))


# ─── Mise à jour ──────────────────────────────────────────────────────────────

@router.put(
    "/societes/identifiants-fournisseur/{record_id}",
    response_model=SocieteIdentifiantFournisseurSchema,
    summary="Mettre à jour une correspondance",
)
def update(record_id: int, payload: SocieteIdentifiantFournisseurUpdate, db: Session = Depends(get_db)):
    _get_or_404(db, record_id)
    updates = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Aucun champ à mettre à jour")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["record_id"] = record_id
    db.execute(text(f"UPDATE {TABLE} SET {set_clause} WHERE id = :record_id"), updates)
    db.commit()
    return _row_to_dict(_get_or_404(db, record_id))


# ─── Suppression ──────────────────────────────────────────────────────────────

@router.delete(
    "/societes/identifiants-fournisseur/{record_id}",
    summary="Supprimer une correspondance",
)
def delete(record_id: int, db: Session = Depends(get_db)):
    _get_or_404(db, record_id)
    db.execute(text(f"DELETE FROM {TABLE} WHERE id = :id"), {"id": record_id})
    db.commit()
    return {"detail": "Supprimé"}
