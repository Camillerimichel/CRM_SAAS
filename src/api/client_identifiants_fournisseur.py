"""
Endpoints CRUD pour la table mariadb_client_identifiants_fournisseur.

Permet de gérer les correspondances entre identifiants externes fournisseurs
et clients CRM. Utilisé en amont de l'import de fichiers fournisseurs.

Endpoints :
  GET    /clients/{client_id}/identifiants-fournisseur        – liste par client
  GET    /identifiants-fournisseur/{id}                       – détail
  POST   /identifiants-fournisseur                            – création
  PUT    /identifiants-fournisseur/{id}                       – mise à jour
  DELETE /identifiants-fournisseur/{id}                       – suppression
  GET    /identifiants-fournisseur/resolve                    – résolution fournisseur → client_id
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.client_identifiant_fournisseur import (
    ClientIdentifiantFournisseurCreate,
    ClientIdentifiantFournisseurSchema,
    ClientIdentifiantFournisseurUpdate,
)

router = APIRouter(tags=["identifiants-fournisseur"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers DB
# ──────────────────────────────────────────────────────────────────────────────

def _get_or_404(db: Session, record_id: int):
    row = db.execute(
        text("SELECT id, client_id, fournisseur, identifiant_externe, date_creation, actif "
             "FROM mariadb_client_identifiants_fournisseur WHERE id = :id"),
        {"id": record_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Identifiant fournisseur introuvable")
    return row


def _row_to_dict(row) -> dict:
    m = row._mapping if hasattr(row, "_mapping") else None
    if m:
        return dict(m)
    return {
        "id": row[0], "client_id": row[1], "fournisseur": row[2],
        "identifiant_externe": row[3], "date_creation": row[4], "actif": row[5],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/clients/{client_id}/identifiants-fournisseur",
    response_model=list[ClientIdentifiantFournisseurSchema],
    summary="Liste des identifiants fournisseurs d'un client",
)
def list_by_client(client_id: int, db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            "SELECT id, client_id, fournisseur, identifiant_externe, date_creation, actif "
            "FROM mariadb_client_identifiants_fournisseur "
            "WHERE client_id = :cid ORDER BY fournisseur"
        ),
        {"cid": client_id},
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get(
    "/identifiants-fournisseur/resolve",
    summary="Résoudre un identifiant externe → client_id CRM",
    response_model=dict,
)
def resolve(
    fournisseur: str = Query(..., description="Code fournisseur (ex: GENERALI)"),
    identifiant_externe: str = Query(..., description="Identifiant du client chez ce fournisseur"),
    db: Session = Depends(get_db),
):
    row = db.execute(
        text(
            "SELECT client_id FROM mariadb_client_identifiants_fournisseur "
            "WHERE fournisseur = :f AND identifiant_externe = :id AND actif = 1"
        ),
        {"f": fournisseur.strip().upper(), "id": identifiant_externe.strip()},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Aucun client trouvé pour cet identifiant fournisseur")
    client_id = row[0] if not hasattr(row, "_mapping") else row._mapping["client_id"]
    return {"client_id": client_id, "fournisseur": fournisseur.upper(), "identifiant_externe": identifiant_externe}


@router.get(
    "/identifiants-fournisseur/{record_id}",
    response_model=ClientIdentifiantFournisseurSchema,
    summary="Détail d'un identifiant fournisseur",
)
def get_one(record_id: int, db: Session = Depends(get_db)):
    return _row_to_dict(_get_or_404(db, record_id))


@router.post(
    "/identifiants-fournisseur",
    response_model=ClientIdentifiantFournisseurSchema,
    status_code=201,
    summary="Créer une correspondance client ↔ fournisseur",
)
def create(payload: ClientIdentifiantFournisseurCreate, db: Session = Depends(get_db)):
    # Vérifier que le client existe
    exists = db.execute(
        text("SELECT id FROM mariadb_clients WHERE id = :id"), {"id": payload.client_id}
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Client {payload.client_id} introuvable")

    # Vérifier l'unicité (fournisseur, identifiant_externe)
    conflict = db.execute(
        text(
            "SELECT id, client_id FROM mariadb_client_identifiants_fournisseur "
            "WHERE fournisseur = :f AND identifiant_externe = :id"
        ),
        {"f": payload.fournisseur, "id": payload.identifiant_externe},
    ).fetchone()
    if conflict:
        cid = conflict[1] if not hasattr(conflict, "_mapping") else conflict._mapping["client_id"]
        raise HTTPException(
            status_code=409,
            detail=f"Cet identifiant externe est déjà attribué au client {cid} chez {payload.fournisseur}",
        )

    db.execute(
        text(
            "INSERT INTO mariadb_client_identifiants_fournisseur "
            "(client_id, fournisseur, identifiant_externe, actif) "
            "VALUES (:cid, :f, :id, :actif)"
        ),
        {
            "cid": payload.client_id,
            "f": payload.fournisseur,
            "id": payload.identifiant_externe,
            "actif": payload.actif,
        },
    )
    db.commit()

    new_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()
    return _row_to_dict(_get_or_404(db, new_id))


@router.put(
    "/identifiants-fournisseur/{record_id}",
    response_model=ClientIdentifiantFournisseurSchema,
    summary="Mettre à jour une correspondance",
)
def update(record_id: int, payload: ClientIdentifiantFournisseurUpdate, db: Session = Depends(get_db)):
    _get_or_404(db, record_id)

    updates = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Aucun champ à mettre à jour")

    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["record_id"] = record_id
    db.execute(
        text(f"UPDATE mariadb_client_identifiants_fournisseur SET {set_clause} WHERE id = :record_id"),
        updates,
    )
    db.commit()
    return _row_to_dict(_get_or_404(db, record_id))


@router.delete(
    "/identifiants-fournisseur/{record_id}",
    summary="Supprimer une correspondance",
)
def delete(record_id: int, db: Session = Depends(get_db)):
    _get_or_404(db, record_id)
    db.execute(
        text("DELETE FROM mariadb_client_identifiants_fournisseur WHERE id = :id"),
        {"id": record_id},
    )
    db.commit()
    return {"detail": "Supprimé"}
