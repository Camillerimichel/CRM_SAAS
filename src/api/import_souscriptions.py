"""Routes FastAPI pour l'import de souscriptions.

Endpoints :
  POST /import/souscriptions/preview        – aperçu 5 lignes + stats (sans écriture)
  POST /import/souscriptions/commit         – import effectif
  POST /import/souscriptions/creer-produit  – crée un produit manquant à la volée
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import get_db
from src.services.import_souscriptions import (
    commit_souscriptions,
    parse_souscriptions_csv,
    preview_souscriptions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import", tags=["import"])


@router.post("/souscriptions/preview", summary="Aperçu import souscriptions (sans écriture)")
async def souscriptions_preview(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="Fournissez un fichier via multipart/form-data.")
    data = await file.read()
    try:
        rows = parse_souscriptions_csv(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not rows:
        raise HTTPException(status_code=400, detail="Aucune souscription trouvée dans le fichier.")
    return preview_souscriptions(db, rows)


@router.post("/souscriptions/commit", summary="Import souscriptions (écriture effective)")
async def souscriptions_commit(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="Fournissez un fichier via multipart/form-data.")
    data = await file.read()
    try:
        rows = parse_souscriptions_csv(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not rows:
        raise HTTPException(status_code=400, detail="Fichier vide ou non parseable.")
    return commit_souscriptions(db, rows)


@router.post("/souscriptions/creer-produit", summary="Crée un produit générique manquant")
async def souscriptions_creer_produit(
    code: str = Form(...),
    nom: str = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    code = code.strip()
    nom  = nom.strip()
    if not code or not nom:
        raise HTTPException(status_code=400, detail="Code et nom sont obligatoires.")

    existing = db.execute(
        text("SELECT id FROM mariadb_affaires_generique WHERE description = :code LIMIT 1"),
        {"code": code},
    ).fetchone()
    if existing:
        pid = existing._mapping["id"] if hasattr(existing, "_mapping") else existing[0]
        return {"created": False, "id": pid, "message": f"Produit '{code}' déjà existant."}

    next_id = db.execute(
        text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_affaires_generique")
    ).scalar()
    db.execute(
        text(
            "INSERT INTO mariadb_affaires_generique (id, nom_contrat, description, actif) "
            "VALUES (:id, :nom, :code, '1')"
        ),
        {"id": next_id, "nom": nom, "code": code},
    )
    db.commit()
    logger.info("SOUS creer-produit – id=%s code=%s nom=%s", next_id, code, nom)
    return {"created": True, "id": next_id, "message": f"Produit '{nom}' ({code}) créé."}
