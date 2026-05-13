"""
Routes FastAPI pour l'import de produits (contrats génériques).

Endpoints :
  POST /import/produits/preview  – aperçu sans écriture
  POST /import/produits/commit   – import effectif
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from src.database import get_db
from src.services.import_produits import commit_produits, parse_produits_csv, preview_produits

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import", tags=["import"])


@router.post("/produits/preview", summary="Aperçu import produits (sans écriture)")
async def produits_preview(
    request: Request,
    file: Optional[UploadFile] = File(None),
    id_ctg: int = Form(1),
    id_societe: Optional[int] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Fournissez un fichier via multipart/form-data (champ 'file').",
        )
    data = await file.read()
    try:
        rows = parse_produits_csv(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing : {exc}")
    if not rows:
        raise HTTPException(status_code=400, detail="Aucun produit trouvé dans le fichier.")
    return preview_produits(db, rows, id_ctg, id_societe)


@router.post("/produits/commit", summary="Import produits (écriture effective)")
async def produits_commit(
    request: Request,
    file: Optional[UploadFile] = File(None),
    id_ctg: int = Form(1),
    id_societe: Optional[int] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Fournissez un fichier via multipart/form-data (champ 'file').",
        )
    data = await file.read()
    try:
        rows = parse_produits_csv(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing : {exc}")
    if not rows:
        raise HTTPException(status_code=400, detail="Fichier vide ou non parseable.")
    return commit_produits(db, rows, id_ctg, id_societe)
