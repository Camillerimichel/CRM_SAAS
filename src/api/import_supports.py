"""Routes FastAPI pour l'import de supports financiers.

Endpoints :
  POST /import/supports/detect   – détecte les colonnes, retourne aperçu 5 lignes (sans écriture)
  POST /import/supports/preview  – aperçu complet avec statuts DB (sans écriture)
  POST /import/supports/commit   – import effectif
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from src.database import get_db
from src.services.import_supports import commit_supports, detect_supports, preview_supports

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import", tags=["import"])


@router.post("/supports/detect", summary="Détecte les colonnes du fichier supports")
async def supports_detect(
    file: Optional[UploadFile] = File(None),
) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="Fournissez un fichier.")
    data = await file.read()
    try:
        return detect_supports(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/supports/preview", summary="Aperçu import supports (sans écriture)")
async def supports_preview(
    file: Optional[UploadFile] = File(None),
    col_isin: int = Form(...),
    col_libelle: int = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="Fournissez un fichier.")
    data = await file.read()
    try:
        return preview_supports(db, data, col_isin, col_libelle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/supports/commit", summary="Import supports (écriture effective)")
async def supports_commit(
    file: Optional[UploadFile] = File(None),
    col_isin: int = Form(...),
    col_libelle: int = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="Fournissez un fichier.")
    data = await file.read()
    try:
        return commit_supports(db, data, col_isin, col_libelle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
