"""
Routes FastAPI pour l'import de clients depuis fichiers fournisseurs.

Endpoints :
  POST /import/clients/preview  – aperçu sans écriture
  POST /import/clients/commit   – import effectif

Formats acceptés : CSV, XLS, XLSX, JSON (détection auto par extension).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.import_clients import ClientImportCommitResult, ClientImportPreviewResult
from src.services.import_clients import commit_clients, detect_and_parse, preview_clients

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import", tags=["import"])


@router.post(
    "/clients/preview",
    response_model=ClientImportPreviewResult,
    summary="Aperçu import clients (sans écriture)",
)
async def clients_preview(
    request: Request,
    file: Optional[UploadFile] = File(None),
    fournisseur: str = Query(..., description="Code fournisseur (ex: AFI ESCA FRANCE)"),
    db: Session = Depends(get_db),
) -> ClientImportPreviewResult:
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Fournissez un fichier via multipart/form-data (champ 'file').",
        )
    data = await file.read()
    try:
        raw_rows = detect_and_parse(file.filename or "", data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing : {exc}")
    return preview_clients(db, raw_rows, fournisseur=fournisseur)


@router.post(
    "/clients/commit",
    response_model=ClientImportCommitResult,
    summary="Import clients (écriture effective)",
)
async def clients_commit(
    request: Request,
    file: Optional[UploadFile] = File(None),
    fournisseur: str = Query(..., description="Code fournisseur (ex: AFI ESCA FRANCE)"),
    db: Session = Depends(get_db),
) -> ClientImportCommitResult:
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Fournissez un fichier via multipart/form-data (champ 'file').",
        )
    data = await file.read()
    try:
        raw_rows = detect_and_parse(file.filename or "", data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing : {exc}")
    if not raw_rows:
        raise HTTPException(status_code=400, detail="Fichier vide ou non parseable.")
    return commit_clients(db, raw_rows, fournisseur=fournisseur)
