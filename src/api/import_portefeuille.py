"""
Routes FastAPI pour l'import de fichiers fournisseurs de portefeuilles.

Endpoints :
  POST /import/inventaire/preview  – aperçu sans écriture
  POST /import/inventaire/commit   – import effectif
  POST /import/mouvements/preview  – aperçu sans écriture
  POST /import/mouvements/commit   – import effectif

Formats acceptés :
  - multipart/form-data avec champ « file » (CSV ou JSON)
  - application/json avec body directement (liste de rows ou {rows: [...]})
  - Query param ?format=csv|json (détection auto si absent)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.import_portefeuille import (
    ImportCommitResult,
    ImportPreviewResult,
)
from src.services.import_inventaire import (
    commit_inventaire,
    parse_inventaire_csv,
    parse_inventaire_json,
    preview_inventaire,
)
from src.services.import_mouvements import (
    commit_mouvements,
    parse_mouvements_csv,
    parse_mouvements_json,
    preview_mouvements,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import", tags=["import"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_format(filename: str | None, content_type: str | None, hint: str | None) -> str:
    if hint in ("csv", "json"):
        return hint
    if filename and filename.lower().endswith(".json"):
        return "json"
    if filename and filename.lower().endswith(".csv"):
        return "csv"
    if content_type and "json" in content_type:
        return "json"
    return "csv"  # default


async def _read_raw_rows(
    request: Request,
    file: UploadFile | None,
    fmt: str | None,
    parse_csv,
    parse_json,
) -> list[dict]:
    content_type = request.headers.get("content-type", "")

    if file is not None:
        data = await file.read()
        detected = _detect_format(file.filename, file.content_type, fmt)
    elif "application/json" in content_type or "text/plain" in content_type:
        data = await request.body()
        detected = "json" if "json" in content_type else (fmt or "csv")
    else:
        raise HTTPException(
            status_code=400,
            detail="Fournissez un fichier via multipart/form-data (champ 'file') ou un body JSON.",
        )

    try:
        if detected == "json":
            return parse_json(data)
        else:
            return parse_csv(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing : {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Inventaire
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/inventaire/preview",
    response_model=ImportPreviewResult,
    summary="Aperçu import inventaire (sans écriture)",
)
async def inventaire_preview(
    request: Request,
    file: Optional[UploadFile] = File(None),
    format: Optional[str] = Query(None, description="csv ou json (détection auto si absent)"),
    db: Session = Depends(get_db),
) -> ImportPreviewResult:
    raw_rows = await _read_raw_rows(
        request, file, format, parse_inventaire_csv, parse_inventaire_json
    )
    return preview_inventaire(db, raw_rows)


@router.post(
    "/inventaire/commit",
    response_model=ImportCommitResult,
    summary="Import inventaire (écriture effective + recalcul)",
)
async def inventaire_commit(
    request: Request,
    file: Optional[UploadFile] = File(None),
    format: Optional[str] = Query(None, description="csv ou json"),
    id_societe_gestion: Optional[int] = Query(None),
    id_client: Optional[int] = Query(None, description="ID du client (id_personne) à rattacher aux affaires créées à vide"),
    db: Session = Depends(get_db),
) -> ImportCommitResult:
    raw_rows = await _read_raw_rows(
        request, file, format, parse_inventaire_csv, parse_inventaire_json
    )
    if not raw_rows:
        raise HTTPException(status_code=400, detail="Fichier vide ou non parseable.")
    return commit_inventaire(db, raw_rows, id_societe_gestion=id_societe_gestion, id_personne=id_client, run_pipeline=False)


# ──────────────────────────────────────────────────────────────────────────────
# Mouvements
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/mouvements/preview",
    response_model=ImportPreviewResult,
    summary="Aperçu import mouvements (sans écriture)",
)
async def mouvements_preview(
    request: Request,
    file: Optional[UploadFile] = File(None),
    format: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> ImportPreviewResult:
    raw_rows = await _read_raw_rows(
        request, file, format, parse_mouvements_csv, parse_mouvements_json
    )
    return preview_mouvements(db, raw_rows)


@router.post(
    "/mouvements/commit",
    response_model=ImportCommitResult,
    summary="Import mouvements (écriture effective + recalcul PRMP + pipeline)",
)
async def mouvements_commit(
    request: Request,
    file: Optional[UploadFile] = File(None),
    format: Optional[str] = Query(None),
    id_societe_gestion: Optional[int] = Query(None),
    id_client: Optional[int] = Query(None, description="ID du client (id_personne) à rattacher aux affaires créées à vide"),
    db: Session = Depends(get_db),
) -> ImportCommitResult:
    raw_rows = await _read_raw_rows(
        request, file, format, parse_mouvements_csv, parse_mouvements_json
    )
    if not raw_rows:
        raise HTTPException(status_code=400, detail="Fichier vide ou non parseable.")
    return commit_mouvements(db, raw_rows, id_societe_gestion=id_societe_gestion, id_personne=id_client, run_pipeline=False)
