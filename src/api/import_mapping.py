"""
Routes FastAPI pour la normalisation de fichiers d'import via des profils de mapping.

Endpoints :
  GET  /import/mapping/page               – Page HTML wizard
  POST /import/mapping/analyser           – Analyse un fichier, détecte ses colonnes
  GET  /import/mapping/profils            – Liste les profils sauvegardés
  POST /import/mapping/profils            – Crée un profil
  PUT  /import/mapping/profils/{id}       – Met à jour un profil
  DELETE /import/mapping/profils/{id}     – Supprime (désactive) un profil
  POST /import/mapping/appliquer/{type}/preview  – Applique le profil + aperçu
  POST /import/mapping/appliquer/{type}/commit   – Applique le profil + import effectif
"""
from __future__ import annotations

import logging

import json as _json

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.database import get_db
from src.api.main import templates
from src.services.import_mapping import (
    CHAMPS_ATTENDUS,
    analyser_fichier,
    appliquer_profil,
    creer_profil,
    get_cached_rows,
    lister_profils,
    maj_profil,
    suggerer_mapping,
    supprimer_profil,
)
from src.services.import_inventaire import preview_inventaire, commit_inventaire
from src.services.import_mouvements import preview_mouvements, commit_mouvements, iter_commit_mouvements
from src.services.import_avis import preview_avis, commit_avis
from src.database import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/import/mapping", tags=["import-mapping"])

_TYPES_SUPPORTES_PREVIEW = {"inventaire", "mouvements", "avis"}
_TYPES_SUPPORTES_COMMIT = {"inventaire", "mouvements", "avis"}


# ──────────────────────────────────────────────────────────────────────────────
# Page HTML
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/page", response_class=HTMLResponse)
def page_mapping(request: Request, db: Session = Depends(get_db)):
    profils = lister_profils(db)
    return templates.TemplateResponse(
        "dashboard_import_mapping.html",
        {
            "request": request,
            "types_import": list(CHAMPS_ATTENDUS.keys()),
            "profils": profils,
            "champs_attendus_json": __import__("json").dumps(CHAMPS_ATTENDUS),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Analyse fichier
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/analyser")
async def analyser(
    file: UploadFile = File(...),
    type_import: str = Query(...),
    db: Session = Depends(get_db),
):
    data = await file.read()
    result = analyser_fichier(data, file.filename or "")
    if not result["colonnes"]:
        raise HTTPException(status_code=400, detail="Aucune colonne détectée dans le fichier.")
    result["suggestions"] = suggerer_mapping(result["colonnes"], type_import)
    result["champs_attendus"] = CHAMPS_ATTENDUS.get(type_import, {})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CRUD profils
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/profils")
def liste_profils(
    type_import: str | None = Query(None),
    fournisseur: str | None = Query(None),
    db: Session = Depends(get_db),
):
    return lister_profils(db, type_import=type_import, fournisseur=fournisseur)


@router.post("/profils")
async def creer(
    request: Request,
    db: Session = Depends(get_db),
):
    body = await request.json()
    try:
        p = creer_profil(
            db,
            nom=body["nom"],
            type_import=body["type_import"],
            mapping=body["mapping"],
            fournisseur=body.get("fournisseur") or None,
            transformations=body.get("transformations"),
            valeurs_fixes=body.get("valeurs_fixes"),
        )
        return p
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Champ manquant : {exc}")


@router.put("/profils/{profil_id}")
async def modifier(
    profil_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    body = await request.json()
    p = maj_profil(
        db,
        profil_id,
        nom=body["nom"],
        type_import=body["type_import"],
        mapping=body["mapping"],
        fournisseur=body.get("fournisseur") or None,
        transformations=body.get("transformations"),
        valeurs_fixes=body.get("valeurs_fixes"),
    )
    if not p:
        raise HTTPException(status_code=404, detail="Profil introuvable")
    return p


@router.delete("/profils/{profil_id}")
def supprimer(
    profil_id: int,
    db: Session = Depends(get_db),
):
    if not supprimer_profil(db, profil_id):
        raise HTTPException(status_code=404, detail="Profil introuvable")
    return {"detail": "Profil supprimé"}


# ──────────────────────────────────────────────────────────────────────────────
# Appliquer un profil → preview / commit
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/appliquer/{type_import}/preview")
async def appliquer_preview(
    type_import: str,
    request: Request,
    session_id: str = Query(...),
    fournisseur: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if type_import not in _TYPES_SUPPORTES_PREVIEW:
        raise HTTPException(status_code=400, detail=f"Type '{type_import}' non supporté pour l'aperçu (valeurs : {', '.join(_TYPES_SUPPORTES_PREVIEW)}).")

    body = await request.json()
    raw_rows = get_cached_rows(session_id)
    if raw_rows is None:
        raise HTTPException(status_code=400, detail="Session expirée — veuillez re-charger le fichier.")

    rows = appliquer_profil(
        raw_rows,
        mapping=body.get("mapping", {}),
        transformations=body.get("transformations", {}),
        valeurs_fixes=body.get("valeurs_fixes", {}),
    )

    if type_import == "inventaire":
        result = preview_inventaire(db, rows, fournisseur=fournisseur)
    elif type_import == "mouvements":
        result = preview_mouvements(db, rows, fournisseur=fournisseur)
    else:
        result = preview_avis(db, rows, fournisseur=fournisseur)

    return result


@router.post("/appliquer/{type_import}/stream-commit")
async def appliquer_stream_commit(
    type_import: str,
    request: Request,
    session_id: str = Query(...),
    fournisseur: str | None = Query(None),
    id_societe_gestion: int | None = Query(None),
    run_pipeline: bool = Query(True),
):
    """Commit avec progression SSE (text/event-stream).
    Chaque événement : data: {"type":"progress"|"done"|"error", ...}
    """
    if type_import not in _TYPES_SUPPORTES_COMMIT:
        raise HTTPException(status_code=400, detail=f"Type '{type_import}' non supporté.")

    body = await request.json()
    raw_rows = get_cached_rows(session_id)
    if raw_rows is None:
        raise HTTPException(status_code=400, detail="Session expirée — veuillez re-charger le fichier.")

    rows = appliquer_profil(
        raw_rows,
        mapping=body.get("mapping", {}),
        transformations=body.get("transformations", {}),
        valeurs_fixes=body.get("valeurs_fixes", {}),
    )

    def generate():
        db = SessionLocal()
        try:
            if type_import == "mouvements":
                gen = iter_commit_mouvements(
                    db, rows,
                    id_societe_gestion=id_societe_gestion,
                    fournisseur=fournisseur,
                    run_pipeline=run_pipeline,
                )
            else:
                # inventaire / avis : commit classique encapsulé en SSE
                def _wrap():
                    if type_import == "inventaire":
                        result = commit_inventaire(
                            db, rows,
                            id_societe_gestion=id_societe_gestion,
                            fournisseur=fournisseur,
                            run_pipeline=run_pipeline,
                        )
                    else:
                        result = commit_avis(
                            db, rows,
                            id_societe_gestion=id_societe_gestion,
                            fournisseur=fournisseur,
                        )
                    yield {'type': 'done', **result}
                gen = _wrap()

            for event in gen:
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/appliquer/{type_import}/commit")
async def appliquer_commit(
    type_import: str,
    request: Request,
    session_id: str = Query(...),
    fournisseur: str | None = Query(None),
    id_societe_gestion: int | None = Query(None),
    run_pipeline: bool = Query(True),
    db: Session = Depends(get_db),
):
    if type_import not in _TYPES_SUPPORTES_COMMIT:
        raise HTTPException(status_code=400, detail=f"Type '{type_import}' non supporté pour l'import (valeurs : {', '.join(_TYPES_SUPPORTES_COMMIT)}).")

    body = await request.json()
    raw_rows = get_cached_rows(session_id)
    if raw_rows is None:
        raise HTTPException(status_code=400, detail="Session expirée — veuillez re-charger le fichier.")

    rows = appliquer_profil(
        raw_rows,
        mapping=body.get("mapping", {}),
        transformations=body.get("transformations", {}),
        valeurs_fixes=body.get("valeurs_fixes", {}),
    )

    if type_import == "inventaire":
        result = commit_inventaire(
            db, rows,
            id_societe_gestion=id_societe_gestion,
            fournisseur=fournisseur,
            run_pipeline=run_pipeline,
        )
    elif type_import == "mouvements":
        result = commit_mouvements(
            db, rows,
            id_societe_gestion=id_societe_gestion,
            fournisseur=fournisseur,
            run_pipeline=run_pipeline,
        )
    else:
        result = commit_avis(
            db, rows,
            id_societe_gestion=id_societe_gestion,
            fournisseur=fournisseur,
        )

    return result
