"""API ESG d'un portefeuille : contrairement à l'API risque/performance, celle-ci lit une table
de référence locale à CRM_SAAS (esg_fonds_norm, synchronisée depuis CRM_ESG) — pas de donnée
client stockée, uniquement une lecture de référentiel de fonds.

Authentification par la même clé API partagée que les endpoints risque/performance (header
X-API-Key, variable d'environnement RISQUE_PERFORMANCE_API_KEY).

Endpoints :
  GET  /api/esg/metriques   – liste des métriques disponibles (code, libellé, sens, notable)
  POST /api/esg/calcul      – calcul ESG pondéré + notes A-G à partir d'un inventaire
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.esg_portefeuille import (
    METRIQUES_DISPONIBLES,
    MetriqueDisponible,
    CalculEsgRequest,
    CalculEsgResponse,
)
from src.services.esg_portefeuille import calculer_esg

router = APIRouter(prefix="/api/esg", tags=["esg-portefeuille"])


def _verifier_api_key(x_api_key: str | None) -> None:
    cle_attendue = os.getenv("RISQUE_PERFORMANCE_API_KEY")
    if not cle_attendue:
        raise HTTPException(status_code=503, detail="API non configurée (RISQUE_PERFORMANCE_API_KEY manquante)")
    if x_api_key != cle_attendue:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante")


@router.get(
    "/metriques",
    response_model=list[MetriqueDisponible],
    summary="Liste les métriques ESG disponibles pour le calcul de portefeuille",
)
async def esg_metriques_disponibles(x_api_key: str | None = Header(default=None)) -> list[MetriqueDisponible]:
    _verifier_api_key(x_api_key)
    return [
        MetriqueDisponible(code=code, **meta)
        for code, meta in METRIQUES_DISPONIBLES.items()
    ]


@router.post(
    "/calcul",
    response_model=CalculEsgResponse,
    summary="Calcule les métriques ESG pondérées et les notes A-G d'un portefeuille",
    description=(
        "Lit la dernière date de l'inventaire fourni (l'ESG est une photo, pas une série "
        "temporelle), calcule pour chaque métrique sélectionnée une valeur pondérée par la "
        "valorisation et une note A-G par rang en septile sur l'ensemble du référentiel "
        "esg_fonds_norm. Aucune donnée transmise n'est stockée."
    ),
)
async def esg_calcul(
    request: CalculEsgRequest,
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> CalculEsgResponse:
    _verifier_api_key(x_api_key)
    try:
        return calculer_esg(db, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
