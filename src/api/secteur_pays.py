"""API d'analyse secteur/pays d'un portefeuille multi-fonds. Appelle en interne les endpoints
publics CRM_ESG (sector/country-breakdown, sans authentification côté CRM_ESG) pour chaque fonds
détenu, à sa propre dernière période ESG connue. Aucune donnée client stockée.

Authentification identique aux autres endpoints stateless (header X-API-Key,
RISQUE_PERFORMANCE_API_KEY).

Endpoint :
  POST /api/secteur-pays/calcul – répartition secteur/pays globale du portefeuille + détail par fonds
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.schemas.secteur_pays import CalculSecteurPaysRequest, CalculSecteurPaysResponse
from src.services.secteur_pays import calculer_secteur_pays

router = APIRouter(prefix="/api/secteur-pays", tags=["secteur-pays"])


def _verifier_api_key(x_api_key: str | None) -> None:
    cle_attendue = os.getenv("RISQUE_PERFORMANCE_API_KEY")
    if not cle_attendue:
        raise HTTPException(status_code=503, detail="API non configurée (RISQUE_PERFORMANCE_API_KEY manquante)")
    if x_api_key != cle_attendue:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante")


@router.post(
    "/calcul",
    response_model=CalculSecteurPaysResponse,
    summary="Calcule la répartition secteur/pays globale d'un portefeuille et le détail par fonds",
    description=(
        "Pondère chaque fonds détenu par son poids à la dernière date de l'inventaire fourni, puis "
        "combine avec la répartition secteur/pays interne de chaque fonds (lue chez CRM_ESG à SA "
        "PROPRE dernière période connue). Un fonds sans composition connue côté CRM_ESG est exclu "
        "de l'agrégation globale (couverture partielle) sans faire échouer la requête."
    ),
)
async def secteur_pays_calcul(
    request: CalculSecteurPaysRequest,
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> CalculSecteurPaysResponse:
    _verifier_api_key(x_api_key)
    try:
        return calculer_secteur_pays(db, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
