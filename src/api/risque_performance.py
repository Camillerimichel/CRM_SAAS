"""API stateless de calcul financier : performance (Dietz/TWR), indicateur de risque
hebdomadaire et rémunération courtier. Aucune donnée n'est lue ni écrite en base — l'appelant
fournit l'inventaire et les mouvements, l'API calcule et renvoie le résultat.

Destinée à être appelée par un site externe indépendant (démo) : authentification par clé
d'API partagée (header X-API-Key), pas par le système de session/RBAC du dashboard CRM_SAAS.

Endpoints :
  POST /api/risque-performance/calcul             – performance Dietz/TWR + indicateur de risque hebdo (inventaire+mouvements par fonds)
  POST /api/risque-performance/calcul-consolide    – idem, à partir d'une série déjà consolidée par date (valorisation + mouvement net), sans détail par fonds
  POST /api/remuneration/calcul                    – rémunération courtier (rétrocession + commission de gestion)
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from src.schemas.risque_performance import (
    CalculPerformanceRisqueRequest,
    CalculPerformanceRisqueResponse,
    CalculPerformanceRisqueConsolideRequest,
    CalculRemunerationRequest,
    CalculRemunerationResponse,
)
from src.services.risque_performance import (
    calculer_performance_risque,
    calculer_performance_risque_consolide,
    calculer_remuneration,
)

router = APIRouter(prefix="/api", tags=["risque-performance"])


def _verifier_api_key(x_api_key: str | None) -> None:
    cle_attendue = os.getenv("RISQUE_PERFORMANCE_API_KEY")
    if not cle_attendue:
        raise HTTPException(status_code=503, detail="API non configurée (RISQUE_PERFORMANCE_API_KEY manquante)")
    if x_api_key != cle_attendue:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante")


@router.post(
    "/risque-performance/calcul",
    response_model=CalculPerformanceRisqueResponse,
    summary="Calcule la performance (Dietz/TWR) et l'indicateur de risque hebdomadaire d'un portefeuille",
    description=(
        "Calculateur stateless : rien n'est lu ni écrit en base. L'appelant fournit un inventaire "
        "hebdomadaire multi-fonds (isin, nbuc, vl) et, optionnellement, des mouvements (nb_uc signé "
        "via une table de libellés) et une table de frais de gestion annuels. La réponse renvoie, "
        "semaine par semaine, la VL équivalente Dietz et Time-Weighted, la volatilité annualisée "
        "glissante (52 semaines) et deux classes de risque (classe_risque_a, classe_risque_b) "
        "dérivées uniquement de cette volatilité — ce ne sont PAS les indicateurs réglementaires "
        "SRRI/SRI au sens strict (pas de bootstrap VaR, pas de Market/Credit Risk Measure) ; voir "
        "methodologie_url dans la réponse pour le détail des choix retenus."
    ),
)
async def risque_performance_calcul(
    request: CalculPerformanceRisqueRequest,
    x_api_key: str | None = Header(default=None),
) -> CalculPerformanceRisqueResponse:
    _verifier_api_key(x_api_key)
    try:
        return calculer_performance_risque(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/risque-performance/calcul-consolide",
    response_model=CalculPerformanceRisqueResponse,
    summary="Calcule la performance (Dietz/TWR) et l'indicateur de risque hebdomadaire à partir d'une série déjà consolidée",
    description=(
        "Calculateur stateless : rien n'est lu ni écrit en base. Variante de /risque-performance/calcul "
        "pour les appelants qui disposent déjà, par leur propre système, d'une série consolidée par "
        "date (valorisation totale du portefeuille + mouvement net signé de la période), sans détail "
        "par fonds (isin/nbuc/vl). Évite de reproduire côté appelant la reconstitution "
        "inventaire+mouvements → valorisation, au prix d'une approximation du TWR (flux net supposé "
        "survenu en fin de période, faute de détail par fonds pour le revaloriser précisément) — voir "
        "hypotheses_appliquees dans la réponse pour le détail des conventions retenues. Aucune "
        "détection de frais net/brut (statut_frais toujours vide, mécanisme intrinsèquement par-isin)."
    ),
)
async def risque_performance_calcul_consolide(
    request: CalculPerformanceRisqueConsolideRequest,
    x_api_key: str | None = Header(default=None),
) -> CalculPerformanceRisqueResponse:
    _verifier_api_key(x_api_key)
    try:
        return calculer_performance_risque_consolide(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/remuneration/calcul",
    response_model=CalculRemunerationResponse,
    summary="Calcule la rémunération courtier (rétrocession par fonds + commissions de gestion fonds euros et UC)",
    description=(
        "Calculateur stateless : rien n'est lu ni écrit en base. Reconstitue le nombre d'UC de "
        "chaque fonds à partir du premier nbuc observé et des mouvements signés (indépendamment de "
        "la fréquence de reporting de chaque société), puis calcule trois composantes cumulatives, "
        "si commission_gestion_courtier est fourni : la rétrocession par fonds UC (table_retrocession, "
        "taux propre à chaque isin, part du courtier définie par taux_courtier), la commission de "
        "gestion sur l'encours fonds euros (taux_commission_fonds_euros_annuel) et la commission de "
        "gestion sur l'encours UC hors fonds euros (taux_commission_gestion_uc_annuel, additionnelle "
        "à la rétrocession) — classification fonds_euro/uc fournie par table_type_support."
    ),
)
async def remuneration_calcul(
    request: CalculRemunerationRequest,
    x_api_key: str | None = Header(default=None),
) -> CalculRemunerationResponse:
    _verifier_api_key(x_api_key)
    try:
        return calculer_remuneration(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
