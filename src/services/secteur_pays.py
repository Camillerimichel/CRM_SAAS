"""Analyse secteur/pays d'un portefeuille multi-fonds : combine le poids de chaque fonds dans le
portefeuille de l'utilisateur (derniere_date_et_poids, réutilisé de esg_portefeuille.py) avec la
répartition secteur/pays interne de chaque fonds, lue via les endpoints publics CRM_ESG
(GET /api/stats/sector-breakdown, /country-breakdown), chaque fonds étant analysé à SA PROPRE
dernière période ESG connue (year/month omis dans l'appel -> résolution automatique côté CRM_ESG).
Aucune donnée client transmise à CRM_ESG au-delà de l'ISIN de chaque fonds détenu.
"""
from __future__ import annotations

import os
from collections import defaultdict

import httpx
from sqlalchemy.orm import Session

from src.services.esg_portefeuille import derniere_date_et_poids, noms_fonds_par_isin
from src.schemas.secteur_pays import (
    CalculSecteurPaysRequest,
    CalculSecteurPaysResponse,
    DetailFonds,
    FondNonCouvert,
    RepartitionPoste,
)

CRM_ESG_API_BASE = os.getenv("CRM_ESG_API_BASE", "https://esgnote.eu").rstrip("/")
CRM_ESG_TIMEOUT = float(os.getenv("CRM_ESG_API_TIMEOUT", "15"))


def _appeler_breakdown(client: httpx.Client, chemin: str, isin: str) -> dict | None:
    """None si le fonds est inconnu côté CRM_ESG (ou toute autre erreur 4xx/5xx) plutôt que de
    faire échouer toute l'analyse — pattern de couverture partielle. Ne lève une exception que
    sur un problème réseau (CRM_ESG injoignable)."""
    try:
        reponse = client.get(chemin, params={"isin": isin}, timeout=CRM_ESG_TIMEOUT)
    except httpx.RequestError as exc:
        raise RuntimeError(f"CRM_ESG injoignable : {exc}") from exc
    if reponse.status_code != 200:
        return None
    return reponse.json()


def calculer_secteur_pays(db: Session, request: CalculSecteurPaysRequest) -> CalculSecteurPaysResponse:
    derniere_date, poids_par_isin = derniere_date_et_poids(request.inventaire)
    isins = list(poids_par_isin.keys())
    noms = noms_fonds_par_isin(db, isins)

    secteurs_bruts: dict[str, float] = defaultdict(float)
    pays_bruts: dict[str, float] = defaultdict(float)
    fonds_non_couverts: list[FondNonCouvert] = []
    detail_par_fonds: list[DetailFonds] = []
    poids_couvert_total = 0.0

    with httpx.Client(base_url=CRM_ESG_API_BASE) as client:
        for isin in isins:
            poids_fonds = poids_par_isin[isin]
            data_secteur = _appeler_breakdown(client, "/api/stats/sector-breakdown", isin)
            data_pays = _appeler_breakdown(client, "/api/stats/country-breakdown", isin) if data_secteur else None

            if data_secteur is None or data_pays is None:
                fonds_non_couverts.append(FondNonCouvert(
                    isin=isin,
                    nom=noms.get(isin),
                    poids_pct=round(poids_fonds * 100, 2),
                    motif="Composition inconnue côté CRM_ESG (portefeuille introuvable)" if data_secteur is None
                          else "Répartition pays indisponible côté CRM_ESG",
                ))
                continue

            poids_couvert_total += poids_fonds

            secteurs_fonds = [
                RepartitionPoste(libelle=s["sector"], poids_pct=s["fund_weight"])
                for s in data_secteur["sectors"]
            ]
            pays_fonds = [
                RepartitionPoste(libelle=c["country"], poids_pct=c["fund_weight"])
                for c in data_pays["countries"]
            ]
            for s in secteurs_fonds:
                secteurs_bruts[s.libelle] += poids_fonds * (s.poids_pct / 100)
            for c in pays_fonds:
                pays_bruts[c.libelle] += poids_fonds * (c.poids_pct / 100)

            nom_fonds = (noms.get(isin) or data_secteur["fund"]["name"] or "").strip() or None
            detail_par_fonds.append(DetailFonds(
                isin=isin,
                nom=nom_fonds,
                poids_pct=round(poids_fonds * 100, 2),
                periode_analyse=data_secteur["periode_analyse"],
                secteurs=sorted(secteurs_fonds, key=lambda r: -r.poids_pct),
                pays=sorted(pays_fonds, key=lambda r: -r.poids_pct),
            ))

    diviseur = poids_couvert_total or 1.0
    secteurs_globaux = sorted(
        (RepartitionPoste(libelle=k, poids_pct=round((v / diviseur) * 100, 2)) for k, v in secteurs_bruts.items()),
        key=lambda r: -r.poids_pct,
    )
    pays_globaux = sorted(
        (RepartitionPoste(libelle=k, poids_pct=round((v / diviseur) * 100, 2)) for k, v in pays_bruts.items()),
        key=lambda r: -r.poids_pct,
    )

    hypotheses = [
        f"Poids de chaque fonds détenu calculé à la dernière date de l'inventaire fourni ({derniere_date}).",
        "Chaque fonds est analysé à SA PROPRE dernière période ESG connue côté CRM_ESG (pas de date "
        "commune imposée à tout le portefeuille) : un fonds en retard n'empêche pas l'analyse des autres — "
        "voir periode_analyse par fonds dans detail_par_fonds.",
        "Un fonds détenu dont la composition est inconnue de CRM_ESG est exclu de la répartition globale "
        "(voir fonds_non_couverts) ; les répartitions globales sont renormalisées à 100% sur la seule "
        "assiette couverte (voir taux_couverture_pct).",
    ]

    return CalculSecteurPaysResponse(
        identifiant=request.identifiant,
        date_analyse_portefeuille=derniere_date,
        hypotheses_appliquees=hypotheses,
        taux_couverture_pct=round(poids_couvert_total * 100, 2),
        secteurs_globaux=secteurs_globaux,
        pays_globaux=pays_globaux,
        fonds_non_couverts=fonds_non_couverts,
        detail_par_fonds=detail_par_fonds,
    )
