"""Schémas pour l'analyse secteur/pays d'un portefeuille multi-fonds : combine le poids de chaque
fonds détenu (dernière date de l'inventaire fourni par l'appelant) avec la répartition secteur/pays
interne de chaque fonds, lue chez CRM_ESG (GET /api/stats/sector-breakdown, /country-breakdown —
endpoints publics, aucune authentification requise côté CRM_ESG), chaque fonds étant analysé à SA
PROPRE dernière période ESG connue (pas de date commune imposée à tout le portefeuille).

Un fonds détenu peut être absent du référentiel CRM_ESG (composition inconnue) : il est alors
exclu de l'agrégation globale et signalé dans fonds_non_couverts, sans faire échouer la requête
(pattern de couverture partielle, cf. taux_couverture_pct de esg_portefeuille.py).
"""
from __future__ import annotations

from datetime import date as date_type
from pydantic import BaseModel, Field

from src.schemas.risque_performance import InventaireHebdoLigne


class CalculSecteurPaysRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    inventaire: list[InventaireHebdoLigne] = Field(..., description="Inventaire multi-fonds ; seule la dernière date est utilisée pour pondérer les fonds détenus")


class RepartitionPoste(BaseModel):
    libelle: str = Field(..., description="Nom du secteur ou du pays")
    poids_pct: float = Field(..., description="Poids de ce secteur/pays, en %")


class FondNonCouvert(BaseModel):
    isin: str
    nom: str | None = None
    poids_pct: float = Field(..., description="Poids de ce fonds dans le portefeuille de l'utilisateur, en %, exclu de l'agrégation")
    motif: str = Field(..., description="Raison de l'exclusion (ex. composition inconnue côté CRM_ESG)")


class DetailFonds(BaseModel):
    isin: str
    nom: str | None = None
    poids_pct: float = Field(..., description="Poids de ce fonds dans le portefeuille de l'utilisateur, en %")
    periode_analyse: date_type = Field(..., description="Dernière période ESG connue pour CE fonds côté CRM_ESG (propre à chaque fonds)")
    secteurs: list[RepartitionPoste] = Field(..., description="Répartition sectorielle interne de ce fonds (totalise 100%)")
    pays: list[RepartitionPoste] = Field(..., description="Répartition géographique interne de ce fonds (totalise 100%)")


class CalculSecteurPaysResponse(BaseModel):
    identifiant: str | None = None
    date_analyse_portefeuille: date_type = Field(..., description="Dernière date de l'inventaire fourni, utilisée pour pondérer les fonds détenus")
    hypotheses_appliquees: list[str] = Field(default=[], description="Tous les choix/paramètres retenus pour ce calcul précis")
    taux_couverture_pct: float = Field(..., description="Part du portefeuille (en valorisation) couverte par un fonds connu de CRM_ESG")
    secteurs_globaux: list[RepartitionPoste] = Field(..., description="Répartition sectorielle globale du portefeuille (renormalisée à 100% sur la seule assiette couverte)")
    pays_globaux: list[RepartitionPoste] = Field(..., description="Répartition géographique globale du portefeuille (renormalisée à 100% sur la seule assiette couverte)")
    fonds_non_couverts: list[FondNonCouvert] = Field(default=[], description="Fonds détenus dont la composition est inconnue de CRM_ESG, exclus de l'agrégation")
    detail_par_fonds: list[DetailFonds] = Field(default=[], description="Répartition secteur/pays propre à chaque fonds couvert")
