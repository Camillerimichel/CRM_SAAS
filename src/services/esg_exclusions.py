"""Calcul de conformité aux exclusions ESG d'un portefeuille : reprend les 5 critères déjà
utilisés par src/services/esg_fund_exclusions.py::check_fund_exclusions, mais les critères sont
choisis directement par l'appelant plutôt que lus depuis un questionnaire client réel (qui
n'existe pas dans ce contexte de démo stateless). Lit esg_fonds_norm/esg_fonds (référentiel local,
aucune donnée client)."""
from __future__ import annotations

from datetime import date as date_type
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from src.schemas.esg_exclusions import (
    CRITERES_EXCLUSION,
    CalculExclusionsRequest,
    CalculExclusionsResponse,
    DivergenceExclusion,
    ExclusionFondsLigne,
)
from src.services.esg_portefeuille import derniere_date_et_poids, noms_fonds_par_isin

METHODOLOGIE_URL = "/methodologie"

# Notes considérées "faibles" pour le critère faible_note_esg — distribution uniforme A→G,
# donc E/F/G = tiers le moins bon (même convention que esg_fund_exclusions.py).
_FAIBLE_ESG_GRADES = {"E", "F", "G"}


def _binary_flag(val: Any) -> bool | None:
    if val is None:
        return None
    try:
        return float(val) >= 1.0
    except (TypeError, ValueError):
        return None


def _est_declenche(code: str, valeur: Any) -> bool:
    if code == "faible_note_esg":
        return valeur is not None and str(valeur).strip().upper() in _FAIBLE_ESG_GRADES
    return _binary_flag(valeur) is True


def _donnees_referentiel(db: Session, isins: list[str]) -> dict[str, dict]:
    if not isins:
        return {}
    rows = db.execute(
        text(
            """
            SELECT n.isin, n.exposure_to_fossil_fuels, n.sfdr_biodiversity_pai,
                   n.controversial_weapons, n.violations_ungc, f.note_esg_grade
            FROM esg_fonds_norm n
            LEFT JOIN esg_fonds f ON f.isin = n.isin
            WHERE n.isin IN :isins
            """
        ).bindparams(bindparam("isins", expanding=True)),
        {"isins": isins},
    ).fetchall()
    resultat: dict[str, dict] = {}
    for r in rows or []:
        m = r._mapping if hasattr(r, "_mapping") else r
        isin = str(m.get("isin") or "")
        if isin and isin not in resultat:
            resultat[isin] = dict(m)
    return resultat


def calculer_exclusions(db: Session, request: CalculExclusionsRequest) -> CalculExclusionsResponse:
    derniere_date, poids = derniere_date_et_poids(request.inventaire)
    isins = list(poids.keys())
    noms = noms_fonds_par_isin(db, isins)
    donnees = _donnees_referentiel(db, isins)

    hypotheses = [
        f"Photo du portefeuille à la dernière date de l'inventaire fourni ({derniere_date}).",
        "Critères d'exclusion choisis manuellement pour cette démo (pas de questionnaire client réel) "
        "— mêmes colonnes et seuils que le mécanisme de vérification utilisé dans CRM_SAAS.",
        "Un fonds sans donnée disponible pour un critère sélectionné est marqué 'donnée manquante' "
        "pour ce critère plutôt que présumé conforme ou non conforme.",
    ]

    criteres_selectionnes = [
        DivergenceExclusion(code=c, libelle=CRITERES_EXCLUSION[c]["libelle"]) for c in request.criteres
    ]

    fonds_result: list[ExclusionFondsLigne] = []
    for isin in sorted(poids, key=lambda i: -poids[i]):
        m = donnees.get(isin)
        divergences: list[DivergenceExclusion] = []
        has_missing = False

        if m is None:
            has_missing = True
        else:
            for code in request.criteres:
                colonne = CRITERES_EXCLUSION[code]["colonne"]
                valeur = m.get(colonne)
                if valeur is None:
                    has_missing = True
                elif _est_declenche(code, valeur):
                    divergences.append(DivergenceExclusion(code=code, libelle=CRITERES_EXCLUSION[code]["libelle"]))

        conforme: bool | None = None if (has_missing and not divergences) else (len(divergences) == 0)

        fonds_result.append(
            ExclusionFondsLigne(
                isin=isin,
                nom=noms.get(isin),
                poids_pct=round(poids[isin] * 100, 4),
                divergences=divergences,
                conforme=conforme,
                donnees_manquantes=has_missing,
            )
        )

    nb_fonds = len(fonds_result)
    nb_conformes = sum(1 for f in fonds_result if f.conforme is True)
    nb_non_conformes = sum(1 for f in fonds_result if f.conforme is False)
    nb_manquantes = sum(1 for f in fonds_result if f.donnees_manquantes and f.conforme is None)
    total_checkable = nb_conformes + nb_non_conformes
    taux = round(nb_conformes / total_checkable * 100, 2) if total_checkable > 0 else None

    poids_conforme = sum(poids[f.isin] for f in fonds_result if f.conforme is True)
    poids_non_conforme = sum(poids[f.isin] for f in fonds_result if f.conforme is False)

    return CalculExclusionsResponse(
        identifiant=request.identifiant,
        methodologie_url=METHODOLOGIE_URL,
        date_analyse=derniere_date,
        hypotheses_appliquees=hypotheses,
        criteres_selectionnes=criteres_selectionnes,
        fonds=fonds_result,
        nb_fonds=nb_fonds,
        nb_conformes=nb_conformes,
        nb_non_conformes=nb_non_conformes,
        nb_donnees_manquantes=nb_manquantes,
        taux_conformite_pct=taux,
        poids_conforme_pct=round(poids_conforme * 100, 2),
        poids_non_conforme_pct=round(poids_non_conforme * 100, 2),
    )
