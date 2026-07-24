"""Calcul ESG d'un portefeuille : lit la table de référence esg_fonds_norm (locale à CRM_SAAS,
déjà synchronisée depuis CRM_ESG — cf. src/services/esg_import.py) pour calculer, à partir de
l'inventaire fourni (dernière date uniquement, l'ESG est une photo pas une série temporelle), une
valeur pondérée et une note A-G pour chaque métrique sélectionnée : échelle publique A-E à seuils
absolus pour note_esg/e/s/g (cf. METRIQUES_ECHELLE_ABSOLUE ci-dessous), rang par septile pour les
autres métriques notables faute d'échelle absolue publiée.

Contrairement aux endpoints risque/performance, celui-ci lit une donnée de référence partagée en
base (aucune donnée client n'est stockée ni lue, uniquement le référentiel de fonds).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date as date_type

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from src.schemas.risque_performance import InventaireHebdoLigne
from src.schemas.esg_portefeuille import (
    METRIQUES_DISPONIBLES,
    GRADE_LETTERS,
    CalculEsgRequest,
    CalculEsgResponse,
    EsgMetriqueResultat,
    EsgHoldingLigne,
)
from src.services.esg_import import _grade_for_score, GRADE_COLUMNS

METHODOLOGIE_URL = "/methodologie"
TABLE_ESG = "esg_fonds_norm"

# note_esg/e/s/g sont les scores normalisés 0-1 décrits par l'échelle publique A-E de la page
# Méthodologie (seuils absolus : A≥0,80, B≥0,65, C≥0,45, D≥0,30, E sinon — cf. _grade_for_score
# dans esg_import.py, même barème que celui appliqué à esg_fonds). Toutes les autres métriques
# "notables" (intensité carbone en tCO2e, score de température en °C, ratio de rémunération, etc.)
# n'ont pas d'échelle absolue publiée : leur note A-G reste un rang relatif par septile sur le
# référentiel courant, pour ne pas inventer un seuil arbitraire.
METRIQUES_ECHELLE_ABSOLUE = set(GRADE_COLUMNS.keys())


def derniere_date_et_poids(inventaire: list[InventaireHebdoLigne]) -> tuple[date_type, dict[str, float]]:
    """Réutilisé par esg_exclusions.py — dernière date de l'inventaire et poids (0-1) par isin,
    calculés sur la valorisation de cette seule date (photo, pas de série temporelle)."""
    derniere_date = max(l.date for l in inventaire)
    lignes = [l for l in inventaire if l.date == derniere_date]
    valo_par_isin: dict[str, float] = defaultdict(float)
    for l in lignes:
        valo_par_isin[l.isin] += l.valo
    total = sum(valo_par_isin.values())
    poids = {isin: (v / total if total else 0.0) for isin, v in valo_par_isin.items()}
    return derniere_date, poids


def _septile_rangs(db: Session, metrique: str, sens_favorable: str) -> dict[str, int]:
    """Classe tous les fonds du référentiel par cette métrique et assigne un rang 1(A)-7(G) par
    septile, dans le sens indiqué (haut = valeur la plus élevée classée en tête / meilleur septile).
    Utilisé uniquement pour les métriques hors METRIQUES_ECHELLE_ABSOLUE (pas d'échelle publique
    absolue pour ces indicateurs)."""
    rows = db.execute(text(f"SELECT isin, `{metrique}` FROM {TABLE_ESG} WHERE `{metrique}` IS NOT NULL")).fetchall()
    valides = [(r[0], float(r[1])) for r in rows if r[0] and r[1] is not None]
    if not valides:
        return {}
    inverse = sens_favorable == "bas"
    ordered = sorted(valides, key=lambda item: (item[1] if inverse else -item[1], item[0]))
    total = len(ordered)
    base = total // len(GRADE_LETTERS)
    extra = total % len(GRADE_LETTERS)
    rangs: dict[str, int] = {}
    idx = 0
    for bucket in range(len(GRADE_LETTERS)):
        taille = base + (1 if bucket < extra else 0)
        for _ in range(taille):
            if idx >= total:
                break
            rangs[ordered[idx][0]] = bucket + 1  # 1=A ... 7=G
            idx += 1
    return rangs


def _valeurs_referentiel(db: Session, isins: list[str], metrique: str) -> dict[str, float]:
    if not isins:
        return {}
    rows = db.execute(
        text(f"SELECT isin, `{metrique}` FROM {TABLE_ESG} WHERE isin IN :isins AND `{metrique}` IS NOT NULL")
        .bindparams(bindparam("isins", expanding=True)),
        {"isins": isins},
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows if r[0] and r[1] is not None}


def noms_fonds_par_isin(db: Session, isins: list[str]) -> dict[str, str]:
    """Réutilisé par esg_exclusions.py."""
    if not isins:
        return {}
    rows = db.execute(
        text("SELECT code_isin, nom FROM mariadb_support WHERE code_isin IN :isins")
        .bindparams(bindparam("isins", expanding=True)),
        {"isins": isins},
    ).fetchall()
    return {r[0]: r[1] for r in rows if r[0] and r[1]}


def calculer_esg(db: Session, request: CalculEsgRequest) -> CalculEsgResponse:
    derniere_date, poids = derniere_date_et_poids(request.inventaire)
    isins = list(poids.keys())
    noms = noms_fonds_par_isin(db, isins)

    hypotheses = [
        f"Photo du portefeuille à la dernière date de l'inventaire fourni ({derniere_date}), pas de série temporelle.",
        "Note ESG globale, E, S et G : échelle publique A-E à seuils absolus (page Méthodologie — "
        "A≥0,80, B≥0,65, C≥0,45, D≥0,30, E sinon), appliquée à la valeur de chaque fonds puis à la "
        "valeur pondérée du portefeuille.",
        "Autres métriques notables : note A-G par rang en septile sur l'ensemble des fonds du "
        "référentiel esg_fonds_norm (A = meilleur septile, G = moins bon) — relative au référentiel "
        "courant, faute d'échelle absolue publiée pour ces indicateurs.",
        "Valeur pondérée = moyenne des valeurs des fonds détenus disposant de la donnée, pondérée par leur "
        "valorisation (le taux de couverture indique la part de la valorisation concernée).",
    ]

    metriques_non_notables = [m for m in request.metriques if not METRIQUES_DISPONIBLES[m]["notable"]]
    if metriques_non_notables:
        hypotheses.append(
            f"Pas de note A-G pour {', '.join(metriques_non_notables)} : le sens 'meilleur/moins bon' "
            "n'est pas clairement défini pour ces champs — valeur brute affichée uniquement."
        )

    resultats: list[EsgMetriqueResultat] = []
    valeurs_par_isin_metrique: dict[str, dict[str, float]] = {}
    notes_par_isin_metrique: dict[str, dict[str, str]] = {}

    for metrique in request.metriques:
        meta = METRIQUES_DISPONIBLES[metrique]
        valeurs = _valeurs_referentiel(db, isins, metrique)
        valeurs_par_isin_metrique[metrique] = valeurs
        echelle_absolue = metrique in METRIQUES_ECHELLE_ABSOLUE

        poids_valeur_total = sum(poids[isin] for isin in valeurs)
        valeur_ponderee = (
            sum(poids[isin] * v for isin, v in valeurs.items()) / poids_valeur_total
            if poids_valeur_total > 0 else None
        )

        note_grade = None
        if echelle_absolue:
            notes_par_isin_metrique[metrique] = {
                isin: grade for isin, v in valeurs.items() if (grade := _grade_for_score(v)) is not None
            }
            note_grade = _grade_for_score(valeur_ponderee) if valeur_ponderee is not None else None
        else:
            rangs: dict[str, int] = {}
            if meta["notable"]:
                rangs = _septile_rangs(db, metrique, meta["sens_favorable"])
            notes_par_isin_metrique[metrique] = {
                isin: GRADE_LETTERS[rangs[isin] - 1] for isin in valeurs if isin in rangs
            }
            if meta["notable"]:
                poids_rang_total = sum(poids[isin] for isin in valeurs if isin in rangs)
                if poids_rang_total > 0:
                    rang_pondere = sum(poids[isin] * rangs[isin] for isin in valeurs if isin in rangs) / poids_rang_total
                    rang_arrondi = max(1, min(7, round(rang_pondere)))
                    note_grade = GRADE_LETTERS[rang_arrondi - 1]

        resultats.append(
            EsgMetriqueResultat(
                code=metrique,
                libelle=meta["libelle"],
                est_pourcentage=meta["est_pourcentage"],
                unite=meta["unite"],
                valeur_ponderee=valeur_ponderee,
                note_grade=note_grade,
                taux_couverture_pct=round(poids_valeur_total * 100, 2),
            )
        )

    detail: list[EsgHoldingLigne] = []
    for isin in sorted(poids, key=lambda i: -poids[i]):
        detail.append(
            EsgHoldingLigne(
                isin=isin,
                nom=noms.get(isin),
                poids_pct=round(poids[isin] * 100, 4),
                valeurs={m: valeurs_par_isin_metrique[m].get(isin) for m in request.metriques},
                notes={m: notes_par_isin_metrique[m].get(isin) for m in request.metriques},
            )
        )

    return CalculEsgResponse(
        identifiant=request.identifiant,
        methodologie_url=METHODOLOGIE_URL,
        date_analyse=derniere_date,
        hypotheses_appliquees=hypotheses,
        resultats_metriques=resultats,
        detail_par_ligne=detail,
    )
