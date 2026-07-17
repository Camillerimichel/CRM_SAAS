"""Moteur de calcul stateless : performance (Dietz/TWR), indicateur de risque hebdomadaire
(dérivé uniquement de la volatilité historique, sans MRM/CRM) et rémunération courtier
(rétrocession par fonds + commission de gestion). Aucune lecture/écriture en base : tout est
calculé à partir des payloads fournis par l'appelant, rien n'est persisté.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date as date_type
from statistics import stdev
from math import sqrt

from src.schemas.risque_performance import (
    InventaireHebdoLigne,
    MouvementLigne,
    LibelleMouvementLigne,
    CalculPerformanceRisqueRequest,
    CalculPerformanceRisqueResponse,
    PerformanceRisqueLigne,
    StatutFraisIsin,
    CalculRemunerationRequest,
    CalculRemunerationResponse,
    RemunerationResultLigne,
    CommissionGestionLigne,
)

METHODOLOGIE_URL = "/methodologie"

FENETRE_VOLATILITE_SEMAINES = 52
SEMAINES_PAR_AN = 52

# Grilles de seuils appliquées à la même volatilité annualisée (fraction, ex 0.10 = 10%).
# Bornes façon SRRI (UCITS, CESR/10-673) — jamais nommées "SRRI" dans les libellés exposés.
SEUILS_CLASSE_A = [(0.005, 1), (0.02, 2), (0.05, 3), (0.10, 4), (0.15, 5), (0.25, 6)]
# Bornes façon Market Risk Measure / SRI (PRIIPs) — jamais nommées "SRI" dans les libellés exposés.
SEUILS_CLASSE_B = [(0.005, 1), (0.05, 2), (0.12, 3), (0.20, 4), (0.30, 5), (0.80, 6)]

TOLERANCE_DETECTION_FRAIS = 0.20   # ±20% autour de l'écart théorique attendu (1/12e du taux annuel)
SEMAINES_MIN_DETECTION_FRAIS = 4    # ~1 mois de données minimum pour tenter la détection


def _classe_risque(volatilite: float | None, seuils: list[tuple[float, int]]) -> int | None:
    if volatilite is None:
        return None
    for borne, classe in seuils:
        if volatilite < borne:
            return classe
    return 7


def _sens_mouvements(table_libelles: list[LibelleMouvementLigne]) -> dict[str, str]:
    return {ligne.libelle: ligne.sens for ligne in table_libelles}


def _signe(sens: str) -> int:
    return 1 if sens == "+" else (-1 if sens == "-" else 0)


def _calendrier(inventaire: list[InventaireHebdoLigne]) -> list[date_type]:
    return sorted({ligne.date for ligne in inventaire})


def _par_date_isin(inventaire: list[InventaireHebdoLigne]) -> dict[date_type, dict[str, InventaireHebdoLigne]]:
    resultat: dict[date_type, dict[str, InventaireHebdoLigne]] = defaultdict(dict)
    for ligne in inventaire:
        resultat[ligne.date][ligne.isin] = ligne
    return dict(resultat)


def _grille_forward_fill(
    inventaire: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    mouvements_par_semaine: dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]],
) -> dict[date_type, dict[str, tuple[float, float]]]:
    """Chaque fonds ne reporte pas forcément à toutes les dates du calendrier consolidé (funds
    entrant/sortant, sociétés différentes). Entre deux dates où l'inventaire donne une valeur
    exacte (vérité terrain, qui prime toujours), le nb_uc est ajusté par les mouvements survenus
    plutôt que simplement reporté à l'identique : un fonds intégralement sorti par arbitrage sans
    nouvelle ligne d'inventaire ensuite doit tomber à ~0, pas rester à sa dernière valeur connue
    (sinon double-comptage avec le fonds où la valeur a été transférée). Le vl, lui, est simplement
    reporté en avant (c'est un prix de marché, pas une quantité qui "s'épuise")."""
    par_isin: dict[str, dict[date_type, InventaireHebdoLigne]] = defaultdict(dict)
    for ligne in inventaire:
        par_isin[ligne.isin][ligne.date] = ligne

    grille: dict[date_type, dict[str, tuple[float, float]]] = defaultdict(dict)
    for isin, par_date in par_isin.items():
        premiere_date = min(par_date.keys())
        nb_uc_courant: float | None = None
        vl_courant: float | None = None
        for d in calendrier:
            if d < premiere_date:
                continue
            ligne_exacte = par_date.get(d)
            if ligne_exacte is not None:
                nb_uc_courant = ligne_exacte.nbuc
                vl_courant = ligne_exacte.vl
            else:
                for mouvement, signe in mouvements_par_semaine.get(d, {}).get(isin, []):
                    nb_uc_courant = (nb_uc_courant or 0.0) + signe * mouvement.nb_uc
            if nb_uc_courant is not None and vl_courant is not None:
                grille[d][isin] = (nb_uc_courant, vl_courant)
    return dict(grille)


def _mouvements_par_semaine_isin(
    mouvements: list[MouvementLigne],
    calendrier: list[date_type],
    sens_map: dict[str, str],
) -> dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]]:
    """Rattache chaque mouvement à la première date du calendrier >= sa date (la semaine où il
    prend effet), avec son signe résolu (+1/-1/0) via la table de libellés."""
    resultat: dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]] = defaultdict(lambda: defaultdict(list))
    for m in mouvements:
        semaine = next((d for d in calendrier if d >= m.date), None)
        if semaine is None:
            continue
        signe = _signe(sens_map.get(m.libelle, "na"))
        resultat[semaine][m.isin].append((m, signe))
    return resultat


def _valo_consolidee(grille: dict[date_type, dict[str, tuple[float, float]]]) -> dict[date_type, float]:
    return {d: sum(nbuc * vl for nbuc, vl in lignes.values()) for d, lignes in grille.items()}


def _reconstituer_nb_uc(
    inventaire: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    mouvements_par_semaine: dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]],
) -> dict[date_type, dict[str, float]]:
    """Pour chaque isin, part du premier nbuc observé puis l'ajuste à chaque mouvement, plutôt que
    de faire confiance au nbuc reporté chaque semaine (hétérogénéité de reporting entre sociétés)."""
    par_date_isin = _par_date_isin(inventaire)
    premiere_date_isin: dict[str, date_type] = {}
    for d in calendrier:
        for isin in par_date_isin.get(d, {}):
            premiere_date_isin.setdefault(isin, d)

    nb_uc_courant: dict[str, float] = {}
    resultat: dict[date_type, dict[str, float]] = defaultdict(dict)

    for d in calendrier:
        for isin, premiere in premiere_date_isin.items():
            if d < premiere:
                continue
            if d == premiere:
                nb_uc_courant[isin] = par_date_isin[d][isin].nbuc
            else:
                for _mouvement, signe in mouvements_par_semaine.get(d, {}).get(isin, []):
                    nb_uc_courant[isin] = nb_uc_courant.get(isin, 0.0) + signe * _mouvement.nb_uc
            resultat[d][isin] = nb_uc_courant.get(isin, 0.0)

    return dict(resultat)


def _vl_depuis_grille(grille: dict[date_type, dict[str, tuple[float, float]]]) -> dict[date_type, dict[str, float]]:
    return {d: {isin: vl for isin, (_nbuc, vl) in lignes.items()} for d, lignes in grille.items()}


def _nbuc_depuis_grille(grille: dict[date_type, dict[str, tuple[float, float]]]) -> dict[date_type, dict[str, float]]:
    return {d: {isin: nbuc for isin, (nbuc, _vl) in lignes.items()} for d, lignes in grille.items()}


def _detecter_statut_frais(
    isin: str,
    inventaire_isin: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    grille: dict[date_type, dict[str, tuple[float, float]]],
    mouvements_par_semaine: dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]],
    frais_gestion_annuel: float,
) -> StatutFraisIsin:
    lignes_triees = sorted(inventaire_isin, key=lambda l: l.date)
    dates_calendrier_isin = [d for d in calendrier if d >= lignes_triees[0].date]
    fenetre = dates_calendrier_isin[:SEMAINES_MIN_DETECTION_FRAIS + 1]

    if len(fenetre) < 2:
        return StatutFraisIsin(
            isin=isin,
            frais_deja_deduits=False,
            ecart_relatif_detecte=0.0,
            methode="Données insuffisantes (< 2 semaines) pour la détection automatique — traité par défaut comme brut",
        )

    nb_uc_depart = lignes_triees[0].nbuc
    date_depart, date_arrivee = fenetre[0], fenetre[-1]

    nb_uc_attendu = nb_uc_depart
    for d in fenetre[1:]:
        for mouvement, signe in mouvements_par_semaine.get(d, {}).get(isin, []):
            nb_uc_attendu += signe * mouvement.nb_uc

    nb_uc_observe_tuple = grille.get(date_arrivee, {}).get(isin)
    nb_uc_observe = nb_uc_observe_tuple[0] if nb_uc_observe_tuple else None
    if nb_uc_observe is None or nb_uc_attendu == 0:
        return StatutFraisIsin(
            isin=isin,
            frais_deja_deduits=False,
            ecart_relatif_detecte=0.0,
            methode="Observation manquante en fin de fenêtre — traité par défaut comme brut",
        )

    ecart_relatif = (nb_uc_attendu - nb_uc_observe) / nb_uc_attendu
    nb_semaines = len(fenetre) - 1
    attendu_theorique = (frais_gestion_annuel / 100) * (nb_semaines / SEMAINES_PAR_AN)

    deja_deduits = abs(ecart_relatif - attendu_theorique) <= TOLERANCE_DETECTION_FRAIS * max(attendu_theorique, 1e-9)

    return StatutFraisIsin(
        isin=isin,
        frais_deja_deduits=deja_deduits,
        ecart_relatif_detecte=round(ecart_relatif * 100, 4),
        methode=(
            f"Détection automatique sur {nb_semaines} semaine(s) ({date_depart} → {date_arrivee}) : "
            f"écart observé {ecart_relatif * 100:.3f}% vs attendu {attendu_theorique * 100:.3f}% "
            f"(tolérance ±{TOLERANCE_DETECTION_FRAIS * 100:.0f}%)"
        ),
    )


def _chainer_performance(
    calendrier: list[date_type],
    valo: dict[date_type, float],
    nbuc_par_date_isin: dict[date_type, dict[str, float]],
    vl_par_date_isin: dict[date_type, dict[str, float]],
    mouvements_par_semaine: dict[date_type, dict[str, list[tuple[MouvementLigne, int]]]],
) -> tuple[dict[date_type, float], dict[date_type, float | None], dict[date_type, float], dict[date_type, float | None]]:
    """Calcule, semaine par semaine : VL équivalente Dietz et TWR (base 100 à la 1ère date) et les
    rendements hebdomadaires correspondants. Les mouvements sont valorisés au vl de la semaine où
    ils prennent effet (le nombre d'UC du mouvement est la seule donnée fiable, cf. schéma)."""
    vl_dietz: dict[date_type, float] = {}
    vl_twr: dict[date_type, float] = {}
    perf_dietz: dict[date_type, float | None] = {}
    perf_twr: dict[date_type, float | None] = {}

    if not calendrier:
        return vl_dietz, perf_dietz, vl_twr, perf_twr

    vl_dietz[calendrier[0]] = 100.0
    vl_twr[calendrier[0]] = 100.0
    perf_dietz[calendrier[0]] = None
    perf_twr[calendrier[0]] = None

    for i in range(1, len(calendrier)):
        d_prev, d_curr = calendrier[i - 1], calendrier[i]
        valo_prev, valo_curr = valo.get(d_prev, 0.0), valo.get(d_curr, 0.0)
        jours_periode = (d_curr - d_prev).days or 1

        flux_net = 0.0
        flux_pondere = 0.0
        for isin, mouvements in mouvements_par_semaine.get(d_curr, {}).items():
            vl_semaine = vl_par_date_isin.get(d_curr, {}).get(isin)
            if vl_semaine is None:
                continue
            for mouvement, signe in mouvements:
                valeur_flux = signe * mouvement.nb_uc * vl_semaine
                flux_net += valeur_flux
                jours_restants = max((d_curr - mouvement.date).days, 0)
                flux_pondere += valeur_flux * (jours_restants / jours_periode)

        denom_dietz = valo_prev + flux_pondere
        r_dietz = (valo_curr - valo_prev - flux_net) / denom_dietz if denom_dietz else 0.0

        numerateur_twr = sum(
            nbuc_par_date_isin.get(d_prev, {}).get(isin, 0.0) * vl_par_date_isin.get(d_curr, {}).get(isin, 0.0)
            for isin in nbuc_par_date_isin.get(d_prev, {})
        )
        r_twr = (numerateur_twr / valo_prev - 1) if valo_prev else 0.0

        vl_dietz[d_curr] = vl_dietz[d_prev] * (1 + r_dietz)
        vl_twr[d_curr] = vl_twr[d_prev] * (1 + r_twr)
        perf_dietz[d_curr] = r_dietz
        perf_twr[d_curr] = r_twr

    return vl_dietz, perf_dietz, vl_twr, perf_twr


def _volatilites_annualisees(perf_twr: dict[date_type, float | None], calendrier: list[date_type]) -> dict[date_type, float | None]:
    resultat: dict[date_type, float | None] = {}
    rendements: list[float] = []
    for d in calendrier:
        r = perf_twr.get(d)
        if r is not None:
            rendements.append(r)
        fenetre = rendements[-FENETRE_VOLATILITE_SEMAINES:]
        if len(fenetre) >= 2:
            resultat[d] = stdev(fenetre) * sqrt(SEMAINES_PAR_AN)
        else:
            resultat[d] = None
    return resultat


def calculer_performance_risque(request: CalculPerformanceRisqueRequest) -> CalculPerformanceRisqueResponse:
    calendrier = _calendrier(request.inventaire)
    sens_map = _sens_mouvements(request.table_libelles)
    mouvements_par_semaine = _mouvements_par_semaine_isin(request.mouvements, calendrier, sens_map)
    grille = _grille_forward_fill(request.inventaire, calendrier, mouvements_par_semaine)
    vl_par_date_isin = _vl_depuis_grille(grille)
    nb_uc_brut_par_date_isin = _nbuc_depuis_grille(grille)

    valo_brute = _valo_consolidee(grille)

    hypotheses = [
        f"Volatilité annualisée : écart-type des rendements hebdomadaires TWR, fenêtre glissante de {FENETRE_VOLATILITE_SEMAINES} semaines, annualisée par racine(52).",
        "classe_risque_a : grille de seuils inspirée SRRI (UCITS/CESR) ; classe_risque_b : grille inspirée SRI/MRM (PRIIPs). Aucune des deux n'est l'indicateur réglementaire strict (pas de bootstrap VaR, pas de MRM/CRM).",
        "Mouvements valorisés au vl de la semaine où ils prennent effet (nombre d'UC = seule donnée fiable).",
        "Un fonds sans ligne d'inventaire exacte à une date donnée n'est pas exclu de la valorisation consolidée : son vl est reporté en avant, et son nb_uc est ajusté par les mouvements survenus depuis la dernière ligne exacte (évite le double-comptage d'un fonds intégralement sorti par arbitrage sans nouvelle ligne d'inventaire ensuite).",
    ]

    statut_frais: list[StatutFraisIsin] = []
    vl_finale_par_date_isin = {d: dict(lignes) for d, lignes in vl_par_date_isin.items()}

    if request.table_frais_gestion:
        for frais in request.table_frais_gestion:
            inventaire_isin = [l for l in request.inventaire if l.isin == frais.isin]
            if not inventaire_isin:
                continue
            statut = _detecter_statut_frais(frais.isin, inventaire_isin, calendrier, grille, mouvements_par_semaine, frais.frais_gestion_annuel)
            statut_frais.append(statut)
            hypotheses.append(
                f"Fonds {frais.isin} : {'données nettes de frais (aucun ajustement)' if statut.frais_deja_deduits else f'données brutes, taux hebdo {frais.frais_gestion_annuel/52:.4f}% appliqué en cascade sur le vl'}."
            )
            if not statut.frais_deja_deduits:
                facteur_hebdo = 1 - (frais.frais_gestion_annuel / 100) / SEMAINES_PAR_AN
                premiere_date = min(l.date for l in inventaire_isin)
                dates_isin = [d for d in calendrier if d >= premiere_date]
                for n, d in enumerate(dates_isin):
                    vl_brut = vl_finale_par_date_isin.get(d, {}).get(frais.isin)
                    if vl_brut is not None:
                        vl_finale_par_date_isin.setdefault(d, {})[frais.isin] = vl_brut * (facteur_hebdo ** n)

    valo_nette = {
        d: sum(nb_uc_brut_par_date_isin.get(d, {}).get(isin, 0.0) * vl for isin, vl in lignes.items())
        for d, lignes in vl_finale_par_date_isin.items()
    }

    vl_dietz, perf_dietz, vl_twr, perf_twr = _chainer_performance(
        calendrier, valo_brute, nb_uc_brut_par_date_isin, vl_par_date_isin, mouvements_par_semaine
    )
    _vl_dietz_n, perf_dietz_nette, _vl_twr_n, perf_twr_nette = _chainer_performance(
        calendrier, valo_nette, nb_uc_brut_par_date_isin, vl_finale_par_date_isin, mouvements_par_semaine
    )

    volatilites = _volatilites_annualisees(perf_twr, calendrier)

    resultats = [
        PerformanceRisqueLigne(
            date=d,
            valo_consolidee=valo_brute.get(d, 0.0),
            vl_equivalente_dietz=vl_dietz.get(d, 0.0),
            vl_equivalente_twr=vl_twr.get(d, 0.0),
            perf_hebdo_dietz=perf_dietz.get(d),
            perf_hebdo_twr=perf_twr.get(d),
            perf_hebdo_dietz_nette=perf_dietz_nette.get(d) if request.table_frais_gestion else None,
            perf_hebdo_twr_nette=perf_twr_nette.get(d) if request.table_frais_gestion else None,
            volatilite_annualisee_52s=volatilites.get(d),
            classe_risque_a=_classe_risque(volatilites.get(d), SEUILS_CLASSE_A),
            classe_risque_b=_classe_risque(volatilites.get(d), SEUILS_CLASSE_B),
        )
        for d in calendrier
    ]

    return CalculPerformanceRisqueResponse(
        identifiant=request.identifiant,
        methodologie_url=METHODOLOGIE_URL,
        hypotheses_appliquees=hypotheses,
        statut_frais=statut_frais,
        resultats=resultats,
    )


def calculer_remuneration(request: CalculRemunerationRequest) -> CalculRemunerationResponse:
    calendrier = _calendrier(request.inventaire)
    sens_map = _sens_mouvements(request.table_libelles)
    mouvements_par_semaine = _mouvements_par_semaine_isin(request.mouvements, calendrier, sens_map)
    grille = _grille_forward_fill(request.inventaire, calendrier, mouvements_par_semaine)
    vl_par_date_isin = _vl_depuis_grille(grille)
    nb_uc_reconstitue = _reconstituer_nb_uc(request.inventaire, calendrier, mouvements_par_semaine)

    hypotheses = [
        "Rémunération calculée sur le nb_uc reconstitué (1er nbuc observé + mouvements signés), pas le nbuc brut de l'inventaire.",
    ]

    taux_retrocession = {r.isin: r.taux_retrocession_annuel for r in request.table_retrocession}
    resultats_retrocession: list[RemunerationResultLigne] = []
    for d in calendrier:
        for isin, nb_uc in nb_uc_reconstitue.get(d, {}).items():
            taux = taux_retrocession.get(isin)
            if taux is None:
                continue
            vl = vl_par_date_isin.get(d, {}).get(isin, 0.0)
            valorisation = nb_uc * vl
            resultats_retrocession.append(
                RemunerationResultLigne(
                    date=d,
                    isin=isin,
                    nb_uc_reconstitue=nb_uc,
                    valorisation=valorisation,
                    remuneration_semaine=valorisation * (taux / 100) / SEMAINES_PAR_AN,
                )
            )
    total_retrocession = sum(r.remuneration_semaine for r in resultats_retrocession)

    resultats_commission: list[CommissionGestionLigne] = []
    total_commission = 0.0
    if request.commission_gestion_courtier and request.table_type_support:
        type_par_isin = {t.isin: t.type_support for t in request.table_type_support}
        params = request.commission_gestion_courtier
        hypotheses.append(
            f"Commission de gestion courtier : {params.taux_commission_fonds_euros_annuel}%/an sur fonds euros, "
            f"{params.taux_commission_uc_annuel}%/an sur UC, appliquée au prorata 1/52e sur la valorisation reconstituée."
        )
        for d in calendrier:
            valo_fonds_euros = 0.0
            valo_uc = 0.0
            for isin, nb_uc in nb_uc_reconstitue.get(d, {}).items():
                type_support = type_par_isin.get(isin)
                if type_support is None:
                    continue
                vl = vl_par_date_isin.get(d, {}).get(isin, 0.0)
                valorisation = nb_uc * vl
                if type_support == "fonds_euro":
                    valo_fonds_euros += valorisation
                else:
                    valo_uc += valorisation
            commission_fonds_euros = valo_fonds_euros * (params.taux_commission_fonds_euros_annuel / 100) / SEMAINES_PAR_AN
            commission_uc = valo_uc * (params.taux_commission_uc_annuel / 100) / SEMAINES_PAR_AN
            resultats_commission.append(
                CommissionGestionLigne(
                    date=d,
                    valo_fonds_euros=valo_fonds_euros,
                    valo_uc=valo_uc,
                    commission_fonds_euros_semaine=commission_fonds_euros,
                    commission_uc_semaine=commission_uc,
                    commission_totale_semaine=commission_fonds_euros + commission_uc,
                )
            )
        total_commission = sum(r.commission_totale_semaine for r in resultats_commission)

    return CalculRemunerationResponse(
        identifiant=request.identifiant,
        hypotheses_appliquees=hypotheses,
        resultats_retrocession=resultats_retrocession,
        total_retrocession=total_retrocession,
        resultats_commission_gestion=resultats_commission,
        total_commission_gestion=total_commission,
        total_remuneration=total_retrocession + total_commission,
    )
