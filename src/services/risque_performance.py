"""Moteur de calcul stateless : performance (Dietz/TWR), indicateur de risque hebdomadaire
(dérivé uniquement de la volatilité historique, sans MRM/CRM) et rémunération courtier
(rétrocession par fonds + commission de gestion). Aucune lecture/écriture en base : tout est
calculé à partir des payloads fournis par l'appelant, rien n'est persisté.
"""
from __future__ import annotations

from bisect import bisect_left
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
    RetrocessionAgregeeLigne,
)

METHODOLOGIE_URL = "/methodologie"

# Une "position" identifie un fonds détenu au sein d'un contrat donné : (isin, numero_contrat).
# Toute la grille interne (forward-fill, reconstitution nb_uc, chaînage Dietz/TWR) est indexée par
# position plutôt que par isin seul, pour que deux contrats détenant le même fonds ne soient jamais
# confondus dans la valorisation consolidée (leur somme reste correcte : c'est une généralisation
# du mécanisme multi-fonds déjà existant, pas un nouveau mécanisme). numero_contrat="" (défaut) pour
# tout l'inventaire reproduit exactement le comportement mono-portefeuille historique.
Position = tuple[str, str]  # (isin, numero_contrat)

FENETRE_VOLATILITE_SEMAINES = 52
SEMAINES_PAR_AN = 52

# Grilles de seuils appliquées à la même volatilité annualisée (fraction, ex 0.10 = 10%).
# Bornes façon SRRI (UCITS, CESR/10-673) — jamais nommées "SRRI" dans les libellés exposés.
SEUILS_CLASSE_A = [(0.005, 1), (0.02, 2), (0.05, 3), (0.10, 4), (0.15, 5), (0.25, 6)]
# Bornes façon Market Risk Measure / SRI (PRIIPs) — jamais nommées "SRI" dans les libellés exposés.
SEUILS_CLASSE_B = [(0.005, 1), (0.05, 2), (0.12, 3), (0.20, 4), (0.30, 5), (0.80, 6)]

TOLERANCE_DETECTION_FRAIS = 0.20   # ±20% autour de l'écart théorique attendu (1/12e du taux annuel)
SEMAINES_MIN_DETECTION_FRAIS = 4    # ~1 mois de données minimum pour tenter la détection

# Au-delà de ce nombre de positions (isin x numero_contrat) éligibles à la rétrocession, le détail
# par fonds+contrat n'est plus renvoyé (résultat agrégé par date uniquement) : à l'échelle d'une
# vision globale portant sur un très grand nombre de contrats, le détail complet produirait des
# millions de lignes, impraticable pour n'importe quelle interface (et coûteux à construire).
SEUIL_DETAIL_POSITIONS_RETROCESSION = 500


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


def _par_date_isin(inventaire: list[InventaireHebdoLigne]) -> dict[date_type, dict[Position, InventaireHebdoLigne]]:
    resultat: dict[date_type, dict[Position, InventaireHebdoLigne]] = defaultdict(dict)
    for ligne in inventaire:
        resultat[ligne.date][(ligne.isin, ligne.numero_contrat)] = ligne
    return dict(resultat)


def _grille_forward_fill(
    inventaire: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    mouvements_par_semaine: dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]],
) -> dict[date_type, dict[Position, tuple[float, float]]]:
    """Chaque fonds (au sein d'un contrat donné) ne reporte pas forcément à toutes les dates du
    calendrier consolidé (funds entrant/sortant, sociétés différentes). Entre deux dates où
    l'inventaire donne une valeur exacte (vérité terrain, qui prime toujours), le nb_uc est ajusté
    par les mouvements survenus plutôt que simplement reporté à l'identique : une position
    intégralement sortie par arbitrage sans nouvelle ligne d'inventaire ensuite doit tomber à ~0,
    pas rester à sa dernière valeur connue (sinon double-comptage avec la position où la valeur a
    été transférée). Le vl, lui, est simplement reporté en avant (c'est un prix de marché, pas une
    quantité qui "s'épuise")."""
    par_position: dict[Position, dict[date_type, InventaireHebdoLigne]] = defaultdict(dict)
    for ligne in inventaire:
        par_position[(ligne.isin, ligne.numero_contrat)][ligne.date] = ligne

    grille: dict[date_type, dict[Position, tuple[float, float]]] = defaultdict(dict)
    for position, par_date in par_position.items():
        premiere_date = min(par_date.keys())
        # bisect plutôt qu'un scan linéaire depuis le début du calendrier : sur un gros calcul
        # multi-contrats (des milliers de positions, chacune n'active que sur une fraction de
        # l'historique consolidé), ça évite de re-comparer les dates antérieures à premiere_date
        # à chaque position — gain significatif quand le calendrier consolidé couvre des décennies.
        index_depart = bisect_left(calendrier, premiere_date)
        nb_uc_courant: float | None = None
        vl_courant: float | None = None
        for d in calendrier[index_depart:]:
            ligne_exacte = par_date.get(d)
            if ligne_exacte is not None:
                nb_uc_courant = ligne_exacte.nbuc
                vl_courant = ligne_exacte.vl
            else:
                for mouvement, signe in mouvements_par_semaine.get(d, {}).get(position, []):
                    nb_uc_courant = (nb_uc_courant or 0.0) + signe * mouvement.nb_uc
            if nb_uc_courant is not None and vl_courant is not None:
                grille[d][position] = (nb_uc_courant, vl_courant)
    return dict(grille)


def _mouvements_par_semaine_isin(
    mouvements: list[MouvementLigne],
    calendrier: list[date_type],
    sens_map: dict[str, str],
) -> dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]]:
    """Rattache chaque mouvement à la première date du calendrier >= sa date (la semaine où il
    prend effet), avec son signe résolu (+1/-1/0) via la table de libellés."""
    resultat: dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]] = defaultdict(lambda: defaultdict(list))
    for m in mouvements:
        semaine = next((d for d in calendrier if d >= m.date), None)
        if semaine is None:
            continue
        signe = _signe(sens_map.get(m.libelle, "na"))
        resultat[semaine][(m.isin, m.numero_contrat)].append((m, signe))
    return resultat


def _valo_consolidee(grille: dict[date_type, dict[Position, tuple[float, float]]]) -> dict[date_type, float]:
    return {d: sum(nbuc * vl for nbuc, vl in lignes.values()) for d, lignes in grille.items()}


def _decomposer_grille(
    grille: dict[date_type, dict[Position, tuple[float, float]]],
) -> tuple[dict[date_type, dict[Position, float]], dict[date_type, dict[Position, float]], dict[date_type, float]]:
    """Équivalent de _vl_depuis_grille + _nbuc_depuis_grille + _valo_consolidee, mais en une seule
    passe sur la grille au lieu de trois — sur un calcul multi-contrats à plusieurs milliers de
    positions, évite de reparcourir la même structure trois fois pour un résultat identique."""
    vl_par_date: dict[date_type, dict[Position, float]] = {}
    nbuc_par_date: dict[date_type, dict[Position, float]] = {}
    valo_par_date: dict[date_type, float] = {}
    for d, lignes in grille.items():
        vl_d: dict[Position, float] = {}
        nbuc_d: dict[Position, float] = {}
        total = 0.0
        for position, (nbuc, vl) in lignes.items():
            vl_d[position] = vl
            nbuc_d[position] = nbuc
            total += nbuc * vl
        vl_par_date[d] = vl_d
        nbuc_par_date[d] = nbuc_d
        valo_par_date[d] = total
    return vl_par_date, nbuc_par_date, valo_par_date


def _reconstituer_nb_uc(
    inventaire: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    mouvements_par_semaine: dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]],
) -> dict[date_type, dict[Position, float]]:
    """Pour chaque position (isin+contrat), part du premier nbuc observé puis l'ajuste à chaque
    mouvement, plutôt que de faire confiance au nbuc reporté chaque semaine (hétérogénéité de
    reporting entre sociétés)."""
    par_date_isin = _par_date_isin(inventaire)
    premiere_date_position: dict[Position, date_type] = {}
    for d in calendrier:
        for position in par_date_isin.get(d, {}):
            premiere_date_position.setdefault(position, d)

    resultat: dict[date_type, dict[Position, float]] = defaultdict(dict)

    # Boucle par position (bisect pour ne parcourir que la portion pertinente du calendrier),
    # même optimisation que _grille_forward_fill — évite de rescanner tout l'historique consolidé
    # pour chaque position sur un calcul multi-contrats à plusieurs milliers de positions.
    for position, premiere in premiere_date_position.items():
        index_depart = bisect_left(calendrier, premiere)
        nb_uc_courant = 0.0
        for d in calendrier[index_depart:]:
            if d == premiere:
                nb_uc_courant = par_date_isin[d][position].nbuc
            else:
                for _mouvement, signe in mouvements_par_semaine.get(d, {}).get(position, []):
                    nb_uc_courant += signe * _mouvement.nb_uc
            resultat[d][position] = nb_uc_courant

    return dict(resultat)


def _vl_depuis_grille(grille: dict[date_type, dict[Position, tuple[float, float]]]) -> dict[date_type, dict[Position, float]]:
    return {d: {position: vl for position, (_nbuc, vl) in lignes.items()} for d, lignes in grille.items()}


def _nbuc_depuis_grille(grille: dict[date_type, dict[Position, tuple[float, float]]]) -> dict[date_type, dict[Position, float]]:
    return {d: {position: nbuc for position, (nbuc, _vl) in lignes.items()} for d, lignes in grille.items()}


def _detecter_statut_frais(
    position: Position,
    inventaire_position: list[InventaireHebdoLigne],
    calendrier: list[date_type],
    grille: dict[date_type, dict[Position, tuple[float, float]]],
    mouvements_par_semaine: dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]],
    frais_gestion_annuel: float,
) -> StatutFraisIsin:
    isin, numero_contrat = position
    lignes_triees = sorted(inventaire_position, key=lambda l: l.date)
    dates_calendrier_isin = [d for d in calendrier if d >= lignes_triees[0].date]
    fenetre = dates_calendrier_isin[:SEMAINES_MIN_DETECTION_FRAIS + 1]

    if len(fenetre) < 2:
        return StatutFraisIsin(
            isin=isin,
            numero_contrat=numero_contrat,
            frais_deja_deduits=False,
            ecart_relatif_detecte=0.0,
            methode="Données insuffisantes (< 2 semaines) pour la détection automatique — traité par défaut comme brut",
        )

    nb_uc_depart = lignes_triees[0].nbuc
    date_depart, date_arrivee = fenetre[0], fenetre[-1]

    nb_uc_attendu = nb_uc_depart
    for d in fenetre[1:]:
        for mouvement, signe in mouvements_par_semaine.get(d, {}).get(position, []):
            nb_uc_attendu += signe * mouvement.nb_uc

    nb_uc_observe_tuple = grille.get(date_arrivee, {}).get(position)
    nb_uc_observe = nb_uc_observe_tuple[0] if nb_uc_observe_tuple else None
    if nb_uc_observe is None or nb_uc_attendu == 0:
        return StatutFraisIsin(
            isin=isin,
            numero_contrat=numero_contrat,
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
        numero_contrat=numero_contrat,
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
    nbuc_par_date_isin: dict[date_type, dict[Position, float]],
    vl_par_date_isin: dict[date_type, dict[Position, float]],
    mouvements_par_semaine: dict[date_type, dict[Position, list[tuple[MouvementLigne, int]]]],
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

        vl_curr = vl_par_date_isin.get(d_curr, {})

        flux_net = 0.0
        flux_pondere = 0.0
        for position, mouvements in mouvements_par_semaine.get(d_curr, {}).items():
            vl_semaine = vl_curr.get(position)
            if vl_semaine is None:
                continue
            for mouvement, signe in mouvements:
                valeur_flux = signe * mouvement.nb_uc * vl_semaine
                flux_net += valeur_flux
                jours_restants = max((d_curr - mouvement.date).days, 0)
                flux_pondere += valeur_flux * (jours_restants / jours_periode)

        denom_dietz = valo_prev + flux_pondere
        r_dietz = (valo_curr - valo_prev - flux_net) / denom_dietz if denom_dietz else 0.0

        # position est garanti présent dans nbuc_prev (c'est la source d'itération) : accès direct
        # par clé plutôt que .get(..., 0.0), et les deux dicts extraits une seule fois par semaine
        # plutôt que ré-évalués à chaque position (gain notable sur un calcul multi-contrats à
        # plusieurs milliers de positions).
        nbuc_prev = nbuc_par_date_isin.get(d_prev, {})
        numerateur_twr = sum(
            nbuc_prev[position] * vl_curr.get(position, 0.0)
            for position in nbuc_prev
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
    vl_par_date_isin, nb_uc_brut_par_date_isin, valo_brute = _decomposer_grille(grille)

    hypotheses = [
        f"Volatilité annualisée : écart-type des rendements hebdomadaires TWR, fenêtre glissante de {FENETRE_VOLATILITE_SEMAINES} semaines, annualisée par racine(52).",
        "classe_risque_a : grille de seuils inspirée SRRI (UCITS/CESR) ; classe_risque_b : grille inspirée SRI/MRM (PRIIPs). Aucune des deux n'est l'indicateur réglementaire strict (pas de bootstrap VaR, pas de MRM/CRM).",
        "Mouvements valorisés au vl de la semaine où ils prennent effet (nombre d'UC = seule donnée fiable).",
        "Un fonds sans ligne d'inventaire exacte à une date donnée n'est pas exclu de la valorisation consolidée : son vl est reporté en avant, et son nb_uc est ajusté par les mouvements survenus depuis la dernière ligne exacte (évite le double-comptage d'un fonds intégralement sorti par arbitrage sans nouvelle ligne d'inventaire ensuite).",
    ]

    statut_frais: list[StatutFraisIsin] = []
    vl_finale_par_date_isin = {d: dict(lignes) for d, lignes in vl_par_date_isin.items()}

    if request.table_frais_gestion:
        positions_par_isin: dict[str, list[Position]] = defaultdict(list)
        for position in {(l.isin, l.numero_contrat) for l in request.inventaire}:
            positions_par_isin[position[0]].append(position)

        for frais in request.table_frais_gestion:
            for position in positions_par_isin.get(frais.isin, []):
                inventaire_position = [l for l in request.inventaire if (l.isin, l.numero_contrat) == position]
                if not inventaire_position:
                    continue
                statut = _detecter_statut_frais(position, inventaire_position, calendrier, grille, mouvements_par_semaine, frais.frais_gestion_annuel)
                statut_frais.append(statut)
                suffixe_contrat = f" (contrat {position[1]})" if position[1] else ""
                hypotheses.append(
                    f"Fonds {frais.isin}{suffixe_contrat} : {'données nettes de frais (aucun ajustement)' if statut.frais_deja_deduits else f'données brutes, taux hebdo {frais.frais_gestion_annuel/52:.4f}% appliqué en cascade sur le vl'}."
                )
                if not statut.frais_deja_deduits:
                    facteur_hebdo = 1 - (frais.frais_gestion_annuel / 100) / SEMAINES_PAR_AN
                    premiere_date = min(l.date for l in inventaire_position)
                    dates_isin = [d for d in calendrier if d >= premiere_date]
                    for n, d in enumerate(dates_isin):
                        vl_brut = vl_finale_par_date_isin.get(d, {}).get(position)
                        if vl_brut is not None:
                            vl_finale_par_date_isin.setdefault(d, {})[position] = vl_brut * (facteur_hebdo ** n)

    valo_nette = {
        d: sum(nb_uc_brut_par_date_isin.get(d, {}).get(position, 0.0) * vl for position, vl in lignes.items())
        for d, lignes in vl_finale_par_date_isin.items()
    }

    vl_dietz, perf_dietz, vl_twr, perf_twr = _chainer_performance(
        calendrier, valo_brute, nb_uc_brut_par_date_isin, vl_par_date_isin, mouvements_par_semaine
    )
    # Sans table_frais_gestion, valo_nette/vl_finale_par_date_isin sont des copies exactes de
    # valo_brute/vl_par_date_isin (aucun ajustement appliqué) : le second chaînage serait
    # rigoureusement identique au premier — et son résultat n'est de toute façon jamais exposé dans
    # ce cas (cf. perf_hebdo_*_nette ci-dessous). Sur un gros portefeuille (des milliers de
    # positions), ce chaînage est le principal poste de temps de calcul : l'éviter quand il est
    # inutile réduit le temps total d'environ moitié.
    if request.table_frais_gestion:
        _vl_dietz_n, perf_dietz_nette, _vl_twr_n, perf_twr_nette = _chainer_performance(
            calendrier, valo_nette, nb_uc_brut_par_date_isin, vl_finale_par_date_isin, mouvements_par_semaine
        )
    else:
        perf_dietz_nette, perf_twr_nette = {}, {}

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

    type_par_isin = {t.isin: t.type_support for t in request.table_type_support}
    taux_retrocession = {r.isin: r.taux_retrocession_annuel for r in request.table_retrocession}

    resultats_retrocession: list[RemunerationResultLigne] = []
    resultats_retrocession_agregee: list[RetrocessionAgregeeLigne] = []
    retrocession_agregee = False
    total_retrocession = 0.0
    if request.commission_gestion_courtier:
        taux_courtier = request.commission_gestion_courtier.taux_courtier
        hypotheses.append(
            f"Rétrocession UC : {taux_courtier}% (taux_courtier) du taux de rétrocession propre à chaque fonds "
            "(table_retrocession) — ne s'applique pas aux fonds euros."
        )

        positions_eligibles = {
            position
            for lignes in nb_uc_reconstitue.values()
            for position in lignes
            if type_par_isin.get(position[0]) == "uc" and position[0] in taux_retrocession
        }
        retrocession_agregee = len(positions_eligibles) > SEUIL_DETAIL_POSITIONS_RETROCESSION

        if retrocession_agregee:
            hypotheses.append(
                f"Rétrocession agrégée par date uniquement ({len(positions_eligibles)} positions fonds x contrat "
                f"détectées, seuil de détail = {SEUIL_DETAIL_POSITIONS_RETROCESSION}) : le détail par fonds et par "
                "contrat n'est pas renvoyé sur une vision portant sur un aussi grand nombre de positions, pour "
                "éviter une réponse de plusieurs millions de lignes."
            )
            for d in calendrier:
                total_semaine = 0.0
                for position, nb_uc in nb_uc_reconstitue.get(d, {}).items():
                    isin = position[0]
                    if type_par_isin.get(isin) != "uc":
                        continue
                    taux_fonds = taux_retrocession.get(isin)
                    if taux_fonds is None:
                        continue
                    vl = vl_par_date_isin.get(d, {}).get(position, 0.0)
                    total_semaine += (nb_uc * vl) * (taux_fonds / 100) * (taux_courtier / 100) / SEMAINES_PAR_AN
                resultats_retrocession_agregee.append(RetrocessionAgregeeLigne(date=d, remuneration_semaine=total_semaine))
            total_retrocession = sum(r.remuneration_semaine for r in resultats_retrocession_agregee)
        else:
            for d in calendrier:
                for position, nb_uc in nb_uc_reconstitue.get(d, {}).items():
                    isin, numero_contrat = position
                    if type_par_isin.get(isin) != "uc":
                        continue
                    taux_fonds = taux_retrocession.get(isin)
                    if taux_fonds is None:
                        continue
                    vl = vl_par_date_isin.get(d, {}).get(position, 0.0)
                    valorisation = nb_uc * vl
                    remuneration = valorisation * (taux_fonds / 100) * (taux_courtier / 100) / SEMAINES_PAR_AN
                    resultats_retrocession.append(
                        RemunerationResultLigne(
                            date=d,
                            isin=isin,
                            numero_contrat=numero_contrat,
                            nb_uc_reconstitue=nb_uc,
                            valorisation=valorisation,
                            remuneration_semaine=remuneration,
                        )
                    )
            total_retrocession = sum(r.remuneration_semaine for r in resultats_retrocession)

    resultats_commission: list[CommissionGestionLigne] = []
    total_commission = 0.0
    if request.commission_gestion_courtier:
        params = request.commission_gestion_courtier
        hypotheses.append(
            f"Commission de gestion courtier sur fonds euros : {params.taux_commission_fonds_euros_annuel}%/an, "
            "appliquée au prorata 1/52e sur la valorisation reconstituée de la poche fonds euros."
        )
        for d in calendrier:
            valo_fonds_euros = 0.0
            for position, nb_uc in nb_uc_reconstitue.get(d, {}).items():
                isin, _numero_contrat = position
                if type_par_isin.get(isin) != "fonds_euro":
                    continue
                vl = vl_par_date_isin.get(d, {}).get(position, 0.0)
                valo_fonds_euros += nb_uc * vl
            commission_fonds_euros = valo_fonds_euros * (params.taux_commission_fonds_euros_annuel / 100) / SEMAINES_PAR_AN
            resultats_commission.append(
                CommissionGestionLigne(
                    date=d,
                    valo_fonds_euros=valo_fonds_euros,
                    commission_fonds_euros_semaine=commission_fonds_euros,
                )
            )
        total_commission = sum(r.commission_fonds_euros_semaine for r in resultats_commission)

    return CalculRemunerationResponse(
        identifiant=request.identifiant,
        hypotheses_appliquees=hypotheses,
        resultats_retrocession=resultats_retrocession,
        resultats_retrocession_agregee=resultats_retrocession_agregee,
        retrocession_agregee=retrocession_agregee,
        total_retrocession=total_retrocession,
        resultats_commission_gestion=resultats_commission,
        total_commission_gestion=total_commission,
        total_remuneration=total_retrocession + total_commission,
    )
