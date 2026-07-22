from typing import Literal
from datetime import date as date_type
from pydantic import BaseModel, Field, field_validator


# --- Entrées : inventaire hebdomadaire et mouvements (données injectées par l'utilisateur, non stockées) ---
# Un portefeuille peut détenir plusieurs fonds (isin) ; la valorisation consolidée à une date
# est la somme des valorisations (nbuc * vl) de toutes les lignes de cette date.

class InventaireHebdoLigne(BaseModel):
    date: date_type = Field(..., description="Fin de semaine de l'inventaire")
    isin: str = Field(..., description="Code ISIN du fonds")
    nbuc: float = Field(..., description="Nombre d'unités de compte détenues à cette date, pour ce fonds")
    vl: float = Field(..., description="Valeur liquidative de l'unité à cette date, pour ce fonds")

    @property
    def valo(self) -> float:
        return self.nbuc * self.vl

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            return date_type.fromisoformat(v)
        return v


# Un mouvement ne porte que sur un nombre d'UC : c'est la seule donnée juridiquement opposable
# (contrairement à un montant en devise, qui suppose un VL de conversion arbitraire). Le sens
# (ajoute / retire / sans incidence) dépend du libellé, résolu via table_libelles ci-dessous.

class MouvementLigne(BaseModel):
    date: date_type = Field(..., description="Date d'effet du mouvement")
    isin: str = Field(..., description="Code ISIN du fonds concerné")
    libelle: str = Field(..., description="Libellé du mouvement tel que reporté par la société (ex. 'Versement libre')")
    nb_uc: float = Field(..., description="Nombre d'UC concernées, en valeur absolue — pas de montant en devise")

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            return date_type.fromisoformat(v)
        return v


# Table de correspondance libellé -> incidence sur le nombre d'UC. Chaque société utilisant ses
# propres libellés, cette table est fournie par l'utilisateur (pas de liste figée côté API) ;
# le site démo se charge de proposer à l'utilisateur les libellés distincts trouvés dans son
# fichier de mouvements pour qu'il leur affecte un sens avant l'envoi du calcul.

class LibelleMouvementLigne(BaseModel):
    libelle: str = Field(..., description="Libellé de mouvement à qualifier")
    sens: Literal["+", "-", "na"] = Field(..., description="+ ajoute des UC, - en retire, na = sans incidence")


# --- Deux tables de taux annuels, sémantiquement distinctes ---
# frais_gestion : coût annuel supporté par le CLIENT (utilisé pour la performance nette)
# retrocession : rémunération annuelle perçue par le COURTIER (utilisée pour le calcul de rémunération)

class FraisGestionUCLigne(BaseModel):
    isin: str = Field(..., description="Code ISIN du fonds")
    nom: str | None = Field(default=None, description="Nom du fonds, pour affichage")
    frais_gestion_annuel: float = Field(..., description="Taux annuel de frais de gestion, en %, coût supporté par le client")


class RetrocessionUCLigne(BaseModel):
    isin: str = Field(..., description="Code ISIN du fonds")
    nom: str | None = Field(default=None, description="Nom du fonds, pour affichage")
    taux_retrocession_annuel: float = Field(..., description="Taux annuel de rétrocession, en %, rémunération perçue par le courtier sur ce fonds")


# --- Calcul performance / risque (portefeuille consolidé, tous fonds confondus) ---
# valo_consolidee(semaine) = somme, sur tous les isin, de nbuc*vl à cette semaine (valeurs telles
# que reportées dans l'inventaire, sans reconstitution).
# Les mouvements servent à isoler les flux externes (nb_uc signé via table_libelles, valorisés au
# vl de la semaine du mouvement) pour le calcul Dietz/TWR.
#
# Si table_frais_gestion est fournie, on ne sait pas a priori si l'assureur a déjà déduit ses frais
# de gestion du nombre d'UC ou non. Détection automatique sur le premier mois de données, par isin :
# on compare le nb_uc réellement observé en fin de mois à celui attendu (nb_uc début de mois +/- les
# mouvements du mois, sans frais). Si l'écart relatif correspond à ~1/12e du taux annuel de frais
# (à la tolérance près), les données sont considérées nettes (frais déjà pris) ; sinon elles sont
# considérées brutes, et le taux hebdomadaire (1/52e du taux annuel) est alors appliqué en cascade
# sur chaque valorisation calculée, avant le calcul Time-Weighted / Dietz net. Ce statut détecté est
# renvoyé par isin (cf. StatutFraisIsin) pour être affiché en tête du tableau de résultats.

class StatutFraisIsin(BaseModel):
    isin: str = Field(..., description="Code ISIN du fonds concerné")
    frais_deja_deduits: bool = Field(..., description="Détecté sur le 1er mois : True = données nettes, False = données brutes")
    ecart_relatif_detecte: float = Field(..., description="Écart mesuré entre nb_uc attendu et observé sur le 1er mois, en %")
    methode: str = Field(
        default="Détection automatique sur le premier mois de données (comparaison nb_uc attendu / observé)",
        description="Explication de la détection appliquée pour ce fonds",
    )


class ConsolideLigne(BaseModel):
    date: date_type = Field(..., description="Fin de semaine")
    valorisation: float = Field(..., description="Valorisation totale du portefeuille à cette date, déjà consolidée côté appelant (tous fonds confondus)")
    mouvement: float = Field(default=0.0, description="Montant net signé (en devise) des mouvements de la semaine (versements positifs, rachats négatifs) ; 0 si aucun mouvement")

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            return date_type.fromisoformat(v)
        return v


# --- Variante "fichier consolidé" : l'appelant fournit directement, pour chaque date, la
# valorisation totale du portefeuille et le mouvement net de la semaine (déjà calculés de son côté,
# par exemple à partir de son propre système de gestion), sans détail par fonds (isin/nbuc/vl).
# Cela évite de reproduire côté appelant la reconstitution inventaire+mouvements -> valorisation
# (source d'erreurs si l'appelant ne maîtrise pas bien ce calcul), au prix d'une perte de précision
# sur le TWR : le moteur "standard" (cf. CalculPerformanceRisqueRequest) revalorise individuellement
# chaque fonds détenu aux nouveaux VL de la semaine pour isoler l'effet marché de l'effet des flux ;
# ce détail n'existe plus ici. Conventions retenues (documentées dans hypotheses_appliquees à chaque
# réponse) :
#   - Dietz : formule "Simple Dietz" standard, flux pondéré à mi-période (poids 0.5) :
#       r = (V1 - V0 - F) / (V0 + F/2)
#   - TWR : le mouvement net de la semaine est supposé survenu en fin de période (convention
#     standard en l'absence de valorisation immédiatement avant/après le flux) :
#       r = (V1 - F) / V0 - 1
#   - Aucune détection de frais net/brut ni statut par fonds (statut_frais toujours vide) : ce
#     mécanisme est intrinsèquement par-isin.
# La réponse réutilise CalculPerformanceRisqueResponse/PerformanceRisqueLigne tels quels (déjà
# consolidés au niveau portefeuille, aucun champ par-isin).

class CalculPerformanceRisqueConsolideRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    consolide: list[ConsolideLigne] = Field(..., description="Série consolidée par date (valorisation + mouvement net), au moins 2 dates")


class CalculPerformanceRisqueRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    inventaire: list[InventaireHebdoLigne] = Field(..., description="Inventaire hebdomadaire multi-fonds, au moins 2 dates")
    mouvements: list[MouvementLigne] = Field(default=[], description="Mouvements (souscriptions/rachats/arbitrages) sur la période")
    table_libelles: list[LibelleMouvementLigne] = Field(default=[], description="Table de correspondance libellé -> sens (+/-/na)")
    table_frais_gestion: list[FraisGestionUCLigne] = Field(default=[], description="Taux de frais de gestion annuels par fonds, pour la performance nette")


class PerformanceRisqueLigne(BaseModel):
    date: date_type
    valo_consolidee: float = Field(..., description="Somme des valorisations (nbuc*vl) de tous les fonds à cette date")
    vl_equivalente_dietz: float = Field(..., description="VL équivalente reconstituée (méthode Dietz modifiée), base 100 à la 1ère date")
    vl_equivalente_twr: float = Field(..., description="VL équivalente reconstituée (méthode Time-Weighted), base 100 à la 1ère date")
    perf_hebdo_dietz: float | None = Field(default=None, description="Rendement hebdomadaire brut, méthode Dietz")
    perf_hebdo_twr: float | None = Field(default=None, description="Rendement hebdomadaire brut, méthode Time-Weighted")
    perf_hebdo_dietz_nette: float | None = Field(default=None, description="Rendement hebdomadaire net de frais de gestion (Dietz), si table_frais_gestion fournie")
    perf_hebdo_twr_nette: float | None = Field(default=None, description="Rendement hebdomadaire net de frais de gestion (TWR), si table_frais_gestion fournie")
    volatilite_annualisee_52s: float | None = Field(default=None, description="Écart-type annualisé des rendements TWR, fenêtre glissante 52 semaines (fraction, ex 0.10 = 10%)")
    classe_risque_a: int | None = Field(default=None, description="Classe de risque 1-7, grille de seuils inspirée SRRI (UCITS) — pas l'indicateur réglementaire strict, cf. methodologie_url")
    classe_risque_b: int | None = Field(default=None, description="Classe de risque 1-7, grille de seuils inspirée SRI/MRM (PRIIPs) — pas l'indicateur réglementaire strict, cf. methodologie_url")


class CalculPerformanceRisqueResponse(BaseModel):
    identifiant: str | None = None
    methodologie_url: str = Field(..., description="Lien vers la page décrivant précisément la méthodologie retenue")
    hypotheses_appliquees: list[str] = Field(default=[], description="Tous les choix/paramètres retenus pour ce calcul précis, à afficher en tête de tableau")
    statut_frais: list[StatutFraisIsin] = Field(default=[], description="Statut net/brut détecté par fonds, à afficher en tête de tableau")
    resultats: list[PerformanceRisqueLigne]


# --- Calcul rémunération courtier ---
# Deux composantes distinctes, cumulatives :
#
# 1) Rétrocession par fonds UC (par isin) — le taux de chaque fonds (table_retrocession) est le
#    taux de rétrocession propre au FONDS, pas la part qui revient au courtier. Le courtier ne
#    touche qu'une quote-part de ce taux, définie par un unique "taux_courtier" (0-100%, propre au
#    courtier, pas au fonds). Ne s'applique qu'aux isin classés "uc" via table_type_support — les
#    fonds euros ne sont jamais concernés par ce mécanisme (cf point 2).
#    Méthode de reconstitution du nb_uc : pour chaque isin, nb_uc_initial = premier nbuc rencontré
#    dans l'inventaire pour ce fonds ; ce nombre est ensuite ajusté à chaque mouvement (+/-/na,
#    résolu via table_libelles), indépendant de la façon dont chaque société reporte ses inventaires
#    (hebdo, mensuel, ...). Le vl reste celui reporté dans l'inventaire (donnée de marché fiable).
#    valorisation(semaine, isin) = nb_uc_reconstitué(semaine) * vl(semaine, isin)
#    remuneration(semaine, isin) = valorisation(semaine, isin)
#                                   * taux_retrocession_annuel(isin) / 100
#                                   * taux_courtier / 100 / 52
#
# 2) Commission de gestion du courtier sur l'encours fonds euros — indépendante de la rétrocession
#    par fonds, à un taux annuel propre au courtier (taux_commission_fonds_euros_annuel), appliqué
#    sur la valorisation reconstituée de la poche fonds euros (somme des isin classés "fonds_euro").
#    valo_fonds_euros(semaine) = somme, sur les isin fonds_euro, de nb_uc_reconstitué(semaine, isin) * vl(semaine, isin)
#    commission(semaine) = valo_fonds_euros(semaine) * taux_commission_fonds_euros_annuel / 100 / 52

class TypeSupportLigne(BaseModel):
    isin: str = Field(..., description="Code ISIN du fonds")
    type_support: Literal["fonds_euro", "uc"] = Field(..., description="Classification du support : détermine quel mécanisme de rémunération s'applique")


class CommissionGestionCourtierParams(BaseModel):
    taux_commission_fonds_euros_annuel: float = Field(..., description="Taux annuel, en %, propre à ce courtier, appliqué sur l'encours fonds euros")
    taux_courtier: float = Field(..., ge=0, le=100, description="Part (0-100%) du taux de rétrocession de chaque fonds UC qui revient au courtier — ne s'applique pas aux fonds euros")


class CalculRemunerationRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    inventaire: list[InventaireHebdoLigne] = Field(..., description="Inventaire hebdomadaire multi-fonds, au moins 2 dates")
    mouvements: list[MouvementLigne] = Field(default=[], description="Mouvements (souscriptions/rachats/arbitrages) sur la période")
    table_libelles: list[LibelleMouvementLigne] = Field(default=[], description="Table de correspondance libellé -> sens (+/-/na)")
    table_retrocession: list[RetrocessionUCLigne] = Field(..., description="Taux de rétrocession annuels par fonds UC (taux du fonds, pas la part du courtier)")
    table_type_support: list[TypeSupportLigne] = Field(..., description="Classification fonds_euro/uc par isin — requise pour savoir quel mécanisme appliquer à chaque fonds")
    commission_gestion_courtier: CommissionGestionCourtierParams | None = Field(default=None, description="Taux propres au courtier (fonds euros + part sur rétrocession UC) ; si absent, seule la rétrocession brute par fonds est calculée")


class RemunerationResultLigne(BaseModel):
    date: date_type
    isin: str
    nb_uc_reconstitue: float = Field(..., description="Nombre d'UC reconstitué (1er nbuc observé + mouvements signés)")
    valorisation: float = Field(..., description="nb_uc_reconstitue * vl à cette date")
    remuneration_semaine: float = Field(..., description="Part courtier de la rétrocession de la semaine pour ce fonds (taux_retrocession_annuel * taux_courtier)")


class CommissionGestionLigne(BaseModel):
    date: date_type
    valo_fonds_euros: float = Field(..., description="Valorisation reconstituée de la poche fonds euros à cette date")
    commission_fonds_euros_semaine: float


class CalculRemunerationResponse(BaseModel):
    identifiant: str | None = None
    hypotheses_appliquees: list[str] = Field(default=[], description="Tous les choix/paramètres retenus pour ce calcul précis, à afficher en tête de tableau")
    resultats_retrocession: list[RemunerationResultLigne]
    total_retrocession: float
    resultats_commission_gestion: list[CommissionGestionLigne] = []
    total_commission_gestion: float = 0.0
    total_remuneration: float = Field(..., description="total_retrocession + total_commission_gestion")
