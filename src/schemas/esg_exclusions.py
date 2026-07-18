from datetime import date as date_type
from pydantic import BaseModel, Field, field_validator

from src.schemas.risque_performance import InventaireHebdoLigne

# Les 5 premiers critères reprennent ceux déjà utilisés par src/services/esg_fund_exclusions.py
# (check_fund_exclusions), où ils sont normalement lus depuis un questionnaire client réel — ici
# choisis directement par l'utilisateur du site démo, faute de client/questionnaire réel dans ce
# contexte. Ce sont des indicateurs PAI au niveau du fonds lui-même (fund-level).
#
# Les critères suivants ("look-through") sont différents dans leur nature : ils ne portent pas sur
# le fonds lui-même mais sur sa composition réelle (ses positions sous-jacentes), synchronisés
# depuis CRM_ESG (cf. src/services/esg_import.py::sync_esg_exclusions_holdings, qui reprend la même
# logique que l'onglet "Fonds" de esgnote.eu/qualification-esg) — un fonds est en divergence dès
# qu'au moins une de ses positions déclenche la catégorie, quel que soit son poids.
CRITERES_EXCLUSION: dict[str, dict] = {
    "fossiles":              {"libelle": "Exposition aux énergies fossiles",              "colonne": "exposure_to_fossil_fuels"},
    "zones_sensibles":       {"libelle": "Exposition aux zones sensibles (biodiversité)",  "colonne": "sfdr_biodiversity_pai"},
    "armes_controversees":   {"libelle": "Exposition aux armes controversées",             "colonne": "controversial_weapons"},
    "violations_pacte_ocde": {"libelle": "Violation du Pacte mondial des Nations Unies (UNGC)", "colonne": "violations_ungc"},
    "faible_note_esg":       {"libelle": "Note ESG faible (E, F ou G)",                    "colonne": "note_esg_grade"},
    "lt_charbon":            {"libelle": "Charbon (position sous-jacente exclue)",         "colonne": "excluded_coal"},
    "lt_petrole_gaz":        {"libelle": "Pétrole et gaz (position sous-jacente exclue)",   "colonne": "excluded_oil_gas"},
    "lt_sables_bitumineux":  {"libelle": "Sables bitumineux (position sous-jacente exclue)", "colonne": "excluded_tar_sands"},
    "lt_tabac":              {"libelle": "Tabac (position sous-jacente exclue)",            "colonne": "excluded_tobacco"},
    "lt_armement":           {"libelle": "Armement (position sous-jacente exclue)",         "colonne": "excluded_weapons"},
    "lt_armes_controversees": {"libelle": "Armes controversées (position sous-jacente exclue)", "colonne": "excluded_weapons_controversial"},
    "lt_jeux_argent":        {"libelle": "Jeux d'argent (position sous-jacente exclue)",    "colonne": "excluded_gambling"},
    "lt_alcool":             {"libelle": "Alcool (position sous-jacente exclue)",           "colonne": "excluded_alcohol"},
    "lt_nucleaire":          {"libelle": "Nucléaire (position sous-jacente exclue)",        "colonne": "excluded_nuclear"},
    "lt_pornographie":       {"libelle": "Pornographie (position sous-jacente exclue)",     "colonne": "excluded_pornography"},
    "lt_production_fossile": {"libelle": "Production d'électricité fossile (position sous-jacente exclue)", "colonne": "excluded_fossil_power_generation"},
    "lt_corruption":         {"libelle": "Corruption (position sous-jacente en infraction)", "colonne": "excluded_corruption"},
    "lt_droits_humains":     {"libelle": "Atteinte aux droits humains (position sous-jacente en infraction)", "colonne": "excluded_human_rights_issue"},
    "lt_travail_force":      {"libelle": "Travail forcé (position sous-jacente en infraction)", "colonne": "excluded_forced_labour"},
    "lt_environnement":      {"libelle": "Atteinte environnementale (position sous-jacente en infraction)", "colonne": "excluded_environmental_issue"},
}


class CritereDisponible(BaseModel):
    code: str
    libelle: str


class CalculExclusionsRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    inventaire: list[InventaireHebdoLigne] = Field(..., description="Inventaire multi-fonds ; seule la dernière date est utilisée (photo, pas de série temporelle)")
    criteres: list[str] = Field(..., description="Codes des critères d'exclusion à vérifier, parmi ceux listés par GET /api/esg/exclusions/criteres")

    @field_validator("criteres")
    @classmethod
    def _valider_criteres(cls, v: list[str]) -> list[str]:
        inconnus = [c for c in v if c not in CRITERES_EXCLUSION]
        if inconnus:
            raise ValueError(f"Critères inconnus : {', '.join(inconnus)}")
        return v


class DivergenceExclusion(BaseModel):
    code: str
    libelle: str


class ExclusionFondsLigne(BaseModel):
    isin: str
    nom: str | None = None
    poids_pct: float
    divergences: list[DivergenceExclusion] = Field(default=[], description="Critères en divergence pour ce fonds")
    conforme: bool | None = Field(default=None, description="None si donnée manquante et aucune divergence détectée par ailleurs")
    donnees_manquantes: bool = False


class CalculExclusionsResponse(BaseModel):
    identifiant: str | None = None
    methodologie_url: str
    date_analyse: date_type
    hypotheses_appliquees: list[str] = Field(default=[])
    criteres_selectionnes: list[DivergenceExclusion]
    fonds: list[ExclusionFondsLigne]
    nb_fonds: int
    nb_conformes: int
    nb_non_conformes: int
    nb_donnees_manquantes: int
    taux_conformite_pct: float | None = Field(default=None, description="Part de fonds conformes parmi ceux avec donnée disponible, en %")
    poids_conforme_pct: float = Field(..., description="Part de la valorisation du portefeuille jugée conforme, en %")
    poids_non_conforme_pct: float = Field(..., description="Part de la valorisation du portefeuille en divergence sur au moins un critère, en %")
