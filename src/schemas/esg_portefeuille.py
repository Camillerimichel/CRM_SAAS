from datetime import date as date_type
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from src.schemas.risque_performance import InventaireHebdoLigne

# Champs numériques disponibles dans esg_fonds_norm (reflète EXPORT_FIELDS de
# src/services/esg_import.py, hors champs textuels non mesurables company_name/sector1).
#
# - est_pourcentage=True : champ nativement sur une échelle 0-1 (affiché en %) ; False : valeur
#   brute affichée avec son unité.
# - sens_favorable : "haut" si une valeur plus élevée est meilleure (ex. renewable_energy),
#   "bas" si une valeur plus faible est meilleure (ex. exposure_to_fossil_fuels, intensité carbone).
#   Détermine le sens du classement par septile pour la note A-G.
# - notable=False : le sens "meilleur/moins bon" n'est pas clairement défini pour ce champ (taille
#   d'entreprise, métriques d'efficacité sans documentation précise) — valeur brute affichée
#   seulement, pas de note A-G calculée pour éviter d'inventer un sens arbitraire.
METRIQUES_DISPONIBLES: dict[str, dict] = {
    "note_esg":                        {"libelle": "Note ESG globale",                 "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "note_e":                          {"libelle": "Note environnementale",            "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "note_s":                          {"libelle": "Note sociale",                      "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "note_g":                          {"libelle": "Note de gouvernance",               "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "exposure_to_fossil_fuels":        {"libelle": "Exposition aux énergies fossiles",  "est_pourcentage": True,  "unite": None,   "sens_favorable": "bas",  "notable": True},
    "renewable_energy":                {"libelle": "Part d'énergies renouvelables",     "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "processes_ungc":                  {"libelle": "Conformité Pacte Mondial (UNGC)",   "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "environmental_good":              {"libelle": "Revenus verts (environnement)",     "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "social_good":                     {"libelle": "Revenus à impact social positif",   "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "environmental_harm":              {"libelle": "Revenus nuisibles (environnement)", "est_pourcentage": True,  "unite": None,   "sens_favorable": "bas",  "notable": True},
    "social_harm":                     {"libelle": "Revenus nuisibles (social)",        "est_pourcentage": True,  "unite": None,   "sens_favorable": "bas",  "notable": True},
    "pollution__positive_revenue":     {"libelle": "Revenus liés à la dépollution",     "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "pollution__negative_revenue":     {"libelle": "Revenus liés à la pollution",       "est_pourcentage": True,  "unite": None,   "sens_favorable": "bas",  "notable": True},
    "climate_change__negative_revenue": {"libelle": "Revenus liés aux énergies fossiles (climat)", "est_pourcentage": True, "unite": None, "sens_favorable": "bas", "notable": True},
    "board_gender_diversity":          {"libelle": "Mixité du conseil d'administration", "est_pourcentage": True, "unite": None,   "sens_favorable": "haut", "notable": True},
    "pct_female_board":                {"libelle": "Part de femmes au conseil",         "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "pct_female_executives":           {"libelle": "Part de femmes parmi les dirigeants", "est_pourcentage": True, "unite": None,  "sens_favorable": "haut", "notable": True},
    "board_independence":              {"libelle": "Indépendance du conseil",           "est_pourcentage": True,  "unite": None,   "sens_favorable": "haut", "notable": True},
    "gender_pay_gap":                  {"libelle": "Écart de rémunération H/F",         "est_pourcentage": True,  "unite": None,   "sens_favorable": "bas",  "notable": True},
    "ghg_intensity_value":             {"libelle": "Intensité carbone (GES)",           "est_pourcentage": False, "unite": "tCO2e", "sens_favorable": "bas", "notable": True},
    "scope_1_and_2_carbon_intensity":  {"libelle": "Intensité carbone scope 1+2",       "est_pourcentage": False, "unite": "tCO2e", "sens_favorable": "bas", "notable": True},
    "scope_3_carbon_intensity":        {"libelle": "Intensité carbone scope 3",         "est_pourcentage": False, "unite": "tCO2e", "sens_favorable": "bas", "notable": True},
    "emissions_to_water":              {"libelle": "Émissions vers l'eau",              "est_pourcentage": False, "unite": None,   "sens_favorable": "bas",  "notable": True},
    "hazardous_waste":                 {"libelle": "Déchets dangereux",                 "est_pourcentage": False, "unite": None,   "sens_favorable": "bas",  "notable": True},
    "temperature_score":               {"libelle": "Score de température",             "est_pourcentage": False, "unite": "°C",    "sens_favorable": "bas",  "notable": True},
    "executive_pay":                   {"libelle": "Ratio rémunération dirigeants",     "est_pourcentage": False, "unite": "x",     "sens_favorable": "bas",  "notable": True},
    "mcap_usd":                        {"libelle": "Capitalisation boursière",          "est_pourcentage": False, "unite": "M$",    "sens_favorable": None,   "notable": False},
    "revenue_usd":                     {"libelle": "Chiffre d'affaires",                "est_pourcentage": False, "unite": "M$",    "sens_favorable": None,   "notable": False},
    "evic":                            {"libelle": "Valeur d'entreprise (EVIC)",        "est_pourcentage": False, "unite": "M$",    "sens_favorable": None,   "notable": False},
    "average_per_employee_spend":      {"libelle": "Dépense moyenne par salarié",       "est_pourcentage": False, "unite": "$",     "sens_favorable": None,   "notable": False},
    "waste_efficiency":                {"libelle": "Efficacité de gestion des déchets", "est_pourcentage": False, "unite": None,    "sens_favorable": None,   "notable": False},
    "water_efficiency":                {"libelle": "Efficacité de gestion de l'eau",    "est_pourcentage": False, "unite": None,    "sens_favorable": None,   "notable": False},
    "avoiding_water_scarcity":         {"libelle": "Prévention du stress hydrique",     "est_pourcentage": False, "unite": None,    "sens_favorable": None,   "notable": False},
    "carbon_trend":                    {"libelle": "Tendance carbone",                  "est_pourcentage": False, "unite": None,   "sens_favorable": None,   "notable": False},
}

GRADE_LETTERS = ("A", "B", "C", "D", "E", "F", "G")


class MetriqueDisponible(BaseModel):
    code: str
    libelle: str
    est_pourcentage: bool
    unite: str | None = None
    sens_favorable: Literal["haut", "bas"] | None = None
    notable: bool = Field(..., description="False = pas de note A-G calculée (sens non défini), valeur brute uniquement")


class CalculEsgRequest(BaseModel):
    identifiant: str | None = Field(default=None, description="Libellé libre pour l'affichage, non stocké côté API")
    inventaire: list[InventaireHebdoLigne] = Field(..., description="Inventaire multi-fonds ; seule la dernière date est utilisée (photo, pas de série temporelle)")
    metriques: list[str] = Field(..., description="Codes des métriques à calculer, parmi celles listées par GET /api/esg/metriques")

    @field_validator("metriques")
    @classmethod
    def _valider_metriques(cls, v: list[str]) -> list[str]:
        inconnues = [m for m in v if m not in METRIQUES_DISPONIBLES]
        if inconnues:
            raise ValueError(f"Métriques inconnues : {', '.join(inconnues)}")
        return v


class EsgMetriqueResultat(BaseModel):
    code: str
    libelle: str
    est_pourcentage: bool
    unite: str | None = None
    valeur_ponderee: float | None = Field(default=None, description="Valeur du portefeuille pondérée par la valorisation des fonds ayant cette donnée")
    note_grade: str | None = Field(default=None, description="Note A (meilleur septile du référentiel) à G (moins bon) ; absente si notable=False pour cette métrique")
    taux_couverture_pct: float = Field(..., description="Part de la valorisation du portefeuille disposant de cette donnée, en %")


class EsgHoldingLigne(BaseModel):
    isin: str
    nom: str | None = None
    poids_pct: float
    valeurs: dict[str, float | None] = Field(default={}, description="Valeur brute par métrique sélectionnée, pour ce fonds")
    notes: dict[str, str | None] = Field(default={}, description="Note A-G par métrique sélectionnée, pour ce fonds")


class CalculEsgResponse(BaseModel):
    identifiant: str | None = None
    methodologie_url: str
    date_analyse: date_type = Field(..., description="Dernière date de l'inventaire fourni, utilisée pour la photo du portefeuille")
    hypotheses_appliquees: list[str] = Field(default=[], description="Tous les choix/paramètres retenus pour ce calcul précis")
    resultats_metriques: list[EsgMetriqueResultat]
    detail_par_ligne: list[EsgHoldingLigne]
