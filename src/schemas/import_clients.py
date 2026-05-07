from __future__ import annotations
from pydantic import BaseModel


class ClientImportAlerte(BaseModel):
    ligne: int | None = None
    code: str
    message: str


class ClientImportRowPreview(BaseModel):
    ligne: int
    ref_client: str
    ref_cgp: str
    societe_id: int | None
    societe_nom: str | None
    nom: str
    prenom: str
    qualite: str
    adresse: str
    cp: str
    ville: str
    statut: str  # nouveau | existant | cgp_inconnu | doublon_ignore
    client_id: int | None = None  # si existant


class ClientImportPreviewResult(BaseModel):
    total_brut: int          # lignes brutes dans le fichier (hors en-tête)
    total_valides: int       # lignes après dédoublonnage
    doublons_ignores: int
    nouveaux: int
    existants: int
    cgp_inconnus: int
    apercu: list[ClientImportRowPreview]
    alertes: list[ClientImportAlerte]


class ClientImportCommitResult(BaseModel):
    crees: int
    mis_a_jour: int
    ignores: int
    alertes: list[ClientImportAlerte]
