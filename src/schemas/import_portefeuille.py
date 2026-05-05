from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Any


class InventaireRow(BaseModel):
    ref_affaire: str | None = None
    id_affaire: int | None = None
    date: str
    code_isin: str
    nbuc: float
    vl: float
    nom_support: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # tolerate uppercase CSV headers
            return {k.lower().strip(): v for k, v in data.items()}
        return data

    @field_validator("code_isin", mode="before")
    @classmethod
    def _upper_isin(cls, v: Any) -> str:
        return (str(v) if v is not None else "").strip().upper()


class MouvementRow(BaseModel):
    ref_affaire: str | None = None
    id_affaire: int | None = None
    date: str
    code_isin: str
    code_mouvement: str
    nbuc: float
    vl: float
    montant_ope: float | None = None
    frais: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k.lower().strip(): v for k, v in data.items()}
        return data

    @field_validator("code_isin", mode="before")
    @classmethod
    def _upper_isin(cls, v: Any) -> str:
        return (str(v) if v is not None else "").strip().upper()

    @field_validator("code_mouvement", mode="before")
    @classmethod
    def _upper_code(cls, v: Any) -> str:
        return (str(v) if v is not None else "").strip().upper()


class ImportAlerte(BaseModel):
    ligne: int | None = None
    code: str
    message: str


class ImportPreviewResult(BaseModel):
    total_lignes: int
    lignes_valides: int
    lignes_invalides: int
    alertes: list[ImportAlerte]
    apercu: list[dict]


class ImportCommitResult(BaseModel):
    insere: int
    mis_a_jour: int
    alertes: list[ImportAlerte]
    avis_generes: int = 0
    affaires_creees: int = 0
    duree_recalcul_s: float = 0.0
