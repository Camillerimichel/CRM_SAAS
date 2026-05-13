"""
Service d'import de souscriptions (ouvertures de contrats) depuis fichier fournisseur.

Format CSV attendu (semicolon, UTF-8 BOM) :
  [SOUS];Nom Contrat;Réf contrat;date ouverture

Logique métier :
  - [SOUS]        = identifiant du client chez l'assureur → résolu via mariadb_client_identifiants_fournisseur
  - Nom Contrat   = produit générique → résolu via mariadb_affaires_generique.nom_contrat
  - Réf contrat   = référence unique du contrat → dédoublonnage sur mariadb_affaires.ref
  - date ouverture = format collé DMMYYYY ou DDMMYYYY (ex: 1071998 = 01/07/1998)
  - Statuts : nouveau | existant | client_inconnu | produit_inconnu
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_COL_SOUS          = "[SOUS]"
_COL_NOM_CONTRAT   = "Nom Contrat"
_COL_REF_CONTRAT   = "Réf contrat"
_COL_DATE_OUVERTURE = "date ouverture"
_REQUIRED_COLS     = [_COL_SOUS, _COL_NOM_CONTRAT, _COL_REF_CONTRAT, _COL_DATE_OUVERTURE]
_PREVIEW_ROWS      = 5


def _normalize_header(h: str) -> str:
    return h.strip().lstrip("﻿").strip('"').strip()


def _parse_date(raw: str) -> Optional[datetime]:
    s = raw.strip()
    if not s:
        return None
    try:
        return datetime.strptime(s.zfill(8), "%d%m%Y")
    except ValueError:
        return None


def parse_souscriptions_csv(data: bytes) -> list[dict]:
    try:
        text_data = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_data = data.decode("latin-1", errors="replace")

    reader = csv.reader(io.StringIO(text_data), delimiter=";")
    rows = list(reader)
    if not rows:
        raise ValueError("Fichier CSV vide")

    headers = [_normalize_header(h) for h in rows[0]]
    missing = [c for c in _REQUIRED_COLS if c not in headers]
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")

    result = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        d = {headers[i]: (row[i].strip() if i < len(row) else "") for i in range(len(headers))}
        result.append(d)
    return result


def _find_client(db: Session, ref_sous: str) -> Optional[int]:
    row = db.execute(
        text(
            "SELECT client_id FROM mariadb_client_identifiants_fournisseur "
            "WHERE identifiant_externe = :id AND actif = 1 LIMIT 1"
        ),
        {"id": ref_sous},
    ).fetchone()
    if not row:
        return None
    return (row._mapping["client_id"] if hasattr(row, "_mapping") else row[0])


def _find_produit(db: Session, code_court: str) -> Optional[int]:
    row = db.execute(
        text("SELECT id FROM mariadb_affaires_generique WHERE description = :code LIMIT 1"),
        {"code": code_court},
    ).fetchone()
    if not row:
        return None
    return (row._mapping["id"] if hasattr(row, "_mapping") else row[0])


def _affaire_exists(db: Session, ref_contrat: str) -> bool:
    row = db.execute(
        text("SELECT id FROM mariadb_affaires WHERE ref = :ref LIMIT 1"),
        {"ref": ref_contrat},
    ).fetchone()
    return row is not None


def _enrich_row(db: Session, raw: dict) -> dict:
    ref_sous     = raw.get(_COL_SOUS, "").strip()
    nom_contrat  = raw.get(_COL_NOM_CONTRAT, "").strip()
    ref_contrat  = raw.get(_COL_REF_CONTRAT, "").strip()
    date_raw     = raw.get(_COL_DATE_OUVERTURE, "").strip()
    date_obj     = _parse_date(date_raw)

    client_id  = _find_client(db, ref_sous) if ref_sous else None
    produit_id = _find_produit(db, nom_contrat) if nom_contrat else None  # nom_contrat = code court dans le CSV
    existe     = _affaire_exists(db, ref_contrat) if ref_contrat else False

    if existe:
        statut = "existant"
    elif client_id is None:
        statut = "client_inconnu"
    elif produit_id is None:
        statut = "produit_inconnu"
    else:
        statut = "nouveau"

    return {
        "ref_sous":      ref_sous,
        "nom_contrat":   nom_contrat,
        "ref_contrat":   ref_contrat,
        "date_affichee": date_obj.strftime("%d/%m/%Y") if date_obj else date_raw,
        "date_raw":      date_raw,
        "client_id":     client_id,
        "produit_id":    produit_id,
        "statut":        statut,
    }


def preview_souscriptions(db: Session, rows: list[dict]) -> dict:
    nb_nouveau = nb_existant = nb_client_inconnu = nb_produit_inconnu = 0
    sample: list[dict] = []
    problemes: list[dict] = []

    for i, raw in enumerate(rows):
        info = _enrich_row(db, raw)
        if info["statut"] == "nouveau":
            nb_nouveau += 1
        elif info["statut"] == "existant":
            nb_existant += 1
        elif info["statut"] == "client_inconnu":
            nb_client_inconnu += 1
            problemes.append(info)
        elif info["statut"] == "produit_inconnu":
            nb_produit_inconnu += 1
            problemes.append(info)
        if i < _PREVIEW_ROWS:
            sample.append(info)

    return {
        "total":             len(rows),
        "nb_nouveau":        nb_nouveau,
        "nb_existant":       nb_existant,
        "nb_client_inconnu": nb_client_inconnu,
        "nb_produit_inconnu":nb_produit_inconnu,
        "sample":            sample,
        "problemes":         problemes,
    }


def commit_souscriptions(db: Session, rows: list[dict]) -> dict:
    inserted = skipped = erreurs = 0

    next_id = db.execute(
        text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_affaires")
    ).scalar()

    for raw in rows:
        info = _enrich_row(db, raw)

        if info["statut"] == "existant":
            skipped += 1
            continue
        if info["client_id"] is None or info["produit_id"] is None:
            erreurs += 1
            continue

        date_obj = _parse_date(info["date_raw"])
        db.execute(
            text(
                "INSERT INTO mariadb_affaires (id, id_personne, ref, date_debut, id_affaire_generique) "
                "VALUES (:id, :cid, :ref, :date, :pid)"
            ),
            {
                "id":   next_id,
                "cid":  info["client_id"],
                "ref":  info["ref_contrat"],
                "date": date_obj,
                "pid":  info["produit_id"],
            },
        )
        next_id += 1
        inserted += 1
        logger.info("IMPORT SOUS – affaire ref=%s client_id=%s", info["ref_contrat"], info["client_id"])

    db.commit()
    return {"inserted": inserted, "skipped": skipped, "erreurs": erreurs}
