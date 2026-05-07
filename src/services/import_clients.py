"""
Service d'import de clients depuis fichiers fournisseurs.

Format attendu (CSV semicolon, UTF-8 BOM) :
  [CLIENTS];Ref CGP;Nom;Prenom;Qualité;Adresse;Adresse 2;CP;Ville

Formats supportés : CSV, XLS, XLSX, JSON.

Logique métier :
  - [CLIENTS]  = identifiant du client chez l'assureur → mariadb_client_identifiants_fournisseur
  - Ref CGP    = identifiant du CGP chez l'assureur → résolu via mariadb_societe_identifiants_fournisseur
  - Dédoublonnage : même ref client → on garde la ligne avec l'adresse la plus complète (CP ≠ 0)
  - Statuts : nouveau | existant | cgp_inconnu | doublon_ignore
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.schemas.import_clients import (
    ClientImportAlerte,
    ClientImportCommitResult,
    ClientImportPreviewResult,
    ClientImportRowPreview,
)

logger = logging.getLogger(__name__)

# ── Colonnes attendues ────────────────────────────────────────────────────────
_COL_REF_CLIENT = "[CLIENTS]"
_COL_REF_CGP    = "Ref CGP"
_COL_NOM        = "Nom"
_COL_PRENOM     = "Prenom"
_COL_QUALITE    = "Qualité"
_COL_ADRESSE    = "Adresse"
_COL_ADRESSE2   = "Adresse 2"
_COL_CP         = "CP"
_COL_VILLE      = "Ville"

_REQUIRED_COLS = [_COL_REF_CLIENT, _COL_REF_CGP, _COL_NOM, _COL_CP]


# ── Parsers ───────────────────────────────────────────────────────────────────

def _normalize_header(h: str) -> str:
    """Supprime BOM, espaces et normalise les guillemets."""
    return h.strip().lstrip("﻿").strip('"').strip()


def _row_to_dict(headers: list[str], values: list[str]) -> dict:
    return {h: (values[i].strip() if i < len(values) else "") for i, h in enumerate(headers)}


def parse_clients_csv(data: bytes) -> list[dict]:
    text_data = data.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text_data), delimiter=";")
    rows = list(reader)
    if not rows:
        raise ValueError("Fichier CSV vide")
    headers = [_normalize_header(h) for h in rows[0]]
    _check_headers(headers)
    return [_row_to_dict(headers, r) for r in rows[1:] if any(c.strip() for c in r)]


def parse_clients_xlsx(data: bytes) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise ValueError("Fichier XLSX vide")
    headers = [_normalize_header(str(h or "")) for h in rows[0]]
    _check_headers(headers)
    result = []
    for row in rows[1:]:
        values = [str(c).strip() if c is not None else "" for c in row]
        if any(v for v in values):
            result.append(_row_to_dict(headers, values))
    return result


def parse_clients_xls(data: bytes) -> list[dict]:
    import xlrd
    wb = xlrd.open_workbook(file_contents=data)
    ws = wb.sheet_by_index(0)
    if ws.nrows == 0:
        raise ValueError("Fichier XLS vide")
    headers = [_normalize_header(str(ws.cell_value(0, c))) for c in range(ws.ncols)]
    _check_headers(headers)
    result = []
    for r in range(1, ws.nrows):
        values = [str(ws.cell_value(r, c)).strip() for c in range(ws.ncols)]
        # xlrd retourne les nombres comme float
        values = [v.rstrip(".0") if v.endswith(".0") else v for v in values]
        if any(v for v in values):
            result.append(_row_to_dict(headers, values))
    return result


def parse_clients_json(data: bytes) -> list[dict]:
    payload = json.loads(data.decode("utf-8-sig", errors="replace"))
    if isinstance(payload, dict):
        payload = payload.get("clients", payload.get("data", []))
    if not payload:
        raise ValueError("Fichier JSON vide")
    headers = list(payload[0].keys()) if payload else []
    _check_headers([_normalize_header(h) for h in headers])
    return [{_normalize_header(k): str(v).strip() for k, v in row.items()} for row in payload]


def _check_headers(headers: list[str]) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in headers]
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")


def detect_and_parse(filename: str, data: bytes) -> list[dict]:
    ext = (filename or "").lower().rsplit(".", 1)[-1]
    if ext == "xlsx":
        return parse_clients_xlsx(data)
    if ext == "xls":
        return parse_clients_xls(data)
    if ext == "json":
        return parse_clients_json(data)
    return parse_clients_csv(data)


# ── Dédoublonnage ─────────────────────────────────────────────────────────────

@dataclass
class _ClientRow:
    ligne: int
    ref_client: str
    ref_cgp: str
    nom: str
    prenom: str
    qualite: str
    adresse: str
    cp: str
    ville: str

    @property
    def adresse_complete(self) -> bool:
        return bool(self.adresse) and self.cp not in ("0", "", "None")

    @property
    def adresse_rue_complete(self) -> str:
        return self.adresse.strip()


def _parse_rows(raw: list[dict]) -> tuple[list[_ClientRow], list[ClientImportAlerte]]:
    rows = []
    alertes = []
    for i, r in enumerate(raw, start=2):  # ligne 1 = header
        ref_client = r.get(_COL_REF_CLIENT, "").strip()
        ref_cgp    = r.get(_COL_REF_CGP, "").strip()
        nom        = r.get(_COL_NOM, "").strip()
        prenom     = r.get(_COL_PRENOM, "").strip()
        qualite    = r.get(_COL_QUALITE, "").strip()
        adresse    = " ".join(filter(None, [
            r.get(_COL_ADRESSE, "").strip(),
            r.get(_COL_ADRESSE2, "").strip(),
        ]))
        cp         = r.get(_COL_CP, "").strip()
        ville      = r.get(_COL_VILLE, "").strip()

        if not ref_client:
            alertes.append(ClientImportAlerte(ligne=i, code="ref_manquant", message="Référence client vide — ligne ignorée"))
            continue

        rows.append(_ClientRow(
            ligne=i,
            ref_client=ref_client,
            ref_cgp=ref_cgp,
            nom=nom,
            prenom=prenom,
            qualite=qualite,
            adresse=adresse,
            cp=cp,
            ville=ville,
        ))
    return rows, alertes


def _deduplicate(rows: list[_ClientRow]) -> tuple[list[_ClientRow], list[_ClientRow]]:
    """
    Par ref_client : garde la ligne avec l'adresse la plus complète.
    En cas d'égalité, garde la dernière occurrence.
    Retourne (lignes_gardees, lignes_ignorees).
    """
    best: dict[str, _ClientRow] = {}
    for row in rows:
        key = row.ref_client
        if key not in best:
            best[key] = row
        else:
            current = best[key]
            # Préférer la ligne avec adresse complète
            if row.adresse_complete and not current.adresse_complete:
                best[key] = row
            elif row.adresse_complete == current.adresse_complete:
                best[key] = row  # dernière occurrence

    kept    = list(best.values())
    kept_set = {id(r) for r in kept}
    ignored = [r for r in rows if id(r) not in kept_set]
    return kept, ignored


# ── DB helpers ────────────────────────────────────────────────────────────────

def _resolve_societe(db: Session, fournisseur: str, ref_cgp: str) -> tuple[int | None, str | None, int | None]:
    """Retourne (societe_id, societe_nom, sif_id) ou (None, None, None).
    identifiant_externe est globalement unique — pas besoin de filtrer par fournisseur."""
    row = db.execute(
        text(
            "SELECT s.id, s.nom, m.id AS sif_id "
            "FROM mariadb_societe_identifiants_fournisseur m "
            "JOIN mariadb_societe_gestion s ON s.id = m.societe_id "
            "WHERE m.identifiant_externe = :id AND m.actif = 1 LIMIT 1"
        ),
        {"id": ref_cgp},
    ).fetchone()
    if not row:
        return None, None, None
    m = row._mapping if hasattr(row, "_mapping") else None
    return (m["id"] if m else row[0]), (m["nom"] if m else row[1]), (m["sif_id"] if m else row[2])


def _find_client(db: Session, fournisseur: str, ref_client: str) -> int | None:
    """Retourne client_id si déjà connu pour ce fournisseur."""
    row = db.execute(
        text(
            "SELECT client_id FROM mariadb_client_identifiants_fournisseur "
            "WHERE fournisseur = :f AND identifiant_externe = :id AND actif = 1 LIMIT 1"
        ),
        {"f": fournisseur.strip().upper(), "id": ref_client},
    ).fetchone()
    return (row[0] if not hasattr(row, "_mapping") else row._mapping["client_id"]) if row else None


# ── Preview ───────────────────────────────────────────────────────────────────

def preview_clients(
    db: Session,
    raw_rows: list[dict],
    fournisseur: str,
) -> ClientImportPreviewResult:
    rows, alertes = _parse_rows(raw_rows)
    kept, ignored = _deduplicate(rows)

    apercu: list[ClientImportRowPreview] = []
    nb_nouveaux = nb_existants = nb_cgp_inconnus = 0

    for row in ignored:
        apercu.append(ClientImportRowPreview(
            ligne=row.ligne,
            ref_client=row.ref_client,
            ref_cgp=row.ref_cgp,
            societe_id=None,
            societe_nom=None,
            nom=row.nom,
            prenom=row.prenom,
            qualite=row.qualite,
            adresse=row.adresse,
            cp=row.cp,
            ville=row.ville,
            statut="doublon_ignore",
        ))

    for row in kept:
        societe_id, societe_nom, _ = _resolve_societe(db, fournisseur, row.ref_cgp)
        client_id = _find_client(db, fournisseur, row.ref_client)

        if societe_id is None:
            nb_cgp_inconnus += 1
            alertes.append(ClientImportAlerte(
                ligne=row.ligne,
                code="cgp_inconnu",
                message=f"Ref CGP '{row.ref_cgp}' inconnue chez '{fournisseur}' — importé sans CGP rattaché",
            ))

        if client_id:
            statut = "existant" if societe_id else "cgp_inconnu_existant"
            nb_existants += 1
        else:
            statut = "nouveau" if societe_id else "cgp_inconnu_nouveau"
            nb_nouveaux += 1

        apercu.append(ClientImportRowPreview(
            ligne=row.ligne,
            ref_client=row.ref_client,
            ref_cgp=row.ref_cgp,
            societe_id=societe_id,
            societe_nom=societe_nom,
            nom=row.nom,
            prenom=row.prenom,
            qualite=row.qualite,
            adresse=row.adresse,
            cp=row.cp,
            ville=row.ville,
            statut=statut,
            client_id=client_id,
        ))

    apercu.sort(key=lambda r: r.ligne)

    return ClientImportPreviewResult(
        total_brut=len(raw_rows),
        total_valides=len(kept),
        doublons_ignores=len(ignored),
        nouveaux=nb_nouveaux,
        existants=nb_existants,
        cgp_inconnus=nb_cgp_inconnus,
        apercu=apercu,
        alertes=alertes,
    )


# ── Commit ────────────────────────────────────────────────────────────────────

def commit_clients(
    db: Session,
    raw_rows: list[dict],
    fournisseur: str,
) -> ClientImportCommitResult:
    rows, alertes = _parse_rows(raw_rows)
    kept, ignored = _deduplicate(rows)

    crees = mis_a_jour = ignores = 0
    ignores += len(ignored)

    fournisseur_upper = fournisseur.strip().upper()

    for row in kept:
        societe_id, _, sif_id = _resolve_societe(db, fournisseur, row.ref_cgp)
        if societe_id is None:
            alertes.append(ClientImportAlerte(
                ligne=row.ligne,
                code="cgp_inconnu",
                message=f"Ref CGP '{row.ref_cgp}' inconnue — importé sans CGP rattaché",
            ))

        client_id = _find_client(db, fournisseur, row.ref_client)

        adresse_rue = row.adresse or None
        adresse_cp  = row.cp if row.cp not in ("0", "") else None
        adresse_ville = row.ville or None

        if client_id is None:
            # Créer le client
            next_id = db.execute(
                text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_clients")
            ).scalar()
            db.execute(
                text(
                    "INSERT INTO mariadb_clients "
                    "(id, qualite, nom, prenom, adresse_rue, adresse_cp, adresse_ville, id_societe_gestion) "
                    "VALUES (:id, :qualite, :nom, :prenom, :adresse_rue, :adresse_cp, :adresse_ville, :sg)"
                ),
                {
                    "id": next_id,
                    "qualite": row.qualite or None,
                    "nom": row.nom or None,
                    "prenom": row.prenom or None,
                    "adresse_rue": adresse_rue,
                    "adresse_cp": adresse_cp,
                    "adresse_ville": adresse_ville,
                    "sg": societe_id,
                },
            )
            # Enregistrer le mapping fournisseur avec le sif_id
            db.execute(
                text(
                    "INSERT IGNORE INTO mariadb_client_identifiants_fournisseur "
                    "(client_id, fournisseur, identifiant_externe, actif, societe_identifiant_id) "
                    "VALUES (:cid, :f, :id, 1, :sif_id)"
                ),
                {"cid": next_id, "f": fournisseur_upper, "id": row.ref_client, "sif_id": sif_id},
            )
            crees += 1
            logger.info("IMPORT CLIENTS – créé client id=%s ref=%s (%s %s)", next_id, row.ref_client, row.nom, row.prenom)
        else:
            updates = {}
            if row.qualite:
                updates["qualite"] = row.qualite
            if row.nom:
                updates["nom"] = row.nom
            if row.prenom:
                updates["prenom"] = row.prenom
            if adresse_rue:
                updates["adresse_rue"] = adresse_rue
            if adresse_cp:
                updates["adresse_cp"] = adresse_cp
            if adresse_ville:
                updates["adresse_ville"] = adresse_ville
            if updates:
                set_clause = ", ".join(f"{k} = :{k}" for k in updates)
                updates["cid"] = client_id
                db.execute(
                    text(f"UPDATE mariadb_clients SET {set_clause} WHERE id = :cid"),
                    updates,
                )
            # Mettre à jour societe_identifiant_id si non encore renseigné
            if sif_id:
                db.execute(
                    text(
                        "UPDATE mariadb_client_identifiants_fournisseur "
                        "SET societe_identifiant_id = :sif_id "
                        "WHERE client_id = :cid AND fournisseur = :f AND societe_identifiant_id IS NULL"
                    ),
                    {"sif_id": sif_id, "cid": client_id, "f": fournisseur_upper},
                )
            mis_a_jour += 1

    db.commit()

    return ClientImportCommitResult(
        crees=crees,
        mis_a_jour=mis_a_jour,
        ignores=ignores,
        alertes=alertes,
    )
