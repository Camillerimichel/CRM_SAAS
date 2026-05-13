"""
Service d'import de fichiers d'avis d'opérations.

Format attendu (CSV ou JSON) :
  date, ref_affaire [, reference, entree, sortie, commentaire, id_client_fournisseur]

Pipeline après commit :
  UPSERT dans la table avis (clé : id_affaire + date + reference).
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.schemas.import_portefeuille import (
    AvisRow,
    ImportAlerte,
    ImportPreviewResult,
    ImportCommitResult,
)
from src.services.import_inventaire import (
    _resolve_affaire_id,
    _resolve_or_create_affaire,
    _resolve_societe_by_fournisseur,
    _parse_date,
)

logger = logging.getLogger(__name__)

_ID_ETAPE_VALIDE = 9  # 'validé' dans avis_regle
_ETAT_DEFAUT     = 1


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_avis_csv(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    for sep in (",", ";", "\t"):
        reader = csv.DictReader(io.StringIO(content), delimiter=sep)
        rows = list(reader)
        if rows and len(rows[0]) > 1:
            return rows
    return list(csv.DictReader(io.StringIO(content)))


def parse_avis_json(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return data
    for key in ("rows", "avis"):
        if key in data:
            return data[key]
    raise ValueError("Format JSON invalide : attendu une liste ou {avis: [...]}")


def _validate_rows(raw_rows: list[dict]) -> tuple[list[AvisRow], list[ImportAlerte]]:
    rows: list[AvisRow] = []
    alertes: list[ImportAlerte] = []
    for i, raw in enumerate(raw_rows, start=1):
        try:
            row = AvisRow.model_validate(raw)
            if not row.ref_affaire and row.id_affaire is None and not row.id_client_fournisseur:
                alertes.append(ImportAlerte(
                    ligne=i, code="missing_affaire",
                    message="ref_affaire, id_affaire et id_client_fournisseur sont tous absents",
                ))
                continue
            if _parse_date(row.date) is None:
                alertes.append(ImportAlerte(
                    ligne=i, code="invalid_date", message=f"Date invalide : {row.date}",
                ))
                continue
            rows.append(row)
        except Exception as exc:
            alertes.append(ImportAlerte(ligne=i, code="parse_error", message=str(exc)))
    return rows, alertes


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_affaire_id_avis(db: Session, row: AvisRow, fournisseur: str | None) -> int | None:
    """Résout l'id_affaire depuis ref_affaire, id_client_fournisseur, ou id_affaire direct."""
    if row.id_affaire is not None:
        r = db.execute(
            text("SELECT id FROM mariadb_affaires WHERE id = :id"),
            {"id": row.id_affaire},
        ).fetchone()
        return r[0] if r else None
    if row.ref_affaire:
        r = db.execute(
            text("SELECT id FROM mariadb_affaires WHERE ref = :ref LIMIT 1"),
            {"ref": row.ref_affaire.strip()},
        ).fetchone()
        if r:
            return r[0]
    if row.id_client_fournisseur and fournisseur:
        societe_id = _resolve_societe_by_fournisseur(db, fournisseur, row.id_client_fournisseur)
        if societe_id:
            r = db.execute(
                text(
                    "SELECT a.id FROM mariadb_affaires a "
                    "JOIN mariadb_affaire_societe ms ON ms.id_affaire = a.id "
                    "WHERE ms.id_societe = :sid LIMIT 1"
                ),
                {"sid": societe_id},
            ).fetchone()
            if r:
                return r[0]
    return None


def _upsert_avis(
    db: Session,
    id_affaire: int,
    date_dt: datetime,
    reference: str,
    entree: str | None,
    sortie: str | None,
    commentaire: str | None,
) -> tuple[int, bool]:
    """Insère ou met à jour un avis. Retourne (id_avis, created)."""
    existing = db.execute(
        text(
            "SELECT id FROM avis "
            "WHERE id_affaire = :aff AND DATE(date) = DATE(:d) AND reference = :ref LIMIT 1"
        ),
        {"aff": id_affaire, "d": date_dt, "ref": reference},
    ).fetchone()
    if existing:
        db.execute(
            text(
                "UPDATE avis SET entree = :e, sortie = :s, commentaire = :c WHERE id = :id"
            ),
            {"e": entree or "", "s": sortie or "", "c": commentaire or "", "id": existing[0]},
        )
        return existing[0], False
    next_id = db.execute(
        text("SELECT COALESCE(MAX(id), 0) + 1 FROM avis")
    ).scalar()
    db.execute(
        text(
            "INSERT INTO avis (id, reference, date, id_affaire, id_etape, etat, entree, sortie, commentaire) "
            "VALUES (:id, :ref, :d, :aff, :etape, :etat, :e, :s, :c)"
        ),
        {
            "id":    next_id,
            "ref":   reference,
            "d":     date_dt,
            "aff":   id_affaire,
            "etape": _ID_ETAPE_VALIDE,
            "etat":  _ETAT_DEFAUT,
            "e":     entree or "",
            "s":     sortie or "",
            "c":     commentaire or "",
        },
    )
    return next_id, True


# ──────────────────────────────────────────────────────────────────────────────
# Preview
# ──────────────────────────────────────────────────────────────────────────────

def preview_avis(
    db: Session,
    raw_rows: list[dict],
    fournisseur: str | None = None,
) -> dict:
    rows, alertes = _validate_rows(raw_rows)
    apercu = []
    for i, row in enumerate(rows):
        id_affaire = _resolve_affaire_id_avis(db, row, fournisseur)
        affaire_trouvee = id_affaire is not None
        apercu.append({
            "ref_affaire":   row.ref_affaire or row.id_client_fournisseur or str(row.id_affaire or ""),
            "date":          row.date,
            "reference":     row.reference or "",
            "entree":        row.entree or "",
            "sortie":        row.sortie or "",
            "commentaire":   row.commentaire or "",
            "affaire_trouvee": affaire_trouvee,
            "affaire_a_creer": not affaire_trouvee and bool(row.ref_affaire),
        })

    return ImportPreviewResult(
        total_lignes=len(raw_rows),
        lignes_valides=len(rows),
        lignes_invalides=len(raw_rows) - len(rows),
        alertes=alertes,
        apercu=apercu,
    ).model_dump()


# ──────────────────────────────────────────────────────────────────────────────
# Commit
# ──────────────────────────────────────────────────────────────────────────────

def commit_avis(
    db: Session,
    raw_rows: list[dict],
    id_societe_gestion: int | None = None,
    fournisseur: str | None = None,
) -> dict:
    import time
    t0 = time.perf_counter()

    rows, alertes = _validate_rows(raw_rows)
    insere = 0
    mis_a_jour = 0
    affaires_creees = 0

    for i, row in enumerate(rows, start=1):
        id_affaire = _resolve_affaire_id_avis(db, row, fournisseur)
        if id_affaire is None:
            if row.ref_affaire:
                id_affaire, _ = _resolve_or_create_affaire(
                    db, row.ref_affaire,
                    id_societe_gestion=id_societe_gestion,
                )
                affaires_creees += 1
            else:
                alertes.append(ImportAlerte(
                    ligne=i, code="affaire_introuvable",
                    message=f"Affaire introuvable pour ref={row.ref_affaire!r} / id_client={row.id_client_fournisseur!r}",
                ))
                continue

        date_dt = _parse_date(row.date)
        if date_dt is None:
            alertes.append(ImportAlerte(ligne=i, code="invalid_date", message=f"Date invalide : {row.date}"))
            continue

        reference = row.reference or f"avis-{row.date}-{id_affaire}"

        try:
            _, created = _upsert_avis(
                db, id_affaire, date_dt, reference,
                entree=row.entree,
                sortie=row.sortie,
                commentaire=row.commentaire,
            )
            if created:
                insere += 1
            else:
                mis_a_jour += 1
        except Exception as exc:
            db.rollback()
            alertes.append(ImportAlerte(ligne=i, code="db_error", message=str(exc)))
            continue

    db.commit()

    return ImportCommitResult(
        insere=insere,
        mis_a_jour=mis_a_jour,
        alertes=alertes,
        avis_generes=0,
        affaires_creees=affaires_creees,
        duree_recalcul_s=round(time.perf_counter() - t0, 2),
    ).model_dump()
