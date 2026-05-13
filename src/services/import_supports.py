"""Service d'import de supports financiers depuis fichier fournisseur CSV."""
from __future__ import annotations

import csv
import io
import re
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

_UUID_RE = re.compile(r'^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$')
_ISIN_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{10}$')


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _parse_rows(data: bytes) -> list[list[str]]:
    """Parse le CSV et retourne uniquement les lignes de la section [SUPPORTS]."""
    reader = csv.reader(io.StringIO(_decode(data)), delimiter=";")
    in_supports = False
    rows = []
    for parts in reader:
        if not parts:
            continue
        first = parts[0].strip()
        if first.upper().startswith("[SUPPORTS]"):
            in_supports = True
            continue
        if first.startswith("[") and in_supports:
            break
        if in_supports and first:
            rows.append([p.strip() for p in parts])
    if not rows:
        raise ValueError(
            "Aucun support trouvé. Vérifiez que le fichier contient une section [SUPPORTS]."
        )
    return rows


def _find_nonempty_cols(rows: list[list[str]]) -> list[int]:
    """Retourne les indices de colonnes avec au moins 10% de valeurs non-vides."""
    if not rows:
        return []
    max_cols = max(len(r) for r in rows)
    result = []
    for col_idx in range(max_cols):
        values = [r[col_idx] if col_idx < len(r) else "" for r in rows]
        if sum(1 for v in values if v) >= max(1, len(rows) * 0.1):
            result.append(col_idx)
    return result


def _detect_types(rows: list[list[str]], col_indices: list[int]) -> dict[str, Optional[int]]:
    """Auto-détecte quel indice de colonne correspond à hex/isin/libelle."""
    detected: dict[str, Optional[int]] = {"hex": None, "isin": None, "libelle": None}
    sample = rows[:30]
    for col_idx in col_indices:
        values = [r[col_idx] if col_idx < len(r) else "" for r in sample]
        non_empty = [v for v in values if v]
        if not non_empty:
            continue
        uuid_ratio = sum(1 for v in non_empty if _UUID_RE.match(v)) / len(non_empty)
        isin_ratio = sum(1 for v in non_empty if _ISIN_RE.match(v)) / len(non_empty)
        if uuid_ratio > 0.5 and detected["hex"] is None:
            detected["hex"] = col_idx
        elif isin_ratio > 0.3 and detected["isin"] is None:
            detected["isin"] = col_idx
        elif detected["libelle"] is None:
            detected["libelle"] = col_idx
    return detected


def detect_supports(data: bytes) -> dict:
    """Parse le fichier, détecte les colonnes et retourne un aperçu des 5 premières lignes."""
    rows = _parse_rows(data)
    col_indices = _find_nonempty_cols(rows)
    detected = _detect_types(rows, col_indices)

    sample = []
    for row in rows[:5]:
        sample.append([row[i] if i < len(row) else "" for i in col_indices])

    col_labels = []
    for i, col_idx in enumerate(col_indices):
        examples = [r[col_idx] if col_idx < len(r) else "" for r in rows[:3] if col_idx < len(r) and r[col_idx]]
        ex = examples[0][:30] if examples else ""
        col_labels.append({"index": col_idx, "label": f"Colonne {col_idx + 1}", "example": ex})

    return {
        "total": len(rows),
        "columns": col_labels,
        "sample": sample,
        "detected": detected,
    }


def preview_supports(
    db: Session,
    data: bytes,
    col_isin: int,
    col_libelle: int,
) -> dict:
    """Retourne l'aperçu complet avec statut existant/nouveau pour chaque support."""
    rows = _parse_rows(data)
    preview = []
    nb_nouveaux = 0
    nb_existants = 0
    nb_ignores = 0

    for i, row in enumerate(rows):
        isin = row[col_isin].strip() if col_isin < len(row) else ""
        libelle = row[col_libelle].strip() if col_libelle < len(row) else ""
        if not isin and not libelle:
            nb_ignores += 1
            continue
        existing = None
        if isin:
            existing = db.execute(
                text("SELECT id FROM mariadb_support WHERE code_isin = :isin LIMIT 1"),
                {"isin": isin},
            ).fetchone()
        statut = "existant" if existing else "nouveau"
        if statut == "nouveau":
            nb_nouveaux += 1
        else:
            nb_existants += 1
        preview.append({
            "ligne": i + 1,
            "code_isin": isin or "—",
            "nom": libelle or "—",
            "statut": statut,
        })

    return {
        "rows": preview,
        "total": len(rows),
        "nb_nouveaux": nb_nouveaux,
        "nb_existants": nb_existants,
        "nb_ignores": nb_ignores,
    }


def commit_supports(
    db: Session,
    data: bytes,
    col_isin: int,
    col_libelle: int,
) -> dict:
    """Insère les nouveaux supports et met à jour les existants dans mariadb_support."""
    rows = _parse_rows(data)
    inserted = 0
    updated = 0
    ignored = 0

    for row in rows:
        isin = row[col_isin].strip() if col_isin < len(row) else ""
        libelle = row[col_libelle].strip() if col_libelle < len(row) else ""
        if not isin and not libelle:
            ignored += 1
            continue
        if isin:
            existing = db.execute(
                text("SELECT id FROM mariadb_support WHERE code_isin = :isin LIMIT 1"),
                {"isin": isin},
            ).fetchone()
        else:
            existing = db.execute(
                text("SELECT id FROM mariadb_support WHERE nom = :nom LIMIT 1"),
                {"nom": libelle},
            ).fetchone()
        if existing:
            if libelle:
                db.execute(
                    text("UPDATE mariadb_support SET nom = :nom WHERE id = :id"),
                    {"nom": libelle, "id": existing[0]},
                )
            updated += 1
        else:
            db.execute(
                text("INSERT INTO mariadb_support (code_isin, nom) VALUES (:isin, :nom)"),
                {"isin": isin or None, "nom": libelle or None},
            )
            inserted += 1

    db.commit()
    return {"inserted": inserted, "updated": updated, "ignored": ignored}
