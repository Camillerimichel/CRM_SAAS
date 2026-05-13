"""
Service d'import de produits (contrats génériques) depuis fichier fournisseur.

Format CSV attendu :
  [PRODUITS];;...
  CODE;NOM_LONG;...
"""
from __future__ import annotations

import csv
import io
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


def parse_produits_csv(data: bytes) -> list[dict]:
    """Parse le CSV fournisseur (CODE;NOM;...) et retourne une liste de dicts {code, nom}.

    Lève ValueError si le fichier ne contient pas de section [PRODUITS].
    """
    try:
        text_data = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_data = data.decode("latin-1", errors="replace")

    reader = csv.reader(io.StringIO(text_data), delimiter=";")
    all_lines = list(reader)

    # Vérification : la première colonne non-vide doit contenir [PRODUITS]
    has_section = any(
        parts and parts[0].strip().upper().startswith("[PRODUITS]")
        for parts in all_lines
    )
    if not has_section:
        raise ValueError(
            "Format de fichier non reconnu : la section [PRODUITS] est absente. "
            "Vérifiez que vous importez bien un fichier produits fournisseur."
        )

    rows = []
    for parts in all_lines:
        if not parts:
            continue
        code = parts[0].strip()
        if not code or code.startswith("["):
            continue
        nom = parts[1].strip() if len(parts) > 1 else ""
        if not nom:
            continue
        rows.append({"code": code, "nom": nom})
    return rows


def preview_produits(
    db: Session,
    rows: list[dict],
    id_ctg: int,
    id_societe: Optional[int],
) -> dict:
    """Retourne un aperçu : quels produits existent déjà, lesquels seront créés."""
    preview = []
    for r in rows:
        existing = db.execute(
            text("SELECT id FROM mariadb_affaires_generique WHERE nom_contrat = :nom LIMIT 1"),
            {"nom": r["nom"]},
        ).fetchone()
        preview.append({
            "code": r["code"],
            "nom": r["nom"],
            "statut": "existant" if existing else "nouveau",
            "id": (existing._mapping["id"] if hasattr(existing, "_mapping") else existing[0]) if existing else None,
        })
    nb_nouveaux = sum(1 for p in preview if p["statut"] == "nouveau")
    nb_existants = sum(1 for p in preview if p["statut"] == "existant")
    return {
        "rows": preview,
        "nb_nouveaux": nb_nouveaux,
        "nb_existants": nb_existants,
        "id_ctg": id_ctg,
        "id_societe": id_societe,
    }


def commit_produits(
    db: Session,
    rows: list[dict],
    id_ctg: int,
    id_societe: Optional[int],
) -> dict:
    """Insère les nouveaux produits dans mariadb_affaires_generique."""
    inserted = 0
    skipped = 0
    for r in rows:
        existing = db.execute(
            text("SELECT id FROM mariadb_affaires_generique WHERE nom_contrat = :nom LIMIT 1"),
            {"nom": r["nom"]},
        ).fetchone()
        if existing:
            skipped += 1
            continue
        next_id = db.execute(
            text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_affaires_generique")
        ).scalar()
        db.execute(
            text(
                "INSERT INTO mariadb_affaires_generique "
                "(id, nom_contrat, description, id_ctg, id_societe, actif) "
                "VALUES (:id, :nom, :desc, :ctg, :soc, '1')"
            ),
            {
                "id":  next_id,
                "nom": r["nom"],
                "desc": r["code"],
                "ctg": id_ctg,
                "soc": id_societe,
            },
        )
        inserted += 1
    db.commit()
    return {"inserted": inserted, "skipped": skipped}
