"""
Service d'import de fichiers d'inventaire de portefeuille.

Format attendu (CSV ou JSON) :
  ref_affaire, date (YYYY-MM-DD), code_isin, nbuc, vl [, nom_support]

Résultat dans mariadb_historique_support_w :
  UPSERT (id_source=id_affaire, id_support, date, nbuc, vl, valo=nbuc*vl)
  Alerte si écrasement d'une ligne existante.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, date as date_type

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.schemas.import_portefeuille import (
    InventaireRow,
    ImportAlerte,
    ImportPreviewResult,
    ImportCommitResult,
)
from src.services.recalcul_portefeuille import run_full_pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {"date", "code_isin", "nbuc", "vl"}
_OPTIONAL_FIELDS = {"ref_affaire", "id_affaire", "nom_support"}


def parse_inventaire_csv(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    for sep in (",", ";", "\t"):
        reader = csv.DictReader(io.StringIO(content), delimiter=sep)
        rows = list(reader)
        if rows and len(rows[0]) > 2:
            return rows
    return list(csv.DictReader(io.StringIO(content)))


def parse_inventaire_json(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, dict) and "inventaire" in data:
        return data["inventaire"]
    raise ValueError("Format JSON invalide : attendu une liste ou {rows: [...]} ou {inventaire: [...]}")


def _parse_date(val: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return None


def _validate_rows(raw_rows: list[dict]) -> tuple[list[InventaireRow], list[ImportAlerte]]:
    rows: list[InventaireRow] = []
    alertes: list[ImportAlerte] = []
    for i, raw in enumerate(raw_rows, start=1):
        try:
            row = InventaireRow.model_validate(raw)
            if not row.ref_affaire and row.id_affaire is None:
                alertes.append(ImportAlerte(
                    ligne=i,
                    code="missing_affaire",
                    message="ref_affaire et id_affaire sont tous les deux absents",
                ))
                continue
            if not row.code_isin:
                alertes.append(ImportAlerte(
                    ligne=i, code="missing_isin", message="code_isin manquant",
                ))
                continue
            if _parse_date(row.date) is None:
                alertes.append(ImportAlerte(
                    ligne=i, code="invalid_date", message=f"Date invalide : {row.date}",
                ))
                continue
            rows.append(row)
        except Exception as exc:
            alertes.append(ImportAlerte(
                ligne=i, code="parse_error", message=str(exc),
            ))
    return rows, alertes


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_client_by_fournisseur(db: Session, fournisseur: str, id_externe: str) -> int | None:
    """Retourne le client_id CRM correspondant à l'identifiant externe chez ce fournisseur."""
    row = db.execute(
        text(
            "SELECT client_id FROM mariadb_client_identifiants_fournisseur "
            "WHERE fournisseur = :f AND identifiant_externe = :id AND actif = 1 LIMIT 1"
        ),
        {"f": fournisseur.strip().upper(), "id": id_externe.strip()},
    ).fetchone()
    return (row[0] if not hasattr(row, "_mapping") else row._mapping["client_id"]) if row else None


def _resolve_societe_by_fournisseur(db: Session, fournisseur: str, id_externe: str) -> int | None:
    """Retourne le societe_id CRM correspondant à l'identifiant externe chez ce fournisseur."""
    row = db.execute(
        text(
            "SELECT societe_id FROM mariadb_societe_identifiants_fournisseur "
            "WHERE fournisseur = :f AND identifiant_externe = :id AND actif = 1 LIMIT 1"
        ),
        {"f": fournisseur.strip().upper(), "id": id_externe.strip()},
    ).fetchone()
    return (row[0] if not hasattr(row, "_mapping") else row._mapping["societe_id"]) if row else None


def _resolve_affaire_id(db: Session, row) -> int | None:  # InventaireRow | MouvementRow
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
        return r[0] if r else None
    return None


def _ensure_type_finalisation(db: Session) -> int:
    """Retourne l'id du type_evenement 'Finalisation de la création du contrat/compte'."""
    row = db.execute(
        text("SELECT id FROM mariadb_type_evenement WHERE libelle LIKE '%Finalisation%création%' LIMIT 1")
    ).fetchone()
    if row:
        return row[0]
    db.execute(
        text(
            "INSERT INTO mariadb_type_evenement (libelle, categorie) "
            "VALUES ('Finalisation de la création du contrat/compte', 'contrat')"
        )
    )
    db.flush()
    row = db.execute(
        text("SELECT id FROM mariadb_type_evenement WHERE libelle LIKE '%Finalisation%création%' LIMIT 1")
    ).fetchone()
    return row[0]


def _create_affaire_vide(
    db: Session,
    ref: str,
    id_societe_gestion: int | None = None,
    id_personne: int | None = None,
) -> int:
    """Crée une affaire à vide avec seulement la référence. Retourne l'id."""
    next_id = db.execute(
        text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_affaires")
    ).scalar()
    db.execute(
        text(
            """
            INSERT INTO mariadb_affaires (id, ref, id_societe_gestion, id_personne)
            VALUES (:id, :ref, :sg, :personne)
            """
        ),
        {"id": int(next_id), "ref": ref, "sg": id_societe_gestion, "personne": id_personne},
    )
    db.flush()
    logger.warning("IMPORT – Affaire créée à vide : ref=%s (id=%s, id_personne=%s)", ref, next_id, id_personne)
    return int(next_id)


def _create_finalisation_task(
    db: Session,
    id_affaire: int,
    id_societe_gestion: int | None = None,
    id_personne: int | None = None,
) -> None:
    """Génère une tâche 'Finalisation de la création du contrat/compte' liée à l'affaire et au client."""
    type_id = _ensure_type_finalisation(db)
    db.execute(
        text(
            """
            INSERT INTO mariadb_evenement
              (type_id, affaire_id, client_id, id_societe_gestion, date_evenement, statut, commentaire)
            VALUES
              (:type_id, :aff, :client, :sg, :dt, 'à faire',
               'Veuillez finir la complétude de la création de l\\'affaire')
            """
        ),
        {
            "type_id": type_id,
            "aff": id_affaire,
            "client": id_personne,
            "sg": id_societe_gestion,
            "dt": datetime.utcnow().strftime("%Y-%m-%d"),
        },
    )
    db.flush()


def _resolve_or_create_affaire(
    db: Session,
    row,  # InventaireRow | MouvementRow
    id_societe_gestion: int | None = None,
    create_if_missing: bool = False,
    id_personne: int | None = None,
    fournisseur: str | None = None,
) -> tuple[int | None, bool]:
    """
    Cherche l'affaire. Si absente et create_if_missing=True, la crée à vide
    et génère une tâche de finalisation.

    Si la row contient id_client_fournisseur et que fournisseur est fourni,
    le client CRM est résolu via mariadb_client_identifiants_fournisseur et
    utilisé comme id_personne (priorité sur le paramètre id_personne).

    Retourne (id_affaire | None, was_created).
    """
    # Résolution du client via la table de mapping fournisseur
    effective_personne = id_personne
    id_client_ext = getattr(row, "id_client_fournisseur", None)
    if id_client_ext and fournisseur:
        resolved = _resolve_client_by_fournisseur(db, fournisseur, id_client_ext)
        if resolved is not None:
            effective_personne = resolved
        else:
            logger.warning(
                "IMPORT – id_client_fournisseur '%s' inconnu chez '%s' – client non résolu",
                id_client_ext, fournisseur,
            )

    id_affaire = _resolve_affaire_id(db, row)
    if id_affaire is not None:
        return id_affaire, False

    if not create_if_missing:
        return None, False

    ref = (row.ref_affaire or "").strip() or f"IMP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    id_affaire = _create_affaire_vide(db, ref, id_societe_gestion, effective_personne)
    _create_finalisation_task(db, id_affaire, id_societe_gestion, effective_personne)
    return id_affaire, True


def _resolve_or_create_support(
    db: Session,
    code_isin: str,
    nom_support: str | None,
) -> tuple[int, bool]:
    """Retourne (id_support, created). Crée le support si ISIN inconnu."""
    row = db.execute(
        text("SELECT id FROM mariadb_support WHERE code_isin = :isin AND id IS NOT NULL LIMIT 1"),
        {"isin": code_isin},
    ).fetchone()
    if row:
        return int(row[0]), False

    # Créer à la volée
    next_id = db.execute(
        text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_support")
    ).scalar()
    db.execute(
        text(
            """
            INSERT INTO mariadb_support (id, code_isin, nom)
            VALUES (:id, :isin, :nom)
            """
        ),
        {"id": int(next_id), "isin": code_isin, "nom": nom_support or code_isin},
    )
    db.flush()
    logger.warning("IMPORT – ISIN inconnu créé : %s (id=%s)", code_isin, next_id)
    return int(next_id), True


def _upsert_historique_support(
    db: Session,
    id_affaire: int,
    id_support: int,
    snap_date: datetime,
    nbuc: float,
    vl: float,
    id_societe_gestion: int | None,
) -> bool:
    """Retourne True si écrasement d'une ligne existante."""
    existing = db.execute(
        text(
            """
            SELECT id FROM mariadb_historique_support_w
            WHERE id_source = :src AND id_support = :sup AND `date` = :dt
            LIMIT 1
            """
        ),
        {"src": id_affaire, "sup": id_support, "dt": snap_date},
    ).fetchone()

    valo = nbuc * vl
    now = datetime.utcnow()

    if existing:
        db.execute(
            text(
                """
                UPDATE mariadb_historique_support_w
                SET nbuc = :nbuc, vl = :vl, valo = :valo, modif_quand = :now
                WHERE id = :id
                """
            ),
            {"nbuc": nbuc, "vl": vl, "valo": valo, "now": now, "id": existing[0]},
        )
        return True
    else:
        next_id = db.execute(
            text("SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_historique_support_w")
        ).scalar()
        db.execute(
            text(
                """
                INSERT INTO mariadb_historique_support_w
                  (id, modif_quand, source, id_source, `date`, id_support, nbuc, vl, valo, id_societe_gestion)
                VALUES
                  (:id, :now, 'import', :src, :dt, :sup, :nbuc, :vl, :valo, :sg)
                """
            ),
            {
                "id": int(next_id),
                "now": now,
                "src": id_affaire,
                "dt": snap_date,
                "sup": id_support,
                "nbuc": nbuc,
                "vl": vl,
                "valo": valo,
                "sg": id_societe_gestion,
            },
        )
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Preview
# ──────────────────────────────────────────────────────────────────────────────

def preview_inventaire(
    db: Session,
    raw_rows: list[dict],
    fournisseur: str | None = None,
) -> ImportPreviewResult:
    rows, alertes = _validate_rows(raw_rows)

    apercu: list[dict] = []
    for i, row in enumerate(rows[:20]):  # max 20 lignes en aperçu
        # Résolution client fournisseur si disponible
        id_client_ext = getattr(row, "id_client_fournisseur", None)
        client_resolu: int | None = None
        if id_client_ext and fournisseur:
            client_resolu = _resolve_client_by_fournisseur(db, fournisseur, id_client_ext)
            if client_resolu is None:
                alertes.append(ImportAlerte(
                    ligne=i + 1,
                    code="client_fournisseur_inconnu",
                    message=f"id_client_fournisseur '{id_client_ext}' introuvable chez '{fournisseur}'",
                ))

        id_affaire = _resolve_affaire_id(db, row)
        support_row = db.execute(
            text("SELECT id, nom FROM mariadb_support WHERE code_isin = :isin LIMIT 1"),
            {"isin": row.code_isin},
        ).fetchone()
        affaire_a_creer = id_affaire is None
        apercu.append({
            "ligne": i + 1,
            "ref_affaire": row.ref_affaire or str(row.id_affaire),
            "id_client_fournisseur": id_client_ext,
            "client_resolu": client_resolu,
            "id_affaire": id_affaire,
            "code_isin": row.code_isin,
            "nom_support": support_row[1] if support_row else (row.nom_support or "INCONNU"),
            "date": row.date,
            "nbuc": row.nbuc,
            "vl": row.vl,
            "valo": round(row.nbuc * row.vl, 4),
            "affaire_trouvee": not affaire_a_creer,
            "affaire_a_creer": affaire_a_creer,
            "support_connu": support_row is not None,
        })
        if affaire_a_creer:
            alertes.append(ImportAlerte(
                ligne=i + 1,
                code="affaire_a_creer",
                message=(
                    f"Affaire '{row.ref_affaire or row.id_affaire}' introuvable – "
                    "sera créée à vide avec tâche de finalisation"
                ),
            ))

    return ImportPreviewResult(
        total_lignes=len(raw_rows),
        lignes_valides=len(rows),
        lignes_invalides=len(raw_rows) - len(rows),
        alertes=alertes,
        apercu=apercu,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Commit
# ──────────────────────────────────────────────────────────────────────────────

def commit_inventaire(
    db: Session,
    raw_rows: list[dict],
    id_societe_gestion: int | None = None,
    id_personne: int | None = None,
    fournisseur: str | None = None,
    identifiant_societe_externe: str | None = None,
    run_pipeline: bool = True,
) -> ImportCommitResult:
    # Résolution automatique de la société via le mapping fournisseur
    if id_societe_gestion is None and fournisseur and identifiant_societe_externe:
        resolved = _resolve_societe_by_fournisseur(db, fournisseur, identifiant_societe_externe)
        if resolved is not None:
            id_societe_gestion = resolved
        else:
            logger.warning(
                "IMPORT – société '%s' inconnue chez '%s' – id_societe_gestion non résolu",
                identifiant_societe_externe, fournisseur,
            )

    rows, alertes = _validate_rows(raw_rows)

    insere = 0
    mis_a_jour = 0
    affaires_creees = 0
    affected_affaire_ids: set[int] = set()

    for i, row in enumerate(rows, start=1):
        id_affaire, was_created = _resolve_or_create_affaire(
            db, row, id_societe_gestion, create_if_missing=True,
            id_personne=id_personne, fournisseur=fournisseur,
        )
        if id_affaire is None:
            alertes.append(ImportAlerte(
                ligne=i,
                code="affaire_inconnue",
                message=f"Impossible de résoudre ou créer l'affaire : {row.ref_affaire or row.id_affaire}",
            ))
            continue
        if was_created:
            affaires_creees += 1
            alertes.append(ImportAlerte(
                ligne=i,
                code="affaire_creee",
                message=(
                    f"Affaire '{row.ref_affaire}' créée à vide (id={id_affaire}) – "
                    "tâche de finalisation générée"
                ),
            ))

        id_support, created = _resolve_or_create_support(db, row.code_isin, row.nom_support)
        if id_support is None:
            alertes.append(ImportAlerte(
                ligne=i, code="isin_error",
                message=f"Support introuvable ou sans id pour ISIN {row.code_isin}",
            ))
            continue
        if created:
            alertes.append(ImportAlerte(
                ligne=i,
                code="unknown_isin",
                message=f"ISIN {row.code_isin} inconnu – support créé automatiquement",
            ))

        snap_date = _parse_date(row.date)
        overwritten = _upsert_historique_support(
            db, id_affaire, id_support, snap_date, row.nbuc, row.vl, id_societe_gestion
        )

        if overwritten:
            mis_a_jour += 1
            alertes.append(ImportAlerte(
                ligne=i,
                code="conflict_date",
                message=f"Ligne existante écrasée pour affaire {id_affaire}, ISIN {row.code_isin}, date {row.date}",
            ))
        else:
            insere += 1

        affected_affaire_ids.add(id_affaire)

    db.commit()

    duree = 0.0
    if run_pipeline and affected_affaire_ids:
        try:
            duree = run_full_pipeline(db, list(affected_affaire_ids))
        except Exception as exc:
            logger.error("Erreur pipeline recalcul : %s", exc)
            alertes.append(ImportAlerte(
                code="recalcul_error",
                message=f"Pipeline recalcul partiel : {exc}",
            ))

    return ImportCommitResult(
        insere=insere,
        mis_a_jour=mis_a_jour,
        alertes=alertes,
        affaires_creees=affaires_creees,
        duree_recalcul_s=round(duree, 2),
    )
