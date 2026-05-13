"""
Service d'import de fichiers de mouvements de portefeuille.

Format attendu (CSV ou JSON) :
  ref_affaire, date (YYYY-MM-DD), code_isin, code_mouvement, nbuc, vl [, montant_ope, frais]

Pipeline après commit :
  1. Insert mouvement + avis (groupés par (id_affaire, date))
  2. Recompute PRMP (CUMP) sur historique_support pour les (id_affaire, id_support) impactés
  3. run_full_pipeline (agrégation + Dietz/SRRI)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from itertools import count as _icount
from typing import Generator

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.schemas.import_portefeuille import (
    MouvementRow,
    ImportAlerte,
    ImportPreviewResult,
    ImportCommitResult,
)
from src.services.recalcul_portefeuille import run_full_pipeline
from src.services.import_inventaire import (
    _resolve_affaire_id,
    _resolve_or_create_affaire,
    _resolve_or_create_support,
    _resolve_societe_by_fournisseur,
    _parse_date,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_mouvements_csv(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    for sep in (",", ";", "\t"):
        reader = csv.DictReader(io.StringIO(content), delimiter=sep)
        rows = list(reader)
        if rows and len(rows[0]) > 2:
            return rows
    return list(csv.DictReader(io.StringIO(content)))


def parse_mouvements_json(content: str | bytes) -> list[dict]:
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return data
    for key in ("rows", "mouvements", "movements"):
        if key in data:
            return data[key]
    raise ValueError("Format JSON invalide : attendu une liste ou {mouvements: [...]}")


def _validate_rows(raw_rows: list[dict]) -> tuple[list[MouvementRow], list[ImportAlerte]]:
    rows: list[MouvementRow] = []
    alertes: list[ImportAlerte] = []
    for i, raw in enumerate(raw_rows, start=1):
        try:
            row = MouvementRow.model_validate(raw)
            if not row.ref_affaire and row.id_affaire is None:
                alertes.append(ImportAlerte(
                    ligne=i, code="missing_affaire",
                    message="ref_affaire et id_affaire sont tous les deux absents",
                ))
                continue
            if not row.code_isin:
                alertes.append(ImportAlerte(
                    ligne=i, code="missing_isin", message="code_isin manquant",
                ))
                continue
            if not row.code_mouvement:
                alertes.append(ImportAlerte(
                    ligne=i, code="missing_code_mouvement", message="code_mouvement manquant",
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

def _load_mouvement_regle(db: Session) -> dict[str, dict]:
    rows = db.execute(
        text("SELECT id, code, titre, sens, investi, prmp FROM mouvement_regle")
    ).fetchall()
    return {
        r[1].strip().upper(): {"id": r[0], "code": r[1], "titre": r[2],
                                "sens": r[3], "investi": r[4], "prmp": r[5]}
        for r in rows if r[1]
    }


def _get_affaire_societe(db: Session, id_affaire: int) -> int | None:
    row = db.execute(
        text("SELECT id_societe_gestion FROM mariadb_affaires WHERE id = :id"),
        {"id": id_affaire},
    ).fetchone()
    return row[0] if row else None


# ──────────────────────────────────────────────────────────────────────────────
# PRMP recomputation (CUMP)
# ──────────────────────────────────────────────────────────────────────────────

def _recompute_prmp(db: Session, id_affaire: int, id_support: int) -> None:
    movements = db.execute(
        text(
            """
            SELECT m.vl_date, m.nb_uc, m.vl, r.sens, r.prmp AS affects_prmp
            FROM mouvement m
            JOIN mouvement_regle r ON r.id = m.id_mouvement_regle
            WHERE m.id_affaire = :aff AND m.id_support = :sup
              AND (m.etat IS NULL OR m.etat != 8)
            ORDER BY m.vl_date ASC
            """
        ),
        {"aff": id_affaire, "sup": id_support},
    ).fetchall()
    if not movements:
        return

    nbuc_cur = 0.0
    prmp_cur = 0.0
    cump_series: list[tuple[datetime, float]] = []

    for m in movements:
        vl_date, nb_uc, vl_str, sens, affects_prmp = m
        try:
            nb_uc = float(str(nb_uc).replace(",", ".")) if nb_uc is not None else 0.0
            vl_val = float(str(vl_str).replace(",", ".")) if vl_str is not None else 0.0
        except (ValueError, TypeError):
            nb_uc = 0.0
            vl_val = 0.0

        if sens is not None and int(sens) > 0 and affects_prmp:
            add = abs(nb_uc)
            total = nbuc_cur + add
            if total > 0:
                prmp_cur = (nbuc_cur * prmp_cur + add * vl_val) / total
            nbuc_cur = total
        elif sens is not None and int(sens) < 0:
            nbuc_cur = max(0.0, nbuc_cur - abs(nb_uc))

        cump_series.append((vl_date, prmp_cur))

    snapshots = db.execute(
        text(
            """
            SELECT id, `date`
            FROM mariadb_historique_support_w
            WHERE id_source = :aff AND id_support = :sup
            ORDER BY `date` ASC
            """
        ),
        {"aff": id_affaire, "sup": id_support},
    ).fetchall()
    if not snapshots:
        return

    cump_sorted = sorted(cump_series, key=lambda x: x[0])
    for snap_id, snap_date in snapshots:
        prmp_val = 0.0
        for mv_date, mv_prmp in cump_sorted:
            if mv_date <= snap_date:
                prmp_val = mv_prmp
            else:
                break
        db.execute(
            text("UPDATE mariadb_historique_support_w SET prmp = :p WHERE id = :id"),
            {"p": prmp_val, "id": snap_id},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Preview
# ──────────────────────────────────────────────────────────────────────────────

def preview_mouvements(
    db: Session,
    raw_rows: list[dict],
    fournisseur: str | None = None,
) -> ImportPreviewResult:
    rows, alertes = _validate_rows(raw_rows)
    regle_map = _load_mouvement_regle(db)

    apercu: list[dict] = []
    for i, row in enumerate(rows[:20]):
        id_affaire = _resolve_affaire_id(db, row)
        regle = regle_map.get(row.code_mouvement.upper())
        support_row = db.execute(
            text("SELECT id, nom FROM mariadb_support WHERE code_isin = :isin LIMIT 1"),
            {"isin": row.code_isin},
        ).fetchone()
        montant = row.montant_ope if row.montant_ope is not None else abs(row.nbuc * row.vl)
        affaire_a_creer = id_affaire is None
        apercu.append({
            "ligne": i + 1,
            "ref_affaire": row.ref_affaire or str(row.id_affaire),
            "id_affaire": id_affaire,
            "date": row.date,
            "code_isin": row.code_isin,
            "nom_support": support_row[1] if support_row else "INCONNU",
            "code_mouvement": row.code_mouvement,
            "libelle_mouvement": regle["titre"] if regle else "INCONNU",
            "nbuc": row.nbuc,
            "vl": row.vl,
            "montant_ope": round(montant, 4),
            "frais": row.frais or 0,
            "affaire_trouvee": not affaire_a_creer,
            "affaire_a_creer": affaire_a_creer,
            "code_mouvement_connu": regle is not None,
        })
        if affaire_a_creer:
            alertes.append(ImportAlerte(
                ligne=i + 1, code="affaire_a_creer",
                message=(
                    f"Affaire '{row.ref_affaire or row.id_affaire}' introuvable – "
                    "sera créée à vide avec tâche de finalisation"
                ),
            ))
        if regle is None:
            alertes.append(ImportAlerte(
                ligne=i + 1, code="code_mouvement_inconnu",
                message=f"Code mouvement inconnu : {row.code_mouvement}",
            ))

    return ImportPreviewResult(
        total_lignes=len(raw_rows),
        lignes_valides=len(rows),
        lignes_invalides=len(raw_rows) - len(rows),
        alertes=alertes,
        apercu=apercu,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Commit — générateur de progression
# ──────────────────────────────────────────────────────────────────────────────

def iter_commit_mouvements(
    db: Session,
    raw_rows: list[dict],
    id_societe_gestion: int | None = None,
    id_personne: int | None = None,
    fournisseur: str | None = None,
    identifiant_societe_externe: str | None = None,
    run_pipeline: bool = True,
) -> Generator[dict, None, None]:
    """
    Générateur qui produit des événements de progression :
      {'type': 'progress', 'phase': str, 'current': int, 'total': int}
    puis un événement final :
      {'type': 'done', 'insere': int, 'mis_a_jour': int, 'alertes': [...], ...}
    """
    t0 = time.perf_counter()

    if id_societe_gestion is None and fournisseur and identifiant_societe_externe:
        resolved = _resolve_societe_by_fournisseur(db, fournisseur, identifiant_societe_externe)
        if resolved is not None:
            id_societe_gestion = resolved
        else:
            logger.warning(
                "IMPORT – société '%s' inconnue chez '%s'",
                identifiant_societe_externe, fournisseur,
            )

    rows, alertes = _validate_rows(raw_rows)
    total = len(rows)
    regle_map = _load_mouvement_regle(db)

    insere = 0
    mis_a_jour = 0
    avis_generes = 0
    affaires_creees = 0
    affected_affaire_ids: set[int] = set()
    affected_pairs: set[tuple[int, int]] = set()
    avis_groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    resolved_rows: list[tuple] = []

    # ── Phase 1 : résolution ────────────────────────────────────────────────
    yield {'type': 'progress', 'phase': 'résolution', 'current': 0, 'total': total}
    for i, row in enumerate(rows, 1):
        if i % 200 == 0:
            yield {'type': 'progress', 'phase': 'résolution', 'current': i, 'total': total}

        id_affaire, was_created = _resolve_or_create_affaire(
            db, row, id_societe_gestion, create_if_missing=True,
            id_personne=id_personne, fournisseur=fournisseur,
        )
        if id_affaire is None:
            alertes.append(ImportAlerte(
                ligne=i, code="affaire_inconnue",
                message=f"Impossible de résoudre ou créer l'affaire : {row.ref_affaire or row.id_affaire}",
            ))
            continue
        if was_created:
            affaires_creees += 1
            alertes.append(ImportAlerte(
                ligne=i, code="affaire_creee",
                message=f"Affaire '{row.ref_affaire}' créée à vide (id={id_affaire})",
            ))

        regle = regle_map.get(row.code_mouvement.upper())
        if regle is None:
            alertes.append(ImportAlerte(
                ligne=i, code="code_mouvement_inconnu",
                message=f"Code mouvement inconnu : {row.code_mouvement}",
            ))
            continue

        id_support, created = _resolve_or_create_support(db, row.code_isin, None)
        if id_support is None:
            alertes.append(ImportAlerte(
                ligne=i, code="isin_error",
                message=f"Support introuvable ou sans id pour ISIN {row.code_isin}",
            ))
            continue
        if created:
            alertes.append(ImportAlerte(
                ligne=i, code="unknown_isin",
                message=f"ISIN {row.code_isin} inconnu – support créé automatiquement",
            ))

        snap_date = _parse_date(row.date)
        montant = row.montant_ope if row.montant_ope is not None else abs(row.nbuc * row.vl)
        frais = row.frais or 0.0

        avis_groups[(id_affaire, row.date)].append({
            "libelle": regle["titre"], "code": regle["code"], "sens": regle["sens"],
            "nbuc": row.nbuc, "vl": row.vl, "montant": montant,
        })
        resolved_rows.append((row, id_affaire, id_support, regle["id"], snap_date, montant, frais))
        affected_affaire_ids.add(id_affaire)
        affected_pairs.add((id_affaire, id_support))

    yield {'type': 'progress', 'phase': 'résolution', 'current': total, 'total': total}
    db.commit()

    # ── Phase 2 : avis (pré-allocation d'IDs — un seul MAX) ────────────────
    total_avis = len(avis_groups)
    next_avis_id = _icount(
        int(db.execute(text("SELECT COALESCE(MAX(id),0)+1 FROM avis")).scalar())
    )
    avis_id_map: dict[tuple[int, str], int] = {}

    yield {'type': 'progress', 'phase': 'avis', 'current': 0, 'total': total_avis}
    for ai, ((id_affaire, date_str), lines_data) in enumerate(avis_groups.items(), 1):
        if ai % 50 == 0:
            yield {'type': 'progress', 'phase': 'avis', 'current': ai, 'total': total_avis}

        entree = [
            f"{d['libelle']} – {d['nbuc']} UC × {d['vl']} = {d['montant']:.2f}€"
            for d in lines_data if (d["sens"] or 0) > 0
        ]
        sortie = [
            f"{d['libelle']} – {d['nbuc']} UC × {d['vl']} = {d['montant']:.2f}€"
            for d in lines_data if (d["sens"] or 0) < 0
        ]
        snap_date = _parse_date(date_str)
        ref = f"IMP-{id_affaire}-{snap_date.strftime('%Y%m%d')}"

        existing = db.execute(
            text("SELECT id FROM avis WHERE reference = :ref AND id_affaire = :aff LIMIT 1"),
            {"ref": ref, "aff": id_affaire},
        ).fetchone()
        if existing:
            avis_id = existing[0]
            db.execute(
                text("UPDATE avis SET entree = :e, sortie = :s WHERE id = :id"),
                {"e": "\n".join(entree) or None, "s": "\n".join(sortie) or None, "id": avis_id},
            )
        else:
            avis_id = next(next_avis_id)
            db.execute(
                text(
                    "INSERT INTO avis (id, reference, `date`, id_affaire, id_etape, etat,"
                    " entree, sortie, commentaire)"
                    " VALUES (:id, :ref, :dt, :aff, 5, 5, :e, :s, 'Import automatique')"
                ),
                {
                    "id": avis_id, "ref": ref, "dt": snap_date, "aff": id_affaire,
                    "e": "\n".join(entree) or None, "s": "\n".join(sortie) or None,
                },
            )
            avis_generes += 1
        avis_id_map[(id_affaire, date_str)] = avis_id

    yield {'type': 'progress', 'phase': 'avis', 'current': total_avis, 'total': total_avis}
    db.flush()

    # ── Phase 3 : mouvements (pré-allocation d'IDs — un seul MAX) ──────────
    total_mouv = len(resolved_rows)
    next_mouv_id = _icount(
        int(db.execute(text("SELECT COALESCE(MAX(id),0)+1 FROM mouvement")).scalar())
    )

    yield {'type': 'progress', 'phase': 'import', 'current': 0, 'total': total_mouv}
    for i, (row, id_affaire, id_support, id_regle, snap_date, montant, frais) in enumerate(resolved_rows, 1):
        if i % 200 == 0:
            yield {'type': 'progress', 'phase': 'import', 'current': i, 'total': total_mouv}

        existing = db.execute(
            text(
                "SELECT id FROM mouvement"
                " WHERE id_affaire = :aff AND id_support = :sup"
                "   AND id_mouvement_regle = :regle AND vl_date = :dt LIMIT 1"
            ),
            {"aff": id_affaire, "sup": id_support, "regle": id_regle, "dt": snap_date},
        ).fetchone()
        if existing:
            db.execute(
                text(
                    "UPDATE mouvement SET montant_ope=:m, frais=:f, vl=:v, nb_uc=:n,"
                    " modif_quand=:now WHERE id=:id"
                ),
                {"m": str(montant), "f": str(frais), "v": str(row.vl), "n": str(row.nbuc),
                 "now": datetime.utcnow(), "id": existing[0]},
            )
            mis_a_jour += 1
            alertes.append(ImportAlerte(
                code="doublon_mouvement",
                message=f"Mouvement mis à jour : affaire {id_affaire}, ISIN {row.code_isin}, date {row.date}",
            ))
        else:
            mid = next(next_mouv_id)
            id_avis = avis_id_map.get((id_affaire, row.date), 0)
            sg = id_societe_gestion or _get_affaire_societe(db, id_affaire)
            db.execute(
                text(
                    "INSERT INTO mouvement"
                    " (id, modif_quand, id_affaire, id_mouvement_regle, id_support, id_avis,"
                    "  montant_ope, frais, vl_date, date_sp, vl, nb_uc, etat)"
                    " VALUES (:id, :now, :aff, :regle, :sup, :avis,"
                    "  :m, :f, :dt, :dt, :v, :n, 5)"
                ),
                {"id": mid, "now": datetime.utcnow(), "aff": id_affaire, "regle": id_regle,
                 "sup": id_support, "avis": id_avis, "m": str(montant), "f": str(frais),
                 "dt": snap_date, "v": str(row.vl), "n": str(row.nbuc)},
            )
            insere += 1

    yield {'type': 'progress', 'phase': 'import', 'current': total_mouv, 'total': total_mouv}
    db.commit()

    # ── Phase 4 : PRMP ─────────────────────────────────────────────────────
    pairs = list(affected_pairs)
    yield {'type': 'progress', 'phase': 'prmp', 'current': 0, 'total': len(pairs)}
    for pi, (id_affaire, id_support) in enumerate(pairs, 1):
        if pi % 10 == 0:
            yield {'type': 'progress', 'phase': 'prmp', 'current': pi, 'total': len(pairs)}
        try:
            _recompute_prmp(db, id_affaire, id_support)
        except Exception as exc:
            logger.warning("PRMP recompute failed (%s,%s): %s", id_affaire, id_support, exc)
    yield {'type': 'progress', 'phase': 'prmp', 'current': len(pairs), 'total': len(pairs)}
    db.commit()

    # ── Phase 5 : pipeline ──────────────────────────────────────────────────
    duree = 0.0
    if run_pipeline and affected_affaire_ids:
        yield {'type': 'progress', 'phase': 'pipeline', 'current': 0,
               'total': len(affected_affaire_ids)}
        try:
            duree = run_full_pipeline(db, list(affected_affaire_ids))
        except Exception as exc:
            logger.error("Erreur pipeline recalcul : %s", exc)
            alertes.append(ImportAlerte(code="recalcul_error", message=f"Pipeline : {exc}"))

    yield {
        'type': 'done',
        'insere': insere,
        'mis_a_jour': mis_a_jour,
        'alertes': [a.model_dump() for a in alertes],
        'avis_generes': avis_generes,
        'affaires_creees': affaires_creees,
        'duree_recalcul_s': round(time.perf_counter() - t0, 2),
    }


def commit_mouvements(
    db: Session,
    raw_rows: list[dict],
    id_societe_gestion: int | None = None,
    id_personne: int | None = None,
    fournisseur: str | None = None,
    identifiant_societe_externe: str | None = None,
    run_pipeline: bool = True,
) -> dict:
    for event in iter_commit_mouvements(
        db, raw_rows,
        id_societe_gestion=id_societe_gestion,
        id_personne=id_personne,
        fournisseur=fournisseur,
        identifiant_societe_externe=identifiant_societe_externe,
        run_pipeline=run_pipeline,
    ):
        if event.get('type') == 'done':
            return ImportCommitResult(
                insere=event['insere'],
                mis_a_jour=event['mis_a_jour'],
                alertes=[ImportAlerte(**a) for a in event['alertes']],
                avis_generes=event['avis_generes'],
                affaires_creees=event['affaires_creees'],
                duree_recalcul_s=event['duree_recalcul_s'],
            ).model_dump()
    return ImportCommitResult(insere=0, mis_a_jour=0, alertes=[]).model_dump()
