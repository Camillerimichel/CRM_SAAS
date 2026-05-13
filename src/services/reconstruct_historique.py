"""
Reconstruction des snapshots historiques mariadb_historique_support_w
à partir des mouvements importés.

Problème : après un import inventaire, chaque affaire n'a qu'une seule date de snapshot
(la date du fichier inventaire). Le pipeline Dietz/SRRI nécessite une série temporelle.
Ce service rejoue les mouvements chronologiquement pour reconstruire les snapshots manquants.

Algorithme par affaire :
  1. Récupère tous les mouvements triés par date
  2. Pour chaque date de mouvement (non déjà présente dans historique_support_w) :
     - Applique les mouvements du jour sur le cumul nbuc par support
     - Crée un snapshot pour TOUS les supports actifs (nbuc > 0)
       avec la dernière VL connue pour chaque support
  3. Insère les nouveaux snapshots
  4. Recompute PRMP pour chaque (affaire, support)
  5. Relance le pipeline A→E
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from itertools import count as _icount
from typing import Generator

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.services.recalcul_portefeuille import run_full_pipeline
from src.services.import_mouvements import _recompute_prmp

logger = logging.getLogger(__name__)


def _reconstruct_affaire_snapshots(
    db: Session,
    id_affaire: int,
    id_societe_gestion: int | None,
    next_id: "_icount",
) -> int:
    """
    Calcule et insère les snapshots historiques pour une affaire.
    Retourne le nombre de lignes insérées.
    """
    movements = db.execute(
        text(
            """
            SELECT m.vl_date, m.id_support, m.nb_uc, m.vl, r.sens
            FROM mouvement m
            JOIN mouvement_regle r ON r.id = m.id_mouvement_regle
            WHERE m.id_affaire = :aff
              AND (m.etat IS NULL OR m.etat != 8)
            ORDER BY m.vl_date ASC
            """
        ),
        {"aff": id_affaire},
    ).fetchall()

    if not movements:
        return 0

    existing_dates: set = set(
        row[0]
        for row in db.execute(
            text(
                "SELECT DISTINCT date FROM mariadb_historique_support_w WHERE id_source = :aff"
            ),
            {"aff": id_affaire},
        ).fetchall()
    )

    # Group movements by date
    movements_by_date: dict = defaultdict(list)
    for m in movements:
        movements_by_date[m[0]].append(m)

    nbuc_cur: dict[int, float] = {}
    last_vl: dict[int, float] = {}
    inserted = 0

    for snap_date in sorted(movements_by_date.keys()):
        for m in movements_by_date[snap_date]:
            _, id_support, nb_uc_str, vl_str, sens = m
            try:
                nb_uc = float(str(nb_uc_str).replace(",", ".")) if nb_uc_str is not None else 0.0
                vl_val = float(str(vl_str).replace(",", ".")) if vl_str is not None else 0.0
            except (ValueError, TypeError):
                nb_uc = 0.0
                vl_val = 0.0

            if sens is not None and int(sens) > 0:
                nbuc_cur[id_support] = nbuc_cur.get(id_support, 0.0) + nb_uc
            else:
                nbuc_cur[id_support] = max(0.0, nbuc_cur.get(id_support, 0.0) - nb_uc)

            if vl_val > 0:
                last_vl[id_support] = vl_val

        if snap_date in existing_dates:
            continue

        for id_support, nbuc in nbuc_cur.items():
            if nbuc <= 0.001:
                continue
            vl_val = last_vl.get(id_support, 0.0)
            valo = nbuc * vl_val
            db.execute(
                text(
                    """
                    INSERT INTO mariadb_historique_support_w
                      (id, modif_quand, source, id_source, date, id_support,
                       nbuc, vl, prmp, valo, id_societe_gestion)
                    VALUES (:id, NOW(), 'import_reconstruit', :id_source, :snap_date, :id_support,
                            :nbuc, :vl, 0.0, :valo, :sg)
                    """
                ),
                {
                    "id": next(next_id),
                    "id_source": id_affaire,
                    "snap_date": snap_date,
                    "id_support": id_support,
                    "nbuc": nbuc,
                    "vl": vl_val,
                    "valo": valo,
                    "sg": id_societe_gestion,
                },
            )
            inserted += 1

    return inserted


def iter_reconstruct_historique(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> Generator[dict, None, None]:
    """
    Reconstruit mariadb_historique_support_w depuis les mouvements pour les affaires
    n'ayant qu'une seule date de snapshot (affaires importées via inventaire).

    Émet des événements de progression SSE :
      {'type': 'progress', 'phase': str, 'current': int, 'total': int}
    Puis un événement final :
      {'type': 'done', 'nb_affaires': int, 'nb_snapshots': int, 'duree_s': float}
    """
    t0 = time.perf_counter()

    if affaire_ids:
        candidates = affaire_ids
    else:
        rows = db.execute(
            text(
                """
                SELECT h.id_source, MAX(a.id_societe_gestion)
                FROM mariadb_historique_support_w h
                LEFT JOIN mariadb_affaires a ON a.id = h.id_source
                GROUP BY h.id_source
                HAVING COUNT(DISTINCT h.date) = 1
                """
            )
        ).fetchall()
        candidates = [r[0] for r in rows if r[0] is not None]
        societe_map: dict[int, int | None] = {r[0]: r[1] for r in rows if r[0] is not None}

    if not candidates:
        yield {"type": "done", "nb_affaires": 0, "nb_snapshots": 0, "duree_s": 0.0}
        return

    if affaire_ids:
        societe_rows = db.execute(
            text(
                "SELECT id, id_societe_gestion FROM mariadb_affaires WHERE id IN :ids"
            ),
            {"ids": tuple(candidates)},
        ).fetchall()
        societe_map = {r[0]: r[1] for r in societe_rows}

    total = len(candidates)
    yield {"type": "progress", "phase": "snapshots", "current": 0, "total": total}

    next_id = _icount(
        int(
            db.execute(
                text("SELECT COALESCE(MAX(id),0)+1 FROM mariadb_historique_support_w")
            ).scalar()
        )
    )

    nb_snapshots = 0
    affected_affaire_ids: list[int] = []
    affected_pairs: set[tuple[int, int]] = set()

    for i, id_affaire in enumerate(candidates, 1):
        sg = societe_map.get(id_affaire)
        inserted = _reconstruct_affaire_snapshots(db, id_affaire, sg, next_id)
        if inserted > 0:
            nb_snapshots += inserted
            affected_affaire_ids.append(id_affaire)
            pairs = db.execute(
                text(
                    "SELECT DISTINCT id_support FROM mariadb_historique_support_w"
                    " WHERE id_source = :aff AND source = 'import_reconstruit'"
                ),
                {"aff": id_affaire},
            ).fetchall()
            for p in pairs:
                affected_pairs.add((id_affaire, p[0]))

        if i % 20 == 0 or i == total:
            yield {"type": "progress", "phase": "snapshots", "current": i, "total": total}

    db.commit()

    # PRMP recomputation for affected pairs
    pairs_list = list(affected_pairs)
    n_pairs = len(pairs_list)
    if n_pairs:
        yield {"type": "progress", "phase": "prmp", "current": 0, "total": n_pairs}
        for pi, (id_affaire, id_support) in enumerate(pairs_list, 1):
            try:
                _recompute_prmp(db, id_affaire, id_support)
            except Exception as exc:
                logger.warning("PRMP recompute failed (%s,%s): %s", id_affaire, id_support, exc)
            if pi % 20 == 0 or pi == n_pairs:
                yield {"type": "progress", "phase": "prmp", "current": pi, "total": n_pairs}
        db.commit()

    # Pipeline A→E
    if affected_affaire_ids:
        yield {
            "type": "progress",
            "phase": "pipeline",
            "current": 0,
            "total": len(affected_affaire_ids),
        }
        try:
            run_full_pipeline(db, affected_affaire_ids)
        except Exception as exc:
            logger.error("Pipeline error in reconstruction: %s", exc)

    yield {
        "type": "done",
        "nb_affaires": len(affected_affaire_ids),
        "nb_snapshots": nb_snapshots,
        "duree_s": round(time.perf_counter() - t0, 2),
    }
