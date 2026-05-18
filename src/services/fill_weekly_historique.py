"""
Comblement hebdomadaire de mariadb_historique_affaire_w.

Pour les affaires importées, le portefeuille n'est snapshottée qu'aux dates
de mouvement (trimestriel/semestriel). Ce service génère des points hebdomadaires
intermédiaires en portant forward le nb_uc de chaque support et en multipliant
par la VL hebdo de mariadb_support_val.

Après fill : avg_days ≈ 7j → _build_freq_map classifie win=52 → SRRI standard.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Generator

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _fill_affaire_weekly(db: Session, aff_id: int, sg: int | None, force_recalc: bool = False) -> int:
    """
    Insère (ou recalcule si force_recalc=True) les points hebdomadaires dans
    historique_affaire_w pour une affaire. Retourne le nombre de lignes touchées.

    Quand force_recalc=True :
    - Met à jour la colonne valo des lignes existantes avec la valeur recalculée
    - Inclut les dates snapshot comme dates d'ancrage supplémentaires
    Toujours : injecte dans mariadb_support_val les VL présentes dans les snapshots
    mais absentes de la table de cours, pour enrichir le suivi futur.
    """
    # Support positions sorted by date
    pos_rows = db.execute(
        text("""
            SELECT date, id_support, nbuc
            FROM mariadb_historique_support_w
            WHERE id_source = :aff AND nbuc > 0
            ORDER BY date
        """),
        {"aff": aff_id},
    ).fetchall()

    if not pos_rows:
        return 0

    positions_by_date: dict = defaultdict(dict)
    for d, id_s, nbuc in pos_rows:
        try:
            positions_by_date[d][id_s] = float(nbuc or 0)
        except (TypeError, ValueError):
            pass

    pos_dates = sorted(positions_by_date.keys())
    support_ids = list({p[1] for p in pos_rows})

    if not support_ids:
        return 0

    # Existing dates — no UNIQUE key so we guard in Python
    existing: set = set(
        r[0]
        for r in db.execute(
            text("SELECT DISTINCT date FROM mariadb_historique_affaire_w WHERE id = :aff"),
            {"aff": aff_id},
        ).fetchall()
    )

    min_date = pos_dates[0]
    max_date = pos_dates[-1]

    placeholder = ",".join(str(s) for s in support_ids)

    # Primary VL: weekly from mariadb_support_val
    vl_rows = db.execute(
        text(f"""
            SELECT date, id_support, valeur
            FROM mariadb_support_val
            WHERE id_support IN ({placeholder})
              AND date > '1970-01-01'
              AND date BETWEEN :dmin AND :dmax
            ORDER BY date
        """),
        {"dmin": min_date, "dmax": max_date},
    ).fetchall()

    # Fallback VL: sparse snapshots from historique_support_w (forward-filled)
    fallback_rows = db.execute(
        text(f"""
            SELECT date, id_support, vl
            FROM mariadb_historique_support_w
            WHERE id_source = :aff
              AND id_support IN ({placeholder})
              AND vl > 0
            ORDER BY date
        """),
        {"aff": aff_id},
    ).fetchall()

    # {date: {id_support: valeur}} — primary weekly VL
    vl_by_date: dict = defaultdict(dict)
    for d, id_s, val in vl_rows:
        try:
            v = float(val)
            if v > 0:
                vl_by_date[d][id_s] = v
        except (TypeError, ValueError):
            pass

    # Inject snapshot VLs into mariadb_support_val when absent, and add snapshot
    # dates as additional anchor dates for the weekly fill.
    to_inject: list[dict] = []
    for d_snap, id_s, vl_snap in fallback_rows:
        try:
            vl_f = float(vl_snap)
        except (TypeError, ValueError):
            continue
        if vl_f <= 0:
            continue
        if id_s not in vl_by_date.get(d_snap, {}):
            to_inject.append({"id_support": id_s, "date": d_snap, "valeur": vl_f})
            vl_by_date[d_snap][id_s] = vl_f  # also use it locally

    if to_inject:
        db.execute(
            text("""
                INSERT IGNORE INTO mariadb_support_val (id_support, date, valeur)
                VALUES (:id_support, :date, :valeur)
            """),
            to_inject,
        )

    # all_weekly_dates = support_val dates + snapshot dates (anchors for recalc)
    all_weekly_dates: set = set(vl_by_date.keys())

    if not all_weekly_dates and not fallback_rows:
        return 0

    # Sort fallback rows; we will forward-fill as we walk weekly dates
    fallback_sorted = sorted(fallback_rows, key=lambda r: r[0])

    current_nbuc: dict[int, float] = {}
    current_fallback_vl: dict[int, float] = {}
    pos_idx = 0
    fb_idx = 0
    to_insert: list[dict] = []
    to_update: list[dict] = []

    for vl_date in sorted(all_weekly_dates):
        # Advance fallback VL forward-fill pointer
        while fb_idx < len(fallback_sorted) and fallback_sorted[fb_idx][0] <= vl_date:
            _fd, _fs, _fv = fallback_sorted[fb_idx]
            try:
                v = float(_fv)
                if v > 0:
                    current_fallback_vl[_fs] = v
            except (TypeError, ValueError):
                pass
            fb_idx += 1

        # Advance nb_uc forward-fill pointer
        while pos_idx < len(pos_dates) and pos_dates[pos_idx] <= vl_date:
            current_nbuc.update(positions_by_date[pos_dates[pos_idx]])
            pos_idx += 1

        if not current_nbuc:
            continue

        vl_on_date = vl_by_date.get(vl_date, {})
        valo = sum(
            nbuc * (vl_on_date.get(id_s) or current_fallback_vl.get(id_s) or 0.0)
            for id_s, nbuc in current_nbuc.items()
            if nbuc > 0 and (id_s in vl_on_date or id_s in current_fallback_vl)
        )

        if valo <= 0:
            continue

        d = vl_date
        anne = d.year if hasattr(d, "year") else int(str(d)[:4])

        if vl_date in existing:
            if force_recalc:
                to_update.append({"valo": valo, "id": aff_id, "date": d})
        else:
            to_insert.append({"id": aff_id, "date": d, "valo": valo, "anne": anne, "sg": sg})

    if to_insert:
        db.execute(
            text("""
                INSERT INTO mariadb_historique_affaire_w
                  (id, `date`, valo, mouvement, sicav, perf_sicav_hebdo, perf_sicav_52,
                   volat, SRRI, anne, id_societe_gestion)
                VALUES (:id, :date, :valo, 0, 0, 0, 0, 0, 0, :anne, :sg)
            """),
            to_insert,
        )

    if to_update:
        db.execute(
            text("""
                UPDATE mariadb_historique_affaire_w
                SET valo = :valo
                WHERE id = :id AND `date` = :date
            """),
            to_update,
        )

    return len(to_insert) + len(to_update)


def iter_fill_weekly_historique(
    db: Session,
    affaire_ids: list[int] | None = None,
    force_recalc: bool = False,
) -> Generator[dict, None, None]:
    """
    Comble mariadb_historique_affaire_w en hebdomadaire pour toutes les affaires
    ayant un historique support (ou la liste passée en paramètre).

    Émet des événements SSE de progression puis un événement 'done'.
    """
    t0 = time.perf_counter()

    if affaire_ids:
        candidates = affaire_ids
        sg_rows = db.execute(
            text("SELECT id, id_societe_gestion FROM mariadb_affaires WHERE id IN :ids"),
            {"ids": tuple(candidates)},
        ).fetchall()
        sg_map = {r[0]: r[1] for r in sg_rows}
    else:
        rows = db.execute(
            text("""
                SELECT h.id_source, MAX(a.id_societe_gestion) as sg
                FROM mariadb_historique_support_w h
                LEFT JOIN mariadb_affaires a ON a.id = h.id_source
                WHERE h.id_source IS NOT NULL
                GROUP BY h.id_source
            """)
        ).fetchall()
        candidates = [r[0] for r in rows if r[0] is not None]
        sg_map = {r[0]: r[1] for r in rows if r[0] is not None}

    if not candidates:
        yield {"type": "done", "nb_affaires": 0, "nb_inserted": 0, "duree_s": 0.0}
        return

    total = len(candidates)
    yield {"type": "progress", "phase": "fill", "current": 0, "total": total}

    nb_inserted = 0
    affected: list[int] = []

    for i, aff_id in enumerate(candidates, 1):
        sg = sg_map.get(aff_id)
        try:
            inserted = _fill_affaire_weekly(db, aff_id, sg, force_recalc=force_recalc)
        except Exception as exc:
            logger.warning("fill_weekly affaire %s: %s", aff_id, exc)
            inserted = 0

        if inserted > 0:
            nb_inserted += inserted
            affected.append(aff_id)

        if i % 20 == 0 or i == total:
            db.commit()
            yield {"type": "progress", "phase": "fill", "current": i, "total": total}

    db.commit()

    yield {
        "type": "done",
        "nb_affaires": len(affected),
        "nb_inserted": nb_inserted,
        "duree_s": round(time.perf_counter() - t0, 2),
    }
