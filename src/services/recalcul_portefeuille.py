"""
Pipeline de recalcul complet après import de portefeuilles.

Ordre d'exécution :
  A. aggregate_support_to_affaire   – consolide historique_support → historique_affaire (valo)
  B. aggregate_mouvement_to_affaire – consolide mouvements → historique_affaire (mouvement)
  C. aggregate_affaire_to_personne  – consolide historique_affaire → historique_personne
  D. recompute_dietz_affaires       – Dietz, perf_52, volat, SRRI sur les affaires
  E. recompute_dietz_clients        – idem sur les clients
"""
from __future__ import annotations

import logging
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# A. Support → Affaire (valo)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_support_to_affaire(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> int:
    """
    Agrège mariadb_historique_support_w → mariadb_historique_affaire_w (colonne valo).

    Pour chaque (id_source, date) present dans historique_support :
      - supprime les lignes existantes dans historique_affaire pour ce (id, date)
      - insère la somme de valo

    Retourne le nombre de lignes insérées.
    """
    scope_clause = ""
    params: dict = {}
    if affaire_ids:
        scope_clause = "WHERE h.id_source IN :ids"
        params["ids"] = tuple(affaire_ids)

    # Collect (id_source, date) to replace
    pairs = db.execute(
        text(
            f"""
            SELECT DISTINCT h.id_source, h.date, h.id_societe_gestion
            FROM mariadb_historique_support_w h
            {scope_clause}
            """
        ),
        params,
    ).fetchall()

    if not pairs:
        return 0

    # Build set of (id, date) to delete then re-insert
    for row in pairs:
        db.execute(
            text(
                """
                DELETE FROM mariadb_historique_affaire_w
                WHERE id = :id AND `date` = :dt
                """
            ),
            {"id": row[0], "dt": row[1]},
        )

    db.flush()

    # Aggregate insert
    scope_clause2 = ""
    if affaire_ids:
        scope_clause2 = "WHERE h.id_source IN :ids"

    db.execute(
        text(
            f"""
            INSERT INTO mariadb_historique_affaire_w (id, `date`, valo, mouvement, anne, id_societe_gestion)
            SELECT
                h.id_source,
                h.date,
                SUM(h.valo)                       AS valo,
                0.0                               AS mouvement,
                YEAR(h.date)                      AS anne,
                MAX(h.id_societe_gestion)         AS id_societe_gestion
            FROM mariadb_historique_support_w h
            {scope_clause2}
            GROUP BY h.id_source, h.date
            """
        ),
        params,
    )
    db.commit()
    return len(pairs)


# ──────────────────────────────────────────────────────────────────────────────
# B. Mouvement → Affaire (mouvement net)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_mouvement_to_affaire(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> None:
    """
    Recalcule la colonne mouvement dans historique_affaire à partir de la table mouvement.

    Regle : mouvement = SUM(montant_ope * sens) pour investi != 0, par (id_affaire, date hebdo).
    La date hebdo = la date du jour dans historique_affaire la plus proche >= vl_date.
    """
    scope = ""
    params: dict = {}
    if affaire_ids:
        scope = "AND m.id_affaire IN :ids"
        params["ids"] = tuple(affaire_ids)

    db.execute(
        text(
            f"""
            UPDATE mariadb_historique_affaire_w AS h
            JOIN (
                SELECT
                    m.id_affaire,
                    COALESCE(
                        (SELECT hh.date FROM mariadb_historique_affaire_w hh
                         WHERE hh.id = m.id_affaire AND hh.date >= DATE(m.vl_date)
                         ORDER BY hh.date ASC LIMIT 1),
                        (SELECT hh.date FROM mariadb_historique_affaire_w hh
                         WHERE hh.id = m.id_affaire AND hh.date < DATE(m.vl_date)
                         ORDER BY hh.date DESC LIMIT 1)
                    ) AS snap_date,
                    SUM(CAST(m.montant_ope AS DECIMAL(20,4)) * r.sens) AS net_mvt
                FROM mouvement m
                JOIN mouvement_regle r ON r.id = m.id_mouvement_regle
                WHERE r.investi != 0
                  AND (m.etat IS NULL OR m.etat != 8)
                  {scope}
                GROUP BY m.id_affaire, snap_date
                HAVING snap_date IS NOT NULL
            ) AS agg
              ON h.id = agg.id_affaire
             AND h.date = agg.snap_date
            SET h.mouvement = agg.net_mvt
            """
        ),
        params,
    )
    db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# C. Affaire → Personne
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_affaire_to_personne(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> int:
    """
    Agrège mariadb_historique_affaire_w → mariadb_historique_personne_w.

    Pour chaque (id_personne, date), sum(valo) et sum(mouvement) sur toutes les affaires du client.
    Scope optionnel sur affaire_ids pour ne recalculer que les clients concernés.
    """
    # Find affected client ids
    if affaire_ids:
        rows = db.execute(
            text("SELECT DISTINCT id_personne FROM mariadb_affaires WHERE id IN :ids"),
            {"ids": tuple(affaire_ids)},
        ).fetchall()
        client_ids = [r[0] for r in rows if r[0] is not None]
    else:
        client_ids = []

    scope_clause = "WHERE a.id_personne IS NOT NULL"
    params: dict = {}
    if client_ids:
        scope_clause = "WHERE a.id_personne IS NOT NULL AND a.id_personne IN :cids"
        params["cids"] = tuple(client_ids)

    # Bulk DELETE for affected persons, then single INSERT SELECT — avoids per-row
    # lock accumulation that causes deadlocks with concurrent affaire_w writers.
    if client_ids:
        pid_placeholder = ",".join(str(p) for p in client_ids)
        db.execute(
            text(f"DELETE FROM mariadb_historique_personne_w WHERE id IN ({pid_placeholder})")
        )
    else:
        db.execute(text("DELETE FROM mariadb_historique_personne_w"))

    result = db.execute(
        text(
            f"""
            INSERT INTO mariadb_historique_personne_w (id, `date`, valo, mouvement, anne, id_societe_gestion)
            SELECT
                a.id_personne               AS id,
                h.date,
                SUM(COALESCE(h.valo, 0))    AS valo,
                SUM(COALESCE(h.mouvement, 0)) AS mouvement,
                YEAR(h.date)                AS anne,
                MAX(h.id_societe_gestion)   AS id_societe_gestion
            FROM mariadb_historique_affaire_w h
            JOIN mariadb_affaires a ON a.id = h.id
            {scope_clause}
            GROUP BY a.id_personne, h.date
            """
        ),
        params,
    )
    db.commit()
    return result.rowcount


# ──────────────────────────────────────────────────────────────────────────────
# D. Recalcul Dietz + SRRI – Affaires
# ──────────────────────────────────────────────────────────────────────────────

_SRRI_CASES = """
CASE
  WHEN rn < 52                 THEN 0
  WHEN volat_raw < 0.005       THEN 1
  WHEN volat_raw < 0.02        THEN 2
  WHEN volat_raw < 0.05        THEN 3
  WHEN volat_raw < 0.10        THEN 4
  WHEN volat_raw < 0.15        THEN 5
  WHEN volat_raw < 0.25        THEN 6
  ELSE 7
END
"""


def _dietz_cte_sql(win: int, id_filter_sql: str, source_table: str = "mariadb_historique_affaire_w") -> str:
    """Génère le SELECT CTE Dietz+SRRI pour une fenêtre et un filtre d'IDs donnés."""
    w = win - 1
    return f"""
            WITH ordered AS (
              SELECT
                id,
                `date`,
                valo                                                   AS valorisation_suiv,
                COALESCE(mouvement, 0)                                 AS mouvement,
                LAG(valo) OVER (PARTITION BY id ORDER BY `date`)       AS prev_valo,
                ROW_NUMBER() OVER (PARTITION BY id ORDER BY `date`)    AS rn,
                CASE
                  WHEN LEAD(valo) OVER (PARTITION BY id ORDER BY `date`) IS NULL
                       AND valo <= 1
                       AND COALESCE(mouvement, 0) < 0
                  THEN 1 ELSE 0
                END AS is_last_redemption
              FROM {source_table}
              WHERE id IN ({id_filter_sql})
            ),
            base AS (
              SELECT
                id, `date`, rn,
                COALESCE(prev_valo, 0)  AS valo,
                mouvement,
                valorisation_suiv,
                CASE
                  WHEN is_last_redemption = 1                             THEN NULL
                  WHEN ABS(COALESCE(prev_valo, 0) + 0.5 * mouvement) < 0.01 THEN NULL
                  ELSE (valorisation_suiv - mouvement - COALESCE(prev_valo, 0))
                       / NULLIF(COALESCE(prev_valo, 0) + 0.5 * mouvement, 0)
                END AS r
              FROM ordered
            ),
            dietz_calc AS (
              SELECT *,
                1 + SUM(COALESCE(CASE WHEN r >= -1 AND r <= 5 THEN r ELSE NULL END, 0)) OVER (
                  PARTITION BY id ORDER BY `date`
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS dietz
              FROM base
            ),
            perf AS (
              SELECT *,
                CASE
                  WHEN LAG(dietz) OVER (PARTITION BY id ORDER BY `date`) IS NULL THEN NULL
                  ELSE (dietz - LAG(dietz) OVER (PARTITION BY id ORDER BY `date`))
                       / NULLIF(LAG(dietz) OVER (PARTITION BY id ORDER BY `date`), 0)
                END AS perf_dietz
              FROM dietz_calc
            ),
            vola AS (
              SELECT *,
                STDDEV_SAMP(CASE WHEN r >= -1 AND r <= 5 THEN r ELSE NULL END) OVER (
                  PARTITION BY id ORDER BY `date`
                  ROWS BETWEEN {w} PRECEDING AND CURRENT ROW
                ) * SQRT({float(win)}) AS volat_raw
              FROM perf
            )
            SELECT
              id, `date`,
              valo,
              mouvement,
              valorisation_suiv,
              r,
              dietz,
              perf_dietz,
              CASE
                WHEN LAG(dietz, {win}) OVER (PARTITION BY id ORDER BY `date`) IS NULL THEN NULL
                ELSE (dietz - LAG(dietz, {win}) OVER (PARTITION BY id ORDER BY `date`))
                     / NULLIF(LAG(dietz, {win}) OVER (PARTITION BY id ORDER BY `date`), 0)
              END                                                AS perf_52,
              CASE WHEN rn < {win} THEN 0 ELSE COALESCE(volat_raw, 0) END AS volat_52,
              CASE WHEN rn < {win} THEN 0
                WHEN COALESCE(volat_raw, 0) < 0.005 THEN 1
                WHEN COALESCE(volat_raw, 0) < 0.02  THEN 2
                WHEN COALESCE(volat_raw, 0) < 0.05  THEN 3
                WHEN COALESCE(volat_raw, 0) < 0.10  THEN 4
                WHEN COALESCE(volat_raw, 0) < 0.15  THEN 5
                WHEN COALESCE(volat_raw, 0) < 0.25  THEN 6
                ELSE 7
              END                                                AS srri,
              rn
            FROM vola
            WHERE id IS NOT NULL
            ORDER BY id, `date`
    """


def _build_freq_map(db: Session, source_table: str, id_filter_sql: str | None = None) -> dict[int, str]:
    """
    Classifie chaque entité (id) par fréquence moyenne de données.
    Retourne {win: 'id1,id2,...'} avec win ∈ {52, 12, 4}.
    """
    where = f"WHERE id IN ({id_filter_sql})" if id_filter_sql else ""
    db.execute(text("DROP TABLE IF EXISTS tmp_dietz_freq"))
    db.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_dietz_freq AS
            SELECT
                id,
                COALESCE(ROUND(AVG(days_gap)), 7) AS avg_days,
                CASE
                  WHEN COALESCE(ROUND(AVG(days_gap)), 7) <= 10 THEN 52
                  WHEN COALESCE(ROUND(AVG(days_gap)), 7) <= 40 THEN 12
                  ELSE 4
                END AS win
            FROM (
                SELECT id,
                       DATEDIFF(`date`,
                           LAG(`date`) OVER (PARTITION BY id ORDER BY `date`)
                       ) AS days_gap
                FROM {source_table}
                {where}
            ) t
            WHERE days_gap IS NOT NULL AND days_gap > 0
            GROUP BY id
            """
        )
    )
    rows = db.execute(text("SELECT win, GROUP_CONCAT(id ORDER BY id) AS ids FROM tmp_dietz_freq GROUP BY win")).fetchall()
    freq_map: dict[int, str] = {}
    for row in rows:
        m = row._mapping if hasattr(row, "_mapping") else {}
        win = int(m.get("win") or row[0])
        ids_str = str(m.get("ids") or row[1] or "")
        if ids_str:
            freq_map[win] = ids_str

    # Entités sans données de fréquence → fenêtre 52 par défaut
    all_ids_row = db.execute(
        text(f"SELECT GROUP_CONCAT(DISTINCT id ORDER BY id) FROM {source_table} {where}")
    ).scalar() or ""
    classified: set[int] = set()
    for ids_str in freq_map.values():
        classified.update(int(i) for i in ids_str.split(",") if i.strip())
    unclassified = [str(i) for i in (int(x) for x in all_ids_row.split(",") if x.strip()) if i not in classified]
    if unclassified:
        existing = freq_map.get(52, "")
        freq_map[52] = ",".join(filter(None, [existing] + unclassified))

    return freq_map


def recompute_dietz_affaires(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> float:
    """
    Recalcule Modified Dietz, perf_sicav_52, volat_52, SRRI pour les affaires.
    Met à jour mariadb_historique_affaire_w et mariadb_affaires.SRI.
    Adapte la fenêtre à la fréquence des données (hebdo/mensuel/trimestriel).
    Retourne la durée en secondes.
    """
    t0 = perf_counter()

    id_filter = ",".join(str(i) for i in affaire_ids) if affaire_ids else None

    freq_map = _build_freq_map(db, "mariadb_historique_affaire_w", id_filter)

    db.execute(text("DROP TABLE IF EXISTS tempo_hist_affaire_import_w"))
    db.execute(
        text(
            """
            CREATE TABLE tempo_hist_affaire_import_w (
              id                INT,
              `date`            DATETIME,
              valo              DECIMAL(38,18),
              mouvement         DECIMAL(38,18),
              valorisation_suiv DECIMAL(38,18),
              r                 DECIMAL(38,18),
              dietz             DECIMAL(38,18),
              perf_dietz        DECIMAL(38,18),
              perf_52           DECIMAL(38,18),
              volat_52          DECIMAL(38,18),
              srri              INT,
              rn                INT,
              PRIMARY KEY (id, rn)
            )
            """
        )
    )

    for win, ids_str in freq_map.items():
        if not ids_str:
            continue
        cte_sql = _dietz_cte_sql(win, ids_str, "mariadb_historique_affaire_w")
        db.execute(
            text(
                f"""
                INSERT INTO tempo_hist_affaire_import_w (
                  id, `date`, valo, mouvement, valorisation_suiv,
                  r, dietz, perf_dietz, perf_52, volat_52, srri, rn
                )
                {cte_sql}
                """
            )
        )

    # Update historique_affaire
    db.execute(
        text(
            """
            UPDATE mariadb_historique_affaire_w AS m
            JOIN tempo_hist_affaire_import_w AS t
              ON m.id = t.id AND m.`date` = t.`date`
            SET
              m.sicav             = t.dietz,
              m.perf_sicav_hebdo  = t.r,
              m.perf_sicav_52     = t.perf_52,
              m.volat             = t.volat_52,
              m.SRRI              = t.srri
            """
        )
    )

    # Update affaires.SRI with latest SRRI
    db.execute(
        text(
            """
            UPDATE mariadb_affaires AS a
            JOIN (
              SELECT id, srri
              FROM (
                SELECT id, srri,
                       ROW_NUMBER() OVER (PARTITION BY id ORDER BY `date` DESC) AS rk
                FROM tempo_hist_affaire_import_w
              ) AS ranked
              WHERE rk = 1
            ) AS t ON a.id = t.id
            SET a.SRI = t.srri
            """
        )
    )

    db.commit()
    db.execute(text("DROP TABLE IF EXISTS tempo_hist_affaire_import_w"))
    db.commit()
    return perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# E. Recalcul Dietz + SRRI – Clients
# ──────────────────────────────────────────────────────────────────────────────

def recompute_dietz_clients(
    db: Session,
    client_ids: list[int] | None = None,
) -> float:
    """
    Recalcule Modified Dietz, perf_sicav_52, volat_52, SRRI pour les clients.
    Met à jour mariadb_historique_personne_w et mariadb_clients.SRI.
    Adapte la fenêtre à la fréquence des données (hebdo/mensuel/trimestriel).
    Retourne la durée en secondes.
    """
    t0 = perf_counter()

    id_filter = ",".join(str(i) for i in client_ids) if client_ids else None

    freq_map = _build_freq_map(db, "mariadb_historique_personne_w", id_filter)

    db.execute(text("DROP TABLE IF EXISTS tempo_hist_personne_import_w"))
    db.execute(
        text(
            """
            CREATE TABLE tempo_hist_personne_import_w (
              id                INT,
              `date`            DATETIME,
              valo              DECIMAL(38,18),
              mouvement         DECIMAL(38,18),
              valorisation_suiv DECIMAL(38,18),
              r                 DECIMAL(38,18),
              dietz             DECIMAL(38,18),
              perf_dietz        DECIMAL(38,18),
              perf_52           DECIMAL(38,18),
              volat_52          DECIMAL(38,18),
              srri              INT,
              rn                INT,
              PRIMARY KEY (id, rn)
            )
            """
        )
    )

    for win, ids_str in freq_map.items():
        if not ids_str:
            continue
        cte_sql = _dietz_cte_sql(win, ids_str, "mariadb_historique_personne_w")
        db.execute(
            text(
                f"""
                INSERT INTO tempo_hist_personne_import_w (
                  id, `date`, valo, mouvement, valorisation_suiv,
                  r, dietz, perf_dietz, perf_52, volat_52, srri, rn
                )
                {cte_sql}
                """
            )
        )

    db.execute(
        text(
            """
            UPDATE mariadb_historique_personne_w AS m
            JOIN tempo_hist_personne_import_w AS t
              ON m.id = t.id AND m.`date` = t.`date`
            SET
              m.sicav             = t.dietz,
              m.perf_sicav_hebdo  = t.r,
              m.perf_sicav_52     = t.perf_52,
              m.volat             = t.volat_52,
              m.srri              = t.srri
            """
        )
    )

    db.execute(
        text(
            """
            UPDATE mariadb_clients AS c
            JOIN (
              SELECT id, srri
              FROM (
                SELECT id, srri,
                       ROW_NUMBER() OVER (PARTITION BY id ORDER BY `date` DESC) AS rk
                FROM tempo_hist_personne_import_w
              ) AS ranked
              WHERE rk = 1
            ) AS t ON c.id = t.id
            SET c.SRI = t.srri
            """
        )
    )

    db.commit()
    db.execute(text("DROP TABLE IF EXISTS tempo_hist_personne_import_w"))
    db.commit()
    return perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> float:
    """
    Lance le pipeline complet A→E et retourne la durée totale en secondes.
    Si affaire_ids est fourni, l'agrégation est scopée ; le recalcul SRRI reste global
    pour garantir la cohérence des fenêtres de 52 semaines.
    """
    t0 = perf_counter()
    aggregate_support_to_affaire(db, affaire_ids)
    aggregate_mouvement_to_affaire(db, affaire_ids)
    aggregate_affaire_to_personne(db, affaire_ids)
    recompute_dietz_affaires(db)   # toujours global pour les fenêtres glissantes
    recompute_dietz_clients(db)
    return perf_counter() - t0
