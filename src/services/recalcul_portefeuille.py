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
                    (
                        SELECT hh.date
                        FROM mariadb_historique_affaire_w hh
                        WHERE hh.id = m.id_affaire
                          AND hh.date >= DATE(m.vl_date)
                        ORDER BY hh.date ASC
                        LIMIT 1
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

    # Collect (personne_id, date) pairs to replace
    pairs = db.execute(
        text(
            f"""
            SELECT DISTINCT a.id_personne, h.date
            FROM mariadb_historique_affaire_w h
            JOIN mariadb_affaires a ON a.id = h.id
            {scope_clause}
            """
        ),
        params,
    ).fetchall()

    if not pairs:
        return 0

    for row in pairs:
        if row[0] is None:
            continue
        db.execute(
            text(
                """
                DELETE FROM mariadb_historique_personne_w
                WHERE id = :pid AND `date` = :dt
                """
            ),
            {"pid": row[0], "dt": row[1]},
        )

    db.flush()

    db.execute(
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
    return len(pairs)


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


def recompute_dietz_affaires(
    db: Session,
    affaire_ids: list[int] | None = None,
) -> float:
    """
    Recalcule Modified Dietz, perf_sicav_52, volat_52, SRRI pour les affaires.
    Met à jour mariadb_historique_affaire_w et mariadb_affaires.SRI.
    Retourne la durée en secondes.
    """
    t0 = perf_counter()

    scope_filter = ""
    params: dict = {}
    if affaire_ids:
        scope_filter = "WHERE id IN :ids"
        params["ids"] = tuple(affaire_ids)

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

    db.execute(
        text(
            f"""
            INSERT INTO tempo_hist_affaire_import_w (
              id, `date`, valo, mouvement, valorisation_suiv,
              r, dietz, perf_dietz, perf_52, volat_52, srri, rn
            )
            WITH ordered AS (
              SELECT
                id,
                `date`,
                valo                                                   AS valorisation_suiv,
                COALESCE(mouvement, 0)                                 AS mouvement,
                LAG(valo) OVER (PARTITION BY id ORDER BY `date`)       AS prev_valo,
                ROW_NUMBER() OVER (PARTITION BY id ORDER BY `date`)    AS rn
              FROM mariadb_historique_affaire_w
              {scope_filter}
            ),
            base AS (
              SELECT
                id, `date`, rn,
                COALESCE(prev_valo, 0)  AS valo,
                mouvement,
                valorisation_suiv,
                CASE
                  WHEN ABS(COALESCE(prev_valo, 0) + 0.5 * mouvement) < 0.01 THEN NULL
                  ELSE (valorisation_suiv - mouvement - COALESCE(prev_valo, 0))
                       / NULLIF(COALESCE(prev_valo, 0) + 0.5 * mouvement, 0)
                END AS r
              FROM ordered
            ),
            dietz_calc AS (
              SELECT *,
                1 + SUM(COALESCE(r, 0)) OVER (
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
                STDDEV_SAMP(perf_dietz) OVER (
                  PARTITION BY id ORDER BY `date`
                  ROWS BETWEEN 51 PRECEDING AND CURRENT ROW
                ) * SQRT(52) AS volat_raw
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
                WHEN LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`) IS NULL THEN NULL
                ELSE (dietz - LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`))
                     / NULLIF(LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`), 0)
              END                                                AS perf_52,
              CASE WHEN rn < 52 THEN 0 ELSE COALESCE(volat_raw, 0) END AS volat_52,
              CASE WHEN rn < 52 THEN 0
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
            ORDER BY id, `date`
            """
        ),
        params,
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
    Retourne la durée en secondes.
    """
    t0 = perf_counter()

    scope_filter = ""
    params: dict = {}
    if client_ids:
        scope_filter = "AND id IN :ids"
        params["ids"] = tuple(client_ids)

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

    db.execute(
        text(
            f"""
            INSERT INTO tempo_hist_personne_import_w (
              id, `date`, valo, mouvement, valorisation_suiv,
              r, dietz, perf_dietz, perf_52, volat_52, srri, rn
            )
            WITH ordered AS (
              SELECT
                id,
                `date`,
                valo                                                   AS valorisation_suiv,
                COALESCE(mouvement, 0)                                 AS mouvement,
                LAG(valo) OVER (PARTITION BY id ORDER BY `date`)       AS prev_valo,
                ROW_NUMBER() OVER (PARTITION BY id ORDER BY `date`)    AS rn
              FROM mariadb_historique_personne_w
              WHERE id IS NOT NULL {scope_filter}
            ),
            base AS (
              SELECT
                id, `date`, rn,
                COALESCE(prev_valo, 0)  AS valo,
                mouvement,
                valorisation_suiv,
                CASE
                  WHEN ABS(COALESCE(prev_valo, 0) + 0.5 * mouvement) < 0.01 THEN NULL
                  ELSE (valorisation_suiv - mouvement - COALESCE(prev_valo, 0))
                       / NULLIF(COALESCE(prev_valo, 0) + 0.5 * mouvement, 0)
                END AS r
              FROM ordered
            ),
            dietz_calc AS (
              SELECT *,
                1 + SUM(COALESCE(r, 0)) OVER (
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
                STDDEV_SAMP(perf_dietz) OVER (
                  PARTITION BY id ORDER BY `date`
                  ROWS BETWEEN 51 PRECEDING AND CURRENT ROW
                ) * SQRT(52) AS volat_raw
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
                WHEN LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`) IS NULL THEN NULL
                ELSE (dietz - LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`))
                     / NULLIF(LAG(dietz, 52) OVER (PARTITION BY id ORDER BY `date`), 0)
              END                                                AS perf_52,
              CASE WHEN rn < 52 THEN 0 ELSE COALESCE(volat_raw, 0) END AS volat_52,
              CASE WHEN rn < 52 THEN 0
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
            ORDER BY id, `date`
            """
        ),
        params,
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
