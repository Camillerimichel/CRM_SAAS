"""
Détection et traitement des variations de valorisation > seuil sur les historiques hebdomadaires.

Algorithme :
  1. Pour chaque affaire : calcul variation nette semaine sur semaine (mouvements neutralisés)
  2. Si variation > seuil : tentatives de correction dans l'ordre :
       a) VL forward-fill (VL NULL / aberrante)
       b) Vérification du mouvement stocké vs mouvement réel (table mouvement)
       c) Reconstruction du nbuc depuis les mouvements en parts
  3. Si toujours > seuil après corrections : tâche Réglementaire créée (une par client)

Flux SSE (dicts) : progress | log | done | error
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Generator

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from src.models.evenement import Evenement
from src.models.type_evenement import TypeEvenement

_CATEGORIE = "Réglementaire"
_TYPE_LIBELLE = "Variation de cours importante"


def _ensure_type(db: Session) -> TypeEvenement:
    t = (
        db.query(TypeEvenement)
        .filter(func.lower(TypeEvenement.libelle) == func.lower(_TYPE_LIBELLE))
        .first()
    )
    if t:
        return t
    t = TypeEvenement(libelle=_TYPE_LIBELLE, categorie=_CATEGORIE)
    db.add(t)
    db.flush()
    return t


def _log(level: str, message: str) -> dict:
    return {"type": "log", "level": level, "message": message}


def _try_fix_vl(
    db: Session, id_affaire: int, date_semaine: datetime
) -> tuple[float | None, str | None]:
    """Tente de corriger la valo en remplaçant les VL NULL/0 par forward-fill depuis la semaine précédente."""
    supports = db.execute(text("""
        SELECT id_support, nbuc, vl
        FROM mariadb_historique_support_w
        WHERE id_source = :id_affaire AND date = :date
    """), {"id_affaire": id_affaire, "date": date_semaine}).fetchall()

    if not supports:
        return None, None

    corrected_valo = 0.0
    any_fix = False

    for s in supports:
        nbuc = s.nbuc or 0.0
        if s.vl and s.vl > 0:
            corrected_valo += nbuc * s.vl
        else:
            prev = db.execute(text("""
                SELECT vl FROM mariadb_historique_support_w
                WHERE id_source = :id_affaire
                  AND id_support = :id_support
                  AND date < :date
                  AND vl IS NOT NULL AND vl > 0
                ORDER BY date DESC LIMIT 1
            """), {"id_affaire": id_affaire, "id_support": s.id_support, "date": date_semaine}).fetchone()
            if prev:
                corrected_valo += nbuc * prev.vl
                any_fix = True

    return (corrected_valo, "vl_forward_fill") if any_fix else (None, None)


def _verify_mouvement(
    db: Session,
    id_affaire: int,
    prev_date: datetime,
    curr_date: datetime,
    prev_valo: float,
    curr_valo: float,
    stored_mouvement: float,
) -> tuple[float, float] | None:
    """
    Recalcule le mouvement réel depuis la table mouvement et retourne (variation_corrigée, mvt_reel).
    Retourne None si la différence est négligeable (< 1 €).
    """
    row = db.execute(text("""
        SELECT COALESCE(SUM(CAST(m.montant_ope AS DECIMAL(18,4)) * mr.sens), 0) AS mvt_reel
        FROM mouvement m
        JOIN mouvement_regle mr ON mr.id = m.id_mouvement_regle
        WHERE m.id_affaire = :id_affaire
          AND mr.investi != 0
          AND m.etat != 8
          AND m.vl_date > :prev_date
          AND m.vl_date <= :curr_date
    """), {"id_affaire": id_affaire, "prev_date": prev_date, "curr_date": curr_date}).fetchone()

    mvt_reel = float(row.mvt_reel or 0.0)
    if abs(mvt_reel - stored_mouvement) < 1.0:
        return None

    new_var = (curr_valo - prev_valo - mvt_reel) / prev_valo
    return new_var, mvt_reel


def _try_fix_nbuc(
    db: Session,
    id_affaire: int,
    curr_date: datetime,
    prev_date: datetime,
    prev_valo: float,
    curr_mouvement: float,
) -> tuple[float | None, str | None]:
    """
    Reconstruit le nbuc attendu pour chaque support depuis les mouvements en parts.
    Retourne (valo_corrigée, motif) si un écart est détecté, sinon (None, None).
    """
    curr_supports = db.execute(text("""
        SELECT id_support, nbuc, vl
        FROM mariadb_historique_support_w
        WHERE id_source = :id_affaire AND date = :date
    """), {"id_affaire": id_affaire, "date": curr_date}).fetchall()

    if not curr_supports:
        return None, None

    prev_supports = {
        s.id_support: float(s.nbuc or 0.0)
        for s in db.execute(text("""
            SELECT id_support, nbuc
            FROM mariadb_historique_support_w
            WHERE id_source = :id_affaire AND date = :prev_date
        """), {"id_affaire": id_affaire, "prev_date": prev_date}).fetchall()
    }

    # Delta nbuc par support sur la période (prev_date, curr_date]
    mvt_rows = db.execute(text("""
        SELECT m.id_support,
               SUM(CAST(m.nb_uc AS DECIMAL(18,6)) * mr.sens) AS delta_nbuc
        FROM mouvement m
        JOIN mouvement_regle mr ON mr.id = m.id_mouvement_regle
        WHERE m.id_affaire = :id_affaire
          AND m.nb_uc IS NOT NULL
          AND CAST(m.nb_uc AS DECIMAL(18,6)) != 0
          AND m.etat != 8
          AND m.vl_date > :prev_date
          AND m.vl_date <= :curr_date
        GROUP BY m.id_support
    """), {"id_affaire": id_affaire, "prev_date": prev_date, "curr_date": curr_date}).fetchall()

    delta_by_support = {r.id_support: float(r.delta_nbuc or 0.0) for r in mvt_rows}

    any_diff = False
    corrected_valo = 0.0

    for s in curr_supports:
        vl = float(s.vl or 0.0)
        stored_nbuc = float(s.nbuc or 0.0)
        prev_nbuc = prev_supports.get(s.id_support, 0.0)
        delta = delta_by_support.get(s.id_support, 0.0)
        expected_nbuc = prev_nbuc + delta

        if vl > 0 and abs(expected_nbuc - stored_nbuc) > 0.001 and expected_nbuc >= 0:
            corrected_valo += expected_nbuc * vl
            any_diff = True
        else:
            corrected_valo += stored_nbuc * vl

    if not any_diff:
        return None, None

    return corrected_valo, "nbuc_reconstruit"


def iter_controle_valorisations(
    db: Session,
    seuil: float = 0.10,
    valo_min: float = 100.0,
    client_ids: list[int] | None = None,
    date_debut: str | None = None,
    date_fin: str | None = None,
) -> Generator[dict, None, None]:
    t0 = time.time()

    # Construction de la requête avec filtres optionnels
    filters = ["h.id IS NOT NULL", "DAYOFWEEK(h.date) = 6"]
    params: dict = {}

    if client_ids:
        placeholders = ", ".join(f":cid_{i}" for i in range(len(client_ids)))
        filters.append(f"a.id_personne IN ({placeholders})")
        params.update({f"cid_{i}": v for i, v in enumerate(client_ids)})

    if date_debut:
        filters.append("h.date >= :date_debut")
        params["date_debut"] = date_debut

    if date_fin:
        filters.append("h.date <= :date_fin")
        params["date_fin"] = date_fin

    join_clause = "JOIN mariadb_affaires a ON a.id = h.id" if client_ids else ""
    where = " AND ".join(filters)
    ids_raw = db.execute(
        text(f"SELECT DISTINCT h.id FROM mariadb_historique_affaire_w h {join_clause} WHERE {where} ORDER BY h.id"),
        params,
    ).fetchall()
    affaire_ids = [r[0] for r in ids_raw]
    total = len(affaire_ids)

    if total == 0:
        yield {"type": "done", "duree_s": 0, "nb_affaires": 0,
               "nb_anomalies": 0, "nb_resolus": 0, "nb_taches": 0}
        return

    yield {"type": "progress", "current": 0, "total": total}
    yield _log("info", f"{total} affaires à contrôler — seuil {seuil:.0%}")

    nb_anomalies = 0
    nb_resolus = 0
    nb_clotures = 0
    client_anomalies: dict[tuple, dict] = {}

    for i, id_affaire in enumerate(affaire_ids):
        yield {"type": "progress", "current": i + 1, "total": total}

        # Exclure les affaires dont le dernier SRRI a augmenté (effet de bord rachat total)
        srri_tail = db.execute(
            text("""
                SELECT SRRI FROM mariadb_historique_affaire_w
                WHERE id = :id AND SRRI > 0
                ORDER BY date DESC LIMIT 2
            """),
            {"id": id_affaire},
        ).fetchall()
        if len(srri_tail) >= 2 and srri_tail[0][0] > srri_tail[1][0]:
            continue

        row_filters = ["id = :id_affaire", "DAYOFWEEK(date) = 6"]
        row_params: dict = {"id_affaire": id_affaire}
        if date_debut:
            row_filters.append("date >= :date_debut")
            row_params["date_debut"] = date_debut
        if date_fin:
            row_filters.append("date <= :date_fin")
            row_params["date_fin"] = date_fin
        rows = db.execute(
            text(f"SELECT id, date, valo, mouvement FROM mariadb_historique_affaire_w WHERE {' AND '.join(row_filters)} ORDER BY date"),
            row_params,
        ).fetchall()

        if not rows:
            continue

        # Suppression du dernier point si valo = 0 ou NULL → clôture du contrat
        last = rows[-1]
        if not last.valo or last.valo == 0:
            date_cloture = last.date
            date_str = date_cloture.strftime("%Y-%m-%d") if date_cloture else "?"
            db.execute(text("""
                UPDATE mariadb_affaires SET date_cle = :date_cle WHERE id = :id
            """), {"date_cle": date_cloture, "id": id_affaire})
            db.execute(text("""
                DELETE FROM mariadb_historique_affaire_w
                WHERE id = :id_affaire AND date = :date
            """), {"id_affaire": id_affaire, "date": date_cloture})
            db.execute(text("""
                DELETE FROM mariadb_historique_support_w
                WHERE id_source = :id_affaire AND date = :date
            """), {"id_affaire": id_affaire, "date": date_cloture})
            db.commit()
            nb_clotures += 1
            yield _log(
                "cloture",
                f"Affaire #{id_affaire} — dernier point valo=0 supprimé, "
                f"date de clôture fixée au {date_str}",
            )
            rows = rows[:-1]

        for j in range(1, len(rows)):
            prev, curr = rows[j - 1], rows[j]
            if not prev.valo or abs(prev.valo) < valo_min:
                continue
            mouvement = float(curr.mouvement or 0.0)
            variation_nette = (curr.valo - prev.valo - mouvement) / prev.valo

            if abs(variation_nette) <= seuil:
                continue

            nb_anomalies += 1
            signe = "+" if variation_nette > 0 else ""
            date_str = curr.date.strftime("%Y-%m-%d") if curr.date else "?"
            yield _log(
                "anomalie",
                f"Affaire #{id_affaire} — {date_str} : "
                f"variation nette {signe}{variation_nette:.1%}  "
                f"({prev.valo:,.0f} → {curr.valo:,.0f} €, mvt {mouvement:+,.0f} €)",
            )

            motif_vl = None
            motif_mvt = None
            motif_nbuc = None

            # 1. Tentative correction VL (forward-fill)
            corrected_valo_vl, motif_vl = _try_fix_vl(db, id_affaire, curr.date)
            if corrected_valo_vl is not None:
                new_var = (corrected_valo_vl - prev.valo - mouvement) / prev.valo
                if abs(new_var) <= seuil:
                    nb_resolus += 1
                    yield _log(
                        "resolu",
                        f"  → Auto-résolu (VL forward-fill) : variation corrigée "
                        f"{'+' if new_var > 0 else ''}{new_var:.1%}",
                    )
                    continue
                yield _log(
                    "warn",
                    f"  → VL forward-fill insuffisant : variation toujours "
                    f"{'+' if new_var > 0 else ''}{new_var:.1%}",
                )

            # 2. Vérification du mouvement stocké vs mouvement réel
            mvt_result = _verify_mouvement(
                db, id_affaire, prev.date, curr.date,
                float(prev.valo), float(curr.valo), mouvement,
            )
            if mvt_result is not None:
                new_var, mvt_reel = mvt_result
                if abs(new_var) <= seuil:
                    nb_resolus += 1
                    yield _log(
                        "resolu",
                        f"  → Auto-résolu (mouvement recalculé {mvt_reel:+,.0f} € "
                        f"vs stocké {mouvement:+,.0f} €) : variation corrigée "
                        f"{'+' if new_var > 0 else ''}{new_var:.1%}",
                    )
                    continue
                motif_mvt = "mouvement_recalcule"
                mouvement = mvt_reel  # mouvement corrigé pour la suite
                yield _log(
                    "warn",
                    f"  → Mouvement recalculé ({mvt_reel:+,.0f} €) : "
                    f"variation toujours {'+' if new_var > 0 else ''}{new_var:.1%}",
                )

            # 3. Correction nbuc depuis les mouvements en parts
            corrected_valo_nbuc, motif_nbuc = _try_fix_nbuc(
                db, id_affaire, curr.date, prev.date, float(prev.valo), mouvement,
            )
            if corrected_valo_nbuc is not None:
                new_var = (corrected_valo_nbuc - prev.valo - mouvement) / prev.valo
                if abs(new_var) <= seuil:
                    nb_resolus += 1
                    yield _log(
                        "resolu",
                        f"  → Auto-résolu (nbuc reconstruit) : variation corrigée "
                        f"{'+' if new_var > 0 else ''}{new_var:.1%}",
                    )
                    continue
                yield _log(
                    "warn",
                    f"  → Correction nbuc insuffisante : variation toujours "
                    f"{'+' if new_var > 0 else ''}{new_var:.1%}",
                )

            # Anomalie persistante : rattachement au client
            affaire_row = db.execute(
                text("SELECT id_personne, ref FROM mariadb_affaires WHERE id = :id"),
                {"id": id_affaire},
            ).fetchone()

            if not affaire_row or not affaire_row.id_personne:
                yield _log("warn", f"  → Affaire #{id_affaire} sans client — ignorée")
                continue

            client_id = affaire_row.id_personne
            ref = affaire_row.ref or f"#{id_affaire}"

            causes = []
            if motif_vl:
                causes.append("VL manquante / aberrante (correction insuffisante)")
            if motif_mvt:
                causes.append("mouvement recalculé (écart persistant)")
            if motif_nbuc:
                causes.append("décalage de parts (correction insuffisante)")
            if not causes:
                causes.append("cause non identifiée automatiquement")

            aff_key = (client_id, id_affaire)
            if aff_key not in client_anomalies:
                client_anomalies[aff_key] = {
                    "client_id": client_id,
                    "ref": ref,
                    "semaines": [],
                    "pire_variation": 0.0,
                    "cause": set(),
                }
            entry = client_anomalies[aff_key]
            entry["semaines"].append(date_str)
            if abs(variation_nette) > abs(entry["pire_variation"]):
                entry["pire_variation"] = variation_nette
            entry["cause"].update(causes)

    # Regroupement des anomalies par client (une tâche par client)
    par_client: dict[int, list[dict]] = {}
    for entry in client_anomalies.values():
        par_client.setdefault(entry["client_id"], []).append(entry)

    nb_taches = 0
    if par_client:
        type_ev = _ensure_type(db)
        for client_id, affaires in par_client.items():
            lignes = []
            for a in affaires:
                pire = a["pire_variation"]
                signe = "+" if pire > 0 else ""
                nb_sem = len(a["semaines"])
                sem_detail = a["semaines"][0] if nb_sem == 1 else f"{a['semaines'][0]} … {a['semaines'][-1]}"
                causes_str = ", ".join(a["cause"])
                lignes.append(
                    f"- [{a['ref']}] {nb_sem} semaine(s) — pire variation : "
                    f"{signe}{pire:.1%} ({sem_detail}) — {causes_str}"
                )
            commentaire = (
                "Variation de cours détectée > 10% sur les affaires suivantes :\n"
                + "\n".join(lignes)
                + "\nContrôle manuel requis."
            )
            db.add(Evenement(
                type_id=type_ev.id,
                client_id=client_id,
                date_evenement=datetime.utcnow(),
                statut="à faire",
                commentaire=commentaire,
            ))
            nb_taches += 1
            yield _log(
                "tache",
                f"  Tâche créée pour client #{client_id} "
                f"({len(affaires)} affaire(s) concernée(s))",
            )

        db.commit()

    duree = round(time.time() - t0, 1)
    yield {
        "type": "done",
        "duree_s": duree,
        "nb_affaires": total,
        "nb_clotures": nb_clotures,
        "nb_anomalies": nb_anomalies,
        "nb_resolus": nb_resolus,
        "nb_taches": nb_taches,
    }
