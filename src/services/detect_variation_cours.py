"""
Détection et traitement des variations de valorisation > seuil sur les historiques hebdomadaires.

Algorithme :
  1. Pour chaque affaire : calcul variation nette semaine sur semaine (mouvements neutralisés)
  2. Si variation > seuil : tentative de correction VL (forward-fill) puis détection décalage nbuc
  3. Recalcul post-correction — si toujours > seuil : tâche Réglementaire créée (une par client)

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


def _detect_nbuc_decalage(
    db: Session, id_affaire: int, date_semaine: datetime
) -> bool:
    """Retourne True si un mouvement avec nb_uc existe dans la fenêtre ±2 semaines autour de la date."""
    date_min = date_semaine - timedelta(weeks=2)
    date_max = date_semaine + timedelta(weeks=2)
    row = db.execute(
        text("""
            SELECT COUNT(*) AS nb
            FROM mouvement
            WHERE id_affaire = :id_affaire
              AND nb_uc IS NOT NULL AND nb_uc != 0
              AND (
                (vl_date BETWEEN :date_min AND :date_max)
                OR (date_sp BETWEEN :date_min AND :date_max)
              )
        """),
        {"id_affaire": id_affaire, "date_min": date_min, "date_max": date_max},
    ).fetchone()
    return bool(row and row.nb > 0)


def iter_controle_valorisations(
    db: Session,
    seuil: float = 0.10,
) -> Generator[dict, None, None]:
    t0 = time.time()

    # Dans la live DB, la colonne affaire s'appelle `id` (pas `id_affaire`)
    # On ne conserve que les vendredis (DAYOFWEEK = 6)
    ids_raw = db.execute(
        text(
            "SELECT DISTINCT id FROM mariadb_historique_affaire_w "
            "WHERE id IS NOT NULL AND DAYOFWEEK(date) = 6 ORDER BY id"
        )
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
    client_anomalies: dict[int, list[dict]] = {}

    for i, id_affaire in enumerate(affaire_ids):
        yield {"type": "progress", "current": i + 1, "total": total}

        rows = db.execute(text("""
            SELECT id, date, valo, mouvement
            FROM mariadb_historique_affaire_w
            WHERE id = :id_affaire AND DAYOFWEEK(date) = 6
            ORDER BY date
        """), {"id_affaire": id_affaire}).fetchall()

        for j in range(1, len(rows)):
            prev, curr = rows[j - 1], rows[j]
            if not prev.valo or prev.valo == 0:
                continue
            mouvement = curr.mouvement or 0.0
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

            # Tentative correction VL
            corrected_valo, motif_vl = _try_fix_vl(db, id_affaire, curr.date)
            if corrected_valo is not None:
                new_var = (corrected_valo - prev.valo - mouvement) / prev.valo
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

            # Détection décalage nbuc
            nbuc_suspect = _detect_nbuc_decalage(db, id_affaire, curr.date)
            if nbuc_suspect:
                yield _log("warn", "  → Décalage nbuc suspect (mouvement ±2 sem.) — vérification manuelle requise")

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
            if nbuc_suspect:
                causes.append("décalage de parts suspect")
            if not causes:
                causes.append("cause non identifiée automatiquement")

            client_anomalies.setdefault(client_id, []).append({
                "ref": ref,
                "date": date_str,
                "variation": f"{signe}{variation_nette:.1%}",
                "valo_avant": f"{prev.valo:,.0f}",
                "valo_apres": f"{curr.valo:,.0f}",
                "cause": ", ".join(causes),
            })

    # Création des tâches (une par client)
    nb_taches = 0
    if client_anomalies:
        type_ev = _ensure_type(db)
        for client_id, details in client_anomalies.items():
            lignes = [
                f"- [{d['ref']}] Semaine {d['date']} : {d['variation']} "
                f"({d['valo_avant']} → {d['valo_apres']} €) — {d['cause']}"
                for d in details
            ]
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
                f"({len(details)} affaire(s) concernée(s))",
            )

        db.commit()

    duree = round(time.time() - t0, 1)
    yield {
        "type": "done",
        "duree_s": duree,
        "nb_affaires": total,
        "nb_anomalies": nb_anomalies,
        "nb_resolus": nb_resolus,
        "nb_taches": nb_taches,
    }
