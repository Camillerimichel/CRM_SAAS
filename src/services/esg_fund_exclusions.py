from __future__ import annotations

from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

# Grades considered "faible" for the faible_note_esg exclusion criterion.
# Distribution is uniform A→G, so E/F/G = bottom third.
_FAIBLE_ESG_GRADES = {"E", "F", "G"}


def get_client_exclusion_codes(db: Session, client_id: int) -> list[dict[str, str]]:
    """Return exclusion codes+labels declared by the client in their latest ESG questionnaire."""
    row = db.execute(
        text(
            """
            SELECT id FROM esg_questionnaire
            WHERE client_ref = :r
            ORDER BY COALESCE(updated_at, saisie_at, created_at, id) DESC
            LIMIT 1
            """
        ),
        {"r": str(client_id)},
    ).fetchone()
    if not row:
        return []
    qid = int(row[0])

    rows = db.execute(
        text(
            """
            SELECT o.code, o.label
            FROM esg_questionnaire_exclusion qe
            JOIN esg_exclusion_option o ON o.id = qe.option_id
            WHERE qe.questionnaire_id = :q AND o.code != 'aucune'
            ORDER BY o.id
            """
        ),
        {"q": qid},
    ).fetchall()
    return [{"code": str(r[0]), "label": str(r[1])} for r in (rows or []) if r[0]]


def check_fund_exclusions(
    db: Session,
    isins: list[str],
    client_id: int,
) -> dict[str, Any]:
    """
    Cross-reference a client's declared ESG exclusions against actual fund data.

    Args:
        isins:     ISINs of funds to check (from portfolio or affaire holdings).
        client_id: Used to fetch the client's latest ESG questionnaire exclusions.

    Returns a dict:
        {
            "client_exclusions": [{"code": ..., "label": ...}],
            "fonds": [
                {
                    "isin": str,
                    "nom": str | None,
                    "divergences": [{"code": ..., "label": ...}],
                    "conforme": bool | None,     # None = données manquantes
                    "donnees_manquantes": bool,
                }
            ],
            "nb_fonds": int,
            "nb_conformes": int,
            "nb_non_conformes": int,
            "nb_donnees_manquantes": int,
            "taux_conformite": float | None,     # sur les fonds avec données
        }
    """
    _empty: dict[str, Any] = {
        "client_exclusions": [],
        "fonds": [],
        "nb_fonds": 0,
        "nb_conformes": 0,
        "nb_non_conformes": 0,
        "nb_donnees_manquantes": 0,
        "taux_conformite": None,
    }

    client_exclusions = get_client_exclusion_codes(db, client_id)
    if not client_exclusions or not isins:
        return {**_empty, "client_exclusions": client_exclusions}

    active_codes = {e["code"] for e in client_exclusions}

    # Fetch fund data: norm table for binary sector flags, fonds table for grade.
    # Also pull commercial name from mariadb_support when available.
    rows = db.execute(
        text(
            """
            SELECT
                n.isin,
                COALESCE(s.nom, n.name) AS nom,
                n.exposure_to_fossil_fuels,
                n.sfdr_biodiversity_pai,
                n.controversial_weapons,
                n.violations_ungc,
                f.note_esg_grade
            FROM esg_fonds_norm n
            LEFT JOIN esg_fonds f ON f.isin = n.isin
            LEFT JOIN mariadb_support s ON s.code_isin = n.isin
            WHERE n.isin IN :isins
            GROUP BY n.isin, s.nom, n.name,
                     n.exposure_to_fossil_fuels, n.sfdr_biodiversity_pai,
                     n.controversial_weapons, n.violations_ungc,
                     f.note_esg_grade
            """
        ).bindparams(bindparam("isins", expanding=True)),
        {"isins": list(isins)},
    ).fetchall()

    fund_data: dict[str, Any] = {}
    for r in rows or []:
        m = r._mapping if hasattr(r, "_mapping") else r
        isin = str(m.get("isin") or "")
        if isin and isin not in fund_data:
            fund_data[isin] = m

    # Labels indexed by code for divergence messages
    excl_by_code = {e["code"]: e for e in client_exclusions}

    def _binary_flag(val: Any) -> bool | None:
        """True if flag = 1, False if 0, None if NULL."""
        if val is None:
            return None
        try:
            return float(val) >= 1.0
        except (TypeError, ValueError):
            return None

    # Per-exclusion check helpers: code → (column, is_triggered_fn)
    _CHECKS: dict[str, tuple[str, Any]] = {
        "fossiles":          ("exposure_to_fossil_fuels", lambda v: _binary_flag(v) is True),
        "zones_sensibles":   ("sfdr_biodiversity_pai",   lambda v: _binary_flag(v) is True),
        "armes_controversees": ("controversial_weapons", lambda v: _binary_flag(v) is True),
        "violations_pacte_ocde": ("violations_ungc",    lambda v: _binary_flag(v) is True),
        "faible_note_esg":   ("note_esg_grade",          lambda v: v is not None and str(v).strip().upper() in _FAIBLE_ESG_GRADES),
    }

    fonds_result: list[dict[str, Any]] = []
    for isin in isins:
        m = fund_data.get(isin)
        if m is None:
            # No ESG data at all for this ISIN
            status_by_code = {e["code"]: "manquant" for e in client_exclusions if e["code"] in _CHECKS}
            fonds_result.append(
                {
                    "isin": isin,
                    "nom": None,
                    "divergences": [],
                    "status_by_code": status_by_code,
                    "conforme": None,
                    "donnees_manquantes": True,
                }
            )
            continue

        nom = str(m.get("nom") or "") or None
        divergences: list[dict[str, str]] = []
        status_by_code: dict[str, str] = {}
        has_missing = False

        for excl in client_exclusions:
            code = excl["code"]
            if code not in _CHECKS:
                continue
            col, is_triggered = _CHECKS[code]
            val = m.get(col) if hasattr(m, "get") else None
            if val is None:
                status_by_code[code] = "manquant"
                has_missing = True
            elif is_triggered(val):
                status_by_code[code] = "divergence"
                divergences.append(excl_by_code[code])
            else:
                status_by_code[code] = "conforme"

        conforme: bool | None = None if has_missing and not divergences else len(divergences) == 0

        fonds_result.append(
            {
                "isin": isin,
                "nom": nom,
                "divergences": divergences,
                "status_by_code": status_by_code,
                "conforme": conforme,
                "donnees_manquantes": has_missing,
            }
        )

    nb_fonds = len(fonds_result)
    nb_conformes = sum(1 for f in fonds_result if f["conforme"] is True)
    nb_non_conformes = sum(1 for f in fonds_result if f["conforme"] is False)
    nb_manquantes = sum(1 for f in fonds_result if f["donnees_manquantes"] and f["conforme"] is None)
    total_checkable = nb_conformes + nb_non_conformes
    taux = round(nb_conformes / total_checkable, 3) if total_checkable > 0 else None

    return {
        "client_exclusions": client_exclusions,
        "fonds": fonds_result,
        "nb_fonds": nb_fonds,
        "nb_conformes": nb_conformes,
        "nb_non_conformes": nb_non_conformes,
        "nb_donnees_manquantes": nb_manquantes,
        "taux_conformite": taux,
    }


def get_client_portfolio_isins(db: Session, client_id: int) -> list[str]:
    """Return ISINs of funds with valo > 0 at the latest portfolio snapshot date."""
    # One global MAX date across all the client's affaires avoids including
    # historical funds that have since been sold (valo dropped to 0).
    snap_dt = db.execute(
        text(
            """
            SELECT MAX(h.date)
            FROM mariadb_historique_support_w h
            JOIN mariadb_affaires a ON a.id = h.id_source
            WHERE a.id_personne = :cid
            """
        ),
        {"cid": client_id},
    ).scalar()
    if not snap_dt:
        return []

    rows = db.execute(
        text(
            """
            SELECT DISTINCT s.code_isin
            FROM mariadb_historique_support_w h
            JOIN mariadb_affaires a ON a.id = h.id_source
            JOIN mariadb_support s ON s.id = h.id_support
            WHERE a.id_personne = :cid
              AND h.date = :d
              AND h.valo > 0
              AND s.code_isin IS NOT NULL AND s.code_isin != ''
            """
        ),
        {"cid": client_id, "d": snap_dt},
    ).fetchall()
    return [str(r[0]) for r in (rows or []) if r[0]]


def get_affaire_portfolio_isins(db: Session, affaire_id: int) -> list[str]:
    """Return ISINs of funds with valo > 0 at the latest snapshot date for this affaire."""
    snap_dt = db.execute(
        text(
            "SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :aid"
        ),
        {"aid": affaire_id},
    ).scalar()
    if not snap_dt:
        return []

    rows = db.execute(
        text(
            """
            SELECT DISTINCT s.code_isin
            FROM mariadb_historique_support_w h
            JOIN mariadb_support s ON s.id = h.id_support
            WHERE h.id_source = :aid
              AND h.date = :d
              AND h.valo > 0
              AND s.code_isin IS NOT NULL AND s.code_isin != ''
            """
        ),
        {"aid": affaire_id, "d": snap_dt},
    ).fetchall()
    return [str(r[0]) for r in (rows or []) if r[0]]
