from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlencode

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

EXPORT_FIELDS = [
    "company_name",
    "sector1",
    "mcap_usd",
    "revenue_usd",
    "processes_ungc",
    "exposure_to_fossil_fuels",
    "renewable_energy",
    "waste_efficiency",
    "water_efficiency",
    "pollution__positive_revenue",
    "environmental_good",
    "social_good",
    "average_per_employee_spend",
    "pct_female_executives",
    "board_gender_diversity",
    "pct_female_board",
    "board_independence",
    "avoiding_water_scarcity",
    "ghg_intensity_value",
    "emissions_to_water",
    "hazardous_waste",
    "scope_1_and_2_carbon_intensity",
    "scope_3_carbon_intensity",
    "carbon_trend",
    "temperature_score",
    "environmental_harm",
    "climate_change__negative_revenue",
    "pollution__negative_revenue",
    "social_harm",
    "gender_pay_gap",
    "executive_pay",
    "note_e",
    "note_s",
    "note_g",
    "note_esg",
    "evic",
]

GRADE_COLUMNS = {
    "note_e": "note_e_grade",
    "note_s": "note_s_grade",
    "note_g": "note_g_grade",
    "note_esg": "note_esg_grade",
}

# Échelle publique ESGSCORE (page Méthodologie, section "Échelle de notation") : bornes absolues
# sur un score normalisé 0-1, indépendantes de la population de fonds présente dans esg_fonds.
# Remplace l'ancien lettrage par septiles (rang relatif sur 7 groupes A-G), qui ne correspondait
# à aucune méthodologie publiée et produisait des lettres (F, G) n'existant pas dans l'échelle A-E.
GRADE_THRESHOLDS = (
    ("A", 0.80),
    ("B", 0.65),
    ("C", 0.45),
    ("D", 0.30),
)
GRADE_FLOOR_LETTER = "E"


def _grade_for_score(score: object) -> str | None:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    for letter, threshold in GRADE_THRESHOLDS:
        if value >= threshold:
            return letter
    return GRADE_FLOOR_LETTER


@dataclass(frozen=True)
class CrmEsgConfig:
    base_url: str
    email: str
    password: str
    timeout: int
    page_size: int
    max_pages: int | None


def load_crm_esg_config() -> CrmEsgConfig:
    base_url = os.getenv("CRM_ESG_API_BASE", "https://esgnote.eu").rstrip("/")
    email = os.getenv("CRM_ESG_API_EMAIL")
    password = os.getenv("CRM_ESG_API_PASSWORD")
    if not email or not password:
        raise RuntimeError("Missing CRM_ESG_API_EMAIL or CRM_ESG_API_PASSWORD")
    timeout = int(os.getenv("CRM_ESG_API_TIMEOUT", "30"))
    page_size = int(os.getenv("CRM_ESG_API_PAGE_SIZE", "1000"))
    max_pages_raw = os.getenv("CRM_ESG_API_MAX_PAGES")
    max_pages = int(max_pages_raw) if max_pages_raw else None
    return CrmEsgConfig(
        base_url=base_url,
        email=email,
        password=password,
        timeout=timeout,
        page_size=page_size,
        max_pages=max_pages,
    )


def _request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _crm_esg_login(config: CrmEsgConfig) -> str:
    payload = {"email": config.email, "password": config.password}
    data = _request_json(
        f"{config.base_url}/api/auth/login",
        method="POST",
        payload=payload,
        headers={"Content-Type": "application/json"},
        timeout=config.timeout,
    )
    token = data.get("access_token")
    if not token:
        raise RuntimeError("CRM_ESG login failed (missing access_token)")
    return token


def _fetch_export_page(
    config: CrmEsgConfig,
    token: str,
    limit: int,
    offset: int,
    isins: Iterable[str] | None = None,
) -> list[dict]:
    params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
    if isins:
        params["isins"] = ",".join(isins)
    url = f"{config.base_url}/api/referentiel/export-esg?{urlencode(params)}"
    payload = _request_json(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {token}"},
        timeout=config.timeout,
    )
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise RuntimeError("CRM_ESG export payload invalid (items missing)")
    return items


def _fetch_esg_fonds_columns(db: Session) -> set[str]:
    rows = db.execute(text("SHOW COLUMNS FROM esg_fonds")).fetchall()
    return {str(r[0]) for r in rows or [] if r and r[0]}


def _normalize_isin(value: object) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip().upper()
    if len(text_value) > 12:
        text_value = text_value[:12]
    return text_value or None


def _normalize_value(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _prepare_rows(items: Iterable[dict], write_columns: list[str]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    for item in items:
        isin = _normalize_isin(item.get("isin"))
        if not isin:
            skipped += 1
            continue
        row = {"isin": isin}
        for col in write_columns:
            row[col] = _normalize_value(item.get(col))
        rows.append(row)
    return rows, skipped


def _build_upsert_sql(write_columns: list[str]) -> str:
    columns = ["isin"] + write_columns
    cols_sql = ", ".join(f"`{c}`" for c in columns)
    vals_sql = ", ".join(f":{c}" for c in columns)
    if write_columns:
        update_sql = ", ".join(f"`{c}` = VALUES(`{c}`)" for c in write_columns)
        return f"INSERT INTO esg_fonds ({cols_sql}) VALUES ({vals_sql}) ON DUPLICATE KEY UPDATE {update_sql}"
    return f"INSERT IGNORE INTO esg_fonds ({cols_sql}) VALUES ({vals_sql})"


def _clear_orphaned_pai_rows(
    db: Session,
    write_columns: list[str],
    grade_columns: list[str],
    fetched_isins: set[str],
    requested_isins: set[str] | None,
) -> list[str]:
    """Un ISIN déjà importé (colonnes PAI renseignées) qui n'apparaît plus dans l'export CRM_ESG a
    disparu du référentiel source (ex: archivage annuel sans recréation de la ligne). sync_esg_fonds
    ne faisait auparavant que des UPSERT : une notation figée d'avant la disparition restait donc
    affichée indéfiniment, sans source vivante côté CRM_ESG. On efface ici les colonnes PAI/notes/
    grades des ISIN orphelins — jamais les colonnes d'exclusions look-through (sync_esg_exclusions_
    holdings), qui restent valables tant que le fonds est suivi comme portefeuille côté CRM_ESG."""
    if not write_columns:
        return []

    if requested_isins is not None:
        # Sync ciblée sur un sous-ensemble d'ISIN : seuls ceux explicitement demandés mais absents
        # de la réponse sont candidats (on n'a pas vu le reste du référentiel dans cet appel).
        candidate_isins = requested_isins - fetched_isins
    else:
        has_data_sql = " OR ".join(f"`{c}` IS NOT NULL" for c in write_columns)
        rows = db.execute(text(f"SELECT isin FROM esg_fonds WHERE {has_data_sql}")).fetchall()
        known_isins = {r[0] for r in rows or [] if r and r[0]}
        candidate_isins = known_isins - fetched_isins

    if not candidate_isins:
        return []

    clear_columns = write_columns + [c for c in grade_columns if c not in write_columns]
    set_sql = ", ".join(f"`{c}` = NULL" for c in clear_columns)
    db.execute(
        text(f"UPDATE esg_fonds SET {set_sql} WHERE isin = :isin"),
        [{"isin": isin} for isin in candidate_isins],
    )
    db.commit()
    return sorted(candidate_isins)


def _recompute_esg_grades(db: Session, table_columns: set[str]) -> dict[str, int]:
    updated_counts: dict[str, int] = {}
    for score_col, grade_col in GRADE_COLUMNS.items():
        if score_col not in table_columns or grade_col not in table_columns:
            continue
        rows = db.execute(
            text(f"SELECT isin, `{score_col}` FROM esg_fonds WHERE `{score_col}` IS NOT NULL")
        ).fetchall()
        updates = [
            {"isin": r[0], "grade": grade}
            for r in (rows or [])
            if r and r[0] is not None and (grade := _grade_for_score(r[1])) is not None
        ]
        if updates:
            db.execute(
                text(f"UPDATE esg_fonds SET `{grade_col}` = :grade WHERE isin = :isin"),
                updates,
            )
        db.execute(
            text(f"UPDATE esg_fonds SET `{grade_col}` = NULL WHERE `{score_col}` IS NULL")
        )
        updated_counts[grade_col] = len(updates)
    db.commit()
    return updated_counts


def sync_esg_fonds(
    db: Session,
    isins: Iterable[str] | None = None,
) -> dict[str, object]:
    config = load_crm_esg_config()
    token = _crm_esg_login(config)
    table_columns = _fetch_esg_fonds_columns(db)
    if "isin" not in table_columns:
        raise RuntimeError("esg_fonds table missing isin column")
    write_columns = [c for c in EXPORT_FIELDS if c in table_columns]
    missing_columns = [c for c in EXPORT_FIELDS if c not in table_columns]
    grade_columns = [c for c in GRADE_COLUMNS.values() if c in table_columns]
    sql = _build_upsert_sql(write_columns)

    requested_isins: set[str] | None = None
    if isins:
        requested_isins = {norm for i in isins if (norm := _normalize_isin(i))}

    total_fetched = 0
    total_written = 0
    total_skipped = 0
    fetched_isins: set[str] = set()
    page = 0
    offset = 0
    while True:
        items = _fetch_export_page(
            config,
            token=token,
            limit=config.page_size,
            offset=offset,
            isins=isins,
        )
        if not items:
            break
        total_fetched += len(items)
        rows, skipped = _prepare_rows(items, write_columns)
        total_skipped += skipped
        fetched_isins.update(row["isin"] for row in rows)
        if rows:
            db.execute(text(sql), rows)
            db.commit()
            total_written += len(rows)
        if len(items) < config.page_size:
            break
        page += 1
        if config.max_pages is not None and page >= config.max_pages:
            break
        offset += config.page_size

    orphaned_cleared: list[str] = []
    try:
        orphaned_cleared = _clear_orphaned_pai_rows(
            db, write_columns, grade_columns, fetched_isins, requested_isins
        )
    except Exception:
        db.rollback()
        orphaned_cleared = []

    grade_updates = {}
    try:
        grade_updates = _recompute_esg_grades(db, table_columns)
    except Exception:
        db.rollback()
        grade_updates = {}

    return {
        "fetched": total_fetched,
        "written": total_written,
        "skipped": total_skipped,
        "missing_columns": missing_columns,
        "write_columns": write_columns,
        "base_url": config.base_url,
        "grade_updates": grade_updates,
        "orphaned_cleared": orphaned_cleared,
    }


# --- Exclusions ESG "look-through" par fonds ---
# sync_esg_fonds (ci-dessus) ne récupère que des indicateurs PAI au niveau du fonds lui-même
# (peu de colonnes d'exclusion, cf. EXPORT_FIELDS). CRM_ESG dispose d'un mécanisme bien plus riche
# (/qualification-esg, onglet Fonds) qui regarde la composition réelle du fonds (ses positions sous-
# jacentes) et compte, pour 14 catégories (business-involvement + conduite), combien de positions
# déclenchent chacune. Ce qui suit synchronise ce comptage sous forme de flags binaires par fonds
# ("au moins une position du fonds déclenche cette catégorie") dans des colonnes esg_fonds dédiées,
# distinctes des colonnes PAI existantes (jamais écrasées).

EXCLUSION_COUNT_TO_COLUMN = {
    "excluded_coal_count": "excluded_coal",
    "excluded_oil_gas_count": "excluded_oil_gas",
    "excluded_tar_sands_count": "excluded_tar_sands",
    "excluded_tobacco_count": "excluded_tobacco",
    "excluded_weapons_count": "excluded_weapons",
    "excluded_controversial_weapons_count": "excluded_weapons_controversial",
    "excluded_gambling_count": "excluded_gambling",
    "excluded_alcohol_count": "excluded_alcohol",
    "excluded_nuclear_count": "excluded_nuclear",
    "excluded_pornography_count": "excluded_pornography",
    "excluded_fossil_power_generation_count": "excluded_fossil_power_generation",
    "corruption_issue_count": "excluded_corruption",
    "human_rights_issue_count": "excluded_human_rights_issue",
    "forced_labour_issue_count": "excluded_forced_labour",
    "environmental_issue_count": "excluded_environmental_issue",
}


def _fetch_fund_exclusions(config: CrmEsgConfig, token: str, isins: Iterable[str] | None = None) -> list[dict]:
    params: dict[str, str] = {}
    if isins:
        params["isins"] = ",".join(isins)
    url = f"{config.base_url}/api/esg-qualification/export/fund-exclusions"
    if params:
        url += f"?{urlencode(params)}"
    payload = _request_json(url, method="GET", headers={"Authorization": f"Bearer {token}"}, timeout=config.timeout)
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise RuntimeError("CRM_ESG fund-exclusions payload invalide (items manquant)")
    return items


def sync_esg_exclusions_holdings(db: Session, isins: Iterable[str] | None = None) -> dict[str, object]:
    """Synchronise les exclusions look-through par fonds (cf. commentaire ci-dessus). Un fonds
    absent de la réponse (pas enregistré comme "portefeuille" avec composition connue côté CRM_ESG)
    n'est simplement pas touché — pas de valeur par défaut supposée."""
    config = load_crm_esg_config()
    token = _crm_esg_login(config)
    table_columns = _fetch_esg_fonds_columns(db)
    if "isin" not in table_columns:
        raise RuntimeError("esg_fonds table missing isin column")
    write_columns = [c for c in EXCLUSION_COUNT_TO_COLUMN.values() if c in table_columns]
    missing_columns = [c for c in EXCLUSION_COUNT_TO_COLUMN.values() if c not in table_columns]

    items = _fetch_fund_exclusions(config, token, isins=isins)

    rows: list[dict] = []
    for item in items:
        isin = _normalize_isin(item.get("fund_isin"))
        if not isin:
            continue
        row = {"isin": isin}
        for count_field, column in EXCLUSION_COUNT_TO_COLUMN.items():
            if column not in write_columns:
                continue
            valeur = item.get(count_field)
            row[column] = None if valeur is None else (1 if float(valeur) > 0 else 0)
        rows.append(row)

    written = 0
    if rows and write_columns:
        sql = _build_upsert_sql(write_columns)
        db.execute(text(sql), rows)
        db.commit()
        written = len(rows)

    return {
        "fetched": len(items),
        "written": written,
        "write_columns": write_columns,
        "missing_columns": missing_columns,
        "base_url": config.base_url,
    }


# --- Note ESG "look-through" (repli pour les fonds jamais notés directement) ---
# Un fonds suivi comme "portefeuille" côté CRM_ESG (composition connue via
# portefeuille_inventaire_ligne) n'est pas forcément noté lui-même dans Referentiel_final (source
# de sync_esg_fonds ci-dessus) — c'est deux systèmes distincts. Pour ces fonds, on calcule une note
# de repli = moyenne pondérée des notes E/S/G/ESG des sociétés réellement détenues (déjà notées
# individuellement dans Referentiel_final), via GET /api/stats/fund-lookthrough-note côté CRM_ESG.
# Ne touche JAMAIS un fonds qui a déjà une notation directe (note_esg_grade non NULL) pour ne pas la
# masquer — à exécuter après sync_esg_fonds dans la même passe pour que ce test reflète l'état frais
# (orphelins déjà nettoyés par sync_esg_fonds à ce stade).

def sync_esg_lookthrough_notes(db: Session, isins: Iterable[str] | None = None) -> dict[str, object]:
    config = load_crm_esg_config()
    table_columns = _fetch_esg_fonds_columns(db)
    if "isin" not in table_columns:
        raise RuntimeError("esg_fonds table missing isin column")

    grade_columns_map = {"note_e": "note_e_grade", "note_s": "note_s_grade", "note_g": "note_g_grade", "note_esg": "note_esg_grade"}
    write_columns = [c for c in grade_columns_map if c in table_columns] + [g for g in grade_columns_map.values() if g in table_columns]

    portfolios_payload = _request_json(f"{config.base_url}/api/stats/portfolios", timeout=config.timeout)
    tracked_isins = [i for p in (portfolios_payload.get("items") or []) if (i := _normalize_isin(p.get("isin")))]
    if isins:
        voulus = {i for val in isins if (i := _normalize_isin(val))}
        tracked_isins = [i for i in tracked_isins if i in voulus]

    deja_notes: set[str] = set()
    if tracked_isins:
        rows_existants = db.execute(
            text("SELECT isin, note_esg_grade FROM esg_fonds WHERE isin IN :isins")
            .bindparams(bindparam("isins", expanding=True)),
            {"isins": tracked_isins},
        ).fetchall()
        deja_notes = {r[0] for r in rows_existants if r[1] is not None}
    candidats = [i for i in tracked_isins if i not in deja_notes]

    rows: list[dict] = []
    for isin in candidats:
        try:
            data = _request_json(
                f"{config.base_url}/api/stats/fund-lookthrough-note?{urlencode({'isin': isin})}",
                timeout=config.timeout,
            )
        except Exception:
            continue
        row: dict[str, object] = {"isin": isin}
        for score_col, grade_col in grade_columns_map.items():
            valeur = data.get(score_col)
            if score_col in write_columns:
                row[score_col] = valeur
            if grade_col in write_columns:
                row[grade_col] = _grade_for_score(valeur) if valeur is not None else None
        rows.append(row)

    written = 0
    if rows and write_columns:
        sql = _build_upsert_sql(write_columns)
        db.execute(text(sql), rows)
        db.commit()
        written = len(rows)

    return {
        "tracked": len(tracked_isins),
        "deja_notes_ignores": len(deja_notes),
        "candidats": len(candidats),
        "written": written,
        "write_columns": write_columns,
        "base_url": config.base_url,
    }
