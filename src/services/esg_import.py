from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlencode

from sqlalchemy import text
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
    sql = _build_upsert_sql(write_columns)

    total_fetched = 0
    total_written = 0
    total_skipped = 0
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

    return {
        "fetched": total_fetched,
        "written": total_written,
        "skipped": total_skipped,
        "missing_columns": missing_columns,
        "write_columns": write_columns,
        "base_url": config.base_url,
    }
