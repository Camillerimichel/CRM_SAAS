from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session


LEGACY_FIELD_REPLACEMENTS = {
    "wasteefficiency": "waste_efficiency",
    "waterefficiency": "water_efficiency",
    "executivepay": "executive_pay",
    "boardindependence": "board_independence",
    "environmentalgood": "environmental_good",
    "environmentalharm": "environmental_harm",
    "socialgood": "social_good",
    "socialharm": "social_harm",
    "numberofemployees": "number_of_employees",
    "avgperemployeespend": "average_per_employee_spend",
    "pctfemaleboard": "pct_female_board",
    "pctfemaleexec": "pct_female_executives",
    "genderpaygap": "gender_pay_gap",
    "boardgenderdiversity": "board_gender_diversity",
    "ghgintensityvalue": "ghg_intensity_value",
    "biodiversity": "biodiversity",
    "emissionstowater": "emissions_to_water",
    "hazardouswaste": "hazardous_waste",
    "scope1and2carbonintensity": "scope_1_and_2_carbon_intensity",
    "scope3carbonintensity": "scope_3_carbon_intensity",
    "carbontrend": "carbon_trend",
    "temperaturescore": "temperature_score",
    "exposuretofossilfuels": "exposure_to_fossil_fuels",
    "renewableenergy": "renewable_energy",
    "climateimpactrevenue": "climate_impact_revenue",
    "climatechangepositive": "climate_change_positive",
    "climatechangenegative": "climate_change_negative",
    "climatechangenet": "climate_change_net",
    "naturalresourcepositive": "natural_resource_positive",
    "naturalresourcenegative": "natural_resource_negative",
    "naturalresourcenet": "natural_resource_net",
    "pollutionpositive": "pollution_positive",
    "pollutionnegative": "pollution_negative",
    "pollutionnet": "pollution_net",
    "avoidingwaterscarcity": "avoiding_water_scarcity",
    "sfdrbiodiversitypai": "sfdr_biodiversity_pai",
    "controversialweapons": "controversial_weapons",
    "violationsungc": "violations_ungc",
    "processesungc": "processes_ungc",
    "notee": "note_e",
    "notes": "note_s",
    "noteg": "note_g",
    "nom": "name",
}


@dataclass
class LegacyMigrationResult:
    tables_scanned: int = 0
    columns_scanned: int = 0
    columns_updated: int = 0
    rows_matched: int = 0
    rows_updated: int = 0
    updated_columns: list[str] = None
    errors: list[str] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "tables_scanned": self.tables_scanned,
            "columns_scanned": self.columns_scanned,
            "columns_updated": self.columns_updated,
            "rows_matched": self.rows_matched,
            "rows_updated": self.rows_updated,
            "updated_columns": self.updated_columns or [],
            "errors": self.errors or [],
        }


def _quote_ident(value: str) -> str:
    safe = str(value).replace("`", "``")
    return f"`{safe}`"


def _find_candidate_columns(db: Session) -> list[tuple[str, str, str]]:
    rows = db.execute(
        text(
            """
            SELECT c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS c
            JOIN INFORMATION_SCHEMA.TABLES t
              ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
             AND t.TABLE_NAME = c.TABLE_NAME
            WHERE c.TABLE_SCHEMA = DATABASE()
              AND t.TABLE_TYPE = 'BASE TABLE'
              AND c.DATA_TYPE IN ('char', 'varchar', 'text', 'mediumtext', 'longtext', 'json')
              AND (
                c.COLUMN_NAME LIKE :p1 OR
                c.COLUMN_NAME LIKE :p2 OR
                c.COLUMN_NAME LIKE :p3
              )
            ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
            """
        ),
        {"p1": "%esg%", "p2": "%field%", "p3": "%indicator%"},
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows or []]


def _build_like_clause(column: str, replacements: Iterable[str]) -> tuple[str, dict[str, str]]:
    clauses = []
    params: dict[str, str] = {}
    for idx, old in enumerate(replacements):
        key = f"like_{idx}"
        clauses.append(f"{_quote_ident(column)} LIKE :{key}")
        params[key] = f"%{old}%"
    return " OR ".join(clauses), params


def _build_replace_expr(column: str, replacements: dict[str, str]) -> tuple[str, dict[str, str]]:
    expr = _quote_ident(column)
    params: dict[str, str] = {}
    for idx, (old, new) in enumerate(replacements.items()):
        expr = f"REPLACE({expr}, :old_{idx}, :new_{idx})"
        params[f"old_{idx}"] = old
        params[f"new_{idx}"] = new
    return expr, params


def migrate_esg_legacy_fields(db: Session) -> LegacyMigrationResult:
    result = LegacyMigrationResult(updated_columns=[], errors=[])
    replacements = LEGACY_FIELD_REPLACEMENTS
    candidates = _find_candidate_columns(db)
    result.tables_scanned = len({t for t, _, _ in candidates})
    result.columns_scanned = len(candidates)

    for table, column, _dtype in candidates:
        try:
            where_sql, where_params = _build_like_clause(column, replacements.keys())
            count = db.execute(
                text(f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE {where_sql}"),
                where_params,
            ).scalar() or 0
            if not count:
                continue
            expr_sql, replace_params = _build_replace_expr(column, replacements)
            params = {}
            params.update(where_params)
            params.update(replace_params)
            update_sql = (
                f"UPDATE {_quote_ident(table)} "
                f"SET {_quote_ident(column)} = {expr_sql} "
                f"WHERE {where_sql}"
            )
            res = db.execute(text(update_sql), params)
            db.commit()
            result.columns_updated += 1
            result.rows_matched += int(count)
            if res.rowcount is not None and res.rowcount >= 0:
                result.rows_updated += int(res.rowcount)
            result.updated_columns.append(f"{table}.{column}")
        except Exception as exc:
            db.rollback()
            result.errors.append(f"{table}.{column}: {exc}")
            continue
    return result
