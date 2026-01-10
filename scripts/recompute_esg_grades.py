#!/usr/bin/env python3
from __future__ import annotations

from src.database import SessionLocal
from src.services.esg_import import _fetch_esg_fonds_columns, _recompute_esg_grades


def main() -> int:
    db = SessionLocal()
    try:
        table_columns = _fetch_esg_fonds_columns(db)
        updates = _recompute_esg_grades(db, table_columns)
        if not updates:
            print("No grade columns found or no updates performed.")
            return 1
        print("ESG grades recomputed:")
        for col, count in updates.items():
            print(f"- {col}: {count} rows")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
