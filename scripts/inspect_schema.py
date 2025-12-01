#!/usr/bin/env python3
"""
Utilitaire simple pour lister les tables et colonnes visibles par SQLAlchemy.
À exécuter depuis la racine du projet : `python3 scripts/inspect_schema.py`.
"""
from pathlib import Path
import sys

from sqlalchemy import inspect

# Ajouter la racine du projet au PYTHONPATH si besoin
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import engine


def main():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if not tables:
        print("Aucune table détectée.")
        return

    for table in sorted(tables):
        print(f"\nTable: {table}")
        columns = inspector.get_columns(table)
        for col in columns:
            name = col.get("name")
            col_type = col.get("type")
            nullable = col.get("nullable")
            default = col.get("default")
            print(
                f"  - {name} ({col_type})"
                f"{' NULL' if nullable else ' NOT NULL'}"
                f"{f' DEFAULT {default}' if default is not None else ''}"
            )


if __name__ == "__main__":
    main()
