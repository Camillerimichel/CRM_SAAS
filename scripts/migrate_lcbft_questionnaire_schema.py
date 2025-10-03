from sqlalchemy import text
from src.database import SessionLocal


EXPECTED_COLUMNS = [
    ("client_ref", "TEXT"),
    ("relation_mode", "TEXT"),
    ("relation_since", "TEXT"),
    ("has_existing_contracts", "INTEGER DEFAULT 0"),
    ("existing_with_our_insurer", "INTEGER DEFAULT 0"),
    ("existing_contract_ref", "TEXT"),
    ("reason_new_contract", "TEXT"),
    ("ppe_self", "INTEGER DEFAULT 0"),
    ("ppe_self_fonction", "TEXT"),
    ("ppe_self_pays", "TEXT"),
    ("ppe_family", "INTEGER DEFAULT 0"),
    ("ppe_family_fonction", "TEXT"),
    ("ppe_family_pays", "TEXT"),
    ("flag_1a", "INTEGER DEFAULT 0"),
    ("flag_1b", "INTEGER DEFAULT 0"),
    ("flag_1c", "INTEGER DEFAULT 0"),
    ("flag_1d", "INTEGER DEFAULT 0"),
    ("flag_2a", "INTEGER DEFAULT 0"),
    ("flag_2b", "INTEGER DEFAULT 0"),
    ("flag_3a", "INTEGER DEFAULT 0"),
    ("computed_risk_level", "INTEGER DEFAULT 1"),
    ("prof_profession", "TEXT"),
    ("prof_statut_professionnel_id", "INTEGER"),
    ("prof_secteur_id", "INTEGER"),
    ("prof_self_ppe", "INTEGER DEFAULT 0"),
    ("prof_self_ppe_fonction", "TEXT"),
    ("prof_self_ppe_pays", "TEXT"),
    ("prof_family_ppe", "INTEGER DEFAULT 0"),
    ("prof_family_ppe_fonction", "TEXT"),
    ("prof_family_ppe_pays", "TEXT"),
    ("operation_objet", "TEXT"),
    ("montant", "REAL"),
    ("patrimoine_pct", "REAL"),
    ("just_fonds", "TEXT"),
    ("just_destination", "TEXT"),
    ("just_finalite", "TEXT"),
    ("just_produits", "TEXT"),
]


def column_exists(db, table: str, col: str) -> bool:
    try:
        rows = db.execute(text(f"PRAGMA table_info({table})")).fetchall()
        names = {str(r[1]).lower() for r in rows}
        return col.lower() in names
    except Exception:
        return False


def run():
    db = SessionLocal()
    try:
        changed = []
        for name, decl in EXPECTED_COLUMNS:
            if not column_exists(db, "LCBFT_questionnaire", name):
                ddl = f"ALTER TABLE LCBFT_questionnaire ADD COLUMN {name} {decl}"
                db.execute(text(ddl))
                changed.append(name)
        if not column_exists(db, "LCBFT_questionnaire", "created_at"):
            # Ensure created_at exists if missing (defensive)
            db.execute(text("ALTER TABLE LCBFT_questionnaire ADD COLUMN created_at DATETIME"))
            changed.append("created_at")
        if not column_exists(db, "LCBFT_questionnaire", "updated_at"):
            db.execute(text("ALTER TABLE LCBFT_questionnaire ADD COLUMN updated_at DATETIME"))
            changed.append("updated_at")
        db.commit()
        print("Migration completed. Added columns:", ", ".join(changed) if changed else "none")
    except Exception as exc:
        db.rollback()
        print(f"Migration failed: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()

