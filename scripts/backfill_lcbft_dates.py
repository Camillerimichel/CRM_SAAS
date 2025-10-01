from sqlalchemy import text
from src.database import SessionLocal


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
        has_created = column_exists(db, "LCBFT_questionnaire", "created_at")
        has_updated = column_exists(db, "LCBFT_questionnaire", "updated_at")
        has_saisie = column_exists(db, "LCBFT_questionnaire", "saisie_at")
        has_obso = column_exists(db, "LCBFT_questionnaire", "obsolescence_at")

        if has_created and has_saisie:
            db.execute(text(
                "UPDATE LCBFT_questionnaire SET created_at = saisie_at WHERE created_at IS NULL AND saisie_at IS NOT NULL"
            ))
        if has_updated and has_obso:
            db.execute(text(
                "UPDATE LCBFT_questionnaire SET updated_at = obsolescence_at WHERE updated_at IS NULL AND obsolescence_at IS NOT NULL"
            ))
        db.commit()
        print("Backfill completed.")
    except Exception as exc:
        db.rollback()
        print(f"Backfill failed: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    run()

