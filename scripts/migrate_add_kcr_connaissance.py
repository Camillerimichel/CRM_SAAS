from sqlalchemy import text
from src.database import SessionLocal


DDL_CREATE = """
CREATE TABLE IF NOT EXISTS KYC_Client_Risque_Connaissance (
  risque_id INTEGER NOT NULL,
  produit_id INTEGER NOT NULL,
  niveau_id INTEGER NOT NULL,
  produit_label TEXT,
  niveau_label TEXT,
  PRIMARY KEY (risque_id, produit_id),
  FOREIGN KEY (risque_id) REFERENCES KYC_Client_Risque(id) ON DELETE CASCADE
)
"""
DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_kcr_connaissance_risque ON KYC_Client_Risque_Connaissance(risque_id)
"""


def run():
    db = SessionLocal()
    try:
        db.execute(text(DDL_CREATE))
        db.execute(text(DDL_INDEX))
        db.commit()
        print("KYC_Client_Risque_Connaissance: migration applied.")
    except Exception as exc:
        db.rollback()
        print(f"Migration failed: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()
