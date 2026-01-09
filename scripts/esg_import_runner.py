#!/usr/bin/env python3
from __future__ import annotations

import sys
from time import perf_counter

from sqlalchemy import text

from src.database import SessionLocal
from src.services.esg_import import sync_esg_fonds


def _ensure_log_tables(db) -> None:
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS log_type (
              id INT AUTO_INCREMENT PRIMARY KEY,
              code VARCHAR(100) NOT NULL UNIQUE,
              label VARCHAR(255) NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    )
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS log_suivi (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              log_type_id INT NOT NULL,
              started_at DATETIME NOT NULL,
              ended_at DATETIME NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'running',
              message TEXT NULL,
              user_id INT NULL,
              ip_address VARCHAR(64) NULL,
              duration_seconds DECIMAL(12, 2) NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              INDEX idx_log_type_started (log_type_id, started_at DESC),
              CONSTRAINT fk_log_suivi_type FOREIGN KEY (log_type_id) REFERENCES log_type(id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    )
    db.commit()


def _upsert_log_type(db, code: str, label: str) -> int | None:
    row = db.execute(text("SELECT id FROM log_type WHERE code = :c LIMIT 1"), {"c": code}).fetchone()
    if row:
        return row[0] if not hasattr(row, "_mapping") else row._mapping.get("id")
    db.execute(
        text(
            """
            INSERT INTO log_type (code, label)
            VALUES (:c, :l)
            ON DUPLICATE KEY UPDATE label = VALUES(label)
            """
        ),
        {"c": code, "l": label},
    )
    db.commit()
    row = db.execute(text("SELECT id FROM log_type WHERE code = :c LIMIT 1"), {"c": code}).fetchone()
    if row:
        return row[0] if not hasattr(row, "_mapping") else row._mapping.get("id")
    return None


def _start_log_entry(db, code: str, label: str) -> int | None:
    _ensure_log_tables(db)
    log_type_id = _upsert_log_type(db, code, label)
    if log_type_id is None:
        return None
    db.execute(
        text(
            """
            INSERT INTO log_suivi (log_type_id, started_at, status)
            VALUES (:ltid, NOW(), 'running')
            """
        ),
        {"ltid": log_type_id},
    )
    db.commit()
    row = db.execute(text("SELECT LAST_INSERT_ID()")).fetchone()
    if row:
        return row[0] if not hasattr(row, "_mapping") else row._mapping.get(list(row._mapping.keys())[0])
    return None


def _finish_log_entry(db, log_id: int | None, status: str, message: str | None, duration: float | None) -> None:
    if log_id is None:
        return
    db.execute(
        text(
            """
            UPDATE log_suivi
            SET
              status = :st,
              message = :msg,
              ended_at = COALESCE(ended_at, NOW()),
              duration_seconds = :dur
            WHERE id = :lid
            """
        ),
        {
            "st": status,
            "msg": message[:2000] if message else None,
            "dur": round(duration, 2) if duration is not None else None,
            "lid": log_id,
        },
    )
    db.commit()


def main() -> int:
    db = SessionLocal()
    log_id = None
    started = perf_counter()
    try:
        log_id = _start_log_entry(db, "esg_import", "Import ESG (CRM_ESG)")
        stats = sync_esg_fonds(db)
        elapsed = perf_counter() - started
        message = (
            "ESG import OK: "
            f"{stats.get('written')} rows written, "
            f"{stats.get('fetched')} fetched, "
            f"{stats.get('skipped')} skipped."
        )
        _finish_log_entry(db, log_id, "ok", message, elapsed)
        print(message)
        return 0
    except Exception as exc:
        elapsed = perf_counter() - started
        _finish_log_entry(db, log_id, "error", f"ESG import failed: {exc}", elapsed)
        print(f"ESG import failed: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
