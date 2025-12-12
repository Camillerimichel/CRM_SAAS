#!/usr/bin/env python3
"""
Utilitaire : insère des métriques SRI factices pour un client afin de forcer l'affichage
du tableau “Scénarios de tension” dans le rapport KYC.

Ensuite, régénère le PDF via `/dashboard/clients/kyc/{client_id}/rapport?pdf=1`.
"""
from datetime import date, datetime
from decimal import Decimal
import sys

from sqlalchemy import text

from src.database import SessionLocal


def seed_sri_for_client(client_id: int) -> None:
    values = {
        "entity_type": "client",
        "entity_id": client_id,
        "as_of_date": date.today(),
        "calc_at": datetime.utcnow(),
        "sri": 3,
        "m0": 902,
        "m1": Decimal("0.001035"),
        "m2": Decimal("0.000123"),
        "m3": Decimal("-0.000001"),
        "m4": Decimal("0.000000"),
        "sigma": Decimal("0.011111"),
        "mu1": Decimal("-0.531076"),
        "mu2": Decimal("7.659020"),
        "n_periods": 902,
        "var_cf": Decimal("-0.231836"),
        "vev": Decimal("0.022733"),
    }

    session = SessionLocal()
    try:
        session.execute(
            text(
                """
                INSERT INTO sri_metrics (
                    entity_type, entity_id, as_of_date, calc_at,
                    sri, m0, m1, m2, m3, m4,
                    sigma, mu1, mu2, n_periods, var_cf, vev
                ) VALUES (
                    :entity_type, :entity_id, :as_of_date, :calc_at,
                    :sri, :m0, :m1, :m2, :m3, :m4,
                    :sigma, :mu1, :mu2, :n_periods, :var_cf, :vev
                )
                ON DUPLICATE KEY UPDATE
                    calc_at = VALUES(calc_at),
                    sri = VALUES(sri),
                    m0 = VALUES(m0),
                    m1 = VALUES(m1),
                    m2 = VALUES(m2),
                    m3 = VALUES(m3),
                    m4 = VALUES(m4),
                    sigma = VALUES(sigma),
                    mu1 = VALUES(mu1),
                    mu2 = VALUES(mu2),
                    n_periods = VALUES(n_periods),
                    var_cf = VALUES(var_cf),
                    vev = VALUES(vev)
                """
            ),
            values,
        )
        session.commit()
        print(f"[seed_client_sri] métriques SRI insérées pour le client {client_id}.")
    except Exception as exc:
        session.rollback()
        print(f"[seed_client_sri] erreur : {exc}")
        raise
    finally:
        session.close()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} CLIENT_ID")
        raise SystemExit(1)
    try:
        client_id = int(sys.argv[1])
    except ValueError:
        print("CLIENT_ID doit être un entier.")
        raise SystemExit(1)
    seed_sri_for_client(client_id)


if __name__ == "__main__":
    main()
