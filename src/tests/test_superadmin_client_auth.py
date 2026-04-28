import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FASTAPI_SKIP_WARM"] = "1"

import src.api.dashboard as dashboard


class _Result:
    def __init__(self, *, scalar_value=None, fetchone_value=None):
        self._scalar_value = scalar_value
        self._fetchone_value = fetchone_value

    def scalar(self):
        return self._scalar_value

    def fetchone(self):
        return self._fetchone_value


class _FakeDB:
    def __init__(self, responses):
        self.responses = list(responses)
        self.execute_calls = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.execute_calls.append((sql, params or {}))
        if self.responses:
            return self.responses.pop(0)
        return _Result()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture(autouse=True)
def _patch_access(monkeypatch):
    monkeypatch.setattr(dashboard, "extract_user_context", lambda request: ("staff", 7, 1))
    monkeypatch.setattr(dashboard, "load_access", lambda *args, **kwargs: {})
    monkeypatch.setattr(dashboard, "pick_scope", lambda *args, **kwargs: 1)
    monkeypatch.setattr(dashboard, "require_permission", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "hash_password", lambda value: f"hashed::{value}")


def test_superadmin_client_auth_updates_client_email_on_create():
    db = _FakeDB(
        [
            _Result(scalar_value=1869),  # client exists
            _Result(fetchone_value=None),  # duplicate login check
            _Result(),  # auth insert
            _Result(scalar_value=321),  # last insert id
            _Result(),  # email sync
            _Result(),  # role insert ignore
        ]
    )

    response = dashboard.dashboard_superadmin_client_auth(
        request=SimpleNamespace(query_params={}),
        societe_id=1,
        client_id=1869,
        client_account_id=None,
        login="client@example.com",
        password="Secret123!",
        status="active",
        db=db,
    )

    assert response.status_code == 303
    assert any("UPDATE mariadb_clients SET email = :email" in sql for sql, _ in db.execute_calls)
    assert db.commits == 1


def test_superadmin_client_auth_updates_client_email_on_edit():
    db = _FakeDB(
        [
            _Result(scalar_value=1869),  # client exists
            _Result(fetchone_value=(1,)),  # existing account
            _Result(fetchone_value=None),  # duplicate login check
            _Result(),  # auth update
            _Result(),  # email sync
        ]
    )

    response = dashboard.dashboard_superadmin_client_auth(
        request=SimpleNamespace(query_params={}),
        societe_id=1,
        client_id=1869,
        client_account_id=7,
        login="new-email@crmsaas.eu",
        password=None,
        status="disabled",
        db=db,
    )

    assert response.status_code == 303
    assert any("UPDATE mariadb_clients SET email = :email" in sql for sql, _ in db.execute_calls)
    assert db.commits == 1


def test_superadmin_client_auth_duplicate_login_redirects_error():
    db = _FakeDB(
        [
            _Result(scalar_value=1869),  # client exists
            _Result(fetchone_value=(1,)),  # duplicate login
        ]
    )

    response = dashboard.dashboard_superadmin_client_auth(
        request=SimpleNamespace(query_params={}),
        societe_id=1,
        client_id=1869,
        client_account_id=None,
        login="duplicate@crmsaas.eu",
        password=None,
        status="active",
        db=db,
    )

    assert response.status_code == 303
    assert "error=1" in response.headers["location"]
    assert "Login+deja+utilise+pour+ce+courtier" in response.headers["location"]
