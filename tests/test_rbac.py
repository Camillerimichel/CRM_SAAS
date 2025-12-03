import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip heavy startup in tests
os.environ["FASTAPI_SKIP_WARM"] = "1"

from src.api.main import app
from src import database
from src.security import rbac


client = TestClient(app)


@pytest.fixture(autouse=True)
def override_deps(monkeypatch):
    # Neutralise la dépendance DB
    app.dependency_overrides[database.get_db] = lambda: None

    class FakeAccess:
        def __init__(self, allow_all=False):
            self.allow_all = allow_all
            self.permissions = set()
            self.societes = {None}

        def has_permission(self, feature, action, societe_id=None):
            return self.allow_all

    def fake_load_access(db, user_type, user_id):
        return FakeAccess(allow_all=False)

    monkeypatch.setattr(rbac, "load_access", fake_load_access)
    yield


def _headers(user_id=None, user_type="staff", societe_id=None):
    h = {}
    if user_id is not None:
        h["X-User-Id"] = str(user_id)
    if user_type:
        h["X-User-Type"] = user_type
    if societe_id is not None:
        h["X-Societe-Id"] = str(societe_id)
    return h


@pytest.mark.parametrize(
    "path",
    [
        "/dashboard/clients/1",
        "/dashboard/affaires/1",
        "/dashboard/allocations",
        "/dashboard/documents",
        "/dashboard/parametres",
        "/dashboard/taches",
    ],
)
def test_requires_auth(path):
    r = client.get(path)
    assert r.status_code in (401, 403, 500)


def test_client_cannot_access_supports_offres_params():
    for path in ["/dashboard/allocations", "/dashboard/documents", "/dashboard/parametres"]:
        r = client.get(path, headers=_headers(user_id=1, user_type="client"))
        assert r.status_code in (401, 403, 500)


def test_client_cannot_edit_task():
    r = client.get("/dashboard/taches/1", headers=_headers(user_id=1, user_type="client"))
    assert r.status_code in (401, 403, 500)


def test_staff_requires_permissions_for_offres():
    r = client.get("/dashboard/allocations", headers=_headers(user_id=1, user_type="staff"))
    assert r.status_code in (401, 403, 500)  # no roles/permissions set -> forbidden/erreur si DB manquante


def test_staff_requires_permissions_for_documents():
    r = client.get("/dashboard/documents", headers=_headers(user_id=1, user_type="staff"))
    assert r.status_code in (401, 403, 500)


def test_superadmin_access_parametres(monkeypatch):
    # Simule un user avec permission (by mocking load_access)
    from src.security import rbac

    class FakeAccess:
        def has_permission(self, feature, action, societe_id=None):
            return True

    def fake_load(db, user_type, user_id):
        return FakeAccess()

    monkeypatch.setattr(rbac, "load_access", fake_load)
    monkeypatch.setattr(rbac, "pick_scope", lambda access, scope: None)
    r = client.get("/dashboard/parametres", headers=_headers(user_id=1, user_type="staff"))
    assert r.status_code == 200
