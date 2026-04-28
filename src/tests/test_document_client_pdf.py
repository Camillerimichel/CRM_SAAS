import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FASTAPI_SKIP_WARM"] = "1"

from src.api.main import app  # noqa: F401
import src.api.dashboard as dashboard
from src.services.document_client import create_document_client
from src.schemas.document_client import DocumentClientCreateSchema
from src.models.client import Client
from src.models.document import Document
from src.models.document_client import DocumentClient


class _FakeQuery:
    def __init__(self, doc):
        self._doc = doc

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._doc


class _FakeDB:
    def __init__(self, doc):
        self._doc = doc

    def query(self, model):
        return _FakeQuery(self._doc)

    def execute(self, *args, **kwargs):
        class _Result:
            def fetchall(self):
                return []

            def fetchone(self):
                return None

            def scalar(self):
                return None

        return _Result()

    def commit(self):
        return None

        def rollback(self):
            return None


class _SendQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _SendDB:
    def __init__(self, doc, client):
        self.doc = doc
        self.client = client

    def query(self, model):
        if model is DocumentClient:
            return _SendQuery(self.doc)
        if model is Client:
            return _SendQuery(self.client)
        return _SendQuery(None)

    def execute(self, *args, **kwargs):
        class _Result:
            def fetchall(self):
                return []

            def fetchone(self):
                return None

            def scalar(self):
                return None

        return _Result()

    def commit(self):
        return None

    def rollback(self):
        return None


class _CreateQuery:
    def __init__(self, responses):
        self._responses = responses
        self._filters = []

    def filter(self, *args, **kwargs):
        self._filters.extend(args)
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        value = self._responses.pop(0) if self._responses else None
        return value


class _CreateDocDB:
    def __init__(self, recent_doc=None):
        self.recent_doc = recent_doc
        self.execute_calls = []
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        if model is DocumentClient:
            return _CreateQuery([self.recent_doc])
        if model is Client:
            return _CreateQuery([type("ClientRow", (), {"nom": "Test", "prenom": "Client"})()])
        if model is Document:
            return _CreateQuery([type("DocumentRow", (), {"id_document_base": 12})()])
        return _CreateQuery([True])

    def execute(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))

        class _Result:
            def scalar(self_inner):
                return 999

            def fetchone(self_inner):
                return None

        return _Result()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakeDoc:
    def __init__(self, *, doc_id, client_id, stored_path=None, stored_filename=None, mime_type="application/pdf"):
        self.id = doc_id
        self.id_client = client_id
        self.stored_path = stored_path
        self.stored_filename = stored_filename
        self.mime_type = mime_type


class _FakeClient:
    def __init__(self, email="client@example.com", nom="Morel", prenom="François"):
        self.email = email
        self.nom = nom
        self.prenom = prenom


def _override_dependencies(monkeypatch):
    monkeypatch.setattr(dashboard, "_require_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "ensure_document_client_storage_columns", lambda db: True)


def test_resolve_document_client_file_with_generated_client_fallback(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    pdf_path = documents_dir / "generated_clients" / "1869" / "Modele_Relance_1869_test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nfake")

    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "DOCUMENTS_DIR", documents_dir)

    resolved = dashboard._resolve_document_client_file(
        None,
        "Modele_Relance_1869_test.pdf",
        1869,
    )

    assert resolved == pdf_path


def test_document_client_pdf_inline_and_download(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    pdf_path = documents_dir / "generated_clients" / "1869" / "Modele_Relance_1869_test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nfake-pdf")

    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "DOCUMENTS_DIR", documents_dir)

    _override_dependencies(monkeypatch)
    response = dashboard.dashboard_document_client_pdf(1, object(), inline=1, db=_FakeDB(
        _FakeDoc(
            doc_id=1,
            client_id=1869,
            stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
            stored_filename="Modele_Relance_1869_test.pdf",
        )
    ))
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("inline; filename=")
    assert response.path == str(pdf_path)

    response = dashboard.dashboard_document_client_pdf(1, object(), inline=0, db=_FakeDB(
        _FakeDoc(
            doc_id=1,
            client_id=1869,
            stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
            stored_filename="Modele_Relance_1869_test.pdf",
        )
    ))
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment; filename=")


def test_document_client_pdf_head_returns_200(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    pdf_path = documents_dir / "generated_clients" / "1869" / "Modele_Relance_1869_test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nfake-pdf")

    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "DOCUMENTS_DIR", documents_dir)
    _override_dependencies(monkeypatch)

    response = dashboard.dashboard_document_client_pdf_head(1, object(), inline=1, db=_FakeDB(
        _FakeDoc(
            doc_id=1,
            client_id=1869,
            stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
            stored_filename="Modele_Relance_1869_test.pdf",
        )
    ))
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("inline; filename=")


def test_document_client_pdf_missing_file_returns_404(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "DOCUMENTS_DIR", documents_dir)

    _override_dependencies(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        dashboard.dashboard_document_client_pdf(2, object(), inline=0, db=_FakeDB(
            _FakeDoc(
                doc_id=2,
                client_id=1869,
                stored_path="generated_clients/1869/missing.pdf",
                stored_filename="missing.pdf",
            )
        ))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Fichier PDF introuvable."


def test_document_client_pdf_missing_document_returns_404(monkeypatch):
    _override_dependencies(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        dashboard.dashboard_document_client_pdf(9999, object(), inline=0, db=_FakeDB(None))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Document client introuvable."


def test_document_client_send_success_returns_diagnostics(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    pdf_path = documents_dir / "generated_clients" / "1869" / "Modele_Relance_1869_test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nfake-pdf")

    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "DOCUMENTS_DIR", documents_dir)
    _override_dependencies(monkeypatch)
    sent = {}

    def fake_send(recipient, subject, body, attachments=()):
        sent["recipient"] = recipient
        sent["subject"] = subject
        sent["body"] = body
        sent["attachments"] = list(attachments)
        return True

    monkeypatch.setattr(dashboard, "send_email_with_attachments", fake_send)

    doc = _FakeDoc(
        doc_id=1,
        client_id=1869,
        stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
        stored_filename="Modele_Relance_1869_test.pdf",
    )
    client = _FakeClient()
    response = asyncio.run(dashboard.dashboard_document_client_send(1, object(), db=_SendDB(doc, client)))

    assert response["recipient"] == "client@example.com"
    assert response["subject"] == "Document client"
    assert response["attachment"]["filename"] == "Modele_Relance_1869_test.pdf"
    assert sent["recipient"] == "client@example.com"
    assert sent["attachments"][0][0] == "Modele_Relance_1869_test.pdf"


def test_document_client_send_missing_email_returns_400(monkeypatch):
    _override_dependencies(monkeypatch)
    doc = _FakeDoc(
        doc_id=1,
        client_id=1869,
        stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
        stored_filename="Modele_Relance_1869_test.pdf",
    )
    client = _FakeClient(email="")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(dashboard.dashboard_document_client_send(1, object(), db=_SendDB(doc, client)))
    assert exc.value.status_code == 400
    assert exc.value.detail == "Aucune adresse email pour ce client."


def test_create_document_client_reuses_recent_generated_document():
    recent_doc = type(
        "RecentDoc",
        (),
        {
            "id": 321,
            "date_creation": datetime.utcnow(),
        },
    )()
    db = _CreateDocDB(recent_doc=recent_doc)
    payload = DocumentClientCreateSchema(
        id_client=1869,
        id_document_base=12,
        nom_document="Modele Relance - Test",
        date_creation=datetime.utcnow(),
        obsolescence="généré",
        stored_filename="Modele_Relance_1869_test.pdf",
        stored_path="generated_clients/1869/Modele_Relance_1869_test.pdf",
        mime_type="application/pdf",
        file_size=1024,
    )

    doc, err = create_document_client(db, payload)

    assert err is None
    assert doc["id"] == 321
    assert db.commits == 0
    assert db.execute_calls == []
