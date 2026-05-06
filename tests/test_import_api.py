"""
Tests des endpoints FastAPI d'import via TestClient.
Les fonctions de service sont patchées : on vérifie le routage,
la sérialisation, et la transmission correcte des paramètres.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FASTAPI_SKIP_WARM"] = "1"

from src.api.main import app
from src import database
from src.schemas.import_portefeuille import (
    ImportPreviewResult,
    ImportCommitResult,
    ImportAlerte,
)

DATA_DIR = Path(__file__).parent / "import_data"

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_db():
    app.dependency_overrides[database.get_db] = lambda: MagicMock()
    yield
    app.dependency_overrides.clear()


def _upload(path: str, filename: str, content: bytes):
    return client.post(path, files={"file": (filename, content, "text/csv")})


def _preview_result(**kwargs):
    defaults = dict(
        total_lignes=2, lignes_valides=2, lignes_invalides=0,
        alertes=[], apercu=[],
    )
    defaults.update(kwargs)
    return ImportPreviewResult(**defaults)


def _commit_result(**kwargs):
    defaults = dict(insere=2, mis_a_jour=0, alertes=[])
    defaults.update(kwargs)
    return ImportCommitResult(**defaults)


# ─── /import/inventaire/preview ───────────────────────────────────────────────

class TestInventairePreview:
    def test_returns_200_with_csv(self):
        with patch("src.api.import_portefeuille.preview_inventaire", return_value=_preview_result()):
            res = _upload("/import/inventaire/preview", "test.csv", b"ref_affaire,date,code_isin,nbuc,vl\n")
        assert res.status_code == 200

    def test_response_schema(self):
        with patch("src.api.import_portefeuille.preview_inventaire", return_value=_preview_result(total_lignes=5)):
            res = _upload("/import/inventaire/preview", "inv.csv", b"ref_affaire,date,code_isin,nbuc,vl\n")
        data = res.json()
        assert data["total_lignes"] == 5
        assert "lignes_valides" in data
        assert "alertes" in data
        assert "apercu" in data

    def test_no_file_returns_400(self):
        res = client.post("/import/inventaire/preview")
        assert res.status_code == 400

    def test_real_csv_file(self):
        content = (DATA_DIR / "test_inventaire.csv").read_bytes()
        with patch("src.api.import_portefeuille.preview_inventaire", return_value=_preview_result(total_lignes=10)):
            res = _upload("/import/inventaire/preview", "test_inventaire.csv", content)
        assert res.status_code == 200


# ─── /import/inventaire/commit ────────────────────────────────────────────────

class TestInventaireCommit:
    def test_returns_200(self):
        with patch("src.api.import_portefeuille.commit_inventaire", return_value=_commit_result()):
            res = _upload("/import/inventaire/commit", "inv.csv",
                          b"ref_affaire,date,code_isin,nbuc,vl\nAFF,2020-01-03,FR0000000001,100,10\n")
        assert res.status_code == 200

    def test_passes_id_client_to_service(self):
        captured = {}

        def fake_commit(db, raw_rows, id_societe_gestion=None, id_personne=None, run_pipeline=True):
            captured["id_personne"] = id_personne
            captured["id_societe_gestion"] = id_societe_gestion
            return _commit_result()

        with patch("src.api.import_portefeuille.commit_inventaire", side_effect=fake_commit):
            _upload(
                "/import/inventaire/commit?id_client=4016&id_societe_gestion=1",
                "inv.csv",
                b"ref_affaire,date,code_isin,nbuc,vl\nAFF,2020-01-03,FR0000000001,100,10\n",
            )
        assert captured["id_personne"] == 4016
        assert captured["id_societe_gestion"] == 1

    def test_empty_file_returns_400(self):
        with patch("src.api.import_portefeuille.commit_inventaire", return_value=_commit_result()):
            res = _upload("/import/inventaire/commit", "empty.csv", b"")
        assert res.status_code == 400

    def test_response_contains_alertes(self):
        alerte = ImportAlerte(code="conflict_date", message="Ligne écrasée")
        with patch("src.api.import_portefeuille.commit_inventaire",
                   return_value=_commit_result(insere=1, mis_a_jour=1, alertes=[alerte])):
            res = _upload("/import/inventaire/commit", "inv.csv",
                          b"ref_affaire,date,code_isin,nbuc,vl\nAFF,2020-01-03,FR0000000001,100,10\n")
        data = res.json()
        assert data["mis_a_jour"] == 1
        assert len(data["alertes"]) == 1
        assert data["alertes"][0]["code"] == "conflict_date"

    def test_affaires_creees_in_response(self):
        with patch("src.api.import_portefeuille.commit_inventaire",
                   return_value=_commit_result(affaires_creees=2)):
            res = _upload("/import/inventaire/commit", "inv.csv",
                          b"ref_affaire,date,code_isin,nbuc,vl\nAFF,2020-01-03,FR0000000001,100,10\n")
        assert res.json()["affaires_creees"] == 2


# ─── /import/mouvements/preview ───────────────────────────────────────────────

class TestMouvementsPreview:
    def test_returns_200(self):
        with patch("src.api.import_portefeuille.preview_mouvements", return_value=_preview_result()):
            res = _upload("/import/mouvements/preview", "mouv.csv",
                          b"ref_affaire,date,code_isin,code_mouvement,nbuc,vl\n")
        assert res.status_code == 200

    def test_no_file_returns_400(self):
        res = client.post("/import/mouvements/preview")
        assert res.status_code == 400

    def test_real_csv_file(self):
        content = (DATA_DIR / "test_mouvements.csv").read_bytes()
        with patch("src.api.import_portefeuille.preview_mouvements", return_value=_preview_result(total_lignes=13)):
            res = _upload("/import/mouvements/preview", "test_mouvements.csv", content)
        assert res.status_code == 200
        assert res.json()["total_lignes"] == 13


# ─── /import/mouvements/commit ────────────────────────────────────────────────

class TestMouvementsCommit:
    def test_returns_200(self):
        with patch("src.api.import_portefeuille.commit_mouvements", return_value=_commit_result()):
            res = _upload("/import/mouvements/commit", "mouv.csv",
                          b"ref_affaire,date,code_isin,code_mouvement,nbuc,vl\n"
                          b"AFF,2020-01-03,FR0000000001,VI,100,10\n")
        assert res.status_code == 200

    def test_passes_id_client_to_service(self):
        captured = {}

        def fake_commit(db, raw_rows, id_societe_gestion=None, id_personne=None, run_pipeline=True):
            captured["id_personne"] = id_personne
            return _commit_result()

        with patch("src.api.import_portefeuille.commit_mouvements", side_effect=fake_commit):
            _upload(
                "/import/mouvements/commit?id_client=4016",
                "mouv.csv",
                b"ref_affaire,date,code_isin,code_mouvement,nbuc,vl\n"
                b"AFF,2020-01-03,FR0000000001,VI,100,10\n",
            )
        assert captured["id_personne"] == 4016

    def test_empty_file_returns_400(self):
        with patch("src.api.import_portefeuille.commit_mouvements", return_value=_commit_result()):
            res = _upload("/import/mouvements/commit", "empty.csv", b"")
        assert res.status_code == 400

    def test_response_contains_avis_generes(self):
        with patch("src.api.import_portefeuille.commit_mouvements",
                   return_value=_commit_result(avis_generes=3)):
            res = _upload("/import/mouvements/commit", "mouv.csv",
                          b"ref_affaire,date,code_isin,code_mouvement,nbuc,vl\n"
                          b"AFF,2020-01-03,FR0000000001,VI,100,10\n")
        assert res.json()["avis_generes"] == 3

    def test_doublon_mouvement_alerte_in_response(self):
        alerte = ImportAlerte(code="doublon_mouvement", message="Mouvement ignoré")
        with patch("src.api.import_portefeuille.commit_mouvements",
                   return_value=_commit_result(insere=0, alertes=[alerte])):
            res = _upload("/import/mouvements/commit", "mouv.csv",
                          b"ref_affaire,date,code_isin,code_mouvement,nbuc,vl\n"
                          b"AFF,2020-01-03,FR0000000001,VI,100,10\n")
        data = res.json()
        assert data["insere"] == 0
        assert data["alertes"][0]["code"] == "doublon_mouvement"
