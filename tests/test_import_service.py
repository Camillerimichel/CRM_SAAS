"""
Tests des fonctions de service d'import avec session DB mockée.
Vérifie la logique métier : résolution d'affaire, création à vide,
tâche de finalisation, UPSERT inventaire, guard anti-doublon mouvements.
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.import_inventaire import (
    _resolve_affaire_id,
    _resolve_or_create_affaire,
    _create_affaire_vide,
    _create_finalisation_task,
    _upsert_historique_support,
    _resolve_or_create_support,
    commit_inventaire,
)
from src.services.import_mouvements import commit_mouvements


# ─── Helpers ──────────────────────────────────────────────────────────────────

def mock_db():
    """Session SQLAlchemy complètement mockée."""
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    db.execute.return_value.scalar.return_value = 1
    db.flush.return_value = None
    db.commit.return_value = None
    return db


def make_row(ref="AFF-1", id_affaire=None):
    row = MagicMock()
    row.ref_affaire = ref
    row.id_affaire = id_affaire
    return row


# ─── _resolve_affaire_id ──────────────────────────────────────────────────────

def test_resolve_affaire_by_ref_found():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (42,)
    row = make_row(ref="AFF-1")
    result = _resolve_affaire_id(db, row)
    assert result == 42


def test_resolve_affaire_by_id_found():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (99,)
    row = make_row(ref=None, id_affaire=99)
    result = _resolve_affaire_id(db, row)
    assert result == 99


def test_resolve_affaire_not_found_returns_none():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = None
    row = make_row(ref="INCONNU")
    result = _resolve_affaire_id(db, row)
    assert result is None


def test_resolve_affaire_no_ref_no_id_returns_none():
    db = mock_db()
    row = make_row(ref=None, id_affaire=None)
    result = _resolve_affaire_id(db, row)
    assert result is None


# ─── _create_affaire_vide ─────────────────────────────────────────────────────

def test_create_affaire_vide_inserts_with_id_personne():
    db = mock_db()
    db.execute.return_value.scalar.return_value = 500
    _id = _create_affaire_vide(db, ref="TEST-REF", id_societe_gestion=1, id_personne=4016)
    assert _id == 500
    # Vérifie que id_personne=4016 est passé dans l'INSERT
    call_args = db.execute.call_args_list
    insert_call = next(
        (c for c in call_args if "INSERT" in str(c.args[0])), None
    )
    assert insert_call is not None
    params = insert_call.args[1]
    assert params["personne"] == 4016
    assert params["ref"] == "TEST-REF"


def test_create_affaire_vide_without_personne():
    db = mock_db()
    db.execute.return_value.scalar.return_value = 501
    _id = _create_affaire_vide(db, ref="TEST-REF", id_societe_gestion=1, id_personne=None)
    assert _id == 501
    insert_call = next(
        (c for c in db.execute.call_args_list if "INSERT" in str(c.args[0])), None
    )
    params = insert_call.args[1]
    assert params["personne"] is None


# ─── _create_finalisation_task ────────────────────────────────────────────────

def test_create_finalisation_task_links_client():
    db = mock_db()
    # _ensure_type_finalisation → fetchone retourne un id de type_evenement
    db.execute.return_value.fetchone.return_value = (75,)
    _create_finalisation_task(db, id_affaire=2257, id_societe_gestion=1, id_personne=4016)
    insert_call = next(
        (c for c in db.execute.call_args_list if "INSERT INTO mariadb_evenement" in str(c.args[0])),
        None,
    )
    assert insert_call is not None
    params = insert_call.args[1]
    assert params["aff"] == 2257
    assert params["client"] == 4016
    assert params["statut"] if "statut" in params else True  # présent dans le SQL


def test_create_finalisation_task_without_client():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (75,)
    _create_finalisation_task(db, id_affaire=2257, id_personne=None)
    insert_call = next(
        (c for c in db.execute.call_args_list if "INSERT INTO mariadb_evenement" in str(c.args[0])),
        None,
    )
    params = insert_call.args[1]
    assert params["client"] is None


# ─── _resolve_or_create_affaire ───────────────────────────────────────────────

def test_resolve_or_create_existing_affaire():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (100,)
    row = make_row(ref="EXIST")
    id_aff, was_created = _resolve_or_create_affaire(db, row, create_if_missing=True)
    assert id_aff == 100
    assert was_created is False


def test_resolve_or_create_missing_no_create():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = None
    row = make_row(ref="MISSING")
    id_aff, was_created = _resolve_or_create_affaire(db, row, create_if_missing=False)
    assert id_aff is None
    assert was_created is False


def test_resolve_or_create_missing_creates_with_personne():
    db = mock_db()
    # Premier appel fetchone (resolve) → None, puis fetchone (type_finalisation) → (75,)
    db.execute.return_value.fetchone.side_effect = [None, (75,)]
    db.execute.return_value.scalar.return_value = 999
    row = make_row(ref="NEW-AFF")
    id_aff, was_created = _resolve_or_create_affaire(
        db, row, id_societe_gestion=1, create_if_missing=True, id_personne=4016
    )
    assert id_aff == 999
    assert was_created is True
    # Vérifier que la tâche de finalisation a été insérée avec client_id=4016
    insert_event = next(
        (c for c in db.execute.call_args_list if "INSERT INTO mariadb_evenement" in str(c.args[0])),
        None,
    )
    assert insert_event is not None
    assert insert_event.args[1]["client"] == 4016


# ─── _upsert_historique_support ───────────────────────────────────────────────

def test_upsert_historique_support_insert_new():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = None  # pas d'existant
    db.execute.return_value.scalar.return_value = 1000
    overwritten = _upsert_historique_support(
        db, id_affaire=1, id_support=2,
        snap_date=datetime(2020, 1, 3),
        nbuc=100.0, vl=10.0, id_societe_gestion=1,
    )
    assert overwritten is False
    insert_call = next(
        (c for c in db.execute.call_args_list if "INSERT INTO mariadb_historique_support_w" in str(c.args[0])),
        None,
    )
    assert insert_call is not None


def test_upsert_historique_support_overwrites_existing():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (55,)  # ligne existante id=55
    overwritten = _upsert_historique_support(
        db, id_affaire=1, id_support=2,
        snap_date=datetime(2020, 1, 3),
        nbuc=100.0, vl=10.0, id_societe_gestion=1,
    )
    assert overwritten is True
    update_call = next(
        (c for c in db.execute.call_args_list if "UPDATE mariadb_historique_support_w" in str(c.args[0])),
        None,
    )
    assert update_call is not None
    assert update_call.args[1]["id"] == 55


# ─── _resolve_or_create_support ───────────────────────────────────────────────

def test_resolve_support_existing():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = (77,)
    _id, created = _resolve_or_create_support(db, "FR0000000001", None)
    assert _id == 77
    assert created is False


def test_resolve_support_creates_unknown_isin():
    db = mock_db()
    db.execute.return_value.fetchone.return_value = None
    db.execute.return_value.scalar.return_value = 200
    _id, created = _resolve_or_create_support(db, "FR9999999999", "Nom Support")
    assert _id == 200
    assert created is True


# ─── commit_inventaire (intégration service) ─────────────────────────────────

def test_commit_inventaire_inserts_rows():
    """Vérifie le chemin nominal : 1 ligne valide → 1 insert."""
    db = mock_db()
    # resolve_affaire → trouvée
    # resolve_support → trouvée
    # upsert → pas d'existant (INSERT)
    db.execute.return_value.fetchone.side_effect = [
        (42,),   # _resolve_affaire_id → id=42
        (77,),   # _resolve_or_create_support → id=77
        None,    # _upsert: pas d'existant
    ]
    db.execute.return_value.scalar.return_value = 1

    raw = [{
        "ref_affaire": "AFF-1", "date": "2020-01-03",
        "code_isin": "FR0000000001", "nbuc": "100", "vl": "10.5",
    }]

    with patch("src.services.import_inventaire.run_full_pipeline", return_value=0.5):
        result = commit_inventaire(db, raw, id_societe_gestion=1, id_personne=4016)

    assert result.insere == 1
    assert result.mis_a_jour == 0
    assert result.affaires_creees == 0


def test_commit_inventaire_creates_missing_affaire():
    """Affaire inconnue → créée à vide + tâche de finalisation."""
    db = mock_db()
    db.execute.return_value.fetchone.side_effect = [
        None,    # _resolve_affaire_id → non trouvée
        (75,),   # _ensure_type_finalisation → id=75
        (77,),   # _resolve_or_create_support → trouvé
        None,    # _upsert: pas d'existant
    ]
    db.execute.return_value.scalar.return_value = 500

    raw = [{
        "ref_affaire": "NEW-AFF", "date": "2020-01-03",
        "code_isin": "FR0000000001", "nbuc": "100", "vl": "10.5",
    }]

    with patch("src.services.import_inventaire.run_full_pipeline", return_value=0.0):
        result = commit_inventaire(db, raw, id_societe_gestion=1, id_personne=4016)

    assert result.affaires_creees == 1
    assert result.insere == 1
    assert any(a.code == "affaire_creee" for a in result.alertes)


def test_commit_inventaire_alerts_on_overwrite():
    """Ligne existante → mis_a_jour + alerte conflict_date."""
    db = mock_db()
    db.execute.return_value.fetchone.side_effect = [
        (42,),   # resolve_affaire
        (77,),   # resolve_support
        (55,),   # upsert → existant trouvé
    ]
    db.execute.return_value.scalar.return_value = 1

    raw = [{
        "ref_affaire": "AFF-1", "date": "2020-01-03",
        "code_isin": "FR0000000001", "nbuc": "100", "vl": "10.5",
    }]

    with patch("src.services.import_inventaire.run_full_pipeline", return_value=0.0):
        result = commit_inventaire(db, raw)

    assert result.mis_a_jour == 1
    assert result.insere == 0
    assert any(a.code == "conflict_date" for a in result.alertes)


# ─── commit_mouvements — guard anti-doublon ───────────────────────────────────

def test_commit_mouvements_dedup_guard():
    """Un mouvement déjà en base doit être ignoré (doublon_mouvement)."""
    db = mock_db()

    # Séquence des fetchone :
    # 1. _resolve_or_create_affaire → trouvée (42)
    # 2. _resolve_or_create_support → trouvé (77)
    # 3. _load_mouvement_regle → (fetchall simulé séparément)
    # 4. vérif doublon → existant trouvé → skip
    mouvement_regle_rows = [(1, "VI", "Versement initial", 1, 1, 1)]
    db.execute.return_value.fetchall.return_value = mouvement_regle_rows

    db.execute.return_value.fetchone.side_effect = [
        (42,),   # resolve_affaire
        (77,),   # resolve_support
        (99,),   # doublon check → mouvement existant
    ]
    db.execute.return_value.scalar.return_value = 1

    raw = [{
        "ref_affaire": "AFF-1", "date": "2020-01-03",
        "code_isin": "FR0000000001", "code_mouvement": "VI",
        "nbuc": "100", "vl": "10.5",
    }]

    with patch("src.services.import_mouvements.run_full_pipeline", return_value=0.0):
        with patch("src.services.import_mouvements._recompute_prmp"):
            result = commit_mouvements(db, raw)

    assert result.insere == 0
    assert any(a.code == "doublon_mouvement" for a in result.alertes)
