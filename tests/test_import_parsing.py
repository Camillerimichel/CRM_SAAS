"""
Tests unitaires pour le parsing et la validation des fichiers d'import.
Aucune dépendance DB — fonctions pures uniquement.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.import_inventaire import (
    parse_inventaire_csv,
    parse_inventaire_json,
    _parse_date,
    _validate_rows as _validate_inv,
)
from src.services.import_mouvements import (
    parse_mouvements_csv,
    parse_mouvements_json,
    _validate_rows as _validate_mouv,
)

DATA_DIR = Path(__file__).parent / "import_data"

# ─── _parse_date ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,expected_year", [
    ("2020-01-03", 2020),
    ("03/01/2020", 2020),
    ("03-01-2020", 2020),
    ("20200103", 2020),
])
def test_parse_date_valid_formats(val, expected_year):
    result = _parse_date(val)
    assert result is not None
    assert result.year == expected_year


def test_parse_date_invalid_returns_none():
    assert _parse_date("not-a-date") is None
    assert _parse_date("") is None
    assert _parse_date("99/99/9999") is None


# ─── parse_inventaire_csv ─────────────────────────────────────────────────────

def test_inventaire_csv_comma():
    csv = "ref_affaire,date,code_isin,nbuc,vl\nAFF-1,2020-01-03,FR0000000001,100.0,10.5\n"
    rows = parse_inventaire_csv(csv)
    assert len(rows) == 1
    assert rows[0]["code_isin"] == "FR0000000001"


def test_inventaire_csv_semicolon():
    csv = "ref_affaire;date;code_isin;nbuc;vl\nAFF-1;2020-01-03;FR0000000001;100.0;10.5\n"
    rows = parse_inventaire_csv(csv)
    assert len(rows) == 1
    assert rows[0]["ref_affaire"] == "AFF-1"


def test_inventaire_csv_bytes_with_bom():
    content = b"\xef\xbb\xbfref_affaire,date,code_isin,nbuc,vl\nAFF-1,2020-01-03,FR0000000001,100,10\n"
    rows = parse_inventaire_csv(content)
    assert len(rows) == 1


def test_inventaire_csv_uppercase_headers():
    csv = "REF_AFFAIRE,DATE,CODE_ISIN,NBUC,VL\nAFF-1,2020-01-03,FR0000000001,100,10\n"
    rows = parse_inventaire_csv(csv)
    rows_validated, alertes = _validate_inv(rows)
    assert len(alertes) == 0
    assert len(rows_validated) == 1


def test_real_inventaire_csv():
    content = (DATA_DIR / "test_inventaire.csv").read_bytes()
    rows = parse_inventaire_csv(content)
    assert len(rows) > 0
    validated, alertes = _validate_inv(rows)
    parse_errors = [a for a in alertes if a.code == "parse_error"]
    assert len(parse_errors) == 0


# ─── parse_inventaire_json ────────────────────────────────────────────────────

def test_inventaire_json_list():
    import json
    data = [{"ref_affaire": "AFF-1", "date": "2020-01-03", "code_isin": "FR0000000001", "nbuc": 100, "vl": 10}]
    rows = parse_inventaire_json(json.dumps(data))
    assert len(rows) == 1


def test_inventaire_json_with_rows_key():
    import json
    data = {"rows": [{"ref_affaire": "AFF-1", "date": "2020-01-03", "code_isin": "FR0000000001", "nbuc": 100, "vl": 10}]}
    rows = parse_inventaire_json(json.dumps(data))
    assert len(rows) == 1


def test_inventaire_json_with_inventaire_key():
    import json
    data = {"inventaire": [{"ref_affaire": "AFF-1", "date": "2020-01-03", "code_isin": "FR0000000001", "nbuc": 100, "vl": 10}]}
    rows = parse_inventaire_json(json.dumps(data))
    assert len(rows) == 1


def test_inventaire_json_invalid_raises():
    import json
    with pytest.raises(ValueError):
        parse_inventaire_json(json.dumps({"bad_key": []}))


# ─── _validate_rows inventaire ────────────────────────────────────────────────

def _inv_row(**kwargs):
    base = {"ref_affaire": "AFF-1", "date": "2020-01-03", "code_isin": "fr0000000001", "nbuc": "100", "vl": "10.5"}
    base.update(kwargs)
    return base


def test_validate_inv_valid():
    rows, alertes = _validate_inv([_inv_row()])
    assert len(rows) == 1
    assert len(alertes) == 0
    assert rows[0].code_isin == "FR0000000001"  # uppercase normalisé


def test_validate_inv_missing_both_affaire_ids():
    row = _inv_row()
    del row["ref_affaire"]
    rows, alertes = _validate_inv([row])
    assert len(rows) == 0
    assert any(a.code == "missing_affaire" for a in alertes)


def test_validate_inv_uses_id_affaire_when_no_ref():
    row = _inv_row()
    del row["ref_affaire"]
    row["id_affaire"] = "42"
    rows, alertes = _validate_inv([row])
    assert len(rows) == 1
    assert rows[0].id_affaire == 42


def test_validate_inv_missing_isin():
    rows, alertes = _validate_inv([_inv_row(code_isin="")])
    assert any(a.code == "missing_isin" for a in alertes)


def test_validate_inv_invalid_date():
    rows, alertes = _validate_inv([_inv_row(date="not-a-date")])
    assert any(a.code == "invalid_date" for a in alertes)


# ─── parse_mouvements_csv ─────────────────────────────────────────────────────

def test_mouvements_csv_comma():
    csv = "ref_affaire,date,code_isin,code_mouvement,nbuc,vl\nAFF-1,2020-01-03,FR0000000001,VI,100,10\n"
    rows = parse_mouvements_csv(csv)
    assert len(rows) == 1
    assert rows[0]["code_mouvement"] == "VI"


def test_real_mouvements_csv():
    content = (DATA_DIR / "test_mouvements.csv").read_bytes()
    rows = parse_mouvements_csv(content)
    assert len(rows) > 0
    validated, alertes = _validate_mouv(rows)
    parse_errors = [a for a in alertes if a.code == "parse_error"]
    assert len(parse_errors) == 0


# ─── _validate_rows mouvements ────────────────────────────────────────────────

def _mouv_row(**kwargs):
    base = {
        "ref_affaire": "AFF-1", "date": "2020-01-03",
        "code_isin": "FR0000000001", "code_mouvement": "vi",
        "nbuc": "100", "vl": "10.5",
    }
    base.update(kwargs)
    return base


def test_validate_mouv_valid():
    rows, alertes = _validate_mouv([_mouv_row()])
    assert len(rows) == 1
    assert rows[0].code_mouvement == "VI"  # uppercase normalisé


def test_validate_mouv_missing_code_mouvement():
    rows, alertes = _validate_mouv([_mouv_row(code_mouvement="")])
    assert any(a.code == "missing_code_mouvement" for a in alertes)


def test_validate_mouv_missing_affaire():
    row = _mouv_row()
    del row["ref_affaire"]
    rows, alertes = _validate_mouv([row])
    assert any(a.code == "missing_affaire" for a in alertes)


def test_validate_mouv_invalid_date():
    rows, alertes = _validate_mouv([_mouv_row(date="32/13/2020")])
    assert any(a.code == "invalid_date" for a in alertes)


def test_validate_mouv_optional_frais_and_montant():
    rows, alertes = _validate_mouv([_mouv_row(frais="12.50", montant_ope="500.0")])
    assert len(rows) == 1
    assert rows[0].frais == 12.50
    assert rows[0].montant_ope == 500.0


def test_validate_mouv_multiple_rows_partial_invalid():
    raw = [
        _mouv_row(),
        _mouv_row(date="bad"),
        _mouv_row(code_isin="FR9999999999"),
    ]
    rows, alertes = _validate_mouv(raw)
    assert len(rows) == 2
    assert len(alertes) == 1
    assert alertes[0].ligne == 2
