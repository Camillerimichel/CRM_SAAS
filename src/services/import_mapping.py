"""
Service de gestion des profils de mapping de fichiers d'import.

Permet de :
- Analyser un fichier entrant (CSV/XLSX) pour détecter ses colonnes et un échantillon
- Suggérer automatiquement la correspondance colonnes → champs attendus
- Appliquer un profil (mapping + transformations) pour normaliser les lignes
- Gérer (CRUD) les profils sauvegardés en base
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Champs attendus par type d'import
# ──────────────────────────────────────────────────────────────────────────────

CHAMPS_ATTENDUS: dict[str, dict[str, dict]] = {
    "inventaire": {
        "date":                  {"label": "Date", "requis": True,  "type": "date"},
        "code_isin":             {"label": "Code ISIN", "requis": True,  "type": "isin"},
        "nbuc":                  {"label": "Nombre de parts (UC)", "requis": True,  "type": "float"},
        "vl":                    {"label": "Valeur liquidative", "requis": True,  "type": "float"},
        "ref_affaire":           {"label": "Réf. contrat", "requis": False, "type": "str"},
        "id_client_fournisseur": {"label": "Réf. client externe", "requis": False, "type": "str"},
        "nom_support":           {"label": "Nom du support", "requis": False, "type": "str"},
    },
    "mouvements": {
        "date":                  {"label": "Date", "requis": True,  "type": "date"},
        "code_isin":             {"label": "Code ISIN", "requis": True,  "type": "isin"},
        "code_mouvement":        {"label": "Type de mouvement", "requis": True,  "type": "str"},
        "nbuc":                  {"label": "Nombre de parts (UC)", "requis": True,  "type": "float"},
        "vl":                    {"label": "Valeur liquidative", "requis": True,  "type": "float"},
        "ref_affaire":           {"label": "Réf. contrat", "requis": False, "type": "str"},
        "id_client_fournisseur": {"label": "Réf. client externe", "requis": False, "type": "str"},
        "montant_ope":           {"label": "Montant opération", "requis": False, "type": "float"},
        "frais":                 {"label": "Frais", "requis": False, "type": "float"},
    },
    "avis": {
        "date":                  {"label": "Date", "requis": True,  "type": "date"},
        "ref_affaire":           {"label": "Réf. contrat", "requis": False, "type": "str"},
        "id_client_fournisseur": {"label": "Réf. client externe", "requis": False, "type": "str"},
        "reference":             {"label": "Référence avis", "requis": False, "type": "str"},
        "entree":                {"label": "Description entrée", "requis": False, "type": "str"},
        "sortie":                {"label": "Description sortie", "requis": False, "type": "str"},
        "commentaire":           {"label": "Commentaire", "requis": False, "type": "str"},
    },
    "supports": {
        "code_isin": {"label": "Code ISIN", "requis": True,  "type": "isin"},
        "libelle":   {"label": "Libellé", "requis": True,  "type": "str"},
    },
    "clients": {
        "ref_cgp":   {"label": "Réf. CGP", "requis": True,  "type": "str"},
        "nom":       {"label": "Nom", "requis": True,  "type": "str"},
        "prenom":    {"label": "Prénom", "requis": True,  "type": "str"},
        "qualite":   {"label": "Qualité (M./Mme)", "requis": False, "type": "str"},
        "adresse":   {"label": "Adresse", "requis": False, "type": "str"},
        "cp":        {"label": "Code postal", "requis": False, "type": "str"},
        "ville":     {"label": "Ville", "requis": False, "type": "str"},
    },
}

# Mots-clés pour la suggestion automatique (champ_cible → mots-clés du nom de colonne)
_MOTS_CLES: dict[str, list[str]] = {
    "date": ["date", "jour", "day", "date_ope", "date op", "date val", "date arrete"],
    "code_isin": ["isin", "code isin", "code_isin", "valeur"],
    "nbuc": ["nbuc", "nb uc", "nb_uc", "nombre", "parts", "quantit", "qty", "qté", "unités", "uc"],
    "vl": ["vl", "val liquid", "valeur liquid", "nav", "prix", "price", "cours"],
    "ref_affaire": ["ref", "contrat", "n° contrat", "num contrat", "numéro", "id contrat", "ref_affaire", "compte", "n°contrat", "numcontrat"],
    "id_client_fournisseur": ["client", "id client", "id_client", "ref client", "client externe", "cli", "tit", "titulaire"],
    "nom_support": ["nom support", "libelle", "libellé", "désignation", "label", "appellation", "denomination", "fonds"],
    "code_mouvement": ["mouvement", "type", "nature", "opération", "operation", "sens", "code ope", "code opé", "type ope"],
    "montant_ope": ["montant", "amount", "total", "brut", "net"],
    "frais": ["frais", "commission", "fee", "charges"],
    "reference": ["reference", "réference", "ref avis", "n° avis", "num avis"],
    "entree": ["entree", "entrée", "souscription", "versement", "achat"],
    "sortie": ["sortie", "rachat", "vente", "retrait"],
    "commentaire": ["commentaire", "note", "observation", "remarque"],
    "libelle": ["libelle", "libellé", "nom", "désignation", "label", "fonds"],
    "ref_cgp": ["ref cgp", "ref_cgp", "cgp", "conseiller", "numéro cgp", "id cgp", "code cgp"],
    "nom": ["nom", "name", "famille"],
    "prenom": ["prenom", "prénom", "first", "firstname"],
    "qualite": ["qualite", "qualité", "civilité", "civility", "genre", "title", "titre"],
    "adresse": ["adresse", "address", "rue"],
    "cp": ["cp", "code postal", "postal", "zip", "codepostal"],
    "ville": ["ville", "city", "commune"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Cache session — rows parsées en mémoire (TTL 1h)
# ──────────────────────────────────────────────────────────────────────────────

_session_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 3600


def _cache_rows(rows: list[dict]) -> str:
    session_id = str(uuid.uuid4())
    _session_cache[session_id] = (time.time(), rows)
    expired = [k for k, (ts, _) in list(_session_cache.items()) if time.time() - ts > _CACHE_TTL]
    for k in expired:
        del _session_cache[k]
    return session_id


def get_cached_rows(session_id: str) -> list[dict] | None:
    entry = _session_cache.get(session_id)
    if not entry:
        return None
    ts, rows = entry
    if time.time() - ts > _CACHE_TTL:
        del _session_cache[session_id]
        return None
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Analyse de fichier
# ──────────────────────────────────────────────────────────────────────────────

def analyser_fichier(data: bytes, filename: str) -> dict:
    """
    Parse un fichier CSV ou XLSX, détecte ses colonnes et renvoie un échantillon.

    Retourne : {colonnes, echantillon, nb_lignes, session_id}
    """
    fname = (filename or "").lower()
    rows: list[dict] = []

    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        rows = _parse_excel(data)
    else:
        rows = _parse_csv_flexible(data)

    if not rows:
        return {"colonnes": [], "echantillon": [], "nb_lignes": 0, "session_id": None}

    colonnes = list(rows[0].keys())
    echantillon = [dict(r) for r in rows[:10]]
    session_id = _cache_rows(rows)

    return {
        "colonnes": colonnes,
        "echantillon": echantillon,
        "nb_lignes": len(rows),
        "session_id": session_id,
    }


def _parse_csv_flexible(data: bytes) -> list[dict]:
    text_data = ""
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text_data = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text_data = data.decode("utf-8", errors="replace")

    for sep in (";", ",", "\t", "|"):
        try:
            reader = csv.DictReader(io.StringIO(text_data), delimiter=sep)
            rows = list(reader)
            if rows and len(rows[0]) > 1:
                # Strip whitespace from keys and values
                return [
                    {k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items() if k is not None}
                    for row in rows
                ]
        except Exception:
            continue

    # Single-column fallback
    reader = csv.DictReader(io.StringIO(text_data))
    return list(reader)


def _parse_excel(data: bytes) -> list[dict]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        rows_raw = list(ws.iter_rows(values_only=True))
        if not rows_raw:
            return []
        headers = [
            str(c).strip() if c is not None else f"col_{i}"
            for i, c in enumerate(rows_raw[0])
        ]
        result = []
        for row in rows_raw[1:]:
            d = {
                headers[i]: (str(v).strip() if v is not None else "")
                for i, v in enumerate(row)
                if i < len(headers)
            }
            if any(v for v in d.values()):
                result.append(d)
        return result
    except ImportError:
        logger.warning("openpyxl non installé — fichiers XLSX non supportés")
        return []
    except Exception as exc:
        logger.error("Erreur lecture Excel : %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Suggestion automatique du mapping
# ──────────────────────────────────────────────────────────────────────────────

def suggerer_mapping(colonnes: list[str], type_import: str) -> dict[str, str | None]:
    """Associe automatiquement les champs attendus aux colonnes détectées."""
    champs = CHAMPS_ATTENDUS.get(type_import, {})
    suggestions: dict[str, str | None] = {c: None for c in champs}

    col_norm = {c: c.lower().strip() for c in colonnes}

    for champ_cible in champs:
        mots = _MOTS_CLES.get(champ_cible, [champ_cible])
        best_col: str | None = None
        best_score = 0

        for col, cn in col_norm.items():
            for mot in mots:
                mot_n = mot.lower()
                if mot_n == cn:
                    score = 100
                elif mot_n in cn or cn in mot_n:
                    score = 50 + len(min(mot_n, cn, key=len))
                else:
                    continue
                if score > best_score:
                    best_score = score
                    best_col = col

        if best_score > 0:
            suggestions[champ_cible] = best_col

    return suggestions


# ──────────────────────────────────────────────────────────────────────────────
# Application du profil / transformations
# ──────────────────────────────────────────────────────────────────────────────

def appliquer_profil(
    rows: list[dict],
    mapping: dict,
    transformations: dict | None = None,
    valeurs_fixes: dict | None = None,
) -> list[dict]:
    """Transforme les lignes brutes en utilisant le profil de mapping."""
    transformations = transformations or {}
    valeurs_fixes = valeurs_fixes or {}
    result = []

    for row in rows:
        row_norm = {k.lower().strip(): v for k, v in row.items()}
        new_row: dict[str, Any] = {}

        for champ_cible, champ_source in mapping.items():
            if not champ_source:
                continue
            val = row.get(champ_source)
            if val is None:
                val = row_norm.get(champ_source.lower().strip())
            if val is None:
                continue
            if champ_cible in transformations:
                val = _appliquer_transformation(val, transformations[champ_cible])
            if val is not None and str(val).strip():
                new_row[champ_cible] = str(val).strip()

        for champ_cible, valeur in valeurs_fixes.items():
            if not new_row.get(champ_cible) and valeur is not None and str(valeur).strip():
                new_row[champ_cible] = str(valeur).strip()

        result.append(new_row)

    return result


def _appliquer_transformation(val: Any, transfo: dict) -> Any:
    t = transfo.get("type", "")
    val_str = str(val).strip() if val is not None else ""

    if t == "date":
        return _normaliser_date(val_str, transfo.get("format", ""))
    elif t == "float_virgule":
        return val_str.replace("\xa0", "").replace(" ", "").replace(",", ".")
    elif t == "float_point":
        return val_str.replace("\xa0", "").replace(" ", "")
    elif t == "uppercase":
        return val_str.upper()
    elif t == "valeur_map":
        vm = transfo.get("mapping", {})
        return vm.get(val_str, vm.get(val_str.upper(), val))
    return val


def _normaliser_date(val: str, fmt: str | None = None) -> str:
    if not val:
        return val
    val = val.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        return val
    if fmt:
        py_fmt = (
            fmt.replace("YYYY", "%Y")
               .replace("YY",   "%y")
               .replace("MM",   "%m")
               .replace("DD",   "%d")
               .replace("M",    "%m")
               .replace("D",    "%d")
        )
        try:
            return datetime.strptime(val, py_fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    for try_fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y%m%d", "%d%m%Y", "%d-%m-%Y", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(val, try_fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return val


# ──────────────────────────────────────────────────────────────────────────────
# CRUD profils (table import_mapping_profil)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_table(db: Session) -> None:
    db.execute(text("""
        CREATE TABLE IF NOT EXISTS import_mapping_profil (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            nom           VARCHAR(255) NOT NULL,
            fournisseur   VARCHAR(255),
            type_import   VARCHAR(50)  NOT NULL,
            mapping       JSON         NOT NULL,
            transformations JSON,
            valeurs_fixes JSON,
            actif         TINYINT(1)   NOT NULL DEFAULT 1,
            cree_quand    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            modif_quand   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_type_fournisseur (type_import, fournisseur)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """))
    db.commit()


def _row_to_dict(row) -> dict:
    m = row._mapping if hasattr(row, "_mapping") else None
    d = dict(m) if m else dict(row)
    for jf in ("mapping", "transformations", "valeurs_fixes"):
        if isinstance(d.get(jf), str):
            try:
                d[jf] = json.loads(d[jf])
            except Exception:
                d[jf] = {}
        elif d.get(jf) is None:
            d[jf] = {}
    # Serialize dates for JSON
    for df in ("cree_quand", "modif_quand"):
        if hasattr(d.get(df), "isoformat"):
            d[df] = d[df].isoformat()
    return d


def lister_profils(
    db: Session,
    type_import: str | None = None,
    fournisseur: str | None = None,
) -> list[dict]:
    _ensure_table(db)
    params: dict = {}
    where = ["actif = 1"]
    if type_import:
        where.append("type_import = :type_import")
        params["type_import"] = type_import
    if fournisseur:
        where.append("fournisseur = :fournisseur")
        params["fournisseur"] = fournisseur
    rows = db.execute(
        text(f"SELECT * FROM import_mapping_profil WHERE {' AND '.join(where)} ORDER BY modif_quand DESC"),
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in (rows or [])]


def creer_profil(
    db: Session,
    nom: str,
    type_import: str,
    mapping: dict,
    fournisseur: str | None = None,
    transformations: dict | None = None,
    valeurs_fixes: dict | None = None,
) -> dict:
    _ensure_table(db)
    db.execute(
        text("""
            INSERT INTO import_mapping_profil
                (nom, fournisseur, type_import, mapping, transformations, valeurs_fixes)
            VALUES
                (:nom, :fournisseur, :type_import, :mapping, :transformations, :valeurs_fixes)
        """),
        {
            "nom": nom,
            "fournisseur": fournisseur,
            "type_import": type_import,
            "mapping": json.dumps(mapping, ensure_ascii=False),
            "transformations": json.dumps(transformations or {}, ensure_ascii=False),
            "valeurs_fixes": json.dumps(valeurs_fixes or {}, ensure_ascii=False),
        },
    )
    db.commit()
    row = db.execute(
        text("SELECT * FROM import_mapping_profil ORDER BY id DESC LIMIT 1")
    ).fetchone()
    return _row_to_dict(row) if row else {}


def maj_profil(
    db: Session,
    profil_id: int,
    nom: str,
    type_import: str,
    mapping: dict,
    fournisseur: str | None = None,
    transformations: dict | None = None,
    valeurs_fixes: dict | None = None,
) -> dict | None:
    _ensure_table(db)
    db.execute(
        text("""
            UPDATE import_mapping_profil
            SET nom = :nom,
                fournisseur = :fournisseur,
                type_import = :type_import,
                mapping = :mapping,
                transformations = :transformations,
                valeurs_fixes = :valeurs_fixes,
                modif_quand = NOW()
            WHERE id = :id
        """),
        {
            "id": profil_id,
            "nom": nom,
            "fournisseur": fournisseur,
            "type_import": type_import,
            "mapping": json.dumps(mapping, ensure_ascii=False),
            "transformations": json.dumps(transformations or {}, ensure_ascii=False),
            "valeurs_fixes": json.dumps(valeurs_fixes or {}, ensure_ascii=False),
        },
    )
    db.commit()
    row = db.execute(
        text("SELECT * FROM import_mapping_profil WHERE id = :id"),
        {"id": profil_id},
    ).fetchone()
    return _row_to_dict(row) if row else None


def supprimer_profil(db: Session, profil_id: int) -> bool:
    _ensure_table(db)
    result = db.execute(
        text("UPDATE import_mapping_profil SET actif = 0 WHERE id = :id"),
        {"id": profil_id},
    )
    db.commit()
    return bool(result.rowcount)
