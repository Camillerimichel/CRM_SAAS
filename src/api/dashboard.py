import logging
import base64
import re
import secrets
import math
from time import perf_counter
from pathlib import Path

from fastapi import APIRouter, Request, Depends, Query, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, bindparam, desc
from sqlalchemy import text
from datetime import datetime, date as _date, timedelta, timezone
from decimal import Decimal, InvalidOperation
from collections import defaultdict
import csv
import io
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl, quote
import ssl
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin


from src.database import get_db
from src.api.main import templates
from src.services.clients import get_clients
from src.services.evenements import (
    create_tache,
    list_statuts,
    add_statut_to_evenement,
    create_envoi,
)
from src.schemas.evenement import TacheCreateSchema
from src.schemas.evenement_statut import EvenementStatutCreateSchema
from src.schemas.evenement_envoi import EvenementEnvoiCreateSchema
from starlette.responses import RedirectResponse

# ---------------- Imports Models ----------------
from src.models.client import Client
from src.models.affaire import Affaire
from src.models.support import Support
from src.models.allocation import Allocation
from src.models.document import Document
from src.models.document_client import DocumentClient
from src.models.historique_personne import HistoriquePersonne
from src.models.historique_affaire import HistoriqueAffaire
from src.models.historique_support import HistoriqueSupport
from src.models.evenement import Evenement
from src.models.type_evenement import TypeEvenement
from src.models.evenement_statut_reclamation import EvenementStatutReclamation
from src.models.administration_groupe_detail import AdministrationGroupeDetail
from src.models.administration_groupe import AdministrationGroupe


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = BASE_DIR / "documents"
VEILLE_FILE = BASE_DIR / "data" / "veille_reglementaire.json"
VEILLE_CACHE = {"ts": None, "items": []}


def rows_to_dicts(rows):
    result = []
    for row in rows:
        data = dict(row._mapping)
        if "prenom" in data and "nom" in data and "full_name" not in data:
            prenom = (data.get("prenom") or "").strip()
            nom = (data.get("nom") or "").strip()
            full_name = (prenom + " " + nom).strip()
            if not full_name:
                full_name = str(data.get("id", "")).strip()
            data["full_name"] = full_name
        result.append(data)
    return result

# ---------------- Veille distante (RSS/HTML fallback) ----------------
KEYWORDS_VEILLE = [
    "priips",
    "mifid",
    "dda",
    "idd",
    "esg",
    "sfdr",
    "amf",
    "acpr",
    "assurance-vie",
    "assurance vie",
    "assurance",
    "courtage",
    "courtier",
    "cif",
    "conseiller en investissements financiers",
    "jurisprudence",
    "sanction",
    "directive",
    "règlement",
]

FINANCE_CACHE: dict = {}
HOME_CACHE: dict = {}
ESG_FIELDS_CACHE: dict = {}


def _matches_keywords(text: str) -> bool:
    content = text.lower()
    return any(k in content for k in KEYWORDS_VEILLE)


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[dict] = []
        self._href = None
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                self._href = href
                self._parts = []

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href:
            txt = "".join(self._parts).strip()
            self.links.append({"href": self._href, "text": txt})
            self._href = None
            self._parts = []

    def handle_data(self, data):
        if self._href:
            self._parts.append(data)


def _parse_html_links(html_text: str, base_url: str):
    parser = _LinkParser()
    try:
        parser.feed(html_text)
    except Exception:
        return []
    items = []
    for link in parser.links:
        href = link.get("href", "")
        text = (link.get("text") or "").strip()
        if not href or not text:
            continue
        full = urljoin(base_url, href)
        items.append({"title": text, "link": full, "date": "", "summary": ""})
    return items


def _parse_rss(xml_text: str):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    for item in root.findall(".//item"):
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "date": (item.findtext("pubDate") or "").strip(),
                "summary": (item.findtext("description") or "").strip(),
            }
        )
    for entry in root.findall(".//atom:entry", ns):
        items.append(
            {
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "link": (entry.find("atom:link", ns).attrib.get("href", "") if entry.find("atom:link", ns) is not None else "").strip(),
                "date": (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip(),
                "summary": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
            }
        )
    return items


def _fetch_url(url: str, insecure: bool = False, proxy: str | None = None, timeout: int = 12) -> str | None:
    import urllib.request
    ctx = ssl._create_unverified_context() if insecure else None
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    if ctx:
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request.build_opener()
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (VeilleReglementaire)")
    try:
        with opener.open(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except Exception:
        return None


def _collect_veille(insecure: bool = True, proxy: str | None = None) -> list[dict]:
    sources = [
        # Régulateurs
        {"name": "AMF", "url": "https://www.amf-france.org/fr/actualites", "html": ["https://www.amf-france.org/fr/espace-presse/communiques-de-presse", "https://www.amf-france.org/fr"]},
        {"name": "ACPR", "url": "https://acpr.banque-france.fr/rss.xml", "html": ["https://acpr.banque-france.fr/communiques-de-presse", "https://acpr.banque-france.fr/actualites"]},
        {"name": "ESMA", "url": "https://www.esma.europa.eu/whats-new", "html": ["https://www.esma.europa.eu/press-news/esma-news", "https://www.esma.europa.eu/press-news/press-releases"]},
        {"name": "EIOPA", "url": "https://www.eiopa.europa.eu/news-events/news", "html": ["https://www.eiopa.europa.eu/media/speeches"]},
        {"name": "EBA", "url": "https://www.eba.europa.eu/press-and-media/news", "html": []},
        {"name": "CNIL", "url": "https://www.cnil.fr/fr/rss.xml", "html": ["https://www.cnil.fr/fr/actualites"]},
        {"name": "CJUE", "url": "https://curia.europa.eu/jcms/jcms/Jo2_7026/rss", "html": []},
        # Presse
        {"name": "Agefi Actifs", "url": "https://www.agefiactifs.com/", "html": []},
        {"name": "Revue Banque", "url": "https://www.revue-banque.fr/", "html": ["https://www.revue-banque.fr/banque-detail/actualites", "https://www.revue-banque.fr/assurance/actualites"]},
        {"name": "Argus Assurance", "url": "https://www.argusdelassurance.com/", "html": []},
        {"name": "Option Finance", "url": "https://www.optionfinance.fr/", "html": ["https://www.optionfinance.fr/accueil/actualites.html", "https://www.optionfinance.fr/asset-management/actualites.html"]},
        # Presse internationale
        {"name": "Bloomberg", "url": "https://www.bloomberg.com/feeds/podcasts/etf_report.xml", "html": ["https://www.bloomberg.com/markets"]},
        {"name": "Reuters", "url": "https://feeds.reuters.com/reuters/worldnews", "html": []},
        {"name": "Financial Times", "url": "https://www.ft.com/?format=rss", "html": []},
    ]
    seen = set()
    items: list[dict] = []
    for src in sources:
        xml = _fetch_url(src["url"], insecure=insecure, proxy=proxy)
        entries = []
        if xml:
            entries = _parse_rss(xml)
        if not entries:
            for alt in src.get("html") or []:
                html = _fetch_url(alt, insecure=insecure, proxy=proxy)
                if html:
                    entries = _parse_html_links(html, alt)
                    if entries:
                        break
        for e in entries:
            title = e.get("title", "")
            link = e.get("link", "")
            summary = e.get("summary", "")
            blob = f"{title} {summary}"
            if not _matches_keywords(blob):
                continue
            key = (title, link)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "source": src["name"],
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": e.get("date", ""),
                    "tags": [src["name"]],
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    items.sort(key=lambda x: x.get("published") or "", reverse=True)
    return items


def _slugify_filename(value: str | None) -> str:
    """Simple ASCII slugification for filenames."""
    if not value:
        return "document"
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip()
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "document"


def fetch_rh_list(db: Session) -> list[dict]:
    """Retrieve RH directory entries ordered by name."""
    try:
        rows = db.execute(
            text("SELECT id, nom, prenom, telephone, mail, niveau_poste, commentaire FROM administration_RH ORDER BY nom, prenom")
        ).fetchall()
        return rows_to_dicts(rows)
    except Exception:
        return []


def _load_groupes_details(db: Session) -> list[dict]:
    """Return all group definitions with member counts and RH names."""
    try:
        rows = db.execute(
            text(
                """
                SELECT d.id,
                       d.type_groupe,
                       d.nom,
                       d.date_creation,
                       d.date_fin,
                       d.responsable_id,
                       d.motif,
                       d.actif,
                       COALESCE(m.nb_membres, 0) AS nb_membres,
                       rh.prenom AS resp_prenom,
                       rh.nom AS resp_nom
                FROM administration_groupe_detail d
                LEFT JOIN (
                    SELECT groupe_id, COUNT(*) AS nb_membres
                    FROM administration_groupe
                    WHERE (COALESCE(actif,1) != 0)
                    GROUP BY groupe_id
                ) m ON m.groupe_id = d.id
                LEFT JOIN administration_RH rh ON rh.id = d.responsable_id
                ORDER BY d.nom
                """
            )
        ).fetchall()
    except Exception:
        return []
    return rows_to_dicts(rows)


def _parse_date_safe(raw):
    if raw in (None, ""):
        return None
    if isinstance(raw, _date):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        text_value = str(raw).strip()
        try:
            return datetime.fromisoformat(text_value).date()
        except ValueError:
            return _date.fromisoformat(text_value[:10])
    except Exception:
        return None


def _align_to_friday(value):
    if value is None:
        return None
    offset = (4 - value.weekday()) % 7
    return value + timedelta(days=offset)


def _compute_reclamations_data(db: Session) -> tuple[dict, dict]:
    """Return aggregated reclamation stats and pivot data."""
    try:
        reclam_total = db.execute(
            text(
                """
                SELECT COUNT(mariadb_evenement.id) AS total
                FROM mariadb_evenement
                JOIN mariadb_type_evenement
                    ON mariadb_type_evenement.id = mariadb_evenement.type_id
                WHERE lower(mariadb_type_evenement.categorie) IN ('reclamation', 'réclamation')
                """
            )
        ).scalar() or 0

        rows_reclam_status = db.execute(
            text(
                """
                SELECT COALESCE(
                        lower(trim(mariadb_evenement_statut_reclamation.libelle)),
                        lower(trim(mariadb_evenement.statut))
                    ) AS statut,
                    COUNT(mariadb_evenement.id) AS nb
                FROM mariadb_evenement
                JOIN mariadb_type_evenement
                    ON mariadb_type_evenement.id = mariadb_evenement.type_id
                LEFT OUTER JOIN mariadb_evenement_statut_reclamation
                    ON mariadb_evenement_statut_reclamation.id = mariadb_evenement.statut_reclamation_id
                WHERE lower(mariadb_type_evenement.categorie) IN ('reclamation', 'réclamation')
                GROUP BY COALESCE(
                    lower(trim(mariadb_evenement_statut_reclamation.libelle)),
                    lower(trim(mariadb_evenement.statut))
                )
                """
            )
        ).fetchall()

        rows_reclam_types = db.execute(
            text(
                """
                SELECT id, libelle
                FROM mariadb_type_evenement
                WHERE lower(categorie) IN ('reclamation', 'réclamation')
                ORDER BY libelle
                """
            )
        ).fetchall()

        rows_reclam_counts = db.execute(
            text(
                """
                SELECT
                    mariadb_type_evenement.id AS type_id,
                    mariadb_type_evenement.libelle AS type_libelle,
                    COALESCE(
                        NULLIF(TRIM(mariadb_evenement_statut_reclamation.libelle), ''),
                        NULLIF(TRIM(mariadb_evenement.statut), ''),
                        'Statut non renseigné'
                    ) AS statut_libelle,
                    COUNT(mariadb_evenement.id) AS nb
                FROM mariadb_evenement
                JOIN mariadb_type_evenement
                    ON mariadb_type_evenement.id = mariadb_evenement.type_id
                LEFT JOIN mariadb_evenement_statut_reclamation
                    ON mariadb_evenement_statut_reclamation.id = mariadb_evenement.statut_reclamation_id
                WHERE lower(mariadb_type_evenement.categorie) IN ('reclamation', 'réclamation')
                GROUP BY
                    mariadb_type_evenement.id,
                    mariadb_type_evenement.libelle,
                    COALESCE(
                        NULLIF(TRIM(mariadb_evenement_statut_reclamation.libelle), ''),
                        NULLIF(TRIM(mariadb_evenement.statut), ''),
                        'Statut non renseigné'
                    )
                """
            )
        ).fetchall()

        reclam_stats = {
            "total": int(reclam_total),
            "by_status": [
                {
                    "label": getattr(row, "statut", None) or "",
                    "count": int(getattr(row, "nb", 0) or 0),
                }
                for row in rows_reclam_status
            ],
        }

        pivot_statuses: set[str] = set()
        pivot_types: dict[int, dict] = {}

        for row in rows_reclam_types:
            try:
                type_id = int(getattr(row, "id", 0) or 0)
            except Exception:
                type_id = 0
            label = (getattr(row, "libelle", "") or "").strip()
            pivot_types[type_id] = {"id": type_id, "label": label, "counts": {}}

        for row in rows_reclam_counts:
            try:
                type_id = int(getattr(row, "type_id", 0) or 0)
            except Exception:
                type_id = 0
            type_label = (getattr(row, "type_libelle", "") or "").strip()
            status_label = (getattr(row, "statut_libelle", "") or "").strip() or "Statut non renseigné"
            count_value = int(getattr(row, "nb", 0) or 0)
            pivot_statuses.add(status_label)
            if type_id not in pivot_types:
                pivot_types[type_id] = {"id": type_id, "label": type_label, "counts": {}}
            pivot_types[type_id]["counts"][status_label] = count_value
            if not pivot_types[type_id].get("label"):
                pivot_types[type_id]["label"] = type_label

        sorted_statuses = sorted(pivot_statuses, key=lambda value: value.lower())
        sorted_types = sorted(pivot_types.values(), key=lambda item: (item.get("label") or "").lower())
        reclam_pivot = {"types": sorted_types, "statuses": sorted_statuses}
        return reclam_stats, reclam_pivot
    except Exception as exc:
        logger.exception("Failed to compute reclamation stats", exc_info=exc)
        return {"total": 0, "by_status": []}, {"types": [], "statuses": []}


def _normalize_valo_token(token: str | float | int) -> float | None:
    if token is None:
        return None
    if isinstance(token, (int, float)):
        try:
            return float(token)
        except Exception:
            return None
    try:
        text_value = str(token).strip().lower()
    except Exception:
        return None
    if not text_value:
        return None
    text_value = text_value.replace(" ", "")
    multiplier = 1.0
    if text_value.endswith("k"):
        multiplier = 1_000.0
        text_value = text_value[:-1]
    elif text_value.endswith("m"):
        multiplier = 1_000_000.0
        text_value = text_value[:-1]
    text_value = text_value.replace(",", ".")
    try:
        return float(text_value) * multiplier
    except ValueError:
        return None


def _parse_valo_filter_expression(expr: str | None) -> dict:
    """Parse valuation filter expression (>100k, 200k-1m, <=500k)."""
    result = {
        "raw": None,
        "min": None,
        "min_inclusive": True,
        "max": None,
        "max_inclusive": True,
    }
    if not expr:
        return result
    raw = str(expr).strip()
    if not raw:
        return result
    result["raw"] = raw
    import re

    tokens = re.findall(r"(>=|<=|>|<|=)?\s*([-+]?[^\s<>=]+)", raw)
    parsed_any = False
    for op, value in tokens:
        num = _normalize_valo_token(value)
        if num is None:
            continue
        parsed_any = True
        if op in (">", ">="):
            result["min"] = num
            result["min_inclusive"] = op == ">="
        elif op in ("<", "<="):
            result["max"] = num
            result["max_inclusive"] = op == "<="
        elif op == "=":
            result["min"] = num
            result["max"] = num
            result["min_inclusive"] = True
            result["max_inclusive"] = True
    if not parsed_any and "-" in raw:
        parts = raw.split("-")
        if len(parts) == 2:
            a = _normalize_valo_token(parts[0])
            b = _normalize_valo_token(parts[1])
            if a is not None and b is not None:
                result["min"] = min(a, b)
                result["max"] = max(a, b)
                parsed_any = True
    if not parsed_any:
        num = _normalize_valo_token(raw)
        if num is not None:
            result["min"] = num
            result["max"] = num
    return result


def _get_veille_context() -> dict:
    import json

    items: list[dict] = []
    now = datetime.now(timezone.utc)
    # Cache only if non-empty
    try:
        if VEILLE_CACHE["ts"] and VEILLE_CACHE.get("items") and (now - VEILLE_CACHE["ts"]).total_seconds() < 300:
            items = VEILLE_CACHE["items"]
        else:
            fetched = _collect_veille(insecure=True, proxy=None)
            if fetched:
                items = fetched
                VEILLE_CACHE["items"] = items
                VEILLE_CACHE["ts"] = now
            else:
                VEILLE_CACHE["ts"] = None
                VEILLE_CACHE["items"] = []
    except Exception:
        items = []
    if not items:
        try:
            if VEILLE_FILE.exists():
                with VEILLE_FILE.open("r", encoding="utf-8") as f:
                    items = json.load(f) or []
        except Exception:
            items = []
    if not items:
        # Dernier recours : quelques éléments statiques pour ne pas afficher une page vide
        items = [
            {
                "source": "Mock Veille",
                "tags": ["divers"],
                "title": "Aucune donnée distante — affichage de secours",
                "link": "#",
                "published": now.strftime("%Y-%m-%d"),
                "summary": "La collecte distante a échoué ou n'a rien renvoyé. Vérifiez l'accès réseau ou le fichier data/veille_reglementaire.json.",
            }
        ]
    grouped: dict = {}
    for it in items:
        tags = it.get("tags") or []
        key = tags[0] if tags else "Divers"
        grouped.setdefault(key, []).append(it)
    for k in grouped:
        grouped[k].sort(key=lambda x: x.get("published") or "", reverse=True)
    recent = sorted(items, key=lambda x: x.get("published") or "", reverse=True)[:15]
    markets = [
        {"name": "CAC 40", "value": "7 250", "var": "+0.8%"},
        {"name": "DAX", "value": "16 150", "var": "+0.5%"},
        {"name": "S&P 500", "value": "4 380", "var": "-0.3%"},
        {"name": "EuroStoxx 50", "value": "4 350", "var": "+0.2%"},
        {"name": "NASDAQ", "value": "13 600", "var": "-0.6%"},
    ]
    markets_meta = {
        "source": "Données marchés (mock local)",
        "timestamp": now.astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
    }
    return {
        "items": items,
        "grouped": grouped,
        "recent": recent,
        "markets": markets,
        "markets_meta": markets_meta,
    }


def _compute_tasks_summary(db: Session, range_days: int = 14) -> dict:
    """Calcule les indicateurs de la section Tâches (vue_suivi_evenement)."""
    from sqlalchemy import text as _text

    if range_days not in (7, 14, 30):
        range_days = 14
    start_date = _date.today() - timedelta(days=range_days - 1)
    try:
        tasks_total = db.execute(_text("SELECT COUNT(1) FROM vue_suivi_evenement")).scalar() or 0
    except Exception:
        tasks_total = 0

    try:
        rows_statut = db.execute(
            _text(
                """
                SELECT COALESCE(NULLIF(TRIM(LOWER(statut)), ''), '(non défini)') AS s, COUNT(1) AS nb
                FROM vue_suivi_evenement
                GROUP BY s
                ORDER BY nb DESC
                """
            )
        ).fetchall()
        tasks_statut = [{"statut": r[0], "nb": int(r[1] or 0)} for r in rows_statut]
    except Exception:
        tasks_statut = []

    try:
        rows_cat = db.execute(
            _text(
                """
                SELECT COALESCE(NULLIF(TRIM(LOWER(categorie)), ''), '(non défini)') AS c, COUNT(1) AS nb
                FROM vue_suivi_evenement
                GROUP BY c
                ORDER BY nb DESC
                """
            )
        ).fetchall()
        tasks_categorie = [{"categorie": r[0], "nb": int(r[1] or 0)} for r in rows_cat]
    except Exception:
        tasks_categorie = []

    try:
        open_count = (
            db.execute(
                _text(
                    """
                    SELECT COUNT(1)
                    FROM vue_suivi_evenement
                    WHERE statut IS NULL OR LOWER(statut) NOT IN ('termine','terminé','cloture','clôturé','annule','annulé')
                    """
                )
            ).scalar()
            or 0
        )
    except Exception:
        open_count = 0

    try:
        rows_days = db.execute(
            _text(
                """
                SELECT DATE(date_evenement) AS d, COUNT(1) AS nb
                FROM vue_suivi_evenement
                WHERE date_evenement >= :start_dt
                GROUP BY d
                ORDER BY d ASC
                """
            ),
            {"start_dt": start_date},
        ).fetchall()
        by_day = {str(r[0])[:10]: int(r[1] or 0) for r in rows_days}
        tasks_days = []
        cur = start_date
        today = _date.today()
        while cur <= today:
            key = cur.isoformat()
            tasks_days.append({"day": key, "nb": by_day.get(key, 0)})
            cur += timedelta(days=1)
    except Exception:
        tasks_days = []

    try:
        rows_avg = db.execute(
            _text(
                """
                WITH ordered AS (
                    SELECT
                        es.evenement_id,
                        se.libelle AS statut,
                        es.date_statut,
                        LEAD(es.date_statut) OVER(PARTITION BY es.evenement_id ORDER BY es.date_statut) AS next_dt
                    FROM mariadb_evenement_statut es
                    JOIN mariadb_statut_evenement se ON se.id = es.statut_id
                )
                SELECT statut, AVG(DATEDIFF(next_dt, date_statut)) AS avg_days
                FROM ordered
                WHERE next_dt IS NOT NULL
                GROUP BY statut
                ORDER BY avg_days DESC
                """
            )
        ).fetchall()
        tasks_avg_by_statut = [{"statut": r[0], "avg_days": float(r[1] or 0)} for r in rows_avg]
    except Exception:
        tasks_avg_by_statut = []

    try:
        row_close = db.execute(
            _text(
                """
                WITH close AS (
                  SELECT es.evenement_id, MIN(es.date_statut) AS close_dt
                  FROM mariadb_evenement_statut es
                  JOIN mariadb_statut_evenement se ON se.id = es.statut_id
                  WHERE LOWER(se.libelle) IN ('termine','terminé','annule','annulé')
                  GROUP BY es.evenement_id
                )
                SELECT AVG(DATEDIFF(close.close_dt, e.date_evenement)) AS avg_close
                FROM close
                JOIN mariadb_evenement e ON e.id = close.evenement_id
                """
            )
        ).scalar()
        tasks_avg_close_days = float(row_close or 0.0)
    except Exception:
        tasks_avg_close_days = 0.0

    try:
        dist_start = _date.today() - timedelta(days=range_days)
        rows_dist = db.execute(
            _text(
                """
                WITH close AS (
                  SELECT es.evenement_id, MIN(es.date_statut) AS close_dt
                  FROM mariadb_evenement_statut es
                  JOIN mariadb_statut_evenement se ON se.id = es.statut_id
                  WHERE LOWER(se.libelle) IN ('termine','terminé','annule','annulé')
                  GROUP BY es.evenement_id
                ), durations AS (
                  SELECT DATEDIFF(c.close_dt, e.date_evenement) AS d, c.close_dt AS cd
                  FROM close c JOIN mariadb_evenement e ON e.id = c.evenement_id
                  WHERE DATE(c.close_dt) >= :dist_start
                )
                SELECT bucket, COUNT(1) AS nb FROM (
                  SELECT CASE
                    WHEN d < 1 THEN '<1j'
                    WHEN d < 3 THEN '1–3j'
                    WHEN d < 7 THEN '3–7j'
                    WHEN d < 14 THEN '7–14j'
                    WHEN d < 30 THEN '14–30j'
                    ELSE '>=30j'
                  END AS bucket
                  FROM durations
                ) x
                GROUP BY bucket
                ORDER BY CASE bucket
                  WHEN '<1j' THEN 0
                  WHEN '1–3j' THEN 1
                  WHEN '3–7j' THEN 2
                  WHEN '7–14j' THEN 3
                  WHEN '14–30j' THEN 4
                  ELSE 5 END
                """
            ),
            {"dist_start": dist_start},
        ).fetchall()
        tasks_close_dist = [{"bucket": r[0], "nb": int(r[1] or 0)} for r in rows_dist]
    except Exception:
        tasks_close_dist = []

    try:
        rows_type_rh = db.execute(
            _text(
                """
                SELECT
                    v.rh_id AS rh_id,
                    COALESCE(NULLIF(TRIM(CONCAT_WS(' ', r.prenom, r.nom)), ''), r.mail, 'Sans responsable') AS rh_nom,
                    v.type_evenement AS type,
                    COUNT(*) AS nb
                FROM vue_suivi_evenement v
                LEFT JOIN administration_RH r ON r.id = v.rh_id
                GROUP BY v.rh_id, rh_nom, v.type_evenement
                ORDER BY rh_nom, v.type_evenement
                """
            )
        ).fetchall()
        tasks_type_by_rh = rows_to_dicts(rows_type_rh)
    except Exception:
        tasks_type_by_rh = []

    try:
        rows_type_total = db.execute(
            _text(
                """
                SELECT
                    v.type_evenement AS type,
                    COUNT(*) AS nb
                FROM vue_suivi_evenement v
                GROUP BY v.type_evenement
                ORDER BY v.type_evenement
                """
            )
        ).fetchall()
        tasks_type_total = rows_to_dicts(rows_type_total)
    except Exception:
        tasks_type_total = []

    return {
        "range": range_days,
        "tasks_total": tasks_total,
        "tasks_open": open_count,
        "tasks_statut": tasks_statut,
        "tasks_categorie": tasks_categorie,
        "tasks_days": tasks_days,
        "tasks_avg_by_statut": tasks_avg_by_statut,
        "tasks_avg_close_days": tasks_avg_close_days,
        "tasks_close_dist": tasks_close_dist,
        "tasks_type_by_rh": tasks_type_by_rh,
        "tasks_type_total": tasks_type_total,
    }

def _get_esg_fields(db: Session, debug: bool = False) -> tuple[list[dict], dict]:
    """Retourne les champs ESG (colonnes de esg_fonds) avec cache (TTL 1h)."""
    now = perf_counter()
    cached = ESG_FIELDS_CACHE.get("fields")
    if cached and (now - cached.get("ts", 0)) < 3600:
        return cached["fields"], cached.get("debug", {}) if debug else {}

    esg_fields: list[dict] = []
    esg_fields_debug: dict = {"source": "", "raw_cols": [], "final": []}
    try:
        ok = False
        # MariaDB/MySQL
        try:
            rows_cols = db.execute(text("SHOW COLUMNS FROM esg_fonds")).fetchall()
            if rows_cols:
                esg_fields_debug = {"source": "SHOW COLUMNS", "raw_cols": [str(getattr(getattr(rc, '_mapping', {}), 'get', lambda *_: rc[0])('Field')) if hasattr(rc, '_mapping') else str(rc[0]) for rc in rows_cols], "final": []}
                for rc in rows_cols:
                    col = rc[0]
                    if str(col).lower() in ("isin", "company name"):
                        continue
                    esg_fields.append({"col": col, "label": col})
                ok = True
        except Exception:
            ok = False
        # SQLite fallback
        if not ok:
            try:
                rows_cols = db.execute(text("PRAGMA table_info(esg_fonds)")).fetchall()
                esg_fields_debug = {"source": "PRAGMA table_info(esg_fonds)", "raw_cols": [str(rc[1]) for rc in (rows_cols or [])], "final": []}
                def _labelize(name: str) -> str:
                    if not name:
                        return name
                    s = str(name).replace('_', ' ')
                    import re as _re
                    s = _re.sub(r'(?<!^)([A-Z])', r' \1', s)
                    return s.strip().capitalize()
                for rc in rows_cols or []:
                    col = rc[1]
                    if str(col).lower() in ("isin", "company name"):
                        continue
                    esg_fields.append({"col": col, "label": _labelize(col)})
                ok = True
            except Exception:
                ok = False
        # Generic fallback
        if not ok:
            try:
                row1 = db.execute(text("SELECT * FROM esg_fonds LIMIT 1")).first()
                if row1 is not None:
                    keys = list(getattr(row1, "_mapping", {}).keys())
                    esg_fields_debug = {"source": "SELECT * LIMIT 1", "raw_cols": [str(k) for k in keys], "final": []}
                    for k in keys:
                        if str(k).lower() in ("isin", "company name"):
                            continue
                        esg_fields.append({"col": k, "label": k})
                ok = True
            except Exception:
                ok = False
    except Exception:
        esg_fields = []
        esg_fields_debug = {"source": "ERROR", "raw_cols": [], "final": []}

    # Déduplique et trie
    seen = set()
    uniq = []
    for it in esg_fields:
        key = str(it.get("col"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    esg_fields = sorted(uniq, key=lambda x: str(x.get("label", "")).lower())
    esg_fields_debug["final"] = esg_fields
    ESG_FIELDS_CACHE["fields"] = {"ts": now, "fields": esg_fields, "debug": esg_fields_debug}
    return esg_fields, (esg_fields_debug if debug else {})


def _compute_conformite_summary(db: Session) -> dict:
    """Calcule les indicateurs Conformité / Documents / Réclamations pour le dashboard (lazy)."""
    try:
        total_clients = db.query(func.count(Client.id)).scalar() or 0
        total_documents = db.query(func.count(DocumentClient.id)).scalar() or 0

        def _doc_stats(doc_name: str) -> tuple[int, int, float, float]:
            total = (
                db.query(func.count(DocumentClient.id))
                .join(Document, Document.id_document_base == DocumentClient.id_document_base)
                .filter(func.lower(Document.documents) == doc_name)
                .scalar()
                or 0
            )
            obsolete = (
                db.query(func.count(DocumentClient.id))
                .join(Document, Document.id_document_base == DocumentClient.id_document_base)
                .filter(func.lower(Document.documents) == doc_name)
                .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
                .scalar()
                or 0
            )
            total_pct = (total / total_clients * 100.0) if total_clients else 0.0
            obs_pct = (obsolete / total_clients * 100.0) if total_clients else 0.0
            return total, obsolete, total_pct, obs_pct

        der_total, der_obs, der_total_pct, der_obs_pct = _doc_stats('der')
        cc_total, cc_obs, cc_total_pct, cc_obs_pct = _doc_stats("recueil d'informations")
        esg_total, esg_obs, esg_total_pct, esg_obs_pct = _doc_stats('questionnaire esg')
        lm_total, lm_obs, lm_total_pct, lm_obs_pct = _doc_stats('lettre de mission')
        la_total, la_obs, la_total_pct, la_obs_pct = _doc_stats("lettre d'adéquation")

        doc_table_name = _resolve_table_name(db, ["courtier_documents_officiels"])
        orias_doc_found = orias_doc_valid = False
        orias_doc_expiry_display = ""
        assoc_doc_found = assoc_doc_valid = False
        assoc_doc_expiry_display = ""
        rcpro_doc_count = rcpro_doc_valid_count = rcpro_doc_expired_count = 0
        rcpro_doc_valid = False
        rcpro_doc_expiry_display = ""

        if doc_table_name:
            row = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 1},
            ).fetchone()
            if row:
                orias_doc_found = True
                expiry_raw = row._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    orias_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= _date.today():
                        orias_doc_valid = True

            row_assoc = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 2},
            ).fetchone()
            if row_assoc:
                assoc_doc_found = True
                expiry_raw = row_assoc._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    assoc_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= _date.today():
                        assoc_doc_valid = True

            rc_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
            ).fetchone()
            count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                    """
                ),
            ).fetchone()
            expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"today": _date.today().isoformat()},
            ).fetchone()
            rcpro_doc_count = int(count_row[0]) if count_row else 0
            rcpro_doc_expired_count = int(expired_row[0]) if expired_row else 0
            rcpro_doc_valid_count = max(rcpro_doc_count - rcpro_doc_expired_count, 0)
            if rc_rows:
                expiry_raw = rc_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    rcpro_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= _date.today():
                        rcpro_doc_valid = True

        reclam_stats, reclam_pivot = _compute_reclamations_data(db)

    except Exception:
        total_documents = 0
        der_total = der_obs = der_total_pct = der_obs_pct = 0.0
        cc_total = cc_obs = cc_total_pct = cc_obs_pct = 0.0
        esg_total = esg_obs = esg_total_pct = esg_obs_pct = 0.0
        lm_total = lm_obs = lm_total_pct = lm_obs_pct = 0.0
        la_total = la_obs = la_total_pct = la_obs_pct = 0.0
        orias_doc_found = orias_doc_valid = False
        orias_doc_expiry_display = ""
        assoc_doc_found = assoc_doc_valid = False
        assoc_doc_expiry_display = ""
        rcpro_doc_count = rcpro_doc_valid_count = rcpro_doc_expired_count = 0
        rcpro_doc_valid = False
        rcpro_doc_expiry_display = ""
        reclam_stats = {"total": 0, "by_status": []}
        reclam_pivot = {"types": [], "statuses": []}

    return {
        "docs_total": total_documents,
        "docs_der_total": der_total,
        "docs_der_obsolete": der_obs,
        "docs_der_total_pct": der_total_pct,
        "docs_der_obsolete_pct": der_obs_pct,
        "docs_cc_total": cc_total,
        "docs_cc_obsolete": cc_obs,
        "docs_cc_total_pct": cc_total_pct,
        "docs_cc_obsolete_pct": cc_obs_pct,
        "docs_esg_total": esg_total,
        "docs_esg_obsolete": esg_obs,
        "docs_esg_total_pct": esg_total_pct,
        "docs_esg_obsolete_pct": esg_obs_pct,
        "docs_lm_total": lm_total,
        "docs_lm_obsolete": lm_obs,
        "docs_lm_total_pct": lm_total_pct,
        "docs_lm_obsolete_pct": lm_obs_pct,
        "docs_la_total": la_total,
        "docs_la_obsolete": la_obs,
        "docs_la_total_pct": la_total_pct,
        "docs_la_obsolete_pct": la_obs_pct,
        "orias_doc_found": orias_doc_found,
        "orias_doc_valid": orias_doc_valid,
        "orias_doc_expiry_display": orias_doc_expiry_display,
        "assoc_doc_found": assoc_doc_found,
        "assoc_doc_valid": assoc_doc_valid,
        "assoc_doc_expiry_display": assoc_doc_expiry_display,
        "rcpro_doc_count": rcpro_doc_count,
        "rcpro_doc_valid": rcpro_doc_valid,
        "rcpro_doc_expiry_display": rcpro_doc_expiry_display,
        "rcpro_doc_expired_count": rcpro_doc_expired_count,
        "rcpro_doc_valid_count": rcpro_doc_valid_count,
        "reclam_stats": reclam_stats,
        "reclam_pivot": reclam_pivot,
    }


@router.get("/veille", response_class=HTMLResponse)
def dashboard_veille(request: Request):
    # Veille dédiée (chargée directement pour cette page)
    veille_ctx = _get_veille_context()
    return templates.TemplateResponse(
        "dashboard_veille.html",
        {
            "request": request,
            "items": veille_ctx.get("items"),
            "grouped": veille_ctx.get("grouped"),
            "recent": veille_ctx.get("recent"),
            "markets": veille_ctx.get("markets"),
            "markets_meta": veille_ctx.get("markets_meta"),
        },
    )

@router.get("/veille/data", response_class=JSONResponse)
def dashboard_veille_data():
    # Veille non chargée sur la page principale pour réduire le temps de rendu ; chargée via /dashboard/veille
    return _get_veille_context()

@router.get("/tasks/summary", response_class=JSONResponse)
def dashboard_tasks_summary(range: int = Query(14), db: Session = Depends(get_db)):
    """Résumé des tâches (lazy load depuis le dashboard)."""
    return _compute_tasks_summary(db, range)

@router.get("/esg/fields", response_class=JSONResponse)
def dashboard_esg_fields(db: Session = Depends(get_db)):
    """Retourne les champs ESG disponibles + allocations (cache 1h)."""
    try:
        alloc_names = [r[0] for r in db.query(Allocation.nom).filter(Allocation.nom.isnot(None)).distinct().order_by(Allocation.nom.asc()).all()]
    except Exception:
        alloc_names = []
    fields, debug = _get_esg_fields(db, debug=True)
    return {
        "status": "ok",
        "esg_fields": fields,
        "alloc_names": alloc_names,
        "debug": debug,
        "dates": [],  # placeholder (si dates spécifiques à renvoyer plus tard)
    }

@router.get("/conformite/summary", response_class=JSONResponse)
def dashboard_conformite_summary(db: Session = Depends(get_db)):
    """Résumé conformité/documents/réclamations pour lazy load."""
    return _compute_conformite_summary(db)


def _fmt_currency(value: float | int | None) -> str:
    try:
        return "{:,.0f}".format(float(value or 0)).replace(",", " ")
    except Exception:
        return "0"


def _build_svg_line_chart(labels: list[str], values: list[float], title: str) -> str | None:
    try:
        vals = [float(v or 0) for v in values or []]
        if not vals:
            return None
        labs = [str(l or "") for l in labels or []]
        if len(vals) > 12:
            vals = vals[-12:]
            labs = labs[-12:]
        W, H, P = 640, 260, 36
        vmax = max(vals) or 1.0
        vmin = min(vals) if vals else 0.0
        if vmax == vmin:
            vmax = vmin + 1
        step = max(1, len(vals) - 1)
        def X(i):
            return P + i * (W - 2 * P) / step
        def Y(v):
            return H - P - ((v - vmin) * (H - 2 * P) / (vmax - vmin))
        points = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
        parts.append(f'<text x="{P}" y="{P-12}" fill="#111" font-size="14" font-weight="bold">{title}</text>')
        for i in range(5):
            y = P + i * (H - 2 * P) / 4
            parts.append(f'<line x1="{P}" y1="{y:.1f}" x2="{W-P}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}" stroke-linejoin="round" stroke-linecap="round" />')
        for i, label in enumerate(labs):
            parts.append(f'<text x="{X(i):.1f}" y="{H-P+16}" text-anchor="middle" font-size="11" fill="#334155">{label}</text>')
        parts.append('</svg>')
        return ''.join(parts)
    except Exception:
        return None


def _build_matplotlib_two_line_chart(
    labels: list[str],
    serie_a: list[float],
    serie_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
) -> str | None:
    try:
        if not labels or not serie_a or not serie_b:
            return None
        if len(labels) != len(serie_a) or len(labels) != len(serie_b):
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:
        return None
    try:
        x = list(range(len(labels)))
        vals_a = [float(v or 0) for v in serie_a]
        vals_b = [float(v or 0) for v in serie_b]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(x, vals_a, label=label_a, color="#2563eb", linewidth=2.2)
        ax.plot(x, vals_b, label=label_b, color="#f97316", linewidth=2.2)
        ax.fill_between(x, vals_b, color="#f973161A")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a")
        ax.set_ylabel("Montant (k€)", fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:.1f}k"))
        ax.legend(loc="upper left", frameon=True)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def _build_svg_two_line_chart(
    labels: list[str],
    serie_a: list[float],
    serie_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
) -> str | None:
    try:
        if not labels or not serie_a or not serie_b:
            return None
        vals_a = [float(v or 0) for v in serie_a]
        vals_b = [float(v or 0) for v in serie_b]
        if len(labels) != len(vals_a) or len(labels) != len(vals_b):
            return None
        combined = vals_a + vals_b
        if not combined:
            return None
        labs = [str(l or "") for l in labels]
        W, H, P = 720, 300, 42
        vmax = max(combined)
        vmin = min(combined)
        if vmax == vmin:
            vmax = vmin + 1.0
        step = max(1, len(labs) - 1)

        def _x(i: int) -> float:
            return P + i * (W - 2 * P) / step

        def _y(v: float) -> float:
            return H - P - ((v - vmin) * (H - 2 * P) / (vmax - vmin))

        def _poly(vals: list[float]) -> str:
            return " ".join(f"{_x(i):.2f},{_y(v):.2f}" for i, v in enumerate(vals))

        parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
        parts.append(f'<style>.grid{{stroke:#dce3f5;stroke-width:1}} .legend-box{{fill:rgba(248,250,252,0.9);stroke:#cbd5f5;stroke-width:1}}</style>')
        parts.append(f'<text x="{P}" y="{P-14}" fill="#0f172a" font-size="15" font-weight="bold">{title}</text>')
        for i in range(5):
            y = P + i * (H - 2 * P) / 4
            parts.append(f'<line class="grid" x1="{P}" y1="{y:.2f}" x2="{W-P}" y2="{y:.2f}" />')
        parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2.6" points="{_poly(vals_a)}" stroke-linejoin="round" stroke-linecap="round" />')
        parts.append(f'<polyline fill="none" stroke="#f97316" stroke-width="2.6" points="{_poly(vals_b)}" stroke-linejoin="round" stroke-linecap="round" />')
        for i, lab in enumerate(labs):
            parts.append(f'<text x="{_x(i):.2f}" y="{H-P+18}" font-size="11" text-anchor="middle" fill="#475569">{lab}</text>')
        def _fmt_k(v: float) -> str:
            return f"{(v/1000):,.1f}".replace(",", " ")
        for i in range(5):
            v = vmin + i * (vmax - vmin) / 4
            y = _y(v)
            parts.append(f'<rect x="{P-55}" y="{y-9:.2f}" width="48" height="16" fill="rgba(255,255,255,0.85)"/>')
            parts.append(f'<text x="{P-12}" y="{y+4:.2f}" font-size="10" text-anchor="end" fill="#64748b">{_fmt_k(v)}</text>')
        legend_x = W - P - 260
        legend_y = P - 28
        parts.append(f'<g transform="translate({legend_x},{legend_y})" class="legend-box">')
        parts.append('<rect x="0" y="-16" width="250" height="28" rx="6" />')
        parts.append(f'<g transform="translate(12,0)"><line x1="0" y1="0" x2="20" y2="0" stroke="#2563eb" stroke-width="2.6"/><text x="28" y="4" font-size="11" fill="#0f172a">{label_a}</text></g>')
        parts.append(f'<g transform="translate(140,0)"><line x1="0" y1="0" x2="20" y2="0" stroke="#f97316" stroke-width="2.6"/><text x="28" y="4" font-size="11" fill="#0f172a">{label_b}</text></g>')
        parts.append('</g>')
        parts.append('</svg>')
        return "".join(parts)
    except Exception:
        return None


def _build_matplotlib_horizontal_bar_chart(labels: list[str], values: list[float], title: str, unit: str | None = None) -> str | None:
    try:
        data = [float(v or 0.0) for v in values or []]
        cats = [str(l or "") for l in labels or []]
        if not data or not cats:
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        positions = list(range(len(cats)))
        height = 0.45
        fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.6 * len(cats) + 1.2)))
        bars = ax.barh([p for p in positions], data, height=height, color="#2563eb", alpha=0.85)
        ax.set_yticks(positions)
        ax.set_yticklabels(cats)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a")
        if unit:
            ax.set_xlabel(unit, fontsize=11)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        max_val = max(data) if data else 0.0
        for bar, value in zip(bars, data):
            xpos = bar.get_width() + (0.01 * max_val if max_val else 0.1)
            if unit and unit.startswith('Valorisation'):
                label = f"{value:,.0f}".replace(',', ' ')
            else:
                label = f"{value:.1f}"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=9, color='#334155')
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None

def _build_matplotlib_perf_vol_chart(
    years: list[int],
    perf_series: list[float],
    vol_series: list[float],
) -> str | None:
    if not years or not perf_series or not vol_series:
        return None
    if not (len(years) == len(perf_series) == len(vol_series)):
        return None

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        def _fmt_pct(value):
            try:
                v = float(value)
                if abs(v) <= 1:
                    v *= 100.0
                return v
            except Exception:
                return 0.0

        perf_pct = [_fmt_pct(v) for v in perf_series]
        vol_pct = [_fmt_pct(v) for v in vol_series]
        years_labels = [str(y) for y in years]
        x = list(range(len(years_labels)))
        width = 0.6
        fig, ax1 = plt.subplots(figsize=(9, 4))
        bars = ax1.bar(x, perf_pct, width=width, color="#2563eb", alpha=0.82, label="Performance 52s (%)")
        ax1.set_ylabel("Performance (%)", color="#1d4ed8", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#1d4ed8")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years_labels, rotation=35, ha="right")
        ax1.set_title("Performances et volatilité annuelles", fontsize=14, fontweight="bold", color="#0f172a")
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1f2937",
            )
        ax2 = ax1.twinx()
        ax2.plot(x, vol_pct, color="#f97316", linewidth=2.4, marker="o", label="Volatilité (%)")
        ax2.fill_between(x, vol_pct, color="#f973161A")
        ax2.set_ylabel("Volatilité (%)", color="#f97316", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#f97316")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def _build_matplotlib_alloc_compare_chart(
    client_series: list[tuple[str | None, float]],
    alloc_series: list[tuple[str | None, float]],
    alloc_label: str,
) -> str | None:
    try:
        if not client_series or not alloc_series:
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except Exception:
        return None

    try:
        def _parse(series: list[tuple[str | None, float]]):
            out_dates = []
            out_vals = []
            for d, v in series:
                if not d:
                    continue
                try:
                    dt = datetime.fromisoformat(str(d))
                except Exception:
                    try:
                        dt = datetime.strptime(str(d), "%Y-%m-%d")
                    except Exception:
                        continue
                out_dates.append(dt)
                try:
                    out_vals.append(float(v or 0.0))
                except Exception:
                    out_vals.append(0.0)
            return out_dates, out_vals

        client_dates, client_vals = _parse(client_series)
        alloc_dates, alloc_vals = _parse(alloc_series)
        if not client_dates or not alloc_dates:
            return None

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(client_dates, client_vals, color="#2563eb", linewidth=2.2, label="Client")
        ax.plot(alloc_dates, alloc_vals, color="#10b981", linewidth=2.0, linestyle="--", label=alloc_label or "Allocation")
        combined_vals = client_vals + alloc_vals
        if not combined_vals:
            return None
        vmin = min(combined_vals)
        vmax = max(combined_vals)
        if vmax == vmin:
            delta = abs(vmax) * 0.05 if vmax != 0 else 5.0
            vmin -= delta
            vmax += delta
        padding = max(1.0, (vmax - vmin) * 0.05)
        ymin = vmin - padding
        ymax = vmax + padding
        ax.set_ylim(ymin, ymax)
        ax.fill_between(alloc_dates, alloc_vals, ymin, color="#10b9811A")
        ax.set_title("Évolution SICAV – Client vs Allocation de référence", fontsize=14, fontweight="bold", color="#0f172a")
        ax.set_ylabel("Indice (base 100)", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper left", frameon=True)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None
    try:
        def _fmt_pct(value):
            try:
                v = float(value)
                if abs(v) <= 1:
                    v *= 100.0
                return v
            except Exception:
                return 0.0
        perf_pct = [_fmt_pct(v) for v in perf_series]
        vol_pct = [_fmt_pct(v) for v in vol_series]
        years_labels = [str(y) for y in years]
        x = np.arange(len(years_labels))
        width = 0.6
        fig, ax1 = plt.subplots(figsize=(9, 4))
        bars = ax1.bar(x, perf_pct, width=width, color="#2563eb", alpha=0.82, label="Performance 52s (%)")
        ax1.set_ylabel("Performance (%)", color="#1d4ed8", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#1d4ed8")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years_labels, rotation=35, ha="right")
        ax1.set_title("Performances et volatilité annuelles", fontsize=14, fontweight="bold", color="#0f172a")
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4),
                         textcoords="offset points",
                         ha="center",
                         va="bottom",
                         fontsize=8,
                         color="#1f2937")
        ax2 = ax1.twinx()
        ax2.plot(x, vol_pct, color="#f97316", linewidth=2.4, marker="o", label="Volatilité (%)")
        ax2.fill_between(x, vol_pct, color="#f973161A")
        ax2.set_ylabel("Volatilité (%)", color="#f97316", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#f97316")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def _build_matplotlib_esg_bar_chart(
    categories: list[str],
    index_values: list[float],
    client_values: list[float],
) -> str | None:
    if not categories:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        positions = list(range(len(categories)))
        height = 0.35
        fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.8 + 0.6 * len(categories))))
        ax.barh([p + height / 2 for p in positions], index_values, height=height, color="#9ca3af", label="Indice (base 100)")
        ax.barh([p - height / 2 for p in positions], client_values, height=height, color="#2563eb", label="Client (base 100)")
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()
        ax.set_xlabel("Base 100")
        ax.set_xlim(left=0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def _normalize_identifier(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _compute_client_esg_comparison(
    db: Session,
    client_id: int,
    fields: list[str],
    alloc: str | None = None,
    alloc_isin: str | None = None,
    as_of: str | None = None,
    alloc_date: str | None = None,
    debug: bool = False,
) -> tuple[dict, dict | None]:
    sel_fields = [f for f in (fields or []) if f]
    payload = {
        "fields": sel_fields,
        "alloc": alloc,
        "alloc_isin": alloc_isin,
        "as_of": as_of,
        "alloc_date": None,
        "results": [],
    }
    debug_info: dict | None = {"client": None, "allocation": None, "esg": None} if debug else None
    if not sel_fields:
        return payload, debug_info

    # Résolution des noms de colonnes (gestion des variantes orthographiques/accents)
    try:
        rows_cols = db.execute(text("PRAGMA table_info(esg_fonds)")).fetchall()
        available_cols = [str(rc[1]) for rc in rows_cols or [] if rc and len(rc) > 1 and rc[1]]
    except Exception:
        available_cols = []
    col_lookup: dict[str, str] = {}
    for col in available_cols:
        key = _normalize_identifier(col)
        if key and key not in col_lookup:
            col_lookup[key] = col
    resolved_pairs: list[tuple[str, str | None]] = []
    seen_cols = set()
    for field_name in sel_fields:
        col = col_lookup.get(_normalize_identifier(field_name))
        if col in seen_cols:
            # éviter les doublons si l'utilisateur fournit plusieurs variantes pour la même colonne
            col = None
        if col:
            seen_cols.add(col)
        resolved_pairs.append((field_name, col))
    valid_pairs = [(orig, col) for orig, col in resolved_pairs if col]
    payload["resolved_fields"] = [col for _, col in valid_pairs]
    payload["resolved_pairs"] = [{"requested": orig, "resolved": col} for orig, col in resolved_pairs]
    payload["unmatched_fields"] = [orig for orig, col in resolved_pairs if not col]
    if not valid_pairs:
        if debug_info is not None:
            debug_info["client"] = {"weights_count": 0, "weights_sum": 0, "affaires": [], "query": None, "isins": []}
            debug_info["allocation"] = {"weights_count": 0, "weights_sum": 0, "isins": [], "date": None, "query": None}
            debug_info["esg"] = {"isin_count": 0, "query": None}
        # Conserver une entrée résultat vide pour chaque champ demandé
        payload["results"] = [
            {
                "field": orig,
                "field_original": orig,
                "resolved_field": None,
                "index": None,
                "client": None,
                "cli_present": 0,
                "idx_present": 0,
            }
            for orig in sel_fields
        ]
        return payload, debug_info

    # Composition client
    client_weights: dict[str, float] = {}
    debug_client: dict | None = {"weights_count": 0, "weights_sum": 0, "affaires": [], "query": None, "isins": []} if debug else None
    try:
        affaire_ids = [rid for (rid,) in db.query(Affaire.id).filter(Affaire.id_personne == client_id).all()]
        sums: dict[str, float] = {}
        total_valo = 0.0
        params_list = []
        weight_sql = (
            "SELECT s.code_isin AS isin, SUM(h.valo) AS somme_valo\n"
            "FROM mariadb_historique_support_w h\n"
            "JOIN mariadb_support s ON s.id = h.id_support\n"
            "WHERE h.id_source = :aid AND h.date = :d\n"
            "GROUP BY s.code_isin"
        )
        for aid in affaire_ids:
            if as_of:
                ref_date = as_of
            else:
                ref_date = db.execute(
                    text("SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :aid"),
                    {"aid": aid},
                ).scalar()
                if not ref_date:
                    continue
            rows = db.execute(text(weight_sql), {"aid": aid, "d": ref_date}).fetchall()
            params_list.append({"aid": aid, "d": str(ref_date) if ref_date is not None else None})
            for r in rows or []:
                isin = getattr(r, "isin", None)
                v = float(getattr(r, "somme_valo", 0) or 0)
                if isin:
                    sums[isin] = sums.get(isin, 0.0) + v
                    total_valo += v
        if total_valo > 0:
            for isin, v in sums.items():
                if v is not None:
                    client_weights[isin] = float(v) / float(total_valo)
        if debug_client is not None:
            debug_client.update({
                "weights_count": len(client_weights),
                "weights_sum": sum(client_weights.values()) if client_weights else 0,
                "affaires": affaire_ids,
                "params": params_list,
                "query": weight_sql,
                "as_of": as_of,
                "isins": sorted(client_weights.keys()),
            })
    except Exception:
        client_weights = {}
    # Composition allocation
    alloc_weights: dict[str, float] = {}
    alloc_date_val = None
    debug_alloc: dict | None = {"weights_count": 0, "weights_sum": 0, "query": None, "isins": [], "date": None} if debug else None
    try:
        import re as _re

        isin_param = alloc_isin or (
            alloc if alloc and _re.match(r"^[A-Z0-9]{9,12}$", str(alloc).strip()) else None
        )
        if isin_param:
            if alloc_date:
                alloc_date_val = db.execute(
                    text("SELECT MAX(date) FROM allocations WHERE ISIN = :i AND date <= :d"),
                    {"i": isin_param, "d": alloc_date},
                ).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(
                        text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"),
                        {"i": isin_param},
                    ).scalar()
            else:
                alloc_date_val = db.execute(
                    text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"),
                    {"i": isin_param},
                ).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE ISIN = :i AND date = :d
                """
            )
            params2 = {"i": isin_param, "d": alloc_date_val}
            rows2 = db.execute(q2, params2).fetchall()
            if debug_alloc is not None:
                debug_alloc["query"] = {"sql": q2.text, "params": {k: (str(v) if v is not None else None) for k, v in params2.items()}}
        elif alloc:
            if alloc_date:
                alloc_date_val = db.execute(
                    text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n)) AND date <= :d"),
                    {"n": alloc, "d": alloc_date},
                ).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(
                        text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"),
                        {"n": alloc},
                    ).scalar()
            else:
                alloc_date_val = db.execute(
                    text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"),
                    {"n": alloc},
                ).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE lower(trim(nom)) = lower(trim(:n)) AND date = :d
                """
            )
            params2 = {"n": alloc, "d": alloc_date_val}
            rows2 = db.execute(q2, params2).fetchall()
            if debug_alloc is not None:
                debug_alloc["query"] = {"sql": q2.text, "params": {k: (str(v) if v is not None else None) for k, v in params2.items()}}
        else:
            rows2 = []
        total2 = sum(float(getattr(r, "v", 0) or 0) for r in rows2) if rows2 else 0.0
        if total2 and total2 > 0:
            for r in rows2:
                isin = getattr(r, "isin", None)
                v = float(getattr(r, "v", 0) or 0)
                if isin:
                    alloc_weights[isin] = v / total2
        if debug_alloc is not None:
            debug_alloc.update({
                "weights_count": len(alloc_weights),
                "weights_sum": sum(alloc_weights.values()) if alloc_weights else 0,
                "isins": sorted(alloc_weights.keys()),
                "date": str(alloc_date_val) if alloc_date_val is not None else None,
            })
    except Exception:
        alloc_weights = {}
    payload["alloc_date"] = str(alloc_date_val) if alloc_date_val is not None else None

    # Rassembler les ISIN
    all_isins = set(client_weights.keys()) | set(alloc_weights.keys())
    if not all_isins:
        if debug_info is not None:
            debug_info["client"] = debug_client
            debug_info["allocation"] = debug_alloc
        return payload, debug_info

    # Chargement des colonnes ESG
    def _quote_col(c: str) -> str:
        c = c.strip()
        return f"`{c}`"

    aliases = [(col, f"c{idx}") for idx, (orig, col) in enumerate(valid_pairs)]
    col_expr = ", ".join(f"{_quote_col(col)} AS {alias}" for col, alias in aliases)
    esg_map: dict[str, dict[str, float]] = {}
    debug_esg: dict | None = {"isin_count": len(all_isins), "query": None} if debug else None
    try:
        isin_list = list(all_isins)
        placeholders = ",".join([f":i{idx}" for idx in range(len(isin_list))])
        params = {f"i{idx}": val for idx, val in enumerate(isin_list)}
        q_esg = text(f"SELECT ISIN, {col_expr} FROM esg_fonds WHERE ISIN IN ({placeholders})")
        rows_esg = db.execute(q_esg, params).fetchall()
        for row in rows_esg or []:
            mm = row._mapping
            isin = mm.get("ISIN")
            if not isin:
                continue
            d = {}
            for (col, alias) in aliases:
                try:
                    val = mm.get(alias)
                except Exception:
                    val = None
                try:
                    d[col] = float(val) if val is not None else None
                except Exception:
                    d[col] = None
            esg_map[isin] = d
        if debug_esg is not None:
            debug_esg["query"] = {"sql": q_esg.text, "params": params}
    except Exception:
        esg_map = {}

    results = []
    for original_name, column_name in valid_pairs:
        cli_val = 0.0
        cli_wsum = 0.0
        for isin, w in client_weights.items():
            v = (esg_map.get(isin) or {}).get(column_name)
            if v is None:
                continue
            cli_val += float(w) * float(v)
            cli_wsum += float(w)
        cli_val = (cli_val / cli_wsum) if cli_wsum > 0 else None

        idx_val = 0.0
        idx_wsum = 0.0
        for isin, w in alloc_weights.items():
            v = (esg_map.get(isin) or {}).get(column_name)
            if v is None:
                continue
            idx_val += float(w) * float(v)
            idx_wsum += float(w)
        idx_val = (idx_val / idx_wsum) if idx_wsum > 0 else None

        cli_present = sum(
            1 for isin, _w in client_weights.items() if (esg_map.get(isin) or {}).get(column_name) is not None
        )
        idx_present = sum(
            1 for isin, _w in alloc_weights.items() if (esg_map.get(isin) or {}).get(column_name) is not None
        )

        if cli_val is None or idx_val is None or idx_val == 0:
            results.append({
                "field": column_name,
                "field_original": original_name,
                "resolved_field": column_name,
                "index": None,
                "client": None,
                "cli_present": cli_present,
                "idx_present": idx_present,
            })
        else:
            results.append({
                "field": column_name,
                "field_original": original_name,
                "resolved_field": column_name,
                "index": 100.0,
                "client": (cli_val / idx_val) * 100.0,
                "cli_present": cli_present,
                "idx_present": idx_present,
            })

    # Ajouter les champs non résolus pour information
    for original_name, column_name in resolved_pairs:
        if column_name:
            continue
        results.append({
            "field": original_name,
            "field_original": original_name,
            "resolved_field": None,
            "index": None,
            "client": None,
            "cli_present": 0,
            "idx_present": 0,
        })

    payload["results"] = results
    if debug_info is not None:
        debug_info["client"] = debug_client
        debug_info["allocation"] = debug_alloc
        if debug_esg is not None:
            debug_esg["resolved_pairs"] = resolved_pairs
        debug_info["esg"] = debug_esg
    return payload, debug_info


def _build_svg_bar_chart(labels: list[str], values: list[float], title: str) -> str | None:
    try:
        vals = [float(v or 0) for v in values or []]
        labs = [str(l or "") for l in labels or []]
        if not vals or not labs:
            return None
        if len(vals) > 6:
            vals = vals[:6]
            labs = labs[:6]
        W, H, P = 520, 240, 36
        vmax = max(vals) or 1.0
        bw = (W - 2 * P) / max(1, len(vals)) * 0.6
        parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
        parts.append(f'<text x="{P}" y="{P-12}" fill="#111" font-size="14" font-weight="bold">{title}</text>')
        for i in range(5):
            y = P + i * (H - 2 * P) / 4
            parts.append(f'<line x1="{P}" y1="{y:.1f}" x2="{W-P}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        colors = ['#2563eb', '#16a34a', '#f97316', '#ef4444', '#8b5cf6', '#0ea5e9']
        for i, v in enumerate(vals):
            height = (H - 2 * P) * (v / vmax)
            x = P + i * ((W - 2 * P) / max(1, len(vals))) + (((W - 2 * P) / max(1, len(vals))) - bw) / 2
            y = H - P - height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{height:.1f}" fill="{colors[i % len(colors)]}" />')
            parts.append(f'<text x="{x + bw/2:.1f}" y="{H-P+16}" text-anchor="middle" font-size="11" fill="#334155">{labs[i]}</text>')
        parts.append('</svg>')
        return ''.join(parts)
    except Exception:
        return None


def _build_finance_analysis(
    db: Session,
    finance_rh_id: int | None,
    finance_date_param: str | None,
    finance_valo_param: str | None = None,
    force_refresh: bool = False,
    allow_fallback: bool = True,
) -> dict:
    """Compute portfolio analysis figures (supports repartition) for dashboard views."""
    # Invalidate previous cache to ensure new params and date calculations are reflected immediately
    FINANCE_CACHE.clear()
    cache_key = (finance_rh_id, finance_date_param or "", finance_valo_param or "")
    now_ts = perf_counter()
    cached = FINANCE_CACHE.get(cache_key)
    if cached and not force_refresh and (now_ts - cached.get("ts", 0)) < 600:
        cached_data = cached.get("data") or {}
        # If new fields (max date) are missing, recompute instead of serving stale cache.
        if cached_data.get("finance_max_date_iso") is None:
            pass
        else:
            return cached_data

    # Date max disponible (pour affichage et fallback)
    finance_max_date = None
    try:
        row_max = db.execute(
            text(
                """
                SELECT MAX(date(h.date)) AS max_date
                FROM mariadb_historique_support_w h
                """
            )
        ).fetchone()
        if row_max:
            max_val = row_max[0] if not hasattr(row_max, "_mapping") else row_max._mapping.get("max_date")
            finance_max_date = _parse_date_safe(max_val)
            logger.info("finance_max_date from mariadb_historique_support_w: %s", finance_max_date)
    except Exception:
        finance_max_date = None

    base_finance_date = _parse_date_safe(finance_date_param)
    if base_finance_date is None:
        base_finance_date = finance_max_date
    if base_finance_date is None:
        base_finance_date = _date.today()

    finance_effective_date = _align_to_friday(base_finance_date) or base_finance_date
    finance_effective_date_iso = finance_effective_date.isoformat()
    finance_date_input = finance_effective_date_iso
    finance_effective_date_display = finance_effective_date.strftime("%d/%m/%Y")

    # Use YYYY-MM-DD to ensure a date-only ISO string for display and SQL literal
    finance_max_date_iso = finance_max_date.strftime("%Y-%m-%d") if finance_max_date else None
    finance_max_date_display = finance_max_date.strftime("%d/%m/%Y") if finance_max_date else None
    finance_supports: list[dict] = []
    finance_total_valo = 0.0
    finance_total_valo_str = "0"
    valo_filter = _parse_valo_filter_expression(finance_valo_param)

    try:
        params = {
            "d_date": finance_effective_date,
        }
        t_sql = perf_counter()
        sql = """
            SELECT
                s.code_isin AS code_isin,
                s.nom AS nom,
                s.cat_principale AS cat_principale,
                s.cat_geo AS cat_geo,
                s.SRRI AS srri,
                SUM(h.valo) AS total_valo
            FROM mariadb_historique_support_w h
            JOIN mariadb_support s ON s.id = h.id_support
            JOIN mariadb_affaires a ON a.id = h.id_source
            LEFT JOIN mariadb_clients c ON c.id = a.id_personne
            WHERE h.date = :d_date
        """
        if finance_rh_id is not None:
            sql += " AND c.commercial_id = :rh_id"
            params["rh_id"] = finance_rh_id
        sql += """
            GROUP BY s.code_isin, s.nom, s.cat_principale, s.cat_geo, s.SRRI
        """
        having_clauses = []
        if valo_filter.get("min") is not None:
            if valo_filter.get("min_inclusive", True):
                having_clauses.append("SUM(h.valo) >= :valo_min")
            else:
                having_clauses.append("SUM(h.valo) > :valo_min")
            params["valo_min"] = valo_filter["min"]
        if valo_filter.get("max") is not None:
            if valo_filter.get("max_inclusive", True):
                having_clauses.append("SUM(h.valo) <= :valo_max")
            else:
                having_clauses.append("SUM(h.valo) < :valo_max")
            params["valo_max"] = valo_filter["max"]
        if having_clauses:
            sql += " HAVING " + " AND ".join(having_clauses)
        sql += " ORDER BY total_valo DESC"
        rows_supports = db.execute(text(sql), params).fetchall()
        logger.info("finance_analysis: supports fetched in %.3fs", perf_counter() - t_sql)
        clients_map: dict[str, list[dict]] = {}
        try:
            t_clients = perf_counter()
            params_clients = {
                "d_date": finance_effective_date,
            }
            sql_clients = """
                SELECT
                    s.code_isin AS code_isin,
                    c.id AS client_id,
                    c.nom AS client_nom,
                    c.prenom AS client_prenom,
                    a.id AS affaire_id,
                    a.ref AS affaire_ref,
                    SUM(h.valo) AS client_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_support s ON s.id = h.id_support
                JOIN mariadb_affaires a ON a.id = h.id_source
                LEFT JOIN mariadb_clients c ON c.id = a.id_personne
                WHERE h.date = :d_date
            """
            if finance_rh_id is not None:
                sql_clients += " AND c.commercial_id = :rh_id"
                params_clients["rh_id"] = finance_rh_id
            sql_clients += """
                GROUP BY s.code_isin, c.id, c.nom, c.prenom, a.id, a.ref
                HAVING SUM(h.valo) > 0
            """
            rows_clients = db.execute(text(sql_clients), params_clients).fetchall()
            logger.info("finance_analysis: clients fetched in %.3fs", perf_counter() - t_clients)
            for r in rows_clients or []:
                m = r._mapping if hasattr(r, "_mapping") else None
                code_isin_client = (m.get("code_isin") if m else r[0]) or None
                if not code_isin_client:
                    continue
                client_valo_raw = float(m.get("client_valo") if m else r[6] or 0.0)
                client_id = m.get("client_id") if m else r[1]
                client_nom = m.get("client_nom") if m else r[2]
                client_prenom = m.get("client_prenom") if m else r[3]
                affaire_id = m.get("affaire_id") if m else r[4]
                affaire_ref = m.get("affaire_ref") if m else r[5]
                fullname_parts = [client_prenom or "", client_nom or ""]
                fullname = " ".join(p.strip() for p in fullname_parts if p and p.strip())
                client_entry = {
                    "client_id": client_id,
                    "client_nom": client_nom,
                    "client_prenom": client_prenom,
                    "client_fullname": fullname or (client_nom or "Client"),
                    "affaire_id": affaire_id,
                    "affaire_ref": affaire_ref,
                    "valorisation": client_valo_raw,
                    "valorisation_str": _fmt_currency(client_valo_raw),
                }
                clients_map.setdefault(code_isin_client, []).append(client_entry)
            for code, items in clients_map.items():
                items.sort(key=lambda it: it.get("valorisation", 0.0), reverse=True)
        except Exception:
            clients_map = {}
        for row in rows_supports or []:
            m = row._mapping if hasattr(row, "_mapping") else None
            code_isin = m.get("code_isin") if m else row[0]
            nom = m.get("nom") if m else row[1]
            cat_principale = m.get("cat_principale") if m else row[2]
            cat_geo = m.get("cat_geo") if m else row[3]
            srri = m.get("srri") if m else row[4]
            total_valo = float(m.get("total_valo") if m else row[5] or 0.0)
            if total_valo > 0:
                finance_total_valo += total_valo
                finance_supports.append(
                    {
                        "code_isin": code_isin,
                        "nom": nom,
                        "cat_principale": cat_principale,
                        "cat_geo": cat_geo,
                        "srri": srri,
                        "total_valo": total_valo,
                        "total_valo_str": "{:,.0f}".format(total_valo).replace(",", " "),
                        "clients": clients_map.get(code_isin, []),
                    }
                )
        finance_total_valo_str = "{:,.0f}".format(finance_total_valo).replace(",", " ")
    except Exception:
        finance_supports = []
        finance_total_valo = 0.0
        finance_total_valo_str = "0"

    # Fallback: si aucune ligne, retenter sur la dernière date dispo (sans dépendre du paramètre fourni)
    if allow_fallback and not finance_supports:
        try:
            fb_row = db.execute(text("SELECT MAX(date(h.date)) AS max_date FROM mariadb_historique_support_w h")).fetchone()
            fb_date_raw = fb_row[0] if fb_row is not None and not hasattr(fb_row, "_mapping") else (fb_row._mapping.get("max_date") if fb_row else None)
            fb_date = _parse_date_safe(fb_date_raw)
            if fb_date and fb_date != finance_effective_date:
                return _build_finance_analysis(
                    db=db,
                    finance_rh_id=finance_rh_id,
                    finance_date_param=fb_date.isoformat(),
                    finance_valo_param=finance_valo_param,
                    force_refresh=True,
                    allow_fallback=False,
                )
        except Exception:
            pass

    result = {
        "finance_supports": finance_supports,
        "finance_total_valo": finance_total_valo,
        "finance_total_valo_str": finance_total_valo_str,
        "finance_date_input": finance_date_input,
        "finance_effective_date_display": finance_effective_date_display,
        "finance_effective_date_iso": finance_effective_date_iso,
        "finance_max_date_iso": finance_max_date_iso,
        "finance_max_date_display": finance_max_date_display,
        "finance_valo_input": valo_filter.get("raw"),
    }
    FINANCE_CACHE[cache_key] = {"ts": now_ts, "data": result}
    return result


def _build_client_synthese_context(db: Session, client_id: int) -> dict | None:
    client = db.query(Client).filter(Client.id == client_id).first()

    rh_list = fetch_rh_list(db)
    if not client:
        return None

    civilite = db.execute(
        text(
            """
            SELECT civilite
            FROM etat_civil_client
            WHERE id_client = :cid
            ORDER BY id DESC
            LIMIT 1
            """
        ),
        {"cid": client_id},
    ).scalar()

    historique = (
        db.query(
            HistoriquePersonne.date,
            HistoriquePersonne.valo,
            HistoriquePersonne.mouvement,
            HistoriquePersonne.sicav,
            HistoriquePersonne.perf_sicav_52,
            HistoriquePersonne.volat,
            HistoriquePersonne.annee,
        )
        .filter(HistoriquePersonne.id == client_id)
        .order_by(HistoriquePersonne.date)
        .all()
    )

    selected_dt = None

    history_labels: list[str] = []
    history_values: list[float] = []
    history_recent: list[dict] = []
    cumul = 0.0
    depots_total = 0.0
    retraits_total = 0.0
    last_valo = 0.0
    first_history_date = None
    last_history_date = None
    duree_historique_str = "-"
    overall_ann_perf_pct = None
    history_mov_raw: list[float] = []
    history_mov_cum: list[float] = []
    client_sicav_series: list[tuple[str | None, float]] = []
    for row in historique:
        dt = None
        if getattr(row, "date", None):
            try:
                dt = row.date.strftime("%Y-%m-%d")
            except Exception:
                dt = str(row.date)[:10]
            if first_history_date is None:
                first_history_date = row.date
            last_history_date = row.date
        else:
            dt = "N/A"
        history_labels.append(dt)
        val = float(getattr(row, "valo", 0.0) or 0.0)
        mov = float(getattr(row, "mouvement", 0.0) or 0.0)
        history_values.append(val)
        last_valo = val
        history_mov_raw.append(mov)
        cumul += mov
        history_mov_cum.append(cumul)
        try:
            sc = float(getattr(row, "sicav", 0.0) or 0.0)
        except Exception:
            sc = 0.0
        client_sicav_series.append((dt, sc))
        if mov > 0:
            depots_total += mov
        elif mov < 0:
            retraits_total += mov
    for dt, val in list(zip(history_labels, history_values))[-12:]:
        history_recent.append({"date": dt, "valo": _fmt_currency(val) + " €"})

    solde_total = depots_total + retraits_total
    depots_str = _fmt_currency(depots_total)
    retraits_str = _fmt_currency(abs(retraits_total))
    retraits_str_signed = _fmt_currency(retraits_total)
    solde_str = _fmt_currency(solde_total)
    last_valo_str = _fmt_currency(last_valo)

    years_map: dict[int, dict] = {}
    for row in historique:
        year = getattr(row, "annee", None)
        if year is None:
            continue
        try:
            y = int(year)
        except Exception:
            continue
        perf = getattr(row, "perf_sicav_52", None)
        vol = getattr(row, "volat", None)
        prev = years_map.get(y)
        if not prev or (row.date and prev.get("date") and row.date > prev["date"]):
            years_map[y] = {"date": getattr(row, "date", None), "perf": perf, "vol": vol}
    years_sorted = sorted(years_map.keys())
    perf_vol_years = [int(y) for y in years_sorted]
    perf_vol_years = [int(y) for y in years_sorted]

    def _to_pct(value):
        if value is None:
            return None
        try:
            n = float(value)
        except Exception:
            return None
        if abs(n) <= 1:
            n *= 100.0
        return n

    def _to_return_decimal(value):
        if value is None:
            return 0.0
        try:
            n = float(value)
        except Exception:
            return 0.0
        if abs(n) <= 1:
            return n
        return n / 100.0

    ann_perf = [_to_pct(years_map[y]["perf"]) for y in years_sorted]
    ann_vol = [_to_pct(years_map[y]["vol"]) for y in years_sorted]
    perf_vol_years = [int(y) for y in years_sorted]
    last_row_hist = historique[-1] if historique else None
    last_perf_pct = _to_pct(getattr(last_row_hist, "perf_sicav_52", None) if last_row_hist else None)
    last_vol_pct = _to_pct(getattr(last_row_hist, "volat", None) if last_row_hist else None)
    current_srri = int(getattr(last_row_hist, "SRRI", None)) if (last_row_hist and getattr(last_row_hist, "SRRI", None) is not None) else None
    client_srri = int(getattr(client, "SRRI", None)) if getattr(client, "SRRI", None) is not None else None

    history_chart_svg = _build_svg_line_chart(history_labels, history_values, "Historique des valorisations")
    perf_chart_svg = None
    if ann_perf:
        perf_labels = [str(y) for y in years_sorted]
        perf_chart_svg = _build_svg_bar_chart(perf_labels[:6], ann_perf[:6], "Performance annuelle (%)")

    # Contrats (affaires)
    subq_aff = (
        db.query(
            HistoriqueAffaire.id.label("affaire_id"),
            func.max(HistoriqueAffaire.date).label("last_date")
        )
        .group_by(HistoriqueAffaire.id)
        .subquery()
    )
    affaires_rows = (
        db.query(
            Affaire.id.label("id"),
            Affaire.ref,
            Affaire.SRRI,
            Affaire.date_cle,
            HistoriqueAffaire.valo.label("last_valo"),
            HistoriqueAffaire.perf_sicav_52.label("last_perf"),
            HistoriqueAffaire.volat.label("last_volat"),
        )
        .join(subq_aff, subq_aff.c.affaire_id == Affaire.id)
        .join(
            HistoriqueAffaire,
            (HistoriqueAffaire.id == subq_aff.c.affaire_id) &
            (HistoriqueAffaire.date == subq_aff.c.last_date)
        )
        .filter(Affaire.id_personne == client_id)
        .all()
    )

    if selected_dt:
        mouvements_rows = db.execute(
            text(
                """
                SELECT h.id AS id_affaire,
                       SUM(COALESCE(h.mouvement, 0)) AS total_mvt
                FROM mariadb_historique_affaire_w h
                WHERE h.date <= :sel
                GROUP BY h.id
                """
            ),
            {"sel": selected_dt},
        ).fetchall()
    else:
        mouvements_rows = db.execute(
            text(
                """
                SELECT h.id AS id_affaire,
                       SUM(COALESCE(h.mouvement, 0)) AS total_mvt
                FROM mariadb_historique_affaire_w h
                GROUP BY h.id
                """
            )
        ).fetchall()

    mouvements_map: dict[int, float] = {}
    for row in mouvements_rows:
        mapping = row._mapping if hasattr(row, "_mapping") else None
        rid_raw = mapping.get("id_affaire") if mapping else (row[0] if len(row) > 0 else None)
        total_raw = mapping.get("total_mvt") if mapping else (row[1] if len(row) > 1 else None)
        try:
            rid = int(rid_raw) if rid_raw is not None else None
        except Exception:
            rid = None
        if rid is None:
            continue
        try:
            mouvements_map[rid] = float(total_raw or 0.0)
        except Exception:
            mouvements_map[rid] = 0.0

    def _to_pct(x):
        if x is None:
            return None
        try:
            v = float(x)
        except Exception:
            return None
        if abs(v) <= 1:
            v *= 100.0
        return v

    contracts = []
    contracts_total_valo = 0.0
    for row in affaires_rows:
        valo_raw = float(row.last_valo or 0.0)
        contracts_total_valo += valo_raw
        contracts.append({
            "ref": row.ref,
            "srri": row.SRRI,
            "valo": _fmt_currency(valo_raw),
            "perf": _to_pct(row.last_perf),
            "vol": _to_pct(row.last_volat),
        })
    contracts_total_valo_str = _fmt_currency(contracts_total_valo)
    nb_contrats_total = len(affaires_rows)
    nb_contrats_fermes = sum(1 for r in affaires_rows if getattr(r, "date_cle", None))
    nb_contrats_ouverts = max(0, nb_contrats_total - nb_contrats_fermes)

    last_support_date = db.execute(text("SELECT MAX(date) FROM mariadb_historique_support_w")).scalar()
    supports_rows = []
    if last_support_date:
        supports_rows = db.execute(
            text(
                """
                SELECT s.code_isin,
                       s.nom,
                       s.cat_principale,
                       s.cat_geo,
                       s.SRRI,
                       SUM(h.valo) AS total_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_support s ON s.id = h.id_support
                JOIN mariadb_affaires a ON a.id = h.id_source
                WHERE a.id_personne = :cid
                  AND DATE(h.date) = DATE(:last_date)
                GROUP BY s.code_isin, s.nom, s.cat_principale, s.cat_geo, s.SRRI
                HAVING SUM(h.valo) > 0
                ORDER BY total_valo DESC
                """
            ),
            {"cid": client_id, "last_date": last_support_date},
        ).fetchall()

    supports = []
    supports_total_valo = 0.0
    for row in supports_rows:
        m = getattr(row, "_mapping", row)
        valo_raw = float(m.get("total_valo") or 0)
        supports_total_valo += valo_raw
        supports.append({
            "code_isin": m.get("code_isin"),
            "nom": m.get("nom"),
            "cat": m.get("cat_principale") or m.get("cat_geo"),
            "cat_principale": m.get("cat_principale"),
            "cat_geo": m.get("cat_geo"),
            "srri": m.get("SRRI"),
            "valo": _fmt_currency(valo_raw),
            "valo_raw": valo_raw,
        })

    supports_total_valo_str = _fmt_currency(supports_total_valo)
    nb_supports_actifs = sum(1 for s in supports if (s.get("valo_raw") or 0) > 0)
    if nb_supports_actifs == 0:
        nb_supports_actifs = len(supports)

    def _compute_support_breakdown(items: list[dict], field_key: str) -> list[dict]:
        total = sum(float(it.get("valo_raw") or 0.0) for it in items)
        if total <= 0:
            return []
        agg: dict[str, float] = {}
        for it in items:
            try:
                val = float(it.get("valo_raw") or 0.0)
            except Exception:
                val = 0.0
            label = it.get(field_key)
            if not label:
                label = "Non renseigné"
            agg[str(label)] = agg.get(str(label), 0.0) + val
        breakdown = []
        for label, val in sorted(agg.items(), key=lambda kv: kv[1], reverse=True):
            pct = (val / total * 100.0) if total > 0 else None
            breakdown.append({
                "label": label,
                "valo": _fmt_currency(val),
                "valo_raw": val,
                "percent": pct,
            })
        return breakdown

    supports_category_breakdown = _compute_support_breakdown(supports, "cat_principale")
    supports_geo_breakdown = _compute_support_breakdown(supports, "cat_geo")
    supports_category_chart_png = None
    if supports_category_breakdown:
        cat_labels = [row.get('label') for row in supports_category_breakdown]
        cat_values = [float(row.get('valo_raw') or 0.0) for row in supports_category_breakdown]
        supports_category_chart_png = _build_matplotlib_horizontal_bar_chart(cat_labels, cat_values, "Répartition par catégorie principale", "Valorisation (€)")
    supports_geo_chart_png = None
    if supports_geo_breakdown:
        geo_labels = [row.get('label') for row in supports_geo_breakdown]
        geo_values = [float(row.get('valo_raw') or 0.0) for row in supports_geo_breakdown]
        supports_geo_chart_png = _build_matplotlib_horizontal_bar_chart(geo_labels, geo_values, "Répartition géographique", "Valorisation (€)")

    # Courbes valorisation & mouvements par année (pour restitution tabulaire)
    curves_year_map: dict[int, dict] = {}
    for idx, row in enumerate(historique):
        dt = getattr(row, "date", None)
        if dt is None:
            continue
        year = dt.year
        mov_cum_val = history_mov_cum[idx] if idx < len(history_mov_cum) else 0.0
        entry = curves_year_map.get(year)
        if entry is None or (entry.get("date") and dt > entry["date"]):
            curves_year_map[year] = {
                "date": dt,
                "valorisation": float(getattr(row, "valo", 0.0) or 0.0),
                "mouvements": float(mov_cum_val or 0.0),
            }
    curves_year_data = []
    curves_values_all: list[float] = []
    curves_year_labels: list[str] = []
    curves_valorisation_series: list[float] = []
    curves_mouvements_series: list[float] = []
    for year in sorted(curves_year_map.keys()):
        entry = curves_year_map[year]
        val_valo = entry.get("valorisation", 0.0) or 0.0
        val_mov = entry.get("mouvements", 0.0) or 0.0
        curves_values_all.extend([val_valo, val_mov])
        curves_year_labels.append(str(year))
        curves_valorisation_series.append(val_valo)
        curves_mouvements_series.append(val_mov)
        curves_year_data.append({
            "year": year,
            "valorisation": _fmt_currency(val_valo),
            "valorisation_raw": val_valo,
            "mouvements": _fmt_currency(val_mov),
            "mouvements_raw": val_mov,
        })
    curves_value_min = min(curves_values_all) if curves_values_all else 0.0
    curves_value_max = max(curves_values_all) if curves_values_all else 0.0
    curves_value_min_str = _fmt_currency(curves_value_min)
    curves_value_max_str = _fmt_currency(curves_value_max)
    curves_chart_png = _build_matplotlib_two_line_chart(
        curves_year_labels,
        curves_valorisation_series,
        curves_mouvements_series,
        "Valorisation",
        "Cumul mouvements",
        "Valorisation & flux cumulés par année",
    )
    curves_chart_svg = None
    if not curves_chart_png:
        curves_chart_svg = _build_svg_two_line_chart(
            curves_year_labels,
            curves_valorisation_series,
            curves_mouvements_series,
            "Valorisation",
            "Cumul mouvements",
            "Valorisation & flux cumulés par année",
        )
    perf_vol_chart_png = _build_matplotlib_perf_vol_chart(perf_vol_years, ann_perf, ann_vol)

    allocation_reference_name: str | None = None
    try:
        row = db.execute(
            text(
                """
                SELECT k.niveau_id
                FROM KYC_Client_Risque k
                WHERE k.client_id = :cid
                ORDER BY k.date_saisie DESC, k.id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        niveau_id = row[0] if row else None
        if niveau_id is not None:
            try:
                niveau_id = int(niveau_id)
            except Exception:
                niveau_id = None
        if niveau_id is not None:
            row_alloc = db.execute(
                text(
                    """
                    SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom
                    FROM allocation_risque ar
                    LEFT JOIN allocations a ON a.nom = ar.allocation_name
                    WHERE ar.risque_id = :rid
                    ORDER BY ar.date_attribution DESC, ar.id DESC
                    LIMIT 1
                    """
                ),
                {"rid": niveau_id},
            ).fetchone()
            if row_alloc:
                allocation_reference_name = row_alloc[0]
    except Exception:
        allocation_reference_name = None

    allocation_series_pairs: list[tuple[str, float]] = []
    alloc_min_dt = None
    alloc_max_dt = None
    if allocation_reference_name:
        try:
            rows_alloc = (
                db.query(Allocation.date, Allocation.sicav)
                .filter(Allocation.nom == allocation_reference_name)
                .order_by(Allocation.date.asc())
                .all()
            )
            for d, val in rows_alloc:
                try:
                    dstr = d.strftime("%Y-%m-%d") if d else None
                except Exception:
                    dstr = str(d)[:10] if d else None
                if not dstr:
                    continue
                try:
                    sval = float(val or 0.0)
                except Exception:
                    sval = 0.0
                allocation_series_pairs.append((dstr, sval))
                try:
                    d_dt = datetime.strptime(dstr, "%Y-%m-%d")
                    if alloc_min_dt is None or d_dt < alloc_min_dt:
                        alloc_min_dt = d_dt
                    if alloc_max_dt is None or d_dt > alloc_max_dt:
                        alloc_max_dt = d_dt
                except Exception:
                    continue
        except Exception:
            allocation_series_pairs = []

    def _within_alloc_range(d: str) -> bool:
        if not d:
            return False
        if alloc_min_dt is None or alloc_max_dt is None:
            return True
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
        except Exception:
            return False
        return alloc_min_dt <= dt <= alloc_max_dt

    client_series_pairs = [(d, v) for d, v in client_sicav_series if d and _within_alloc_range(d)]

    def _to_base100(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
        base = None
        for _, val in series:
            try:
                fval = float(val)
            except Exception:
                continue
            if fval != 0:
                base = fval
                break
        if base is None or base == 0:
            return series
        normalized: list[tuple[str, float]] = []
        for d, val in series:
            try:
                fval = float(val or 0.0)
            except Exception:
                fval = 0.0
            normalized.append((d, (fval / base) * 100.0))
        return normalized

    allocation_series_pairs = _to_base100(allocation_series_pairs)
    client_series_pairs = _to_base100(client_series_pairs)
    alloc_compare_chart_png = _build_matplotlib_alloc_compare_chart(
        client_series_pairs,
        allocation_series_pairs,
        allocation_reference_name or "Allocation",
    )

    top_supports = supports[:5]
    supports_chart_svg = None
    if top_supports:
        supports_chart_svg = _build_svg_bar_chart(
            [s.get("code_isin") or "?" for s in top_supports],
            [s.get("valo_raw") or 0 for s in top_supports],
            "Top supports par valorisation",
        )
    for s in supports:
        s.pop("valo_raw", None)

    # Reporting pluriannuel (tableau solde / valorisation / perfs)
    year_data: dict[int, dict] = {}
    for row in historique:
        date = getattr(row, "date", None)
        year = getattr(row, "annee", None)
        if year is None and date:
            try:
                year = date.year
            except Exception:
                year = None
        if year is None:
            continue
        info = year_data.setdefault(year, {"mov": 0.0, "last": None, "vols": []})
        try:
            info["mov"] += float(getattr(row, "mouvement", 0.0) or 0.0)
        except Exception:
            pass
        vol = getattr(row, "volat", None)
        if vol is not None:
            try:
                info["vols"].append(float(vol))
            except Exception:
                pass
        last = info.get("last")
        if last is None:
            info["last"] = row
        else:
            try:
                cur_dt = getattr(row, "date", None)
                last_dt = getattr(last, "date", None)
                if cur_dt and last_dt:
                    if cur_dt > last_dt:
                        info["last"] = row
                elif cur_dt and not last_dt:
                    info["last"] = row
            except Exception:
                pass

    reporting_years = []
    cum_factor = 1.0
    year_count = 0
    for year in sorted(year_data.keys()):
        info = year_data[year]
        mov = info.get("mov", 0.0) or 0.0
        last_row = info.get("last")
        last_val = float(getattr(last_row, "valo", 0.0) or 0.0) if last_row else 0.0
        perf_raw = getattr(last_row, "perf_sicav_52", None) if last_row else None
        perf_pct = _to_pct(perf_raw)
        ann_return = _to_return_decimal(perf_raw)
        try:
            cum_factor *= (1.0 + ann_return)
        except Exception:
            pass
        year_count += 1
        cum_perf_pct = (cum_factor - 1.0) * 100.0
        try:
            ann_perf_pct = ((cum_factor ** (1.0 / max(1, year_count))) - 1.0) * 100.0
        except Exception:
            ann_perf_pct = None
        vols_pct = [_to_pct(v) for v in info.get("vols", []) if _to_pct(v) is not None]
        avg_vol_pct = sum(vols_pct) / len(vols_pct) if vols_pct else None
        reporting_years.append({
            "year": year,
            "solde_str": _fmt_currency(mov),
            "last_valo_str": _fmt_currency(last_val),
            "cum_perf_pct": cum_perf_pct,
            "ann_perf_pct": ann_perf_pct,
            "avg_vol_pct": avg_vol_pct,
        })

    if first_history_date and last_history_date:
        total_months = (last_history_date.year - first_history_date.year) * 12 + (last_history_date.month - first_history_date.month)
        if last_history_date.day < first_history_date.day:
            total_months -= 1
        years_span_int = total_months // 12
        months_span = total_months % 12
        if years_span_int <= 0 and months_span <= 0:
            try:
                days_span = max(0, (last_history_date - first_history_date).days)
            except Exception:
                days_span = 0
            duree_historique_str = f"{days_span} jours"
        else:
            parts = []
            if years_span_int > 0:
                parts.append(f"{years_span_int} ans")
            if months_span > 0:
                parts.append(f"{months_span} mois")
            duree_historique_str = ", ".join(parts) if parts else "-"
        try:
            span_years = max(1e-6, (last_history_date - first_history_date).days / 365.25)
            overall_ann_perf_pct = ((float(cum_factor) ** (1.0 / span_years)) - 1.0) * 100.0
        except Exception:
            overall_ann_perf_pct = None
    else:
        duree_historique_str = "-"

    responsable_label = "—"
    if getattr(client, "commercial_id", None):
        try:
            target_id = int(client.commercial_id)
        except Exception:
            target_id = None
        if target_id is not None:
            for rh in rh_list:
                try:
                    if int(rh.get("id")) == target_id:
                        prenom = rh.get("prenom") or ""
                        nom = rh.get("nom") or ""
                        label = f"{prenom} {nom}".strip()
                        responsable_label = label if label else "—"
                        break
                except Exception:
                    continue

    synthese_card = {
        "nb_contrats_ouverts": nb_contrats_ouverts,
        "nb_contrats_fermes": nb_contrats_fermes,
        "duree_historique": duree_historique_str,
        "responsable": responsable_label,
    }
    valorisation_card = {
        "last_valo": last_valo_str,
        "depots": depots_str,
        "retraits": retraits_str_signed,
        "solde": solde_str,
    }
    indicateurs_card = {
        "nb_contrats_ouverts": nb_contrats_ouverts,
        "nb_supports_actifs": nb_supports_actifs,
        "last_perf_pct": last_perf_pct,
        "last_vol_pct": last_vol_pct,
        "client_srri": client_srri,
        "current_srri": current_srri,
        "overall_ann_perf_pct": overall_ann_perf_pct,
    }

    report_date = datetime.utcnow()
    try:
        esg_comparison_alloc_value = esg_comparison_alloc
    except NameError:
        esg_comparison_alloc_value = None
    try:
        esg_comparison_rows_value = esg_comparison_rows
    except NameError:
        esg_comparison_rows_value = None
    try:
        esg_comparison_chart_value = esg_comparison_chart_png
    except NameError:
        esg_comparison_chart_value = None
    try:
        esg_unmatched_fields_value = esg_unmatched_fields
    except NameError:
        esg_unmatched_fields_value = None
    try:
        esg_default_fields_value = esg_default_fields
    except NameError:
        esg_default_fields_value = None
    try:
        esg_field_labels_value = esg_field_labels_client
    except NameError:
        esg_field_labels_value = {}

    return {
        "client": client,
        "civilite": civilite,
        "report_date": report_date,
        "report_date_str": report_date.strftime("%d/%m/%Y"),
        "summary": {
            "total_valo": last_valo_str,
            "depots": depots_str,
            "retraits": _fmt_currency(abs(retraits_total)),
            "solde": solde_str,
        },
        "contracts_total_valo": contracts_total_valo_str,
        "supports_total_valo": supports_total_valo_str,
        "history_recent": history_recent,
        "history_chart_svg": history_chart_svg,
        "performance_chart_svg": perf_chart_svg,
        "supports_chart_svg": supports_chart_svg,
        "curves_year_data": curves_year_data,
        "curves_value_min": curves_value_min,
        "curves_value_max": curves_value_max,
        "curves_value_min_str": curves_value_min_str,
        "curves_value_max_str": curves_value_max_str,
        "curves_chart_png": curves_chart_png,
        "curves_chart_svg": curves_chart_svg,
        "perf_vol_chart_png": perf_vol_chart_png,
        "alloc_compare_chart_png": alloc_compare_chart_png,
        "allocation_reference_name": allocation_reference_name,
        "esg_comparison_alloc": esg_comparison_alloc_value,
        "esg_comparison_rows": esg_comparison_rows_value,
        "esg_comparison_chart_png": esg_comparison_chart_value,
        "esg_unmatched_fields": esg_unmatched_fields_value,
        "esg_default_fields": esg_default_fields_value,
        "esg_field_labels": esg_field_labels_value,
        "contracts": contracts,
        "supports": supports,
        "supports_category_breakdown": supports_category_breakdown,
        "supports_geo_breakdown": supports_geo_breakdown,
        "supports_category_chart_png": supports_category_chart_png,
        "supports_geo_chart_png": supports_geo_chart_png,
        "history_years": [str(y) for y in years_sorted],
        "history_perf": ann_perf,
        "history_vol": ann_vol,
        "reporting_years": reporting_years,
        "totals": {
            "total_valo": last_valo_str,
            "depots": depots_str,
            "retraits": _fmt_currency(abs(retraits_total)),
            "solde": solde_str,
        },
        "synthese_card": synthese_card,
        "valorisation_card": valorisation_card,
        "indicateurs_card": indicateurs_card,
    }
def _build_client_adequation_context(db: Session, client_id: int) -> dict | None:
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return None

    report_date = datetime.utcnow()
    report_logo_png = None
    try:
        logo_path = Path(__file__).resolve().parents[1] / "logo" / "logo.png"
        if logo_path.exists():
            report_logo_png = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except Exception:
        report_logo_png = None

    etat_civil_row = None
    try:
        row = db.execute(
            text(
                """
                SELECT civilite,
                       situation_familiale,
                       date_naissance,
                       lieu_naissance,
                       nationalite,
                       profession,
                       commentaire
                FROM etat_civil_client
                WHERE id_client = :cid
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            data = row._mapping if hasattr(row, "_mapping") else {}
            etat_civil_row = {
                "civilite": data.get("civilite"),
                "situation_familiale": data.get("situation_familiale"),
                "date_naissance": data.get("date_naissance"),
                "lieu_naissance": data.get("lieu_naissance"),
                "nationalite": data.get("nationalite"),
                "profession": data.get("profession"),
                "commentaire": data.get("commentaire"),
            }
    except Exception:
        etat_civil_row = None

    situations_matrimoniales: list[dict] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT nb_enfants,
                       situation_id,
                       sm.libelle AS situation_libelle,
                       date_saisie,
                       date_expiration
                FROM KYC_Client_Situation_Matrimoniale m
                LEFT JOIN ref_situation_matrimoniale sm ON sm.id = m.situation_id
                WHERE m.client_id = :cid
                ORDER BY (m.date_saisie IS NULL), m.date_saisie DESC, m.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            situations_matrimoniales.append(
                {
                    "nb_enfants": data.get("nb_enfants") or 0,
                    "situation_id": data.get("situation_id"),
                    "situation_libelle": data.get("situation_libelle"),
                    "date_saisie": data.get("date_saisie"),
                    "date_expiration": data.get("date_expiration"),
                }
            )
    except Exception:
        situations_matrimoniales = []

    nb_enfants_latest = None
    try:
        row = db.execute(
            text(
                """
                SELECT nb_enfants
                FROM KYC_Client_Situation_Matrimoniale
                WHERE client_id = :cid
                ORDER BY (date_saisie IS NULL), date_saisie DESC, id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row is not None:
            nb_enfants_latest = row[0]
    except Exception:
        nb_enfants_latest = None

    synthese_latest = None
    try:
        row = db.execute(
            text(
                """
                SELECT total_revenus,
                       total_charges,
                       total_actif,
                       total_passif,
                       commentaire
                FROM KYC_Client_Synthese
                WHERE client_id = :cid
                ORDER BY date_saisie DESC, id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            data = row._mapping if hasattr(row, "_mapping") else {}
            synthese_latest = {
                "total_revenus": data.get("total_revenus"),
                "total_charges": data.get("total_charges"),
                "total_actif": data.get("total_actif"),
                "total_passif": data.get("total_passif"),
                "commentaire": data.get("commentaire"),
            }
    except Exception:
        synthese_latest = None

    risque_latest = None
    try:
        row = db.execute(
            text(
                """
                SELECT
                  r.libelle AS niveau_risque,
                  k.niveau_id AS niveau_id,
                  k.duree AS horizon_placement,
                  k.experience,
                  k.connaissance,
                  k.commentaire,
                  k.confirmation_client
                FROM KYC_Client_Risque k
                JOIN ref_niveau_risque r ON k.niveau_id = r.id
                WHERE k.client_id = :cid
                ORDER BY k.date_saisie DESC, k.id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            risque_latest = dict(row._mapping)
    except Exception:
        risque_latest = None

    contrat_choisi_nom = None
    contrat_choisi_societe = None
    try:
        row = db.execute(
            text(
                """
                SELECT g.nom_contrat, COALESCE(s.nom,'') AS societe_nom
                FROM KYC_contrat_choisi k
                LEFT JOIN mariadb_affaires_generique g ON g.id = k.id_contrat
                LEFT JOIN mariadb_societe s ON s.id = g.id_societe
                WHERE k.id_client = :cid
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            data = row._mapping if hasattr(row, "_mapping") else {}
            contrat_choisi_nom = data.get("nom_contrat") or row[0]
            contrat_choisi_societe = data.get("societe_nom") if "societe_nom" in data else (row[1] if len(row) > 1 else None)
    except Exception:
        contrat_choisi_nom = None
        contrat_choisi_societe = None

    adequation_contracts: list[dict] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT g.id,
                       g.nom_contrat,
                       COALESCE(s.nom, '') AS societe_nom,
                       COALESCE(g.frais_gestion_assureur,0) + COALESCE(g.frais_gestion_courtier,0) AS total_frais
                FROM mariadb_affaires_generique g
                LEFT JOIN mariadb_societe s ON s.id = g.id_societe
                WHERE COALESCE(g.actif, 1) = 1
                ORDER BY s.nom, g.nom_contrat
                """
            )
        ).fetchall()
        adequation_contracts = [dict(r._mapping) for r in rows]
    except Exception:
        adequation_contracts = []

    adequation_objectifs: list[dict] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT o.objectif_id,
                       ro.libelle AS objectif_libelle,
                       o.horizon_investissement,
                       o.commentaire,
                       COALESCE(o.niveau_id, 9999) AS _niv
                FROM KYC_Client_Objectifs o
                LEFT JOIN ref_objectif ro ON ro.id = o.objectif_id
                WHERE o.client_id = :cid
                ORDER BY _niv, ro.libelle, o.id
                """
            ),
            {"cid": client_id},
        ).fetchall()
        adequation_objectifs = [dict(r._mapping) for r in rows]
    except Exception:
        adequation_objectifs = []

    allocation_reference_name = None
    adequation_allocation_html = None
    if risque_latest and risque_latest.get("niveau_id") is not None:
        try:
            row = db.execute(
                text(
                    """
                    SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom,
                           ar.texte
                    FROM allocation_risque ar
                    LEFT JOIN allocations a ON a.nom = ar.allocation_name
                    WHERE ar.risque_id = :rid
                    ORDER BY ar.date_attribution DESC, ar.id DESC
                    LIMIT 1
                    """
                ),
                {"rid": int(risque_latest.get("niveau_id"))},
            ).fetchone()
            if row:
                allocation_reference_name = row[0]
                texte = row[1]
                if texte:
                    try:
                        import html as _html
                        import re

                        esc = _html.escape(str(texte))
                        esc = re.sub(r"^###\s+(.*)$", r"<h5>\1</h5>", esc, flags=re.M)
                        esc = re.sub(r"^##\s+(.*)$", r"<h4>\1</h4>", esc, flags=re.M)
                        esc = re.sub(r"^#\s+(.*)$", r"<h3>\1</h3>", esc, flags=re.M)
                        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
                        esc = re.sub(r"\*(.+?)\*", r"<em>\1</em>", esc)
                        lines = esc.split("\n")
                        out: list[str] = []
                        in_ul = False
                        for line in lines:
                            if re.match(r"^\s*-\s+", line):
                                if not in_ul:
                                    out.append("<ul>")
                                    in_ul = True
                                out.append("<li>" + re.sub(r"^\s*-\s+", "", line) + "</li>")
                            else:
                                if in_ul:
                                    out.append("</ul>")
                                    in_ul = False
                                if line.strip():
                                    out.append("<p>" + line + "</p>")
                        if in_ul:
                            out.append("</ul>")
                        adequation_allocation_html = "\n".join(out)
                    except Exception:
                        adequation_allocation_html = str(texte).replace("\n", "<br/>")
        except Exception:
            adequation_allocation_html = None

    revenues = float(synthese_latest.get("total_revenus") or 0.0) if synthese_latest else 0.0
    charges = float(synthese_latest.get("total_charges") or 0.0) if synthese_latest else 0.0
    actif = float(synthese_latest.get("total_actif") or 0.0) if synthese_latest else 0.0
    passif = float(synthese_latest.get("total_passif") or 0.0) if synthese_latest else 0.0
    budget = revenues - charges
    patrimoine_net = actif - passif

    civilite = None
    situation_familiale = None
    if etat_civil_row:
        civilite = etat_civil_row.get("civilite")
        situation_familiale = etat_civil_row.get("situation_familiale")
    if not civilite:
        civilite = getattr(client, "civilite", None)

    nb_enfants = nb_enfants_latest
    if nb_enfants is None and situations_matrimoniales:
        nb_enfants = situations_matrimoniales[0].get("nb_enfants")
    if nb_enfants is None:
        nb_enfants = 0

    contexte = {
        "client": client,
        "report_date": report_date,
        "report_date_str": report_date.strftime("%d/%m/%Y"),
        "report_logo_png": report_logo_png,
        "civilite": civilite,
        "situation_familiale": situation_familiale,
        "nb_enfants": nb_enfants,
        "revenus_total": revenues,
        "charges_total": charges,
        "actif_total": actif,
        "passif_total": passif,
        "budget_disponible": budget,
        "patrimoine_net": patrimoine_net,
        "revenus_total_str": _fmt_currency(revenues),
        "charges_total_str": _fmt_currency(charges),
        "actif_total_str": _fmt_currency(actif),
        "passif_total_str": _fmt_currency(passif),
        "budget_disponible_str": _fmt_currency(budget),
        "patrimoine_net_str": _fmt_currency(patrimoine_net),
        "risque_latest": risque_latest,
        "horizon_placement": risque_latest.get("horizon_placement") if risque_latest else None,
        "niveau_risque": risque_latest.get("niveau_risque") if risque_latest else None,
        "experience": risque_latest.get("experience") if risque_latest else None,
        "connaissance": risque_latest.get("connaissance") if risque_latest else None,
        "allocation_reference_name": allocation_reference_name,
        "contrat_choisi_nom": contrat_choisi_nom,
        "contrat_choisi_societe": contrat_choisi_societe,
        "adequation_contracts": adequation_contracts,
        "adequation_objectifs": adequation_objectifs,
        "adequation_allocation_html": adequation_allocation_html,
        "synthese_latest": synthese_latest,
        "etat_civil_latest": etat_civil_row,
        "kyc_etat_civil": etat_civil_row,
        "kyc_situations_matrimoniales": situations_matrimoniales,
    }
    return contexte
def _build_affaire_synthese_context(db: Session, affaire_id: int) -> dict | None:
    affaire = db.query(Affaire).filter(Affaire.id == affaire_id).first()
    if not affaire:
        return None

    client = None
    civilite = None
    if getattr(affaire, "id_personne", None):
        client = db.query(Client).filter(Client.id == affaire.id_personne).first()
        try:
            civilite = db.execute(
                text(
                    """
                    SELECT civilite
                    FROM etat_civil_client
                    WHERE id_client = :cid
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {"cid": affaire.id_personne},
            ).scalar()
        except Exception:
            civilite = None

    client_id = getattr(client, "id", None) if client else getattr(affaire, "id_personne", None)
    rh_list = fetch_rh_list(db)

    historique = (
        db.query(
            HistoriqueAffaire.date,
            HistoriqueAffaire.valo,
            HistoriqueAffaire.mouvement,
            HistoriqueAffaire.sicav,
            HistoriqueAffaire.perf_sicav_52,
            HistoriqueAffaire.volat,
            HistoriqueAffaire.annee,
        )
        .filter(HistoriqueAffaire.id == affaire_id)
        .order_by(HistoriqueAffaire.date.asc())
        .all()
    )

    history_labels: list[str] = []
    history_values: list[float] = []
    history_recent: list[dict] = []
    history_mov_cum: list[float] = []
    client_sicav_series: list[tuple[str | None, float]] = []
    depots_total = 0.0
    retraits_total = 0.0
    cumul = 0.0
    last_valo = 0.0
    first_history_date = None
    last_history_date = None

    for row in historique:
        dt_str = None
        if getattr(row, "date", None):
            try:
                dt_str = row.date.strftime("%Y-%m-%d")
            except Exception:
                dt_str = str(row.date)[:10]
            if first_history_date is None:
                first_history_date = row.date
            last_history_date = row.date
        else:
            dt_str = "N/A"
        history_labels.append(dt_str)
        try:
            val = float(getattr(row, "valo", 0.0) or 0.0)
        except Exception:
            val = 0.0
        history_values.append(val)
        last_valo = val
        mouvement_raw = float(getattr(row, "mouvement", 0.0) or 0.0)
        cumul += mouvement_raw
        history_mov_cum.append(cumul)
        if mouvement_raw > 0:
            depots_total += mouvement_raw
        elif mouvement_raw < 0:
            retraits_total += mouvement_raw
        try:
            sicav_val = float(getattr(row, "sicav", 0.0) or 0.0)
        except Exception:
            sicav_val = 0.0
        client_sicav_series.append((dt_str, sicav_val))

    for dt, val in list(zip(history_labels, history_values))[-12:]:
        history_recent.append({"date": dt, "valo": _fmt_currency(val) + " €"})

    solde_total = depots_total + retraits_total
    depots_str = _fmt_currency(depots_total)
    retraits_str_signed = _fmt_currency(retraits_total)
    solde_str = _fmt_currency(solde_total)
    last_valo_str = _fmt_currency(last_valo)

    years_map: dict[int, dict] = {}
    for row in historique:
        year = getattr(row, "annee", None)
        if year is None:
            continue
        try:
            y = int(year)
        except Exception:
            continue
        perf = getattr(row, "perf_sicav_52", None)
        vol = getattr(row, "volat", None)
        prev = years_map.get(y)
        if not prev or (row.date and prev.get("date") and row.date > prev["date"]):
            years_map[y] = {"date": getattr(row, "date", None), "perf": perf, "vol": vol}

    def _to_pct(value):
        if value is None:
            return None
        try:
            n = float(value)
        except Exception:
            return None
        if abs(n) <= 1:
            n *= 100.0
        return n

    def _to_return_decimal(value):
        if value is None:
            return 0.0
        try:
            n = float(value)
        except Exception:
            return 0.0
        if abs(n) <= 1:
            return n
        return n / 100.0

    years_sorted = sorted(years_map.keys())
    perf_vol_years = [int(y) for y in years_sorted]
    ann_perf = [_to_pct(years_map[y]["perf"]) for y in years_sorted]
    ann_vol = [_to_pct(years_map[y]["vol"]) for y in years_sorted]
    last_row_hist = historique[-1] if historique else None
    last_perf_pct = _to_pct(getattr(last_row_hist, "perf_sicav_52", None) if last_row_hist else None)
    last_vol_pct = _to_pct(getattr(last_row_hist, "volat", None) if last_row_hist else None)

    def _srri_from_vol(vol_value):
        if vol_value is None:
            return None
        try:
            v = float(vol_value)
        except Exception:
            return None
        if abs(v) <= 1:
            v *= 100.0
        if v <= 0.5:
            return 1
        if v <= 2:
            return 2
        if v <= 5:
            return 3
        if v <= 10:
            return 4
        if v <= 15:
            return 5
        if v <= 25:
            return 6
        return 7

    current_srri = _srri_from_vol(getattr(last_row_hist, "volat", None) if last_row_hist else None)
    client_srri = getattr(affaire, "SRRI", None)

    history_chart_svg = _build_svg_line_chart(history_labels, history_values, "Historique des valorisations")
    perf_chart_svg = None
    if ann_perf:
        perf_labels = [str(y) for y in years_sorted]
        perf_chart_svg = _build_svg_bar_chart(perf_labels[:6], ann_perf[:6], "Performance annuelle (%)")

    contracts = [{
        "ref": getattr(affaire, "ref", None),
        "srri": getattr(affaire, "SRRI", None),
        "valo": _fmt_currency(last_valo),
        "perf": last_perf_pct,
        "vol": last_vol_pct,
    }]
    contracts_total_valo = last_valo
    contracts_total_valo_str = _fmt_currency(contracts_total_valo)
    nb_contrats_ouverts = 1 if getattr(affaire, "date_cle", None) in (None, "") else 0
    nb_contrats_fermes = 1 - nb_contrats_ouverts

    last_support_date = db.execute(
        text("SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :aid"),
        {"aid": affaire_id},
    ).scalar()

    supports_rows = []
    if last_support_date:
        supports_rows = db.execute(
            text(
                """
                SELECT s.code_isin,
                       s.nom,
                       s.cat_principale,
                       s.cat_geo,
                       s.SRRI,
                       SUM(h.valo) AS total_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_support s ON s.id = h.id_support
                WHERE h.id_source = :aid
                  AND DATE(h.date) = DATE(:last_date)
                GROUP BY s.code_isin, s.nom, s.cat_principale, s.cat_geo, s.SRRI
                HAVING SUM(h.valo) > 0
                ORDER BY total_valo DESC
                """
            ),
            {"aid": affaire_id, "last_date": last_support_date},
        ).fetchall()

    supports = []
    supports_total_valo = 0.0
    for row in supports_rows or []:
        m = getattr(row, "_mapping", row)
        try:
            valo_raw = float(m.get("total_valo") or 0.0)
        except Exception:
            valo_raw = 0.0
        supports_total_valo += valo_raw
        supports.append({
            "code_isin": m.get("code_isin"),
            "nom": m.get("nom"),
            "cat": m.get("cat_principale") or m.get("cat_geo"),
            "cat_principale": m.get("cat_principale"),
            "cat_geo": m.get("cat_geo"),
            "srri": m.get("SRRI"),
            "valo": _fmt_currency(valo_raw),
            "valo_raw": valo_raw,
        })

    supports_total_valo_str = _fmt_currency(supports_total_valo)
    nb_supports_actifs = sum(1 for s in supports if (s.get("valo_raw") or 0) > 0)
    if nb_supports_actifs == 0:
        nb_supports_actifs = len(supports)

    def _compute_support_breakdown(items: list[dict], field_key: str) -> list[dict]:
        total = sum(float(it.get("valo_raw") or 0.0) for it in items)
        if total <= 0:
            return []
        agg: dict[str, float] = {}
        for it in items:
            label = it.get(field_key) or "Non renseigné"
            agg[str(label)] = agg.get(str(label), 0.0) + float(it.get("valo_raw") or 0.0)
        breakdown = []
        for label, val in sorted(agg.items(), key=lambda kv: kv[1], reverse=True):
            pct = (val / total * 100.0) if total > 0 else None
            breakdown.append({
                "label": label,
                "valo": _fmt_currency(val),
                "valo_raw": val,
                "percent": pct,
            })
        return breakdown

    supports_category_breakdown = _compute_support_breakdown(supports, "cat_principale")
    supports_geo_breakdown = _compute_support_breakdown(supports, "cat_geo")

    supports_category_chart_png = None
    if supports_category_breakdown:
        cat_labels = [row.get("label") for row in supports_category_breakdown]
        cat_values = [float(row.get("valo_raw") or 0.0) for row in supports_category_breakdown]
        supports_category_chart_png = _build_matplotlib_horizontal_bar_chart(
            cat_labels,
            cat_values,
            "Répartition par catégorie principale",
            "Valorisation (€)",
        )

    supports_geo_chart_png = None
    if supports_geo_breakdown:
        geo_labels = [row.get("label") for row in supports_geo_breakdown]
        geo_values = [float(row.get("valo_raw") or 0.0) for row in supports_geo_breakdown]
        supports_geo_chart_png = _build_matplotlib_horizontal_bar_chart(
            geo_labels,
            geo_values,
            "Répartition géographique",
            "Valorisation (€)",
        )

    curves_year_map: dict[int, dict] = {}
    for idx, row in enumerate(historique):
        dt = getattr(row, "date", None)
        if dt is None:
            continue
        year = dt.year
        mov_cum_val = history_mov_cum[idx] if idx < len(history_mov_cum) else 0.0
        entry = curves_year_map.get(year)
        if entry is None or (entry.get("date") and dt > entry["date"]):
            curves_year_map[year] = {
                "date": dt,
                "valorisation": float(getattr(row, "valo", 0.0) or 0.0),
                "mouvements": float(mov_cum_val or 0.0),
            }
    curves_year_data: list[dict] = []
    curves_values_all: list[float] = []
    curves_year_labels: list[str] = []
    curves_valorisation_series: list[float] = []
    curves_mouvements_series: list[float] = []
    for year in sorted(curves_year_map.keys()):
        entry = curves_year_map[year]
        val_valo = entry.get("valorisation", 0.0) or 0.0
        val_mov = entry.get("mouvements", 0.0) or 0.0
        curves_values_all.extend([val_valo, val_mov])
        curves_year_labels.append(str(year))
        curves_valorisation_series.append(val_valo)
        curves_mouvements_series.append(val_mov)
        curves_year_data.append({
            "year": year,
            "valorisation": _fmt_currency(val_valo),
            "mouvements": _fmt_currency(val_mov),
        })
    curves_value_min = min(curves_values_all) if curves_values_all else None
    curves_value_max = max(curves_values_all) if curves_values_all else None
    curves_value_min_str = _fmt_currency(curves_value_min) if curves_value_min is not None else None
    curves_value_max_str = _fmt_currency(curves_value_max) if curves_value_max is not None else None
    curves_chart_png = None
    curves_chart_svg = None
    if curves_year_labels and curves_valorisation_series and curves_mouvements_series:
        curves_chart_png = _build_matplotlib_two_line_chart(
            curves_year_labels,
            curves_valorisation_series,
            curves_mouvements_series,
            "Valorisation",
            "Cumul mouvements",
            "Valorisation & flux cumulés par année",
        )
        if not curves_chart_png:
            curves_chart_svg = _build_svg_two_line_chart(
                curves_year_labels,
                curves_valorisation_series,
                curves_mouvements_series,
                "Valorisation",
                "Cumul mouvements",
                "Valorisation & flux cumulés par année",
            )
    else:
        curves_value_min = curves_value_max = None
        curves_value_min_str = curves_value_max_str = None

    perf_vol_chart_png = _build_matplotlib_perf_vol_chart(perf_vol_years, ann_perf, ann_vol)

    supports_chart_svg = None
    top_supports = supports[:5]
    if top_supports:
        supports_chart_svg = _build_svg_bar_chart(
            [s.get("code_isin") or "?" for s in top_supports],
            [s.get("valo_raw") or 0 for s in top_supports],
            "Top supports par valorisation",
        )

    for s in supports:
        s.pop("valo_raw", None)

    allocation_reference_name: str | None = None
    alloc_compare_chart_png = None
    if client_id is not None:
        try:
            row = db.execute(
                text(
                    """
                    SELECT k.niveau_id
                    FROM KYC_Client_Risque k
                    WHERE k.client_id = :cid
                    ORDER BY k.date_saisie DESC, k.id DESC
                    LIMIT 1
                    """
                ),
                {"cid": client_id},
            ).fetchone()
            niveau_id = row[0] if row else None
            if niveau_id is not None:
                try:
                    niveau_id = int(niveau_id)
                except Exception:
                    niveau_id = None
            if niveau_id is not None:
                row_alloc = db.execute(
                    text(
                        """
                        SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom
                        FROM allocation_risque ar
                        LEFT JOIN allocations a ON a.nom = ar.allocation_name
                        WHERE ar.risque_id = :rid
                        ORDER BY ar.date_attribution DESC, ar.id DESC
                        LIMIT 1
                        """
                    ),
                    {"rid": niveau_id},
                ).fetchone()
                if row_alloc:
                    allocation_reference_name = row_alloc[0]
        except Exception:
            allocation_reference_name = None

        allocation_series_pairs: list[tuple[str, float]] = []
        alloc_min_dt = None
        alloc_max_dt = None
        if allocation_reference_name:
            try:
                rows_alloc = (
                    db.query(Allocation.date, Allocation.sicav)
                    .filter(Allocation.nom == allocation_reference_name)
                    .order_by(Allocation.date.asc())
                    .all()
                )
                for d, val in rows_alloc:
                    try:
                        dstr = d.strftime("%Y-%m-%d") if d else None
                    except Exception:
                        dstr = str(d)[:10] if d else None
                    if not dstr:
                        continue
                    try:
                        sval = float(val or 0.0)
                    except Exception:
                        sval = 0.0
                    allocation_series_pairs.append((dstr, sval))
                    try:
                        d_dt = datetime.strptime(dstr, "%Y-%m-%d")
                        if alloc_min_dt is None or d_dt < alloc_min_dt:
                            alloc_min_dt = d_dt
                        if alloc_max_dt is None or d_dt > alloc_max_dt:
                            alloc_max_dt = d_dt
                    except Exception:
                        continue
            except Exception:
                allocation_series_pairs = []

        def _within_alloc_range(d: str) -> bool:
            if not d:
                return False
            if alloc_min_dt is None or alloc_max_dt is None:
                return True
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
            except Exception:
                return False
            return alloc_min_dt <= dt <= alloc_max_dt

        client_series_pairs = [(d, v) for d, v in client_sicav_series if d and _within_alloc_range(d)]

        def _to_base100(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
            base = None
            for _, val in series:
                try:
                    fval = float(val)
                except Exception:
                    continue
                if fval != 0:
                    base = fval
                    break
            if base is None or base == 0:
                return series
            normalized: list[tuple[str, float]] = []
            for d, val in series:
                try:
                    fval = float(val or 0.0)
                except Exception:
                    fval = 0.0
                normalized.append((d, (fval / base) * 100.0))
            return normalized

        allocation_series_pairs = _to_base100(allocation_series_pairs)
        client_series_pairs = _to_base100(client_series_pairs)
        alloc_compare_chart_png = _build_matplotlib_alloc_compare_chart(
            client_series_pairs,
            allocation_series_pairs,
            allocation_reference_name or "Allocation",
        )

    year_data: dict[int, dict] = {}
    for idx, row in enumerate(historique):
        dt = getattr(row, "date", None)
        if dt is None:
            continue
        year = dt.year
        mov_cum_val = history_mov_cum[idx] if idx < len(history_mov_cum) else 0.0
        entry = year_data.get(year)
        if entry is None or (entry.get("date") and dt > entry["date"]):
            year_data[year] = {
                "date": dt,
                "valorisation": float(getattr(row, "valo", 0.0) or 0.0),
                "mouvements": float(mov_cum_val or 0.0),
                "perf": getattr(row, "perf_sicav_52", None),
                "vol": getattr(row, "volat", None),
            }

    reporting_years = []
    cum_factor = 1.0
    year_count = 0
    for year in sorted(year_data.keys()):
        info = year_data[year]
        mov = info.get("mouvements", 0.0) or 0.0
        last_val = info.get("valorisation", 0.0) or 0.0
        perf_pct_val = _to_pct(info.get("perf"))
        ann_return = _to_return_decimal(info.get("perf"))
        try:
            cum_factor *= (1.0 + ann_return)
        except Exception:
            pass
        year_count += 1
        cum_perf_pct = (cum_factor - 1.0) * 100.0
        try:
            ann_perf_pct = ((cum_factor ** (1.0 / max(1, year_count))) - 1.0) * 100.0
        except Exception:
            ann_perf_pct = None
        vols_pct = [_to_pct(info.get("vol"))] if info.get("vol") is not None else []
        avg_vol_pct = sum(v for v in vols_pct if v is not None) / len(vols_pct) if vols_pct else None
        reporting_years.append({
            "year": year,
            "solde_str": _fmt_currency(mov),
            "last_valo_str": _fmt_currency(last_val),
            "cum_perf_pct": cum_perf_pct,
            "ann_perf_pct": ann_perf_pct,
            "avg_vol_pct": avg_vol_pct,
        })

    duree_historique_str = "-"
    overall_ann_perf_pct = None
    if first_history_date and last_history_date:
        total_months = (last_history_date.year - first_history_date.year) * 12 + (last_history_date.month - first_history_date.month)
        if last_history_date.day < first_history_date.day:
            total_months -= 1
        years_span_int = total_months // 12
        months_span = total_months % 12
        if years_span_int <= 0 and months_span <= 0:
            try:
                days_span = max(0, (last_history_date - first_history_date).days)
            except Exception:
                days_span = 0
            duree_historique_str = f"{days_span} jours"
        else:
            parts = []
            if years_span_int > 0:
                parts.append(f"{years_span_int} ans")
            if months_span > 0:
                parts.append(f"{months_span} mois")
            duree_historique_str = ", ".join(parts) if parts else "-"
        try:
            span_years = max(1e-6, (last_history_date - first_history_date).days / 365.25)
            overall_ann_perf_pct = ((float(cum_factor) ** (1.0 / span_years)) - 1.0) * 100.0
        except Exception:
            overall_ann_perf_pct = None

    responsable_label = "—"
    if client and getattr(client, "commercial_id", None):
        try:
            target_id = int(client.commercial_id)
        except Exception:
            target_id = None
        if target_id is not None:
            for rh in rh_list or []:
                try:
                    if int(rh.get("id")) == target_id:
                        prenom = (rh.get("prenom") or "").strip()
                        nom = (rh.get("nom") or "").strip()
                        label = f"{prenom} {nom}".strip()
                        responsable_label = label if label else "—"
                        break
                except Exception:
                    continue

    synthese_card = {
        "nb_contrats_ouverts": nb_contrats_ouverts,
        "nb_contrats_fermes": nb_contrats_fermes,
        "duree_historique": duree_historique_str,
        "responsable": responsable_label,
    }
    valorisation_card = {
        "last_valo": last_valo_str,
        "depots": depots_str,
        "retraits": retraits_str_signed,
        "solde": solde_str,
    }
    indicateurs_card = {
        "nb_contrats_ouverts": nb_contrats_ouverts,
        "nb_supports_actifs": nb_supports_actifs,
        "last_perf_pct": last_perf_pct,
        "last_vol_pct": last_vol_pct,
        "client_srri": client_srri,
        "current_srri": current_srri,
        "overall_ann_perf_pct": overall_ann_perf_pct,
    }

    report_date = datetime.utcnow()

    return {
        "client": client,
        "civilite": civilite,
        "report_date": report_date,
        "report_date_str": report_date.strftime("%d/%m/%Y"),
        "summary": {
            "total_valo": last_valo_str,
            "depots": depots_str,
            "retraits": _fmt_currency(abs(retraits_total)),
            "solde": solde_str,
        },
        "contracts_total_valo": contracts_total_valo_str,
        "supports_total_valo": supports_total_valo_str,
        "history_recent": history_recent,
        "history_chart_svg": history_chart_svg,
        "performance_chart_svg": perf_chart_svg,
        "supports_chart_svg": supports_chart_svg,
        "curves_year_data": curves_year_data,
        "curves_value_min": curves_value_min,
        "curves_value_max": curves_value_max,
        "curves_value_min_str": curves_value_min_str,
        "curves_value_max_str": curves_value_max_str,
        "curves_chart_png": curves_chart_png,
        "curves_chart_svg": curves_chart_svg,
        "perf_vol_chart_png": perf_vol_chart_png,
        "alloc_compare_chart_png": alloc_compare_chart_png,
        "allocation_reference_name": allocation_reference_name,
        "esg_comparison_alloc": None,
        "esg_comparison_rows": None,
        "esg_comparison_chart_png": None,
        "esg_unmatched_fields": None,
        "esg_default_fields": None,
        "esg_field_labels": {},
        "contracts": contracts,
        "supports": supports,
        "supports_category_breakdown": supports_category_breakdown,
        "supports_geo_breakdown": supports_geo_breakdown,
        "supports_category_chart_png": supports_category_chart_png,
        "supports_geo_chart_png": supports_geo_chart_png,
        "history_years": [str(y) for y in years_sorted],
        "history_perf": ann_perf,
        "history_vol": ann_vol,
        "reporting_years": reporting_years,
        "totals": {
            "total_valo": last_valo_str,
            "depots": depots_str,
            "retraits": _fmt_currency(abs(retraits_total)),
            "solde": solde_str,
        },
        "synthese_card": synthese_card,
        "valorisation_card": valorisation_card,
        "indicateurs_card": indicateurs_card,
        "affaire": affaire,
    }
def _fetch_task_types(db: Session) -> dict[str, list[dict]]:
    rows = db.execute(
        text(
            """
            SELECT id, libelle, categorie
            FROM mariadb_type_evenement
            ORDER BY categorie, libelle
            """
        )
    ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        data = dict(getattr(row, "_mapping", row))
        cat = (data.get("categorie") or "tache").strip().lower()
        grouped.setdefault(cat, []).append(data)
    return grouped


@router.get("/finance/export")
def dashboard_finance_export(request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    finance_rh_param = request.query_params.get("finance_rh")
    finance_rh_id: int | None = None
    if finance_rh_param not in (None, ""):
        try:
            finance_rh_id = int(finance_rh_param)
        except (TypeError, ValueError):
            finance_rh_id = None

    finance_ctx = _build_finance_analysis(
        db=db,
        finance_rh_id=finance_rh_id,
        finance_date_param=request.query_params.get("finance_date"),
        finance_valo_param=request.query_params.get("finance_valo"),
        force_refresh=True,
    )
    # Si aucune donnée remontée (cache potentiellement obsolète), refaire un calcul frais
    if not finance_ctx.get("finance_supports"):
        finance_ctx = _build_finance_analysis(
            db=db,
            finance_rh_id=finance_rh_id,
            finance_date_param=request.query_params.get("finance_date"),
            finance_valo_param=request.query_params.get("finance_valo"),
            force_refresh=True,
        )
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')
    writer.writerow(["Code ISIN", "Support", "Catégorie principale", "Catégorie géo", "SRRI", "Valorisation (€)"])
    for sup in finance_ctx["finance_supports"]:
        writer.writerow([
            sup.get("code_isin") or "",
            sup.get("nom") or "",
            sup.get("cat_principale") or "",
            sup.get("cat_geo") or "",
            sup.get("srri") if sup.get("srri") is not None else "",
            "{:,.0f}".format(sup.get("total_valo") or 0).replace(",", " "),
        ])
    writer.writerow([])
    writer.writerow(["", "", "", "", "Total", "{:,.0f}".format(finance_ctx["finance_total_valo"]).replace(",", " ")])
    output.seek(0)
    filename_date = finance_ctx["finance_date_input"]
    rh_suffix = f"_rh_{finance_rh_id}" if finance_rh_id is not None else ""
    filename = f"analyse_financiere_{filename_date}{rh_suffix}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reclamations/export")
def dashboard_reclamations_export(db: Session = Depends(get_db)) -> StreamingResponse:
    reclam_stats, reclam_pivot = _compute_reclamations_data(db)
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')

    statuses = reclam_pivot.get("statuses", [])
    header = ["Type d'événement"] + statuses
    writer.writerow(header)

    for type_row in reclam_pivot.get("types", []):
        label = (type_row.get("label") or "Type non renseigné")
        counts = [
            type_row.get("counts", {}).get(status, 0)
            for status in statuses
        ]
        writer.writerow([label, *counts])

    writer.writerow([])
    writer.writerow(["Total réclamations", reclam_stats.get("total", 0)])

    output.seek(0)
    csv_bytes = output.getvalue().encode("utf-8-sig")
    filename = f"reclamations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/synthese")
def dashboard_api_synthese(
    request: Request,
    id_client: int = Query(..., alias="id_client"),
    db: Session = Depends(get_db),
):
    ctx = _build_client_synthese_context(db, id_client)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    if not ctx.get('esg_comparison_rows'):
        try:
            esg_alloc = ctx.get('allocation_reference_name') or None
            esg_fields = ctx.get('esg_default_fields') or []
            if esg_alloc and not esg_fields:
                esg_fields = [
                    col
                    for col in (ctx.get('esg_field_labels') or {}).keys()
                ][:4]
            if esg_alloc and not esg_fields:
                esg_fields = [
                    'Avoiding water scarcity',
                    'Board independence',
                    'GHG intensity-Value',
                    'Gender pay gap'
                ]
            if esg_fields:
                ctx['esg_default_fields'] = esg_fields[:4]
            if esg_alloc and esg_fields:
                payload_tmp, _ = _compute_client_esg_comparison(
                    db=db,
                    client_id=id_client,
                    fields=esg_fields,
                    alloc=esg_alloc,
                    alloc_isin=None,
                    as_of=None,
                    alloc_date=None,
                    debug=False,
                )
                rows_payload = payload_tmp.get('results') or []
                labels_map = ctx.get('esg_field_labels') or {}
                resolved_pairs = payload_tmp.get('resolved_pairs') or []
                if not labels_map and resolved_pairs:
                    for pair in resolved_pairs:
                        req = pair.get('requested')
                        res = pair.get('resolved')
                        if res and req:
                            labels_map[res] = req
                formatted_rows = []
                for row in rows_payload:
                    if not isinstance(row, dict):
                        continue
                    field = row.get('field')
                    label = row.get('label') or labels_map.get(field) or row.get('field_original') or field
                    row['label'] = label
                    formatted_rows.append(row)
                ctx['esg_comparison_rows'] = formatted_rows
                ctx['esg_unmatched_fields'] = payload_tmp.get('unmatched_fields') or []
                ctx['esg_default_fields'] = payload_tmp.get('resolved_fields') or esg_fields[:4]
                ctx['esg_comparison_alloc'] = esg_alloc
                ctx['esg_field_labels'] = labels_map
                rows_chart = [
                    r for r in formatted_rows
                    if r and r.get('index') is not None and r.get('client') is not None
                ]
                if rows_chart:
                    cats = [str(r.get('label') or r.get('field')) for r in rows_chart]
                    idx_vals = [float(r.get('index') or 0.0) for r in rows_chart]
                    cli_vals = [float(r.get('client') or 0.0) for r in rows_chart]
                    try:
                        ctx['esg_comparison_chart_png'] = _build_matplotlib_esg_bar_chart(cats, idx_vals, cli_vals)
                    except Exception:
                        pass
        except Exception:
            pass

    if not ctx.get("report_logo_png"):
        try:
            logo_path = Path(__file__).resolve().parents[1] / "logo" / "logo.png"
            if logo_path.exists():
                ctx["report_logo_png"] = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        except Exception:
            pass

    ctx_render = dict(ctx)
    ctx_render["request"] = request
    ctx_render["title"] = f"Synthèse {ctx['client'].prenom or ''} {ctx['client'].nom or ''}".strip()
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"WeasyPrint indisponible: {exc}")

    html = templates.get_template("synthese_report.html").render(ctx_render)
    pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
    filename = f"synthese_{id_client}_{ctx['report_date'].strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/affaires/{affaire_id}/synthese/pdf")
def dashboard_affaire_synthese_pdf(affaire_id: int, request: Request, db: Session = Depends(get_db)):
    ctx = _build_affaire_synthese_context(db, affaire_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Affaire introuvable.")

    client_obj = ctx.get("client")
    if not client_obj:
        raise HTTPException(status_code=404, detail="Client associé introuvable pour ce contrat.")

    client_id = getattr(client_obj, "id", None)
    if client_id and not ctx.get("esg_comparison_rows"):
        try:
            esg_alloc = ctx.get("allocation_reference_name") or None
            esg_fields = ctx.get("esg_default_fields") or []
            if esg_alloc and not esg_fields:
                esg_fields = [
                    col
                    for col in (ctx.get("esg_field_labels") or {}).keys()
                ][:4]
            if esg_alloc and not esg_fields:
                esg_fields = [
                    'Avoiding water scarcity',
                    'Board independence',
                    'GHG intensity-Value',
                    'Gender pay gap'
                ]
            if esg_fields:
                ctx['esg_default_fields'] = esg_fields[:4]
            if esg_alloc and esg_fields:
                payload_tmp, _ = _compute_client_esg_comparison(
                    db=db,
                    client_id=client_id,
                    fields=esg_fields,
                    alloc=esg_alloc,
                    alloc_isin=None,
                    as_of=None,
                    alloc_date=None,
                    debug=False,
                )
                rows_payload = payload_tmp.get('results') or []
                labels_map = ctx.get('esg_field_labels') or {}
                resolved_pairs = payload_tmp.get('resolved_pairs') or []
                if not labels_map and resolved_pairs:
                    for pair in resolved_pairs:
                        req = pair.get('requested')
                        res = pair.get('resolved')
                        if res and req:
                            labels_map[res] = req
                formatted_rows = []
                for row in rows_payload:
                    if not isinstance(row, dict):
                        continue
                    field = row.get('field')
                    label = row.get('label') or labels_map.get(field) or row.get('field_original') or field
                    row['label'] = label
                    formatted_rows.append(row)
                ctx['esg_comparison_rows'] = formatted_rows
                ctx['esg_unmatched_fields'] = payload_tmp.get('unmatched_fields') or []
                ctx['esg_default_fields'] = payload_tmp.get('resolved_fields') or esg_fields[:4]
                ctx['esg_comparison_alloc'] = esg_alloc
                ctx['esg_field_labels'] = labels_map
                rows_chart = [
                    r for r in formatted_rows
                    if r and r.get('index') is not None and r.get('client') is not None
                ]
                if rows_chart:
                    cats = [str(r.get('label') or r.get('field')) for r in rows_chart]
                    idx_vals = [float(r.get('index') or 0.0) for r in rows_chart]
                    cli_vals = [float(r.get('client') or 0.0) for r in rows_chart]
                    try:
                        ctx['esg_comparison_chart_png'] = _build_matplotlib_esg_bar_chart(cats, idx_vals, cli_vals)
                    except Exception:
                        pass
        except Exception:
            pass

    if not ctx.get("report_logo_png"):
        try:
            logo_path = Path(__file__).resolve().parents[1] / "logo" / "logo.png"
            if logo_path.exists():
                ctx["report_logo_png"] = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        except Exception:
            pass

    ctx_render = dict(ctx)
    ctx_render["request"] = request
    affaire_ref = getattr(ctx.get("affaire"), "ref", None)
    prenom = getattr(client_obj, "prenom", "") or ""
    nom = getattr(client_obj, "nom", "") or ""
    client_label = f"{prenom} {nom}".strip()
    title_parts = ["Synthèse contrat"]
    if affaire_ref:
        title_parts.append(str(affaire_ref))
    if client_label:
        title_parts.append(client_label)
    ctx_render["title"] = " – ".join([p for p in title_parts if p])

    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"WeasyPrint indisponible: {exc}")

    html = templates.get_template("synthese_report.html").render(ctx_render)
    pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
    filename = f"synthese_affaire_{affaire_id}_{ctx['report_date'].strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/clients/{client_id}/der/pdf")
async def dashboard_client_der_pdf(client_id: int, request: Request, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    der_context = _build_der_context(db)
    context = {
        "request": request,
        "client": client,
        "client_id": client_id,
        "fatca_today": _date.today().isoformat(),
        "lm_today": _date.today().strftime('%d/%m/%Y'),
    }
    context.update(der_context)
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"WeasyPrint indisponible: {exc}")

    html = templates.get_template("der_report.html").render(context)
    pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
    filename = f"der_{client_id}_{_date.today().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@router.get("/clients/{client_id}/mission/pdf")
async def dashboard_client_mission_pdf(client_id: int, request: Request, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    der_context = _build_der_context(db)
    mission_context = _build_mission_context(db, client_id)
    context = {
        "request": request,
        "client": client,
        "client_id": client_id,
    }
    context.update(der_context)
    context.update(mission_context)
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"WeasyPrint indisponible: {exc}")

    html = templates.get_template("mission_report.html").render(context)
    pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
    filename = f"lettre_mission_{client_id}_{_date.today().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@router.get("/clients/{client_id}/adequation/pdf")
async def dashboard_client_adequation_pdf(client_id: int, request: Request, db: Session = Depends(get_db)):
    ctx = _build_client_adequation_context(db, client_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    if not ctx.get("report_logo_png"):
        try:
            logo_path = Path(__file__).resolve().parents[1] / "logo" / "logo.png"
            if logo_path.exists():
                ctx["report_logo_png"] = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        except Exception:
            pass

    ctx_render = dict(ctx)
    ctx_render["request"] = request
    client_obj = ctx.get("client")
    client_label = ""
    if client_obj:
        prenom = getattr(client_obj, "prenom", "") or ""
        nom = getattr(client_obj, "nom", "") or ""
        client_label = f"{prenom} {nom}".strip()
    title_parts = ["Lettre d’adéquation"]
    if client_label:
        title_parts.append(client_label)
    ctx_render["title"] = " – ".join([p for p in title_parts if p])

    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"WeasyPrint indisponible: {exc}")

    html = templates.get_template("adequation_report.html").render(ctx_render)
    pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
    filename = f"lettre_adequation_{client_id}_{ctx['report_date'].strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


# ---------------- KYC Report (HTML/PDF) ----------------
@router.get("/clients/kyc/{client_id}/rapport")
async def dashboard_client_kyc_report(
    client_id: int,
    request: Request,
    pdf: int = 0,
    engine: str | None = None,
    db: Session = Depends(get_db),
):
    client = db.query(Client).filter(Client.id == client_id).first()

    if not client:
        return templates.TemplateResponse(
            "kyc_report.html",
            {"request": request, "error": "Client introuvable.", "report_logo_png": report_logo_png},
        )

    def _fmt_amount(v):
        try:
            return "{:,.2f}".format(float(v)).replace(",", " ")
        except Exception:
            return "-"

    # Etat civil (dernier)
    etat = None
    try:
        row = db.execute(text(
            "SELECT * FROM etat_civil_client WHERE id_client = :cid ORDER BY id DESC LIMIT 1"
        ), {"cid": client_id}).fetchone()
        etat = dict(row._mapping) if row else None
    except Exception:
        etat = None

    # Adresses
    adresses = rows_to_dicts(db.execute(text(
        """
        SELECT a.*, COALESCE(t.libelle,'Non renseigné') AS type_libelle
        FROM KYC_Client_Adresse a
        LEFT JOIN ref_type_adresse t ON t.id = a.type_adresse_id
        WHERE a.client_id = :cid
        ORDER BY (a.date_saisie IS NULL), a.date_saisie DESC, a.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])

    # Situation matrimoniale
    matrimoniales = rows_to_dicts(db.execute(text(
        """
        SELECT m.*, sm.libelle AS situation_libelle, sc.libelle AS convention_libelle
        FROM KYC_Client_Situation_Matrimoniale m
        LEFT JOIN ref_situation_matrimoniale sm ON sm.id = m.situation_id
        LEFT JOIN ref_situation_matrimoniale_convention sc ON sc.id = m.convention_id
        WHERE m.client_id = :cid
        ORDER BY (m.date_saisie IS NULL), m.date_saisie DESC, m.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])

    # Situation professionnelle
    professionnelles = rows_to_dicts(db.execute(text(
        """
        SELECT p.*, ps.libelle AS secteur_libelle, st.libelle AS statut_libelle
        FROM KYC_Client_Situation_Professionnelle p
        LEFT JOIN ref_profession_secteur ps ON ps.id = p.secteur_id
        LEFT JOIN ref_statut_professionnel st ON st.id = p.statut_id
        WHERE p.client_id = :cid
        ORDER BY (p.date_saisie IS NULL), p.date_saisie DESC, p.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])

    # Patrimoine et revenus
    actifs = rows_to_dicts(db.execute(text(
        """
        SELECT a.*, COALESCE(t.libelle,'Non renseigné') AS type_libelle
        FROM KYC_Client_Actif a LEFT JOIN ref_type_actif t ON t.id=a.type_actif_id
        WHERE a.client_id=:cid ORDER BY (a.date_saisie IS NULL), a.date_saisie DESC, a.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])
    passifs = rows_to_dicts(db.execute(text(
        """
        SELECT p.*, COALESCE(t.libelle,'Non renseigné') AS type_libelle
        FROM KYC_Client_Passif p LEFT JOIN ref_type_passif t ON t.id=p.type_passif_id
        WHERE p.client_id=:cid ORDER BY (p.date_saisie IS NULL), p.date_saisie DESC, p.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])
    # Revenus: joindre ref_type_revenu si la table existe
    try:
        has_ref_rev = bool(db.execute(text("PRAGMA table_info('ref_type_revenu')")).fetchall())
    except Exception:
        has_ref_rev = False
    if has_ref_rev:
        revenus = rows_to_dicts(db.execute(text(
            """
            SELECT r.*, COALESCE(t.libelle,'Non renseigné') AS type_libelle
            FROM KYC_Client_Revenus r LEFT JOIN ref_type_revenu t ON t.id=r.type_revenu_id
            WHERE r.client_id=:cid ORDER BY (r.date_saisie IS NULL), r.date_saisie DESC, r.id DESC
            """
        ), {"cid": client_id}).fetchall() or [])
    else:
        revenus = rows_to_dicts(db.execute(text(
            """
            SELECT r.*, '' AS type_libelle
            FROM KYC_Client_Revenus r
            WHERE r.client_id=:cid ORDER BY (r.date_saisie IS NULL), r.date_saisie DESC, r.id DESC
            """
        ), {"cid": client_id}).fetchall() or [])
    charges = rows_to_dicts(db.execute(text(
        """
        SELECT c.*, COALESCE(t.libelle,'Non renseigné') AS type_libelle
        FROM KYC_Client_Charges c LEFT JOIN ref_type_charge t ON t.id=c.type_charge_id
        WHERE c.client_id=:cid ORDER BY (c.date_saisie IS NULL), c.date_saisie DESC, c.id DESC
        """
    ), {"cid": client_id}).fetchall() or [])
    actif_total = sum([float(x.get("valeur") or 0) for x in actifs])
    passif_total = sum([float(x.get("montant_rest_du") or 0) for x in passifs])
    revenu_total = sum([float(x.get("montant_annuel") or 0) for x in revenus])
    charge_total = sum([float(x.get("montant_annuel") or 0) for x in charges])
    patrimoine_net = actif_total - passif_total
    budget_net = revenu_total - charge_total

    # Donut datasets (aggregations par type)
    from collections import defaultdict as _dd
    def _agg(items: list[dict], label_key: str, value_key: str) -> dict:
        sums = _dd(float)
        for it in items or []:
            try:
                label = str(it.get(label_key) or 'Autre')
            except Exception:
                label = 'Autre'
            try:
                val = float(it.get(value_key) or 0)
            except Exception:
                val = 0.0
            if val:
                sums[label] += val
        # Sort by descending value
        labels_vals = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
        labels = [kv[0] for kv in labels_vals]
        values = [kv[1] for kv in labels_vals]
        return { 'labels': labels, 'values': values }

    donut_data = {
        'actifs': _agg(actifs, 'type_libelle', 'valeur'),
        'passifs': _agg(passifs, 'type_libelle', 'montant_rest_du'),
        'revenus': _agg(revenus, 'type_libelle', 'montant_annuel'),
        'charges': _agg(charges, 'type_libelle', 'montant_annuel'),
    }

    # SVG fallbacks for charts (usable by WeasyPrint and HTML without JS)
    def _svg_donut(labels: list[str], values: list[float]) -> str | None:
        try:
            total = sum([float(v or 0) for v in (values or [])])
            if not labels or not values or total <= 0:
                return None
            W,H,cx,cy,r,th = 260, 200, 130, 90, 70, 26
            import math
            C = 2*math.pi*r
            acc = 0.0
            palette = ['#2563eb','#16a34a','#f59e0b','#ef4444','#8b5cf6','#10b981','#f97316','#0ea5e9','#eab308','#dc2626']
            parts = [
                f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">',
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef2ff" stroke-width="{th}" />'
            ]
            for i, v in enumerate(values):
                v = float(v or 0)
                if v <= 0: continue
                seg = C * (v/total)
                color = palette[i % len(palette)]
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{th}" '
                    f'stroke-dasharray="{seg:.4f} {C-seg:.4f}" stroke-dashoffset="{-acc:.4f}" '
                    f'transform="rotate(-90 {cx} {cy})" stroke-linecap="butt" />'
                )
                acc += seg
            parts.append('</svg>')
            return ''.join(parts)
        except Exception:
            return None

    def _svg_bar(labels: list[str], values: list[float], title: str) -> str | None:
        try:
            vals = [float(v or 0) for v in values or []]
            if not vals: return None
            W,H,P = 640, 260, 32
            vmax = max(vals) or 1.0
            bw = (W - 2*P) / max(1, len(vals)) * 0.7
            colors = ['#2563eb','#ef4444','#16a34a','#f59e0b']
            parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
            parts.append(f'<text x="{P}" y="{P-10}" fill="#111" font-size="14" font-weight="bold">{title}</text>')
            # grid
            for i in range(5):
                y = P + i*(H-2*P)/4
                parts.append(f'<line x1="{P}" y1="{y:.1f}" x2="{W-P}" y2="{y:.1f}" stroke="#eef2ff" stroke-width="1" />')
            for i, v in enumerate(vals):
                h = (H-2*P) * (v / vmax)
                x = P + i*((W-2*P)/max(1,len(vals))) + ((W-2*P)/max(1,len(vals)) - bw)/2
                y = H-P - h
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[i%len(colors)]}" />')
                parts.append(f'<text x="{x + bw/2:.1f}" y="{H-P+16}" text-anchor="middle" font-size="12" fill="#111">{labels[i]}</text>')
            parts.append('</svg>')
            return ''.join(parts)
        except Exception:
            return None

    def _svg_pie_two(a: float, b: float, labels=('Actifs','Passifs'), title='Patrimoine (Actifs/Passifs)') -> str | None:
        try:
            a = max(0.0, float(a or 0)); b = max(0.0, float(b or 0)); total = a + b
            if total <= 0: return None
            W,H,cx,cy,r,th = 320, 240, 160, 110, 80, 36
            import math
            C = 2*math.pi*r
            parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
            parts.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#111">{title}</text>')
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef2ff" stroke-width="{th}" />')
            segA = C * (a/total); segB = C - segA
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#2563eb" stroke-width="{th}" stroke-dasharray="{segA:.4f} {C-segA:.4f}" transform="rotate(-90 {cx} {cy})" />')
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#ef4444" stroke-width="{th}" stroke-dasharray="{segB:.4f} {C-segB:.4f}" stroke-dashoffset="{-segA:.4f}" transform="rotate(-90 {cx} {cy})" />')
            parts.append('</svg>')
            return ''.join(parts)
        except Exception:
            return None

    # Objectifs (sélectionnés)
    # Tenter de joindre libellés via ref_objectif (nom de table exact) ou, à défaut, via risque_objectif_option
    try:
        has_ref_obj = bool(db.execute(text("PRAGMA table_info('ref_objectif')")).fetchall())
    except Exception:
        has_ref_obj = False
    try:
        has_risque_obj = bool(db.execute(text("PRAGMA table_info('risque_objectif_option')")).fetchall())
    except Exception:
        has_risque_obj = False
    if has_ref_obj:
        objectifs = rows_to_dicts(db.execute(text(
            """
            SELECT o.id, o.objectif_id, ro.libelle AS objectif_libelle, o.niveau_id, o.horizon_investissement, o.commentaire
            FROM KYC_Client_Objectifs o
            LEFT JOIN ref_objectif ro ON ro.id = o.objectif_id
            WHERE o.client_id=:cid
            ORDER BY (o.niveau_id IS NULL), o.niveau_id ASC, o.id DESC
            """
        ), {"cid": client_id}).fetchall() or [])
    elif has_risque_obj:
        objectifs = rows_to_dicts(db.execute(text(
            """
            SELECT o.id, o.objectif_id, r.label AS objectif_libelle, o.niveau_id, o.horizon_investissement, o.commentaire
            FROM KYC_Client_Objectifs o
            LEFT JOIN risque_objectif_option r ON r.id = o.objectif_id
            WHERE o.client_id=:cid
            ORDER BY (o.niveau_id IS NULL), o.niveau_id ASC, o.id DESC
            """
        ), {"cid": client_id}).fetchall() or [])
    else:
        objectifs = rows_to_dicts(db.execute(text(
            """
            SELECT o.id, o.objectif_id, NULL AS objectif_libelle, o.niveau_id, o.horizon_investissement, o.commentaire
            FROM KYC_Client_Objectifs o
            WHERE o.client_id=:cid
            ORDER BY (o.niveau_id IS NULL), o.niveau_id ASC, o.id DESC
            """
        ), {"cid": client_id}).fetchall() or [])

    # Fallback: si aucun objectif enregistré, tenter depuis le dernier questionnaire risque (objectifs principaux)
    try:
        if not objectifs:
            rq = db.execute(text("SELECT id FROM risque_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"), {"r": str(client_id)}).fetchone()
            if rq:
                rqid = int(rq[0])
                rows = db.execute(text(
                    """
                    SELECT o.id AS option_id, o.label AS libelle
                    FROM risque_questionnaire_objectif q
                    LEFT JOIN risque_objectif_option o ON o.id = q.option_id
                    WHERE q.questionnaire_id = :qid
                    ORDER BY q.id ASC
                    """
                ), {"qid": rqid}).fetchall()
                if rows:
                    tmp = []
                    pr = 1
                    for r in rows:
                        lib = getattr(r, 'libelle', None) or (r[1] if len(r)>1 else None)
                        oid = getattr(r, 'option_id', None) or (r[0] if len(r)>0 else None)
                        tmp.append({
                            'id': None,
                            'objectif_id': oid,
                            'objectif_libelle': lib,
                            'niveau_id': pr,
                            'horizon_investissement': None,
                            'commentaire': None,
                        })
                        pr += 1
                    objectifs = tmp
    except Exception:
        pass

    # Risque (KYC_Client_Risque dernier + connaissance produits + allocation)
    risque = None
    try:
        row = db.execute(text(
            "SELECT * FROM KYC_Client_Risque WHERE client_id=:cid ORDER BY date_saisie DESC, id DESC LIMIT 1"
        ), {"cid": client_id}).fetchone()
        risque = dict(row._mapping) if row else None
        # Libellé du niveau de risque (ref)
        try:
            if risque and (risque.get("niveau_id") is not None):
                r_lbl = db.execute(text("SELECT libelle FROM ref_niveau_risque WHERE id = :i"), {"i": risque.get("niveau_id")}).fetchone()
                if r_lbl and r_lbl[0] is not None:
                    risque["niveau_label"] = r_lbl[0]
        except Exception:
            pass
        if risque:
            rows_c = db.execute(text(
                "SELECT produit_id, produit_label, niveau_id, niveau_label FROM KYC_Client_Risque_Connaissance WHERE risque_id=:rid ORDER BY produit_label, produit_id"
            ), {"rid": risque.get("id")}).fetchall()
            items = rows_to_dicts(rows_c)
            # Enrichir labels manquants via la table de référence des produits
            try:
                missing_ids = sorted({int(it.get('produit_id')) for it in items if it.get('produit_id') and (not it.get('produit_label'))})
                if missing_ids:
                    q = text("SELECT id, label FROM risque_connaissance_produit_option WHERE id IN (%s)" % ",".join(str(x) for x in missing_ids))
                    rows_map = db.execute(q).fetchall()
                    ref_map = {int(r[0]): (r[1] or '') for r in rows_map}
                    for it in items:
                        pid = it.get('produit_id')
                        if pid and not it.get('produit_label'):
                            try:
                                it['produit_label'] = ref_map.get(int(pid)) or it.get('produit_label')
                            except Exception:
                                pass
            except Exception:
                pass
            risque["connaissance_produits"] = items
            # Allocation liée au niveau
            try:
                row_alloc = db.execute(text(
                    """
                    SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom
                          , ar.texte AS allocation_texte
                    FROM allocation_risque ar
                    LEFT JOIN allocations a ON a.nom = ar.allocation_name
                    WHERE ar.risque_id = :rid
                    ORDER BY ar.date_attribution DESC, ar.id DESC
                    LIMIT 1
                    """
                ), {"rid": risque.get("niveau_id")}).fetchone()
                if row_alloc:
                    risque["allocation_nom"] = row_alloc[0]
                    # Convert markdown to simple HTML for conformity template
                    try:
                        md = row_alloc[1]
                        if md:
                            import re, html as _html
                            text_md = _html.escape(str(md))
                            text_md = re.sub(r"^###\s+(.*)$", r"<h5>\1</h5>", text_md, flags=re.M)
                            text_md = re.sub(r"^##\s+(.*)$", r"<h4>\1</h4>", text_md, flags=re.M)
                            text_md = re.sub(r"^#\s+(.*)$", r"<h3>\1</h3>", text_md, flags=re.M)
                            text_md = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text_md)
                            text_md = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text_md)
                            lines = text_md.split('\n')
                            out = []
                            in_ul = False
                            for ln in lines:
                                if re.match(r"^\s*-\s+", ln):
                                    if not in_ul:
                                        out.append("<ul>"); in_ul=True
                                    out.append("<li>" + re.sub(r"^\s*-\s+", "", ln) + "</li>")
                                else:
                                    if in_ul:
                                        out.append("</ul>"); in_ul=False
                                    if ln.strip(): out.append("<p>"+ln+"</p>")
                            if in_ul: out.append("</ul>")
                            risque["allocation_html"] = "\n".join(out)
                    except Exception:
                        pass
            except Exception:
                pass
            # Disponibilité (fallback depuis risque_questionnaire si absente)
            try:
                if not risque.get("disponibilite"):
                    row_dispo = db.execute(text(
                        """
                        SELECT d.label AS dispo, du.label AS duree
                        FROM risque_questionnaire rq
                        LEFT JOIN risque_disponibilite_option d ON d.id = rq.disponibilite_option_id
                        LEFT JOIN risque_duree_option du ON du.id = rq.duree_option_id
                        WHERE rq.client_ref = :r
                        ORDER BY rq.updated_at DESC LIMIT 1
                        """
                    ), {"r": str(client_id)}).fetchone()
                    if row_dispo:
                        m = row_dispo._mapping
                        if m.get("dispo"):
                            risque["disponibilite"] = m.get("dispo")
                        if (not risque.get("duree")) and m.get("duree"):
                            risque["duree"] = m.get("duree")
            except Exception:
                pass
    except Exception:
        risque = None

    # Risque synthétique via jointure (niveau libellé, horizon, expérience, connaissance)
    risque_latest_info = None
    try:
        row = db.execute(text(
            """
            SELECT
              r.libelle AS niveau_risque,
              k.duree AS horizon_placement,
              k.experience,
              k.connaissance,
              k.commentaire
            FROM KYC_Client_Risque k
            JOIN ref_niveau_risque r
              ON k.niveau_id = r.id
            WHERE k.client_id = :cid
            ORDER BY k.date_saisie DESC, k.id DESC
            LIMIT 1
            """
        ), {"cid": client_id}).fetchone()
        if row:
            risque_latest_info = dict(row._mapping)
    except Exception:
        risque_latest_info = None

    # ESG (dernier) + exclusions
    esg = None
    esg_exclusions: list[str] = []
    try:
        row = db.execute(text(
            "SELECT * FROM esg_questionnaire WHERE client_ref=:r ORDER BY updated_at DESC LIMIT 1"
        ), {"r": str(client_id)}).fetchone()
        esg = dict(row._mapping) if row else None
        if esg and esg.get("id"):
            try:
                rows_ex = db.execute(text(
                    """
                    SELECT COALESCE(o.label, o.code) AS label
                    FROM esg_questionnaire_exclusion qe
                    LEFT JOIN esg_exclusion_option o ON o.id = qe.option_id
                    WHERE qe.questionnaire_id = :qid
                    ORDER BY o.label
                    """
                ), {"qid": esg.get("id")}).fetchall()
                esg_exclusions = [str(r[0]) for r in rows_ex]
            except Exception:
                esg_exclusions = []
    except Exception:
        esg = None
        esg_exclusions = []

    ctx = {
        "request": request,
        "client": client,
        "today": _date.today().isoformat(),
        "etat": etat,
        "adresses": adresses,
        "matrimoniales": matrimoniales,
        "professionnelles": professionnelles,
        "actifs": actifs,
        "passifs": passifs,
        "revenus": revenus,
        "charges": charges,
        "actif_total": _fmt_amount(actif_total),
        "passif_total": _fmt_amount(passif_total),
        "revenu_total": _fmt_amount(revenu_total),
        "charge_total": _fmt_amount(charge_total),
        "patrimoine_net": _fmt_amount(patrimoine_net),
        "budget_net": _fmt_amount(budget_net),
        "objectifs": objectifs,
        "risque": risque,
        "esg": esg,
        "esg_exclusions": esg_exclusions,
        "want_charts": int(pdf or 0) == 0,
        "chart_data": {
            "labels": ["Actifs", "Passifs", "Revenus", "Charges"],
            "values": [actif_total, passif_total, revenu_total, charge_total],
        },
        "donut_data": donut_data,
    }

    # Inject SVG fallbacks for charts into context (independent of Matplotlib)
    try:
        ctx['donut_actifs_svg'] = _svg_donut(donut_data['actifs']['labels'], donut_data['actifs']['values']) or None
        ctx['donut_passifs_svg'] = _svg_donut(donut_data['passifs']['labels'], donut_data['passifs']['values']) or None
        ctx['donut_revenus_svg'] = _svg_donut(donut_data['revenus']['labels'], donut_data['revenus']['values']) or None
        ctx['donut_charges_svg'] = _svg_donut(donut_data['charges']['labels'], donut_data['charges']['values']) or None
        ctx['chart_synth_svg'] = _svg_bar(["Actifs","Passifs","Revenus","Charges"], [actif_total, passif_total, revenu_total, charge_total], 'Synthèse') or None
        ctx['chart_pie_svg'] = _svg_pie_two(actif_total, passif_total) or None
        # Allocation line+vol (SVG fallback)
        series = ctx.get('alloc_series_data')
        if series:
            try:
                dates = [s.get('date') for s in series]
                svals = [float(s.get('sicav') or 0) for s in series]
                vvals = [float(s.get('vol') or 0) for s in series]
                # Build simple SVG with dual axis
                W,H,P = 650, 260, 36
                try:
                    import math
                except Exception:
                    math = None
                xmin,xmax = 0, max(1, len(svals)-1)
                smin,smax = min(svals), max(svals) if svals else (0,1)
                if smax == smin: smax = smin + 1
                vmin,vmax = min(vvals), max(vvals) if vvals else (0,1)
                if vmax == vmin: vmax = vmin + 1
                def X(i):
                    return P + (i*(W-2*P))/max(1,(len(svals)-1))
                def Y1(val):
                    return H-P - ((val - smin)*(H-2*P))/(smax - smin)
                def Y2(val):
                    return H-P - ((val - vmin)*(H-2*P))/(vmax - vmin)
                pts1 = ' '.join([f"{X(i):.1f},{Y1(v):.1f}" for i,v in enumerate(svals)])
                pts2 = ' '.join([f"{X(i):.1f},{Y2(v):.1f}" for i,v in enumerate(vvals)])
                svg = []
                svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">')
                # grid
                for i in range(5):
                    y = P + i*(H-2*P)/4
                    svg.append(f'<line x1="{P}" y1="{y:.1f}" x2="{W-P}" y2="{y:.1f}" stroke="#eef2ff" stroke-width="1" />')
                svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts1}" />')
                svg.append(f'<polyline fill="none" stroke="#ef4444" stroke-width="1.5" points="{pts2}" />')
                svg.append(f'<text x="{P}" y="{P-10}" fill="#111" font-size="13" font-weight="bold">Performance et volatilité – {ctx.get("risque",{}).get("allocation_nom") or "Allocation"}</text>')
                svg.append(f'<text x="{W-P}" y="{P-10}" fill="#ef4444" font-size="11" text-anchor="end">Volatilité (%)</text>')
                svg.append('</svg>')
                ctx['chart_alloc_svg'] = ''.join(svg)
            except Exception:
                ctx['chart_alloc_svg'] = None
    except Exception:
        pass

    # Pied de page: informations cabinet (DER_courtier)
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            m = row._mapping
            nom_cabinet = (m.get('nom_cabinet') or '').strip()
            adresse_rue = (m.get('adresse_rue') or '').strip()
            adresse_cp = (m.get('adresse_cp') or '')
            adresse_ville = (m.get('adresse_ville') or '').strip()
            nom_responsable = (m.get('nom_responsable') or '').strip()
            courriel = (m.get('courriel') or '').strip()
            numero_orias = (m.get('numero_orias') or '').strip()
            assoc_id = m.get('association_prof')
            num_adh = (m.get('num_adh_assoc') or '').strip()
            statut_id = m.get('statut_social')
            capital_social = m.get('capital_social')
            siren = (m.get('siren') or '').strip()
            rcs = (m.get('rcs') or '').strip()

            # Assoc label
            assoc_label = None
            try:
                refs = _fetch_ref_list(db, ["DER_association_professionnelle", "DER_association_prof"]) or []
                for it in refs:
                    try:
                        if int(it.get('id')) == int(assoc_id):
                            assoc_label = it.get('libelle')
                            break
                    except Exception:
                        if str(it.get('id')) == str(assoc_id):
                            assoc_label = it.get('libelle')
                            break
            except Exception:
                assoc_label = None

            # Statut label
            statut_label = None
            try:
                srefs = _fetch_ref_list(db, ["DER_statut_social", "DER_statuts_sociaux"]) or []
                for it in srefs:
                    try:
                        if int(it.get('id')) == int(statut_id):
                            statut_label = it.get('libelle')
                            break
                    except Exception:
                        if str(it.get('id')) == str(statut_id):
                            statut_label = it.get('libelle')
                            break
            except Exception:
                statut_label = None

            # Compose lines
            line1 = nom_cabinet or None
            # Ligne "Statuts au capital ..."
            cap_str = _fmt_amount(capital_social) if (capital_social is not None) else None
            stat_parts = []
            if statut_label:
                stat_parts.append(str(statut_label))
            if cap_str and cap_str != '-':
                stat_parts.append(f"au capital social de {cap_str} €")
            line2 = " ".join(stat_parts) if stat_parts else None
            # Ligne suivante: "Siren : ..." et/ou "RCS ..."
            s_parts = []
            if siren:
                s_parts.append(f"SIREN {siren}")
            if rcs:
                s_parts.append(f"RCS {rcs}")
            line2b = " — ".join(s_parts) if s_parts else None

            # Adresse: "<rue>, <CP> <Ville>"
            addr_parts = []
            if adresse_rue:
                addr_parts.append(adresse_rue)
            cp_ville = (f"{adresse_cp} {adresse_ville}" if (adresse_cp or adresse_ville) else "").strip()
            if cp_ville:
                if addr_parts:
                    addr_parts[-1] = addr_parts[-1] + ", " + cp_ville
                else:
                    addr_parts.append(cp_ville)
            line3 = addr_parts[0] if addr_parts else None
            # Responsable & courriel
            r_parts = [p for p in [nom_responsable, courriel] if p]
            line4 = " & ".join(r_parts) if r_parts else None
            # ORIAS et Association sur deux lignes
            line5 = f"Numéro Orias {numero_orias}" if numero_orias else None
            line6 = None
            if assoc_label:
                line6 = f"Association professionnelle {assoc_label}"
                if num_adh:
                    line6 += f" {num_adh}"

            if nom_cabinet:
                ctx['footer_text'] = nom_cabinet
            if line1:
                ctx['footer_line1'] = line1
            if line2:
                ctx['footer_line2'] = line2
            if line2b:
                ctx['footer_line2b'] = line2b
            if line3:
                ctx['footer_line3'] = line3
            if line4:
                ctx['footer_line4'] = line4
            if line5:
                ctx['footer_line5'] = line5
            if line6:
                ctx['footer_line6'] = line6

            # Logo (src/logo/*)
            try:
                import os, base64
                logo_dir = os.path.join(os.path.dirname(__file__), '..', 'logo')
                logo_dir = os.path.abspath(logo_dir)
                if os.path.isdir(logo_dir):
                    for fn in os.listdir(logo_dir):
                        lower = fn.lower()
                        if lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                            path = os.path.join(logo_dir, fn)
                            mime = 'image/png'
                            if lower.endswith('.jpg') or lower.endswith('.jpeg'):
                                mime = 'image/jpeg'
                            elif lower.endswith('.gif'):
                                mime = 'image/gif'
                            elif lower.endswith('.svg'):
                                mime = 'image/svg+xml'
                            with open(path, 'rb') as f:
                                b64 = base64.b64encode(f.read()).decode('ascii')
                            ctx['logo_url'] = f"data:{mime};base64,{b64}"
                            break
            except Exception:
                pass
    except Exception:
        pass

    # Allocation counts for diagnostics in report (exact and normalized)
    try:
        alloc_name = ctx.get('risque', {}).get('allocation_nom') if ctx.get('risque') else None
        if alloc_name:
            try:
                row_cnt = db.execute(text("SELECT COUNT(*) FROM allocations WHERE nom = :n"), {"n": alloc_name}).fetchone()
                ctx['alloc_count_exact'] = int(row_cnt[0]) if row_cnt else 0
            except Exception:
                ctx['alloc_count_exact'] = 0
            try:
                row_cnt2 = db.execute(text("SELECT COUNT(*) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"), {"n": alloc_name}).fetchone()
                ctx['alloc_count_norm'] = int(row_cnt2[0]) if row_cnt2 else 0
            except Exception:
                ctx['alloc_count_norm'] = 0
        else:
            ctx['alloc_count_exact'] = 0
            ctx['alloc_count_norm'] = 0
    except Exception:
        ctx['alloc_count_exact'] = 0
        ctx['alloc_count_norm'] = 0

    # Prepare allocation series for client-side chart fallback (no matplotlib)
    try:
        alloc_name = ctx.get('risque', {}).get('allocation_nom') if ctx.get('risque') else None
        if alloc_name:
            rows_series = (
                db.query(Allocation.date, Allocation.sicav, Allocation.volat)
                .filter(func.lower(func.trim(Allocation.nom)) == alloc_name.strip().lower())
                .order_by(Allocation.date.asc())
                .all()
            )
            if rows_series:
                ser = []
                for d, s, v in rows_series:
                    try:
                        ds = d.strftime('%Y-%m-%d')
                    except Exception:
                        ds = str(d)[:10]
                    try:
                        sic = float(s or 0.0)
                    except Exception:
                        sic = 0.0
                    try:
                        vol = float(v or 0.0)
                    except Exception:
                        vol = 0.0
                    if abs(vol) <= 1:
                        vol *= 100.0
                    ser.append({'date': ds, 'sicav': sic, 'vol': vol})
                if ser:
                    ctx['alloc_series_data'] = ser
    except Exception:
        pass

    # Build chart images for both HTML preview and PDF (Matplotlib if available)
    try:
        import importlib.util as _ilu
        if _ilu.find_spec('matplotlib') is None:
            raise SystemExit  # silently skip chart generation
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore
        import io, base64

        def _png_bytes(fig) -> bytes:
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150); plt.close(fig); return buf.getvalue()

        # Bar chart synthèse
        try:
            fig, ax = plt.subplots(figsize=(6, 3))
            vals = [float(actif_total), float(passif_total), float(revenu_total), float(charge_total)]
            ax.bar(["Actifs", "Passifs", "Revenus", "Charges"], vals, color=['#2563eb','#ef4444','#16a34a','#f59e0b'])
            ax.set_ylabel('€'); ax.set_title('Synthèse'); ax.tick_params(axis='x', rotation=0)
            chart_img_synth = _png_bytes(fig)
            ctx['chart_img_synth_b64'] = base64.b64encode(chart_img_synth).decode('ascii')
        except Exception as _e1:
            logger.debug('Chart synthese error: %s', _e1, exc_info=True)

        # Pie chart Actifs/Passifs
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie([max(0.0,float(actif_total)), max(0.0,float(passif_total))], labels=['Actifs','Passifs'], autopct='%1.0f%%', startangle=90)
            ax.axis('equal'); ax.set_title('Patrimoine (Actifs/Passifs)')
            chart_img_pie = _png_bytes(fig)
            ctx['chart_img_pie_b64'] = base64.b64encode(chart_img_pie).decode('ascii')
        except Exception as _e2:
            logger.debug('Chart pie error: %s', _e2, exc_info=True)

        # Courbe allocation (si allocation dispo)
        try:
            alloc_name = ctx.get('risque',{}).get('allocation_nom') if ctx.get('risque') else None
            if alloc_name:
                rows_series = (
                    db.query(Allocation.date, Allocation.sicav, Allocation.volat)
                    .filter(func.lower(func.trim(Allocation.nom)) == alloc_name.strip().lower())
                    .order_by(Allocation.date.asc())
                    .all()
                )
                logger.debug("Alloc chart preview: %s rows for '%s'", len(rows_series or []), alloc_name)
                if rows_series:
                    dates, svals, vvals = [], [], []
                    for d, s, v in rows_series:
                        dates.append(d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d)[:10])
                        svals.append(float(s or 0)); vol=float(v or 0); vvals.append(vol*100.0 if abs(vol)<=1 else vol)
                    fig, ax1 = plt.subplots(figsize=(6.5, 3.2)); ax2 = ax1.twinx()
                    ax1.plot(dates, svals, color='#2563eb', linewidth=1.5, label=str(alloc_name))
                    ax2.plot(dates, vvals, color='#ef4444', linewidth=1.2, label='Volatilité (%)')
                    ax1.set_ylabel(str(alloc_name)); ax2.set_ylabel('Volatilité (%)'); ax1.set_title(f'Performance et volatilité – {alloc_name}')
                    # Masquer l'échelle de l'axe des X (dates)
                    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    chart_img_alloc = _png_bytes(fig)
                    ctx['chart_img_alloc_b64'] = base64.b64encode(chart_img_alloc).decode('ascii')
        except Exception as _e3:
            logger.debug('Chart alloc error: %s', _e3, exc_info=True)
    except SystemExit:
        pass
    except Exception as _emat:
        logger.debug('Matplotlib error (charts skipped): %s', _emat)

    # Render HTML
    # Template selection (style)
    style = (request.query_params.get('style') or '').lower()
    template_name = "kyc_conformite_report.html" if style in ("conformite", "full", "complete") else "kyc_report.html"
    # Auto-print flag for browser PDF
    auto_print = (request.query_params.get('print') in ('1','true','yes'))
    if auto_print:
        ctx['auto_print'] = True
    html = templates.get_template(template_name).render(ctx)
    if int(pdf or 0) == 1:
        # Par défaut on privilégie WeasyPrint pour un PDF identique à l'HTML
        prefer_weasy = (engine or 'weasy').lower() == 'weasy'
        prefer_reportlab = (engine or '').lower() == 'reportlab'
        # Build chart images (PNG base64) with Matplotlib if available
        chart_img_synth: bytes | None = None
        chart_img_pie: bytes | None = None
        chart_img_alloc: bytes | None = None
        try:
            import importlib.util as _ilu
            if _ilu.find_spec('matplotlib') is None:
                raise SystemExit
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt  # type: ignore
            import io, base64

            def _png_bytes(fig) -> bytes:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                return buf.getvalue()

            # Synthèse barres
            try:
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = ["Actifs", "Passifs", "Revenus", "Charges"]
                vals = [float(actif_total), float(passif_total), float(revenu_total), float(charge_total)]
                colors = ['#2563eb', '#ef4444', '#16a34a', '#f59e0b']
                ax.bar(labels, vals, color=colors)
                ax.set_ylabel('€')
                ax.set_title('Synthèse')
                ax.tick_params(axis='x', rotation=0)
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f"{int(x):,}".replace(',', ' ')))
                chart_img_synth = _png_bytes(fig)
            except Exception:
                chart_img_synth = None

            # Pie chart Actifs vs Passifs
            try:
                fig, ax = plt.subplots(figsize=(4, 4))
                sizes = [max(0.0, float(actif_total)), max(0.0, float(passif_total))]
                lbls = ['Actifs', 'Passifs']
                ax.pie(sizes, labels=lbls, autopct='%1.0f%%', startangle=90)
                ax.axis('equal')
                ax.set_title('Patrimoine (Actifs/Passifs)')
                chart_img_pie = _png_bytes(fig)
            except Exception:
                chart_img_pie = None

            # Donuts répartitions (pour PDF)
            def _fmt_eur(x: float) -> str:
                try:
                    return f"{int(round(x)):,}".replace(',', ' ') + " €"
                except Exception:
                    return str(x)
            donut_imgs: dict[str, bytes] = {}
            donut_svgs: dict[str, str] = {}
            try:
                palettes = [
                    '#2563eb', '#16a34a', '#f59e0b', '#ef4444',
                    '#8b5cf6', '#10b981', '#f97316', '#eab308', '#0ea5e9', '#dc2626'
                ]
                def make_donut(title: str, labels: list[str], values: list[float]) -> bytes | None:
                    if not labels or not values or sum(values) <= 0:
                        return None
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(values, labels=None, startangle=90,
                           colors=palettes[:max(1, len(values))],
                           wedgeprops={'width':0.35, 'edgecolor':'white'})
                    # Petite légende en dessous
                    try:
                        lbls = [f"{l} — {_fmt_eur(v)}" for l, v in zip(labels, values)]
                        ax.legend(lbls, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)
                    except Exception:
                        pass
                    ax.set_title(title)
                    ax.axis('equal')
                    return _png_bytes(fig)

                def make_donut_svg(labels: list[str], values: list[float]) -> str | None:
                    if not labels or not values:
                        return None
                    total = sum([float(v or 0) for v in values])
                    if total <= 0:
                        return None
                    W,H,cx,cy,r,th = 260, 200, 130, 90, 70, 26
                    import math
                    C = 2*math.pi*r
                    acc = 0.0
                    parts = [
                        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">',
                        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef2ff" stroke-width="{th}" />'
                    ]
                    for i, v in enumerate(values):
                        v = float(v or 0)
                        if v <= 0:
                            continue
                        seg = C * (v/total)
                        color = palettes[i % len(palettes)]
                        parts.append(
                            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{th}" '
                            f'stroke-dasharray="{seg:.4f} {C-seg:.4f}" stroke-dashoffset="{-acc:.4f}" '
                            f'transform="rotate(-90 {cx} {cy})" stroke-linecap="butt" />'
                        )
                        acc += seg
                    parts.append('</svg>')
                    return ''.join(parts)

                donut_imgs['actifs'] = make_donut('Répartition des actifs', donut_data['actifs']['labels'], donut_data['actifs']['values']) or None
                donut_imgs['passifs'] = make_donut('Répartition des passifs', donut_data['passifs']['labels'], donut_data['passifs']['values']) or None
                donut_imgs['revenus'] = make_donut('Répartition des revenus', donut_data['revenus']['labels'], donut_data['revenus']['values']) or None
                donut_imgs['charges'] = make_donut('Répartition des charges', donut_data['charges']['labels'], donut_data['charges']['values']) or None
                # Always prepare SVG fallback (usable by WeasyPrint)
                donut_svgs['actifs'] = make_donut_svg(donut_data['actifs']['labels'], donut_data['actifs']['values']) or ''
                donut_svgs['passifs'] = make_donut_svg(donut_data['passifs']['labels'], donut_data['passifs']['values']) or ''
                donut_svgs['revenus'] = make_donut_svg(donut_data['revenus']['labels'], donut_data['revenus']['values']) or ''
                donut_svgs['charges'] = make_donut_svg(donut_data['charges']['labels'], donut_data['charges']['values']) or ''
            except Exception:
                donut_imgs = {}
                donut_svgs = {}

            # Courbe Allocation (SICAV + Vol) si allocation connue
            try:
                alloc_name = None
                if 'risque' in ctx and ctx['risque'] and ctx['risque'].get('allocation_nom'):
                    alloc_name = ctx['risque'].get('allocation_nom')
                if alloc_name:
                    rows_series = (
                        db.query(Allocation.date, Allocation.sicav, Allocation.volat)
                        .filter(func.lower(func.trim(Allocation.nom)) == alloc_name.strip().lower())
                        .order_by(Allocation.date.asc())
                        .all()
                    )
                    if rows_series:
                        dates = []
                        sicav_vals = []
                        vol_vals = []
                        for d, s, v in rows_series:
                            try:
                                dates.append(d.strftime('%Y-%m-%d'))
                            except Exception:
                                dates.append(str(d)[:10])
                            sicav_vals.append(float(s or 0))
                            vol = float(v or 0)
                            if abs(vol) <= 1:
                                vol *= 100.0
                            vol_vals.append(vol)
                        if dates:
                            fig, ax1 = plt.subplots(figsize=(6.5, 3.2))
                            ax2 = ax1.twinx()
                            ax1.plot(dates, sicav_vals, color='#2563eb', linewidth=1.5, label=alloc_name)
                            ax2.plot(dates, vol_vals, color='#ef4444', linewidth=1.2, label='Volatilité (%)')
                            ax1.set_ylabel(alloc_name)
                            ax2.set_ylabel('Volatilité (%)')
                            ax1.set_title(f'Performance et volatilité – {alloc_name}')
                            # Masquer l'échelle de l'axe des X (dates)
                            ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                            ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                            chart_img_alloc = _png_bytes(fig)
            except Exception:
                chart_img_alloc = None

            # For WeasyPrint HTML, embed base64 images in template context
            if chart_img_synth:
                ctx['chart_img_synth_b64'] = base64.b64encode(chart_img_synth).decode('ascii')
            if chart_img_pie:
                ctx['chart_img_pie_b64'] = base64.b64encode(chart_img_pie).decode('ascii')
            if chart_img_alloc:
                ctx['chart_img_alloc_b64'] = base64.b64encode(chart_img_alloc).decode('ascii')
            # Donuts (PDF)
            for key, img in (donut_imgs or {}).items():
                if img:
                    ctx[f'donut_{key}_b64'] = base64.b64encode(img).decode('ascii')
            # SVG fallbacks (if Matplotlib pngs absent)
            for key, svg in (donut_svgs or {}).items():
                if svg and not ctx.get(f'donut_{key}_b64'):
                    ctx[f'donut_{key}_svg'] = svg
            # Re-render HTML with images using the same selected template
            html = templates.get_template(template_name).render(ctx)
        except SystemExit:
            # Matplotlib absent: conserver le HTML déjà rendu, sans images supplémentaires
            pass
        except Exception:
            pass
        # 1) Try WeasyPrint (preferred by default)
        if prefer_weasy:
            try:
                from weasyprint import HTML  # type: ignore
                pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
                return StreamingResponse(
                    iter([pdf_bytes]), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=rapport_kyc_{client_id}.pdf"},
                )
            except Exception as _exc_wp:
                logger.debug("WeasyPrint export failed (preferred): %s", _exc_wp, exc_info=True)
                # Activer la branche complète ReportLab ci-dessous
                prefer_reportlab = True
        # 2) ReportLab uniquement si explicitement demandé
        if prefer_reportlab:
            try:
                import io
                from reportlab.lib.pagesizes import A4  # type: ignore
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage  # type: ignore
                from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
                from reportlab.lib import colors  # type: ignore

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4)
                styles = getSampleStyleSheet()
                H1 = styles['Heading1']; H2 = styles['Heading2']; H3 = styles['Heading3']; P = styles['BodyText']
                story = []

                # Titre
                story.append(Paragraph(f"Rapport KYC", H1))
                story.append(Paragraph(f"Pour {client.prenom} {client.nom}", H2))
                story.append(Paragraph(f"Date: {ctx['today']}", P))
                story.append(Spacer(1, 12))

                # Chapitre 1: Etat civil
                story.append(Paragraph("1. Etat civil", H2))
                if etat:
                    story.append(Paragraph(f"Civilité: {etat.get('civilite') or '-'}", P))
                    story.append(Paragraph(f"Date de naissance: {etat.get('date_naissance') or '-'}", P))
                    story.append(Paragraph(f"Lieu de naissance: {etat.get('lieu_naissance') or '-'}", P))
                    story.append(Paragraph(f"Nationalité: {etat.get('nationalite') or '-'}", P))
                story.append(Spacer(1, 8))

                # Adresses tableau
                if adresses:
                    data = [["Type", "Adresse"]] + [[a.get('type_libelle') or '-', f"{a.get('rue') or ''} {a.get('complement') or ''} {a.get('code_postal') or ''} {a.get('ville') or ''} {a.get('pays') or ''}"] for a in adresses]
                    tbl = Table(data, hAlign='LEFT')
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#eef2ff')),
                        ('TEXTCOLOR',(0,0),(-1,0), colors.HexColor('#1d4ed8')),
                        ('GRID',(0,0),(-1,-1), 0.25, colors.grey),
                        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                    ]))
                    story.append(tbl)
                story.append(Spacer(1, 12))

                # Chapitre 2: Patrimoine et revenu (totaux)
                story.append(Paragraph("2. Patrimoine et revenu", H2))
                story.append(Paragraph(f"Actifs: {ctx['actif_total']} € — Passifs: {ctx['passif_total']} €", P))
                story.append(Paragraph(f"Revenus: {ctx['revenu_total']} € — Charges: {ctx['charge_total']} €", P))
                story.append(Paragraph(f"Patrimoine net: {ctx['patrimoine_net']} €", P))
                story.append(Paragraph(f"Solde budget: {ctx['budget_net']} €", P))
                # Charts (if available)
                try:
                    if chart_img_synth:
                        story.append(Spacer(1, 8))
                        story.append(RLImage(io.BytesIO(chart_img_synth), width=480, height=220))
                    if chart_img_pie:
                        story.append(Spacer(1, 6))
                        story.append(RLImage(io.BytesIO(chart_img_pie), width=260, height=260))
                except Exception:
                    pass
                story.append(PageBreak())

                # Chapitre 3: Objectifs
                story.append(Paragraph("3. Objectifs", H2))
                if objectifs:
                    data = [["Objectif", "Priorité", "Horizon", "Commentaire"]]
                    for o in objectifs:
                        data.append([
                            o.get('objectif_libelle') or '-', str(o.get('niveau_id') or '-'),
                            o.get('horizon_investissement') or '-', o.get('commentaire') or '-',
                        ])
                    tbl = Table(data, hAlign='LEFT', colWidths=[140, 60, 120, 200])
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#eef2ff')),
                        ('TEXTCOLOR',(0,0),(-1,0), colors.HexColor('#1d4ed8')),
                        ('GRID',(0,0),(-1,-1), 0.25, colors.grey),
                        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                    ]))
                    story.append(tbl)
                story.append(PageBreak())

                # Chapitre 4: Connaissances financières (récap synthétique)
                story.append(Paragraph("4. Connaissances financières", H2))
                if risque:
                    story.append(Paragraph(f"Niveau final: {risque.get('niveau_id')} — Allocation: {risque.get('allocation_nom') or '-'}", P))
                    story.append(Paragraph(f"Expérience: {risque.get('experience') or '-'} — Connaissance: {risque.get('connaissance') or '-'}", P))
                    story.append(Paragraph(f"Perte acceptée: {risque.get('contraintes') or '-'} — Confirmation: {risque.get('confirmation_client') or '-'}", P))
                    if risque.get('commentaire'):
                        story.append(Paragraph(f"Commentaire: {risque.get('commentaire')}", P))
                    # Allocation chart
                    try:
                        if chart_img_alloc:
                            story.append(Spacer(1, 8))
                            story.append(RLImage(io.BytesIO(chart_img_alloc), width=500, height=240))
                    except Exception:
                        pass
                    story.append(Spacer(1, 12))

                # Chapitre 5: ESG
                story.append(Paragraph("5. Sensibilité ESG", H2))
                if esg:
                    story.append(Paragraph(f"Environnement — Importance: {esg.get('env_importance') or '-'}; Réduction GES: {esg.get('env_ges_reduc') or '-'}", P))
                    story.append(Paragraph(f"Social — Droits humains: {esg.get('soc_droits_humains') or '-'}; Parité: {esg.get('soc_parite') or '-'}", P))
                    story.append(Paragraph(f"Gouvernance — Transparence: {esg.get('gov_transparence') or '-'}; Contrôle éthique: {esg.get('gov_controle_ethique') or '-'}", P))
                    if esg_exclusions:
                        story.append(Paragraph("Exclusions:", H3))
                        for e in esg_exclusions:
                            story.append(Paragraph(f"• {e}", P))

                doc.build(story)
                pdf_bytes = buf.getvalue()
                buf.close()
                return StreamingResponse(
                    iter([pdf_bytes]), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=rapport_kyc_{client_id}.pdf"},
                )
            except Exception as _exc_rl:
                logger.debug("ReportLab export failed: %s", _exc_rl, exc_info=True)
        # 3) Try WeasyPrint (fallback if not preferred or after RL failure)
        try:
            from weasyprint import HTML  # type: ignore
            pdf_bytes = HTML(string=html, base_url=str(request.url)).write_pdf()
            return StreamingResponse(
                iter([pdf_bytes]), media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=rapport_kyc_{client_id}.pdf"},
            )
        except Exception as _exc_wp:
            logger.debug("WeasyPrint export failed (final): %s", _exc_wp, exc_info=True)
            return templates.TemplateResponse(
                template_name,
                ctx | {"error": "Export PDF indisponible (WeasyPrint non installé). Le rendu HTML reflète fidèlement le rapport."},
            )
    return templates.TemplateResponse(template_name, ctx)


# ---------------- Paramètres (référentiels) ----------------
@router.get("/parametres", response_class=HTMLResponse)
def dashboard_parametres(request: Request, db: Session = Depends(get_db)):
    """Affiche la page Paramètres. Données minimales pour l'accès à la page.

    Les actions de création/mise à jour/suppression référencées par le template
    (/dashboard/parametres/...) ne sont pas encore implémentées ici. Cette route
    permet au moins d'accéder à la page, en attendant les endpoints POST.
    """
    open_section = request.query_params.get("open") or None
    saved_success = True if (request.query_params.get("saved") in ("1", "true", "yes")) else False
    saved_error = True if (request.query_params.get("error") in ("1", "true", "yes")) else False
    saved_error_msg = request.query_params.get("errmsg") or None
    raw_edit_doc_id = request.query_params.get("editDoc")
    edit_doc_id: int | None = None
    try:
        if raw_edit_doc_id is not None and str(raw_edit_doc_id).strip() != "":
            edit_doc_id = int(str(raw_edit_doc_id))
    except Exception:
        edit_doc_id = None

    # Valeurs par défaut pour garantir le rendu même si certaines tables manquent
    contrats_generiques: list[dict] = []
    societes: list[dict] = []
    contrat_categories: list[dict] = []
    societe_categories: list[dict] = []
    contrat_supports: list[dict] = []
    # Courtier identity (single row) and reference lists
    courtier: dict | None = None
    statut_sociaux: list[dict] = []
    associations_prof: list[dict] = []
    autorites_mediation: list[dict] = []
    # Courtier Détails: Activité
    activite_refs: list[dict] = []
    activite_items: list[dict] = []
    # Courtier Détails: Relation commerciale
    relation_items: list[dict] = []
    # Courtier Détails: Rémunération
    remun_refs: list[dict] = []
    remun_items: list[dict] = []
    remun_extra_cols: list[dict] = []
    # Courtier Détails: Assurances & garanties (normes)
    garanties_normes: list[dict] = []
    supports: list[dict] = []
    # Administration (relations / RH)
    admin_types: list[dict] = []
    admin_intervenants: list[dict] = []
    groupes_details: list[dict] = []
    # Courtier Documents officiels
    compliance_activites: list[dict] = []
    courtier_documents: list[dict] = []
    courtier_document_edit: dict | None = None
    # KPIs portefeuille
    total_clients = 0
    total_affaires = 0
    total_valo = 0.0
    total_valo_str = "0"

    # Chargement des données si les tables existent
    from sqlalchemy import text as _text

    try:
        total_clients = db.query(func.count(Client.id)).scalar() or 0
    except Exception:
        total_clients = 0
    try:
        total_affaires = db.query(func.count(Affaire.id)).scalar() or 0
    except Exception:
        total_affaires = 0
    # Analyse finance retirée de la page Management pour accélérer le rendu
    total_valo = 0.0
    try:
        total_valo_str = "{:,.0f}".format(float(total_valo or 0)).replace(",", " ")
    except Exception:
        total_valo_str = "0"

    # Analyse financière (supports) pour Paramètres > Portefeuille
    finance_rh_options: list[dict[str, str | int]] = []
    try:
        for rh in fetch_rh_list(db):
            try:
                rid = int(rh.get("id"))
            except Exception:
                continue
            prenom = (rh.get("prenom") or "").strip()
            nom = (rh.get("nom") or "").strip()
            mail = (rh.get("mail") or "").strip()
            label_parts = [p for p in [prenom, nom] if p]
            if label_parts:
                label = " ".join(label_parts)
            elif mail:
                label = mail
            else:
                label = f"RH #{rid}"
            finance_rh_options.append({"id": rid, "label": label})
    except Exception:
        finance_rh_options = []
    finance_rh_options.sort(key=lambda x: str(x.get("label", "")).lower())

    finance_rh_param = request.query_params.get("finance_rh")
    finance_rh_id: int | None = None
    if finance_rh_param not in (None, ""):
        try:
            finance_rh_id = int(finance_rh_param)
        except (TypeError, ValueError):
            finance_rh_id = None

    # Bloc finance neutralisé pour Management
    finance_supports = []
    finance_total_valo = 0.0
    finance_total_valo_str = "0"
    finance_date_input = None
    finance_effective_date_display = None
    finance_effective_date_iso = None
    finance_max_date_display = None
    finance_max_date_iso = None
    finance_valo_input = None
    finance_sql_literal = None

    def _load_admin_types_list() -> list[dict]:
        try:
            rows = rows_to_dicts(
                db.execute(
                    _text(
                        """
                        SELECT id, nom, inverse_id
                        FROM administration_type_relation
                        ORDER BY nom
                        """
                    )
                ).fetchall()
            )
        except Exception:
            return []
        inv = {item.get("id"): item.get("nom") for item in rows}
        for item in rows:
            inv_id = item.get("inverse_id")
            item["inverse_nom"] = inv.get(inv_id)
        return rows
    try:
        societes = rows_to_dicts(
            db.execute(
                _text(
                    """
                    SELECT id, nom, id_ctg, contact, telephone, email, commentaire
                    FROM mariadb_societe
                    ORDER BY nom
                    """
                )
            ).fetchall()
        )
    except Exception:
        societes = []

    try:
        # Typologies des contrats: table mariadb_affaires_generique_ctg
        contrat_categories = rows_to_dicts(
            db.execute(
                _text(
                    "SELECT id, libelle, description FROM mariadb_affaires_generique_ctg ORDER BY libelle"
                )
            ).fetchall()
        )
    except Exception:
        contrat_categories = []

    try:
        societe_categories = rows_to_dicts(
            db.execute(
                _text(
                    "SELECT id, libelle, description FROM mariadb_societe_ctg ORDER BY libelle"
                )
            ).fetchall()
        )
    except Exception:
        societe_categories = []

    try:
        contrats_generiques = rows_to_dicts(
            db.execute(
                _text(
                    """
                    SELECT g.id,
                           g.nom_contrat,
                           g.id_societe,
                           g.id_ctg,
                           g.frais_gestion_assureur,
                           g.frais_gestion_courtier,
                           s.nom AS societe_nom
                    FROM mariadb_affaires_generique g
                    LEFT JOIN mariadb_societe s ON s.id = g.id_societe
                    WHERE COALESCE(g.actif, 1) = 1
                    ORDER BY s.nom, g.nom_contrat
                    """
                )
            ).fetchall()
        )
    except Exception:
        contrats_generiques = []

    try:
        supports = rows_to_dicts(
            db.execute(
                _text("SELECT id, nom, code_isin FROM mariadb_support ORDER BY nom")
            ).fetchall()
        )
    except Exception:
        supports = []

    admin_rh: list[dict] = []
    try:
        admin_rh = rows_to_dicts(
            db.execute(
                _text(
                    """
                    SELECT id, nom, prenom, telephone, mail, niveau_poste, commentaire
                    FROM administration_RH
                    ORDER BY nom, prenom
                    """
                )
            ).fetchall()
        )
    except Exception:
        admin_rh = []

    admin_types = _load_admin_types_list()

    def _load_admin_intervenants_list() -> list[dict]:
        try:
            rows = rows_to_dicts(
                db.execute(
                    _text(
                        """
                        SELECT id, nom, type_niveau, type_personne, telephone, mail
                        FROM administration_intervenant
                        ORDER BY nom
                        """
                    )
                ).fetchall()
            )
        except Exception:
            return []
        return rows

    admin_intervenants = _load_admin_intervenants_list()

    groupes_details = _load_groupes_details(db)

    try:
        # Carte des supports par contrat (avec enrichissements nécessaires au tableau/CSV)
        contrat_supports = rows_to_dicts(
            db.execute(
                _text(
                    """
                    SELECT cs.id,
                           cs.id_affaire_generique,
                           cs.id_support,
                           cs.taux_retro,
                           s.nom AS support_nom,
                           s.code_isin,
                           s.cat_gene,
                           s.cat_geo,
                           s.promoteur,
                           g.nom_contrat AS contrat_nom,
                           so.nom AS societe_nom
                    FROM mariadb_contrat_supports cs
                    JOIN mariadb_support s ON s.id = cs.id_support
                    JOIN mariadb_affaires_generique g ON g.id = cs.id_affaire_generique
                    LEFT JOIN mariadb_societe so ON so.id = g.id_societe
                    ORDER BY g.nom_contrat, s.nom
                    """
                )
            ).fetchall()
        )
    except Exception:
        contrat_supports = []

    # ----- Courtier: identité (1 seule ligne) + référentiels -----
    try:
        from sqlalchemy import text as _text
        row = db.execute(_text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            courtier = dict(row._mapping)
    except Exception:
        courtier = None
    statut_sociaux = []
    associations_prof = []
    autorites_mediation = []
    try:
        statut_sociaux = _fetch_ref_list(db, [
            "DER_statut_social",
            "DER_statuts_sociaux",
        ])
    except Exception:
        pass
    try:
        associations_prof = _fetch_ref_list(db, [
            "DER_association_professionnelle",
            "DER_association_prof",
        ])
    except Exception:
        pass
    try:
        autorites_mediation = _fetch_ref_list(db, [
            "DER_courtier_ref_autorite",
            "DER_autorite_mediation",
            "DER_courtier_autorite",
        ])
    except Exception:
        pass

    # -------- Courtier: Documents officiels
    orias_detail: dict | None = None
    compliance_table = _resolve_table_name(db, ["Compliance_rc_pro", "compliance_rc_pro"]) or "Compliance_rc_pro"
    try:
        compliance_activites = rows_to_dicts(
            db.execute(
                text(
                    f"""
                    SELECT id, activite
                    FROM {compliance_table}
                    ORDER BY activite
                    """
                )
            ).fetchall()
        )
    except Exception:
        compliance_activites = []

    try:
        orias_detail_row = db.execute(
            text(
                f"""
                SELECT *
                FROM {compliance_table}
                WHERE id = :oid
                LIMIT 1
                """
            ),
            {"oid": 1},
        ).fetchone()
        orias_detail = dict(orias_detail_row._mapping) if orias_detail_row else None
    except Exception:
        orias_detail = None

    doc_table_name = _ensure_courtier_documents_table(db)
    if doc_table_name:
        try:
            raw_docs = rows_to_dicts(
                db.execute(
                    text(
                        f"""
                        SELECT d.id,
                               d.activite_id,
                               d.date_validite,
                               d.original_filename,
                               d.stored_filename,
                               d.stored_path,
                               d.mime_type,
                               d.file_size,
                               d.created_at,
                               d.updated_at,
                               c.activite AS activite_libelle
                        FROM {doc_table_name} d
                        LEFT JOIN Compliance_rc_pro c ON c.id = d.activite_id
                        ORDER BY COALESCE(c.activite, ''), d.date_validite DESC, d.id DESC
                        """
                    )
                ).fetchall()
            )
            today = _date.today()
            for item in raw_docs:
                parsed_date = _parse_date_safe(item.get("date_validite"))
                if parsed_date:
                    item["date_validite_display"] = parsed_date.strftime("%d/%m/%Y")
                    item["is_expired"] = parsed_date < today
                else:
                    item["date_validite_display"] = ""
                    item["is_expired"] = False
                iid = item.get("id")
                item["download_url"] = f"/dashboard/parametres/courtier/documents/{iid}/download"
                item["view_url"] = f"/dashboard/parametres/courtier/documents/{iid}/view"
                item["delete_url"] = f"/dashboard/parametres/courtier/documents/{iid}/delete"
                item["edit_url"] = f"/dashboard/parametres?open=courtierDocuments&editDoc={iid}"
                item["date_validite_value"] = parsed_date.isoformat() if parsed_date else ""
                if edit_doc_id is not None and iid == edit_doc_id and courtier_document_edit is None:
                    courtier_document_edit = dict(item)
                courtier_documents.append(item)
        except Exception as exc:
            logger.error(f"load courtier documents failed: {exc}")
    if edit_doc_id is not None and courtier_document_edit is None:
        if doc_table_name:
            try:
                row = _get_courtier_document(db, doc_table_name, edit_doc_id)
            except Exception:
                row = None
            if row:
                parsed_date = _parse_date_safe(row.get("date_validite"))
                courtier_document_edit = {
                    "id": row.get("id"),
                    "activite_id": row.get("activite_id"),
                    "date_validite_value": parsed_date.isoformat() if parsed_date else "",
                    "original_filename": row.get("original_filename") or row.get("stored_filename"),
                }

    # -------- Détails: Activité

    act_ref_table = _resolve_table_name(db, ["DER_courtier_activite_ref"])
    if act_ref_table:
        try:
            activite_refs = _fetch_ref_list(db, [act_ref_table])
        except Exception:
            activite_refs = []

    act_table = _resolve_table_name(db, ["DER_courtier_activite"])
    if act_table:
        id_col, fk_ref_col, fk_cour_col, creation_col, modification_col = _courtier_activite_columns(db, act_table)
        try:
            from sqlalchemy import text as _text
            where_clause = ""
            params = {}
            if fk_cour_col and courtier and courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = courtier.get("id")
            pk_expr = id_col or "id"
            rows = db.execute(_text(f"SELECT {pk_expr} AS __pk, * FROM {act_table}{where_clause}"), params).fetchall()
            # detect statut column
            statut_col = None
            cols = _sqlite_table_columns(db, act_table)
            for c in cols:
                nm = (c.get("name") or "").lower()
                if nm == "statut":
                    statut_col = c.get("name")
                    break
            for r in rows:
                m = r._mapping
                rid = m.get("__pk") if "__pk" in m else (m.get(id_col) if id_col else m.get("id"))
                ref_id = m.get(fk_ref_col)
                ref_label = None
                for ref in activite_refs:
                    if (ref.get("id") == ref_id) or (str(ref.get("id")) == str(ref_id)):
                        ref_label = ref.get("libelle")
                        break
                activite_items.append({
                    "id": rid,
                    "ref_id": ref_id,
                    "ref_label": ref_label,
                    "statut": m.get(statut_col) if statut_col else None,
                })
        except Exception:
            activite_items = []

    # Load relations commerciales
    rel_table = _resolve_table_name(db, ["DER_courtier_relation_commerciale"])
    if rel_table:
        try:
            id_col, fk_soc_col, fk_cour_col, creation_col, modification_col = _courtier_relation_columns(db, rel_table)
            where_clause = ""
            params = {}
            if fk_cour_col and courtier and courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = courtier.get("id")
            pk_expr = id_col or "id"
            rows = db.execute(text(f"SELECT {pk_expr} AS __pk, * FROM {rel_table}{where_clause}"), params).fetchall()
            for r in rows:
                m = r._mapping
                rid = m.get("__pk") if "__pk" in m else (m.get(id_col) if id_col else m.get("id"))
                sid = m.get(fk_soc_col)
                # find label from societes already loaded
                s_label = None
                for s in societes:
                    if (s.get("id") == sid) or (str(s.get("id")) == str(sid)):
                        s_label = s.get("nom")
                        break
                relation_items.append({"id": rid, "societe_id": sid, "societe_nom": s_label})
        except Exception as e:
            logger.error(f"load relation commerciales failed: {e}")

    # Load rémunération (mode de facturation)
    try:
        remun_refs = _fetch_ref_list(db, ["DER_courtier_mode_facturation_ref"]) or []
    except Exception:
        remun_refs = []
    rem_table = _resolve_table_name(db, ["DER_courtier_mode_facturation"])
    if rem_table:
        try:
            # infer columns similar to activite/relation
            cols = _sqlite_table_columns(db, rem_table)
            colnames = [c.get("name") for c in cols]
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
            # fk to ref
            import unicodedata
            def norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s)
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            norm_map = {name: norm(name) for name in colnames}
            inv_map = {v: k for k, v in norm_map.items()}
            fk_ref_col = None
            for cand in ["id_ref","id_mode_facturation_ref","id_courtier_mode_facturation_ref","mode_ref","mode","id_mode_ref"]:
                if cand in inv_map:
                    fk_ref_col = inv_map[cand]
                    break
            if not fk_ref_col:
                for name in colnames:
                    if name != id_col:
                        fk_ref_col = name
                        break
            # fk to courtier
            fk_cour_col = None
            for cand in ["id_courtier","courtier_id","id_der_courtier","id_cabinet"]:
                if cand in inv_map:
                    fk_cour_col = inv_map[cand]
                    break
            # detect creation/modification date cols to exclude from UI
            creation_col = next((c for c in ["date_creation", "date_création", "dateCreation"] if c in colnames), None)
            modification_col = next((c for c in ["date_obsolescence", "date_modification", "date_modif", "date_maj", "dateMiseAJour"] if c in colnames), None)

            # Extra editable columns = all except id / fk_ref / fk_courtier / date cols / 'type'
            extras = []
            for c in cols:
                nm = c.get("name")
                if nm in (id_col, fk_ref_col, fk_cour_col, creation_col, modification_col):
                    continue
                if (nm or '').lower() == 'type':
                    continue
                extras.append({"name": nm, "type": (c.get("type") or "")})
            remun_extra_cols = extras

            where_clause = ""
            params = {}
            if fk_cour_col and courtier and courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = courtier.get("id")
            pk_expr = id_col or "id"
            rows = db.execute(text(f"SELECT {pk_expr} AS __pk, * FROM {rem_table}{where_clause}"), params).fetchall()
            for r in rows:
                m = r._mapping
                rid = m.get("__pk") if "__pk" in m else (m.get(id_col) if id_col else m.get("id"))
                # ref id direct si colonne présente
                ref_id_fk = m.get(fk_ref_col) if fk_ref_col else None
                # Dérive le domaine depuis 'type' en base
                raw_type = str(m.get('type') or '')
                import unicodedata
                def _norm_u(s: str) -> str:
                    s2 = unicodedata.normalize('NFKD', s or '')
                    s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                    return s2.upper()
                def _map_type(val: str) -> str:
                    v = _norm_u(val)
                    if 'HONOR' in v or 'PONCT' in v:
                        return 'HONORAIRES'
                    if 'ENTREE' in v:
                        return 'FRAIS_ENTREE'
                    if 'GESTION' in v:
                        return 'FRAIS_GESTION'
                    return ''
                tdom = _map_type(raw_type)
                # Choisir ref_id prioritairement via type, sinon via FK
                ref_id = None
                ref_label = None
                if tdom:
                    for ref in remun_refs:
                        lbl = _map_type(str(ref.get('libelle') or ''))
                        if lbl == tdom:
                            ref_id = ref.get('id')
                            ref_label = ref.get('libelle')
                            break
                if ref_id is None and ref_id_fk is not None:
                    ref_id = ref_id_fk
                    for ref in remun_refs:
                        if (ref.get('id') == ref_id) or (str(ref.get('id')) == str(ref_id)):
                            ref_label = ref.get('libelle')
                            break
                extra_vals = {}
                for ex in extras:
                    extra_vals[ex["name"]] = m.get(ex["name"]) if ex["name"] in m else None
                # convenience fields for display
                montant_val = extra_vals.get("montant") if "montant" in extra_vals else extra_vals.get("valeur")
                pct_val = extra_vals.get("pourcentage") if "pourcentage" in extra_vals else extra_vals.get("taux")
                type_raw = m.get('type') if 'type' in colnames else None
                remun_items.append({
                    "id": rid,
                    "ref_id": ref_id,
                    "ref_label": ref_label,
                    "extra": extra_vals,
                    "montant": montant_val,
                    "pourcentage": pct_val,
                    "type": type_raw,
                })
        except Exception as e:
            logger.error(f"load remuneration failed: {e}")

    # Load garanties normes
    gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
    if gar_table:
        try:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            # detect columns by normalized names
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(n): n for n in colnames }
            c_type = inv.get('type_garantie') or inv.get('type') or inv.get('garantie') or colnames[0]
            c_ias = inv.get('ias') or 'IAS'
            c_iobsp = inv.get('iobsp') or 'IOBSP'
            c_immo = inv.get('immo') or 'IMMO'
            # determine primary key column if any
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            for r in rows:
                m = r._mapping
                garanties_normes.append({
                    'id': m.get('__pk') if '__pk' in m else (m.get(id_col) if id_col else None),
                    'type_garantie': m.get(c_type),
                    'IAS': m.get(c_ias),
                    'IOBSP': m.get(c_iobsp),
                    'IMMO': m.get(c_immo),
                    '_c_ias': c_ias,
                    '_c_iobsp': c_iobsp,
                    '_c_immo': c_immo,
                })
        except Exception as e:
            logger.error(f"load garanties normes failed: {e}")

    # Types/Statuts pour tâches groupées (onglet Groupes)
    try:
        evt_types_rows = db.execute(text("SELECT libelle, categorie FROM mariadb_type_evenement ORDER BY categorie, libelle")).fetchall()
        evt_types = rows_to_dicts(evt_types_rows)
        evt_categories = sorted({(r.get("categorie") or "").strip() for r in evt_types if r.get("categorie")})
    except Exception:
        evt_types = []
        evt_categories = []
    try:
        evt_statuts_rows = db.execute(text("SELECT id, libelle FROM mariadb_statut_evenement ORDER BY libelle")).fetchall()
        evt_statuts = rows_to_dicts(evt_statuts_rows)
    except Exception:
        evt_statuts = []

    return templates.TemplateResponse(
        "dashboard_parametres.html",
        {
            "request": request,
            "open_section": open_section,
            "saved_success": saved_success,
            "saved_error": saved_error,
            "saved_error_msg": saved_error_msg,
            "contrats_generiques": contrats_generiques,
            "societes": societes,
            "contrat_categories": contrat_categories,
            "societe_categories": societe_categories,
            "contrat_supports": contrat_supports,
            "supports": supports,
            "courtier": courtier,
            "form_courtier": None,
            "form_errors": None,
            "statut_sociaux": statut_sociaux,
            "associations_prof": associations_prof,
            "autorites_mediation": autorites_mediation,
            "activite_refs": activite_refs,
            "activite_items": activite_items,
            "relation_items": relation_items,
            "remun_refs": remun_refs,
            "remun_items": remun_items,
            "remun_extra_cols": remun_extra_cols,
            "garanties_normes": garanties_normes,
            "compliance_activites": compliance_activites,
            "courtier_documents": courtier_documents,
            "courtier_document_edit": courtier_document_edit,
            "admin_types": admin_types,
            "admin_intervenants": admin_intervenants,
            "groupes_details": groupes_details,
            "admin_rh": admin_rh,
            "portfolio_total_clients": total_clients,
            "portfolio_total_affaires": total_affaires,
            "portfolio_total_valo": total_valo,
            "portfolio_total_valo_str": total_valo_str,
            "finance_supports": finance_supports,
            "finance_total_valo": finance_total_valo,
            "finance_total_valo_str": finance_total_valo_str,
            "finance_date_input": finance_date_input,
            "finance_effective_date_display": finance_effective_date_display,
            "finance_effective_date_iso": finance_effective_date_iso,
            "finance_max_date_display": finance_max_date_display,
            "finance_max_date_iso": finance_max_date_iso,
            "finance_rh_options": finance_rh_options,
            "finance_rh_selected": finance_rh_id,
            "finance_valo_input": finance_valo_input,
            "evt_types": evt_types,
            "evt_categories": evt_categories,
            "evt_statuts": evt_statuts,
        },
    )


@router.post("/parametres/courtier/documents", response_class=HTMLResponse)
async def upload_courtier_document(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    raw_doc_id = form.get("doc_id")
    raw_activite = form.get("activite_id") or form.get("activite")
    raw_date = form.get("date_validite")
    upload: UploadFile | None = form.get("document_file") or form.get("document_pdf") or form.get("document")

    def _parse_int(value) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except Exception:
            return None

    doc_id = _parse_int(raw_doc_id)

    async def _error(message: str, keep_edit: bool = False) -> RedirectResponse:
        from urllib.parse import quote

        if upload is not None:
            try:
                await upload.close()
            except Exception:
                pass
        url = "/dashboard/parametres?open=courtierDocuments"
        if keep_edit and doc_id is not None:
            url += f"&editDoc={doc_id}"
        url += f"&error=1&errmsg={quote(message)}"
        return RedirectResponse(url=url, status_code=303)

    activite_id = _parse_int(raw_activite)
    date_value = raw_date.strip() if isinstance(raw_date, str) else raw_date
    parsed_date = _parse_date_safe(date_value)
    if date_value and parsed_date is None:
        return await _error("Date de validité invalide.", keep_edit=True)
    if activite_id is None:
        return await _error("Type de document requis.", keep_edit=True)

    has_new_file = bool(upload and getattr(upload, "filename", ""))
    if doc_id is None and not has_new_file:
        return await _error("Merci de sélectionner un fichier PDF.")

    if has_new_file:
        mime_type = (upload.content_type or "").lower()
        filename_lower = (upload.filename or "").lower()
        if not filename_lower.endswith(".pdf") and mime_type != "application/pdf":
            return await _error("Le fichier doit être un PDF.", keep_edit=True)
    else:
        mime_type = None

    try:
        exists = db.execute(
            text("SELECT 1 FROM Compliance_rc_pro WHERE id = :id LIMIT 1"), {"id": activite_id}
        ).fetchone()
    except Exception:
        exists = None
    if not exists:
        return await _error("Type de document introuvable.", keep_edit=True)

    table_name = _ensure_courtier_documents_table(db)
    if not table_name:
        return await _error("Impossible de préparer le stockage des documents.", keep_edit=True)

    existing_doc = None
    if doc_id is not None:
        existing_doc = _get_courtier_document(db, table_name, doc_id)
        if not existing_doc:
            return await _error("Document introuvable.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    original_filename: str
    content_type: str
    stored_filename: str | None
    stored_path: str | None
    file_size: int
    new_dest_path: Path | None = None
    previous_path: Path | None = None

    if has_new_file:
        original_filename = upload.filename or "document.pdf"
        content_type = upload.content_type or "application/pdf"
        base_slug = _slugify_filename(Path(original_filename).stem)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        stored_filename = f"{base_slug}-{timestamp}-{secrets.token_hex(4)}.pdf"
        stored_path = str(Path("documents") / stored_filename)
        try:
            file_bytes = await upload.read()
        finally:
            try:
                await upload.close()
            except Exception:
                pass
        if not file_bytes:
            return await _error("Le fichier est vide.", keep_edit=True)
        dest_path = DOCUMENTS_DIR / stored_filename
        try:
            with open(dest_path, "wb") as buffer:
                buffer.write(file_bytes)
        except Exception as exc:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass
            logger.error(f"failed to write uploaded document: {exc}")
            return await _error("Enregistrement du fichier impossible.", keep_edit=True)
        file_size = len(file_bytes)
        new_dest_path = dest_path
    else:
        if existing_doc is None:
            return await _error("Merci de sélectionner un fichier PDF.")
        original_filename = (
            existing_doc.get("original_filename")
            or existing_doc.get("stored_filename")
            or "document.pdf"
        )
        content_type = existing_doc.get("mime_type") or "application/pdf"
        stored_filename = existing_doc.get("stored_filename")
        stored_path = existing_doc.get("stored_path") or stored_filename
        file_size = existing_doc.get("file_size") or 0
        try:
            previous_path = _resolve_document_path(existing_doc)
        except Exception:
            previous_path = None

    date_iso = parsed_date.isoformat() if parsed_date else None

    if doc_id is None:
        params = {
            "activite_id": activite_id,
            "date_validite": date_iso,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "stored_path": stored_path,
            "mime_type": content_type,
            "file_size": file_size,
        }
        try:
            db.execute(
                text(
                    f"""
                    INSERT INTO {table_name} (
                        activite_id, date_validite, original_filename, stored_filename,
                        stored_path, mime_type, file_size, created_at, updated_at
                    ) VALUES (
                        :activite_id, :date_validite, :original_filename, :stored_filename,
                        :stored_path, :mime_type, :file_size, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    """
                ),
                params,
            )
            db.commit()
        except Exception as exc:
            logger.error(f"failed to persist courtier document metadata: {exc}")
            try:
                if new_dest_path and new_dest_path.exists():
                    new_dest_path.unlink()
            except Exception:
                pass
            return await _error("Impossible d'enregistrer le document.")
    else:
        update_params = {
            "id": doc_id,
            "activite_id": activite_id,
            "date_validite": date_iso,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "stored_path": stored_path,
            "mime_type": content_type,
            "file_size": file_size,
        }
        try:
            db.execute(
                text(
                    f"""
                    UPDATE {table_name}
                    SET activite_id = :activite_id,
                        date_validite = :date_validite,
                        original_filename = :original_filename,
                        stored_filename = :stored_filename,
                        stored_path = :stored_path,
                        mime_type = :mime_type,
                        file_size = :file_size,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                    """
                ),
                update_params,
            )
            db.commit()
        except Exception as exc:
            logger.error(f"failed to update courtier document metadata: {exc}")
            try:
                if new_dest_path and new_dest_path.exists():
                    new_dest_path.unlink()
            except Exception:
                pass
            return await _error("Impossible de mettre à jour le document.", keep_edit=True)
        if has_new_file and previous_path and previous_path.exists() and previous_path != new_dest_path:
            try:
                previous_path.unlink()
            except Exception as exc:
                logger.warning(f"failed to remove previous document file {previous_path}: {exc}")

    return RedirectResponse(
        url="/dashboard/parametres?open=courtierDocuments&saved=1",
        status_code=303,
    )


@router.post("/parametres/courtier/documents/{doc_id}/delete", response_class=HTMLResponse)
async def delete_courtier_document(doc_id: int, request: Request, db: Session = Depends(get_db)):
    table_name = _ensure_courtier_documents_table(db)
    if not table_name:
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/parametres?open=courtierDocuments&error=1&errmsg={quote('Aucune table de documents disponible.')}",
            status_code=303,
        )
    row = _get_courtier_document(db, table_name, doc_id)
    if not row:
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/parametres?open=courtierDocuments&error=1&errmsg={quote('Document introuvable.')}",
            status_code=303,
        )
    file_path = _resolve_document_path(row)
    try:
        db.execute(
            text(f"DELETE FROM {table_name} WHERE id = :id"),
            {"id": doc_id},
        )
        db.commit()
    except Exception as exc:
        logger.error(f"failed to delete courtier document metadata: {exc}")
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/parametres?open=courtierDocuments&error=1&errmsg={quote('Suppression impossible.')}",
            status_code=303,
        )

    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as exc:
        logger.warning(f"failed to remove document file {file_path}: {exc}")

    return RedirectResponse(
        url="/dashboard/parametres?open=courtierDocuments&saved=1",
        status_code=303,
    )


@router.get("/parametres/courtier/documents/{doc_id}/download")
def download_courtier_document(doc_id: int, db: Session = Depends(get_db)):
    return _serve_courtier_document(doc_id, inline=False, db=db)


@router.get("/parametres/courtier/documents/{doc_id}/view")
def view_courtier_document(doc_id: int, db: Session = Depends(get_db)):
    return _serve_courtier_document(doc_id, inline=True, db=db)


@router.post("/parametres/courtier/garanties/{row_id}", response_class=HTMLResponse)
async def update_courtier_garanties(row_id: str, request: Request, db: Session = Depends(get_db)):
    table = _resolve_table_name(db, ["DER_courtier_garanties_normes"]) or "DER_courtier_garanties_normes"
    cols = _sqlite_table_columns(db, table)
    colnames = [c.get("name") for c in cols]
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
    if not id_col:
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote('Impossible de déterminer la clé primaire des garanties.')}",
            status_code=303,
        )
    import unicodedata
    def _norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s or '')
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()
    inv = { _norm(n): n for n in colnames }
    c_ias = inv.get('ias') or 'IAS'
    c_iobsp = inv.get('iobsp') or 'IOBSP'
    c_immo = inv.get('immo') or 'IMMO'
    form = await request.form()
    def _as_int_like(raw: str | None) -> int | None:
        v = _as_float(raw, None)
        if v is None:
            return None
        try:
            return int(round(v))
        except Exception:
            return None
    params = {
        '__pk__': row_id,
        '___ias': _as_int_like(form.get(c_ias)),
        '___iobsp': _as_int_like(form.get(c_iobsp)),
        '___immo': _as_int_like(form.get(c_immo)),
    }
    # Build set clause with actual column names
    set_parts = [ f"{c_ias} = :___ias", f"{c_iobsp} = :___iobsp", f"{c_immo} = :___immo" ]
    try:
        db.execute(text(f"UPDATE {table} SET {', '.join(set_parts)} WHERE {id_col} = :__pk__"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"update garanties failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


def _get_db_dialect(db: Session) -> str:
    """Return the lower-cased SQLAlchemy dialect name (mysql, sqlite, etc.)."""
    bind = None
    try:
        bind = db.get_bind()
    except Exception:
        bind = getattr(db, "bind", None)
    if not bind:
        return ""
    dialect = getattr(getattr(bind, "dialect", None), "name", "") or ""
    return str(dialect).lower()


def _sqlite_table_columns(db: Session, table: str) -> list[dict]:
    """Return PRAGMA-like metadata for the target table on both SQLite and MariaDB/MySQL."""
    from sqlalchemy import text as _text

    dialect = _get_db_dialect(db)
    if dialect in ("mysql", "mariadb"):
        sql = _text(
            """
            SELECT
                ordinal_position AS cid,
                column_name      AS name,
                column_type      AS type,
                CASE WHEN is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                column_default   AS dflt_value,
                CASE WHEN column_key = 'PRI' THEN 1 ELSE 0 END AS pk
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = :tbl
            ORDER BY ordinal_position
            """
        )
        try:
            rows = db.execute(sql, {"tbl": table}).fetchall()
            return rows_to_dicts(rows)
        except Exception:
            return []

    # SQLite (or unknown) fallback via PRAGMA
    try:
        cols = rows_to_dicts(db.execute(_text(f"PRAGMA table_info('{table}')")).fetchall())
        return cols
    except Exception:
        return []


def _table_primary_key(db: Session, table: str) -> str | None:
    """Return the first primary key column name or a sensible fallback (id/first column)."""
    cols = _sqlite_table_columns(db, table)
    if not cols:
        return None
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    if pk_cols:
        return pk_cols[0]
    colnames = [c.get("name") for c in cols if c.get("name")]
    if "id" in colnames:
        return "id"
    return colnames[0] if colnames else None


def _last_insert_id(db: Session) -> int | None:
    """Return the last inserted row identifier for the current session/connection."""
    from sqlalchemy import text as _text

    dialect = _get_db_dialect(db)
    query = "SELECT LAST_INSERT_ID()" if dialect in ("mysql", "mariadb") else "SELECT last_insert_rowid()"
    try:
        row = db.execute(_text(query)).fetchone()
    except Exception:
        return None
    if not row:
        return None
    value = None
    try:
        value = row[0]
    except Exception:
        try:
            mapping = getattr(row, "_mapping", {})
            value = next(iter(mapping.values())) if mapping else None
        except Exception:
            value = None
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return value


def _coerce_value(raw: str | None, col_type: str) -> object | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    t = (col_type or "").upper()
    try:
        if any(x in t for x in ["INT"]):
            return int(s)
        if any(x in t for x in ["REAL", "FLOA", "DOUB", "DEC", "NUM"]):
            return float(s.replace(",", "."))
        if any(x in t for x in ["DATE", "TIME"]):
            return s  # laisser tel quel (ISO conseillé)
        return s
    except Exception:
        return s


def _allowed_courtier_tables(db: Session) -> set[str]:
    from sqlalchemy import text as _text
    names = set()
    dialect = _get_db_dialect(db)
    if dialect in ("mysql", "mariadb"):
        query = _text(
            """
            SELECT table_name AS name
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name LIKE 'DER_courtier%'
            """
        )
    else:
        query = _text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'DER_courtier%'"
        )
    try:
        rows = db.execute(query).fetchall()
        for r in rows:
            name = None
            try:
                name = r[0]
            except Exception:
                name = getattr(getattr(r, "_mapping", {}), "get", lambda *_: None)("name")
            if name:
                names.add(name)
    except Exception:
        pass
    return names


def _fetch_ref_list(db: Session, candidates: list[str]) -> list[dict]:
    """Retourne [{id, libelle}] depuis la première table existante parmi candidates.
    Détection robuste de la colonne libellé (insensible à la casse/accents) parmi: libelle, libellé, label, nom, name, intitulé, intitule.
    """
    from sqlalchemy import text as _text
    import unicodedata

    def _norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s)
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()

    table = _resolve_table_name(db, candidates)
    if not table:
        return []

    cols = _sqlite_table_columns(db, table)
    if not cols:
        return []

    # Déterminer colonne id: priorité PK sinon 'id', sinon première
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else None
    if not id_col:
        # rechercher 'id' normalisé
        name_map = {c.get("name"): _norm(c.get("name") or "") for c in cols}
        inv_map = {v: k for k, v in name_map.items()}
        id_col = inv_map.get("id", cols[0].get("name"))
    # Déterminer colonne label
    # Préférence: libellé/libelle/label/nom/name/intitulé/intitule; sinon première non-id
    name_map = {c.get("name"): _norm(c.get("name") or "") for c in cols}
    inv_map = {v: k for k, v in name_map.items()}
    for pref in ["libelle", _norm("libellé"), "label", "nom", "name", "intitule", _norm("intitulé")]:
        if pref in inv_map:
            label_col = inv_map[pref]
            break
    else:
        label_col = next((c.get("name") for c in cols if c.get("name") != id_col), id_col)

    raw_rows = db.execute(_text(f"SELECT * FROM {table}")).fetchall()
    items = []
    for row in raw_rows:
        m = row._mapping
        try:
            iid = m.get(id_col) if id_col and id_col in m else m.get("id")
            lbl = m.get(label_col)
            if lbl is None:
                # Dernier secours: première colonne non id
                for k in m.keys():
                    if k not in (id_col, "id"):
                        lbl = m.get(k)
                        if lbl is not None:
                            break
            items.append({"id": iid, "libelle": lbl})
        except Exception:
            continue
    # Trier par libellé (sécurité)
    try:
        items.sort(key=lambda x: (str(x.get("libelle") or "").lower(), str(x.get("id"))))
    except Exception:
        pass
    return items


def _resolve_table_name(db: Session, candidates: list[str]) -> str | None:
    """Return the first existing table/view name matching one of candidates (case-insensitive)."""
    from sqlalchemy import text as _text

    bind = db.get_bind()
    dialect = (getattr(bind, "dialect", None).name or "").lower() if bind else ""
    if dialect.startswith("sqlite"):
        lookup_sql = (
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND lower(name) = lower(:n) LIMIT 1"
        )
    elif dialect in ("mysql", "mariadb"):
        lookup_sql = (
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND lower(table_name) = lower(:n) "
            "LIMIT 1"
        )
    else:
        # Fallback: try ANSI INFORMATION_SCHEMA if available
        lookup_sql = (
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE lower(table_name) = lower(:n) LIMIT 1"
        )

    for cand in candidates:
        row = db.execute(_text(lookup_sql), {"n": cand}).fetchone()
        if row:
            try:
                return row[0]
            except Exception:
                return row._mapping.get("name")
    return None


def _ensure_courtier_documents_table(db: Session) -> str | None:
    """Ensure the storage table for official broker documents exists and return its name."""
    table = _resolve_table_name(db, ["courtier_documents_officiels"])
    if table:
        return table
    try:
        db.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS courtier_documents_officiels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activite_id INTEGER NOT NULL,
                    date_validite TEXT,
                    original_filename TEXT,
                    stored_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT,
                    file_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (activite_id) REFERENCES Compliance_rc_pro(id)
                )
                """
            )
        )
        db.commit()
        return "courtier_documents_officiels"
    except Exception as exc:
        logger.error(f"failed to ensure courtier_documents_officiels table: {exc}")
        return None


def _get_courtier_document(db: Session, table: str, doc_id: int) -> dict | None:
    try:
        row = db.execute(
            text(f"SELECT * FROM {table} WHERE id = :id"),
            {"id": doc_id},
        ).fetchone()
        if row:
            return dict(row._mapping)
    except Exception as exc:
        logger.error(f"fetch courtier document failed for id={doc_id}: {exc}")
    return None


@router.get("/compliance_rc_pro/{item_id}", response_class=JSONResponse)
async def dashboard_compliance_rc_pro_detail(item_id: int, db: Session = Depends(get_db)):
    """Return the Compliance RC Pro record as JSON for dynamic displays."""
    table_name = _resolve_table_name(db, ["Compliance_rc_pro", "compliance_rc_pro"]) or "Compliance_rc_pro"
    try:
        row = db.execute(
            text(
                f"""
                SELECT *
                FROM {table_name}
                WHERE id = :oid
                LIMIT 1
                """
            ),
            {"oid": item_id},
        ).fetchone()
    except Exception as exc:
        logger.error(f"fetch compliance rc pro failed: {exc}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des données de conformité.")
    if not row:
        raise HTTPException(status_code=404, detail="Enregistrement introuvable.")

    def _normalize(val):
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, _date):
            return val.isoformat()
        if isinstance(val, Decimal):
            try:
                return float(val)
            except (TypeError, InvalidOperation):
                return str(val)
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8")
            except Exception:
                return base64.b64encode(val).decode("ascii")
        return val

    payload = {key: _normalize(value) for key, value in row._mapping.items()}
    return JSONResponse(payload)


def _resolve_document_path(row: dict) -> Path:
    """Compute safe absolute path for a stored document entry."""
    base = DOCUMENTS_DIR.resolve()
    candidates: list[Path] = []
    stored_filename = row.get("stored_filename")
    stored_path = row.get("stored_path")
    if stored_filename:
        candidates.append((base / stored_filename))
    if stored_path:
        sp = Path(stored_path)
        if sp.is_absolute():
            candidates.append(sp)
        else:
            candidates.append((base / sp))
            candidates.append((BASE_DIR / sp))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        try:
            resolved.relative_to(base)
            return resolved
        except Exception:
            continue
    fallback_name = stored_filename or Path(stored_path or "document.pdf").name
    return base / fallback_name


def _serve_courtier_document(doc_id: int, inline: bool, db: Session) -> FileResponse:
    table = _ensure_courtier_documents_table(db)
    if not table:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    row = _get_courtier_document(db, table, doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    file_path = _resolve_document_path(row)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable sur le serveur.")
    mime_type = row.get("mime_type") or "application/pdf"
    original_filename = (row.get("original_filename") or file_path.name or "document.pdf").replace('"', "")
    if inline:
        headers = {"Content-Disposition": f'inline; filename="{original_filename}"'}
        return FileResponse(str(file_path), media_type=mime_type, headers=headers)
    return FileResponse(str(file_path), media_type=mime_type, filename=original_filename)


def _courtier_activite_columns(db: Session, table: str) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Infer (id_col, fk_ref_col, fk_courtier_col, creation_col, modification_col) for DER_courtier_activite-like table."""
    cols = _sqlite_table_columns(db, table)
    colnames = [c.get("name") for c in cols]
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
    import unicodedata
    def norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s)
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()
    norm_map = {name: norm(name) for name in colnames}
    inv_map = {v: k for k, v in norm_map.items()}
    fk_ref_col = None
    for cand in ["id_ref", "id_activite_ref", "id_courtier_activite_ref", "id_der_courtier_activite_ref", "ref", "activite", "activite_ref"]:
        if cand in inv_map:
            fk_ref_col = inv_map[cand]
            break
    if not fk_ref_col:
        for name in colnames:
            if name != id_col:
                fk_ref_col = name
                break
    # detect fk to courtier
    fk_courtier_col = None
    for cand in [
        "id_courtier", "id_der_courtier", "id_cabinet", "id_courtier_fk", "courtier_id",
    ]:
        if cand in inv_map:
            fk_courtier_col = inv_map[cand]
            break
    creation_col = next((c for c in ["date_creation", "date_création", "dateCreation"] if c in colnames), None)
    modification_col = next((c for c in ["date_obsolescence", "date_modification", "date_modif", "date_maj", "dateMiseAJour"] if c in colnames), None)
    return id_col, fk_ref_col, fk_courtier_col, creation_col, modification_col


def _build_der_context(db: Session) -> dict:
    """Assemble DER-related data used by client detail modal and DER PDF export."""
    context: dict = {
        "DER_courtier": None,
        "DER_statut_social": None,
        "DER_courtier_garanties_normes": [],
        "DER_sql_params_activite": {":cid": None},
        "lm_remunerations": [],
        "centre_mediation_lib": None,
        "DER_activites": [],
        "footer_text": None,
        "footer_line1": None,
        "footer_line2": None,
        "footer_line2b": None,
        "footer_line3": None,
        "footer_line4": None,
        "footer_line5": None,
        "footer_line6": None,
    }
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            der_courtier = dict(row._mapping)
            context["DER_courtier"] = der_courtier
            # Statut social resolution
            ss_label = None
            raw_val = None
            for key in ("statut_social", "statut", "statut_soc", "statut_social_lib"):
                if key in der_courtier and der_courtier.get(key) not in (None, ""):
                    raw_val = der_courtier.get(key)
                    break
            if raw_val is not None:
                try:
                    ss_id = int(str(raw_val).strip())
                    r2 = db.execute(text("SELECT lib FROM DER_statut_social WHERE id = :i"), {"i": ss_id}).fetchone()
                    if r2 and r2[0]:
                        ss_label = r2[0]
                except Exception:
                    ss_label = str(raw_val)
            context["DER_statut_social"] = {"lib": ss_label} if ss_label is not None else None
            # Params for DER activite SQL
            try:
                cid_val = der_courtier.get("id")
                context["DER_sql_params_activite"] = {":cid": cid_val} if cid_val is not None else {":cid": None}
            except Exception:
                context["DER_sql_params_activite"] = {":cid": None}
            # Centre de médiation libellé
            centre_mediation_lib = None
            try:
                raw_cm = der_courtier.get("centre_mediation")
                if raw_cm not in (None, ""):
                    try:
                        cm_id = int(str(raw_cm).strip())
                        row_cm = db.execute(text("SELECT lib FROM DER_courtier_ref_autorite WHERE id = :i"), {"i": cm_id}).fetchone()
                        if row_cm and row_cm[0]:
                            centre_mediation_lib = row_cm[0]
                    except Exception:
                        centre_mediation_lib = str(raw_cm)
            except Exception:
                centre_mediation_lib = None
            context["centre_mediation_lib"] = centre_mediation_lib
            # Footer lines (reuse in PDF rapports)
            try:
                nom_cabinet = (der_courtier.get("nom_cabinet") or der_courtier.get("raison_sociale") or "").strip()
                statut_id = der_courtier.get("statut_social")
                capital_social = der_courtier.get("capital_social")
                siren = (der_courtier.get("siren") or "").strip()
                rcs = (der_courtier.get("rcs") or "").strip()
                adresse_rue = (der_courtier.get("adresse_rue") or der_courtier.get("adresse") or "").strip()
                adresse_cp = (der_courtier.get("adresse_cp") or "").strip()
                adresse_ville = (der_courtier.get("adresse_ville") or "").strip()
                nom_responsable = (der_courtier.get("nom_responsable") or "").strip()
                courriel = (der_courtier.get("courriel") or der_courtier.get("email") or "").strip()
                numero_orias = (der_courtier.get("numero_orias") or "").strip()
                assoc_id = der_courtier.get("association_prof")
                num_adh = (der_courtier.get("num_adh_assoc") or "").strip()

                def _fmt_amount(val) -> str | None:
                    if val in (None, "", "-"):
                        return None
                    try:
                        f = float(val)
                    except Exception:
                        return str(val)
                    try:
                        return "{:,.0f}".format(f).replace(",", " ")
                    except Exception:
                        return str(val)

                assoc_label = None
                if assoc_id not in (None, ""):
                    try:
                        refs = _fetch_ref_list(db, ["DER_association_professionnelle", "DER_association_prof"]) or []
                        for it in refs:
                            try:
                                if int(it.get("id")) == int(assoc_id):
                                    assoc_label = it.get("libelle")
                                    break
                            except Exception:
                                if str(it.get("id")) == str(assoc_id):
                                    assoc_label = it.get("libelle")
                                    break
                    except Exception:
                        assoc_label = None

                statut_label = None
                if statut_id not in (None, ""):
                    try:
                        srefs = _fetch_ref_list(db, ["DER_statut_social", "DER_statuts_sociaux"]) or []
                        for it in srefs:
                            try:
                                if int(it.get("id")) == int(statut_id):
                                    statut_label = it.get("libelle")
                                    break
                            except Exception:
                                if str(it.get("id")) == str(statut_id):
                                    statut_label = it.get("libelle")
                                    break
                    except Exception:
                        statut_label = None

                line1 = nom_cabinet or None
                cap_str = _fmt_amount(capital_social)
                stat_parts: list[str] = []
                if statut_label:
                    stat_parts.append(str(statut_label))
                if cap_str and cap_str != "-":
                    stat_parts.append(f"au capital social de {cap_str} €")
                line2 = " ".join(stat_parts) if stat_parts else None
                s_parts: list[str] = []
                if siren:
                    s_parts.append(f"SIREN {siren}")
                if rcs:
                    s_parts.append(f"RCS {rcs}")
                line2b = " — ".join(s_parts) if s_parts else None
                addr_parts: list[str] = []
                if adresse_rue:
                    addr_parts.append(adresse_rue)
                cp_ville = (f"{adresse_cp} {adresse_ville}").strip()
                if cp_ville:
                    if addr_parts:
                        addr_parts[-1] = f"{addr_parts[-1]}, {cp_ville}"
                    else:
                        addr_parts.append(cp_ville)
                line3 = addr_parts[0] if addr_parts else None
                r_parts = [p for p in [nom_responsable, courriel] if p]
                line4 = " & ".join(r_parts) if r_parts else None
                line5 = f"Numéro Orias {numero_orias}" if numero_orias else None
                line6 = None
                if assoc_label:
                    line6 = f"Association professionnelle {assoc_label}"
                    if num_adh:
                        line6 += f" {num_adh}"

                if nom_cabinet:
                    context["footer_text"] = nom_cabinet
                context["footer_line1"] = line1 or context.get("footer_line1")
                context["footer_line2"] = line2 or context.get("footer_line2")
                context["footer_line2b"] = line2b or context.get("footer_line2b")
                context["footer_line3"] = line3 or context.get("footer_line3")
                context["footer_line4"] = line4 or context.get("footer_line4")
                context["footer_line5"] = line5 or context.get("footer_line5")
                context["footer_line6"] = line6 or context.get("footer_line6")
            except Exception:
                pass
    except Exception:
        context["DER_courtier"] = None
        context["DER_statut_social"] = None
        context["centre_mediation_lib"] = None

    # Garanties professionnelles
    try:
        gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
        if gar_table:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = {_norm(name): name for name in colnames}
            c_type = inv.get("type_garantie") or inv.get("type") or inv.get("garantie") or (colnames[0] if colnames else None)
            c_ias = inv.get("ias") or "IAS"
            c_iobsp = inv.get("iobsp") or "IOBSP"
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            garanties = []
            for r in rows:
                m = r._mapping
                garanties.append({
                    "type_garantie": m.get(c_type) if c_type else None,
                    "IAS": m.get(c_ias),
                    "IOBSP": m.get(c_iobsp),
                })
            context["DER_courtier_garanties_normes"] = garanties
    except Exception:
        context["DER_courtier_garanties_normes"] = []

    # Modes de rémunération
    context["lm_remunerations"] = []
    try:
        der_courtier = context.get("DER_courtier")
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone() if der_courtier is None else None
        if row and not der_courtier:
            der_courtier = dict(row._mapping)
            context["DER_courtier"] = der_courtier
        cid = der_courtier.get("id") if der_courtier else None
        if cid is not None:
            rows = db.execute(text("SELECT type, montant, pourcentage FROM DER_courtier_mode_facturation WHERE courtier_id = :cid"), {"cid": cid}).fetchall()
        else:
            rows = db.execute(text("SELECT type, montant, pourcentage FROM DER_courtier_mode_facturation")).fetchall()
        try:
            ref_modes = db.execute(text("SELECT id, mode FROM DER_courtier_mode_facturation_ref")).fetchall()
        except Exception:
            ref_modes = []
        import unicodedata, re as _re
        def _normtxt(s: str | None) -> str:
            if s is None:
                return ""
            t = unicodedata.normalize('NFKD', str(s))
            t = ''.join(ch for ch in t if not unicodedata.combining(ch))
            t = t.lower().replace("_", " ").replace("'", " ")
            t = _re.sub(r"[^a-z0-9]+", " ", t)
            return _re.sub(r"\s+", " ", t).strip()
        ref_map = {}
        for rm in ref_modes:
            try:
                key = _normtxt(rm._mapping.get("mode") if hasattr(rm, "_mapping") else rm[1])
                ref_map[key] = rm._mapping.get("mode") if hasattr(rm, "_mapping") else rm[1]
            except Exception:
                continue
        def _label_for_type(t: str | None) -> str | None:
            n = _normtxt(t)
            if not n:
                return None
            if "honor" in n:
                return ref_map.get("honoraires") or "Honoraires"
            if "forfait" in n:
                return ref_map.get("forfait") or "Forfait"
            return ref_map.get(n) or (t if t else None)
        remunerations = []
        for row in rows or []:
            m = row._mapping
            orig_type = m.get("type")
            label = _label_for_type(orig_type)
            display = label or (str(orig_type).replace("_", " ").title() if orig_type else None)
            remunerations.append({
                "type": orig_type,
                "mode": display,
                "montant": m.get("montant"),
                "pourcentage": m.get("pourcentage"),
            })
        try:
            import unicodedata
            def _normv(v: str | None) -> str:
                if not v:
                    return ""
                s2 = unicodedata.normalize("NFKD", str(v))
                s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.upper()
            remunerations.sort(
                key=lambda r: (
                    0 if "HONOR" in _normv(r.get("mode") or r.get("type")) else 1,
                    _normv(r.get("mode") or r.get("type"))
                )
            )
        except Exception:
            pass
        context["lm_remunerations"] = remunerations
    except Exception:
        context["lm_remunerations"] = []

    # Activités
    try:
        der_courtier = context.get("DER_courtier")
        cid_val = der_courtier.get("id") if der_courtier else None
        if cid_val is not None:
            rows_der_act = db.execute(
                text(
                    """
                    SELECT a.activite_id, a.statut,
                           r.code, r.libelle, r.domaine, r.sous_categorie, r.description
                    FROM DER_courtier_activite a
                    JOIN DER_courtier_activite_ref r ON r.id = a.activite_id
                    WHERE a.courtier_id = :cid
                    ORDER BY r.domaine, r.libelle
                    """
                ),
                {"cid": cid_val},
            ).fetchall()
            activites = []
            for rr in rows_der_act or []:
                mm = rr._mapping
                activites.append({
                    "activite_id": mm.get("activite_id"),
                    "statut": mm.get("statut"),
                    "code": mm.get("code"),
                    "libelle": mm.get("libelle"),
                    "domaine": mm.get("domaine"),
                    "sous_categorie": mm.get("sous_categorie"),
                    "description": mm.get("description"),
                })
            context["DER_activites"] = activites
    except Exception:
        context["DER_activites"] = []

    return context


def _build_mission_context(db: Session, client_id: int) -> dict:
    """Collect data required for the Lettre de Mission modal and PDF."""
    context: dict = {
        "lm_objectifs": [],
        "lm_total_invest": 0.0,
        "lm_total_invest_str": "0",
        "lm_entretien_date": None,
        "lm_objectifs_summary": "",
        "risque_latest": None,
        "lm_today": _date.today().strftime('%d/%m/%Y'),
        "date_signature": None,
        "signature_id": None,
        "report_logo_png": None,
    }
    # Risque latest
    try:
        row = db.execute(
            text(
                """
                SELECT
                  r.libelle AS niveau_risque,
                  k.niveau_id AS niveau_id,
                  k.duree AS horizon_placement,
                  k.experience,
                  k.connaissance,
                  k.commentaire,
                  k.confirmation_client
                FROM KYC_Client_Risque k
                JOIN ref_niveau_risque r ON k.niveau_id = r.id
                WHERE k.client_id = :cid
                ORDER BY k.date_saisie DESC, k.id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            context["risque_latest"] = dict(row._mapping)
    except Exception:
        context["risque_latest"] = None

    # Objectifs client
    lm_objectifs: list[dict] = []
    lm_total_invest = 0.0
    lm_total_invest_str = "0"
    lm_entretien_date: str | None = None
    _lm_latest_date: _date | None = None
    try:
        objectif_labels: dict[int, str] = {}
        try:
            rows_labels = db.execute(text("SELECT id, libelle FROM ref_objectif")).fetchall()
            objectif_labels = {int(r.id): str(r.libelle) for r in rows_labels}
        except Exception:
            objectif_labels = {}
        if not objectif_labels:
            try:
                rows_labels = db.execute(text("SELECT id, label FROM risque_objectif_option")).fetchall()
                objectif_labels = {int(r.id): str(r.label) for r in rows_labels}
            except Exception:
                objectif_labels = {}
        niveau_labels: dict[int, str] = {}
        try:
            rows_niveau = db.execute(text("SELECT id, libelle FROM ref_niveau_risque")).fetchall()
            niveau_labels = {int(r.id): str(r.libelle) for r in rows_niveau}
        except Exception:
            niveau_labels = {}
        rows = db.execute(
            text(
                """
                SELECT id,
                       objectif_id,
                       horizon_investissement,
                       niveau_id,
                       commentaire,
                       date_saisie,
                       date_expiration,
                       montant
                FROM KYC_Client_Objectifs
                WHERE client_id = :cid
                ORDER BY COALESCE(date_saisie, '0000-00-00') DESC, id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows:
            m = row._mapping
            montant_val = float(m.get("montant") or 0.0)
            lm_total_invest += montant_val
            objectif_id = m.get("objectif_id")
            libelle = None
            if objectif_id is not None:
                try:
                    libelle = objectif_labels.get(int(objectif_id))
                except Exception:
                    libelle = None
            niveau_libelle = None
            niveau_id = m.get("niveau_id")
            if niveau_id is not None:
                try:
                    niveau_libelle = niveau_labels.get(int(niveau_id))
                except Exception:
                    niveau_libelle = None
            date_saisie_raw = m.get("date_saisie")
            date_expiration_raw = m.get("date_expiration")
            parsed_date = _parse_date_safe(date_saisie_raw)
            if parsed_date is None and isinstance(date_saisie_raw, datetime):
                parsed_date = date_saisie_raw.date()
            if parsed_date:
                if _lm_latest_date is None or parsed_date > _lm_latest_date:
                    _lm_latest_date = parsed_date
            lm_objectifs.append(
                {
                    "id": m.get("id"),
                    "libelle": libelle or (f"Objectif {objectif_id}" if objectif_id is not None else "Objectif"),
                    "montant": montant_val,
                    "montant_str": "{:,.0f}".format(montant_val).replace(",", " "),
                    "horizon": m.get("horizon_investissement") or "",
                    "niveau": niveau_libelle,
                    "commentaire": m.get("commentaire") or "",
                    "date_saisie": parsed_date.strftime("%d/%m/%Y") if parsed_date else (str(date_saisie_raw) if date_saisie_raw else ""),
                    "date_expiration": (
                        _parse_date_safe(date_expiration_raw).strftime("%d/%m/%Y")
                        if _parse_date_safe(date_expiration_raw)
                        else (str(date_expiration_raw) if date_expiration_raw else "")
                    ),
                }
            )
        if lm_objectifs:
            lm_total_invest_str = "{:,.0f}".format(lm_total_invest).replace(",", " ")
        else:
            lm_total_invest = 0.0
            lm_total_invest_str = "0"
        if _lm_latest_date:
            lm_entretien_date = _lm_latest_date.strftime("%d/%m/%Y")
    except Exception:
        lm_objectifs = []
        lm_total_invest = 0.0
        lm_total_invest_str = "0"
        lm_entretien_date = None

    context["lm_objectifs"] = lm_objectifs
    context["lm_total_invest"] = lm_total_invest
    context["lm_total_invest_str"] = lm_total_invest_str
    context["lm_entretien_date"] = lm_entretien_date
    context["lm_objectifs_summary"] = ", ".join([obj["libelle"] for obj in lm_objectifs if obj.get("libelle")]) if lm_objectifs else ""

    return context


def _courtier_relation_columns(db: Session, table: str) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Infer (id_col, fk_societe_col, fk_courtier_col, creation_col, modification_col) for DER_courtier_relation_commerciale-like table."""
    cols = _sqlite_table_columns(db, table)
    colnames = [c.get("name") for c in cols]
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
    import unicodedata
    def norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s)
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()
    norm_map = {name: norm(name) for name in colnames}
    inv_map = {v: k for k, v in norm_map.items()}
    fk_societe_col = None
    for cand in ["id_societe", "societe_id", "id_mariadb_societe", "id_der_societe"]:
        if cand in inv_map:
            fk_societe_col = inv_map[cand]
            break
    if not fk_societe_col:
        for name in colnames:
            if name != id_col:
                fk_societe_col = name
                break
    fk_courtier_col = None
    for cand in ["id_courtier", "id_der_courtier", "id_cabinet", "courtier_id"]:
        if cand in inv_map:
            fk_courtier_col = inv_map[cand]
            break
    creation_col = next((c for c in ["date_creation", "date_création", "dateCreation"] if c in colnames), None)
    modification_col = next((c for c in ["date_obsolescence", "date_modification", "date_modif", "date_maj", "dateMiseAJour"] if c in colnames), None)
    return id_col, fk_societe_col, fk_courtier_col, creation_col, modification_col


# ---- Courtier > Relation commerciale ----
@router.post("/parametres/courtier/relation", response_class=HTMLResponse)
async def create_courtier_relation(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    soc_id_raw = form.get("societe_id")
    try:
        societe_id = int(soc_id_raw) if soc_id_raw not in (None, "") else None
    except Exception:
        societe_id = None
    table = _resolve_table_name(db, ["DER_courtier_relation_commerciale"]) or "DER_courtier_relation_commerciale"
    id_col, fk_soc_col, fk_cour_col, creation_col, modification_col = _courtier_relation_columns(db, table)
    if not fk_soc_col:
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&error=1&errmsg=Aucune%20colonne%20societe%20detectee", status_code=303)
    params: dict = {fk_soc_col: societe_id}
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    if creation_col:
        params[creation_col] = now
    if modification_col:
        params[modification_col] = now
    # attach current courtier id if column exists
    try:
        row = db.execute(text("SELECT id FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if fk_cour_col and row and row[0] is not None:
            params[fk_cour_col] = row[0]
    except Exception:
        pass
    # Satisfy NOT NULL columns generically
    try:
        cols = _sqlite_table_columns(db, table)
        for c in cols:
            name = c.get("name")
            if not name or name in params or name == id_col:
                continue
            if c.get("notnull"):
                if name == creation_col or name == modification_col:
                    params[name] = now
                    continue
                if name == fk_soc_col or name == fk_cour_col:
                    continue
                ctype = (c.get("type") or "").upper()
                if any(x in ctype for x in ["INT", "REAL", "FLOA", "DOUB", "DEC", "NUM"]):
                    params[name] = 0
                else:
                    params[name] = ""
    except Exception:
        pass
    from sqlalchemy import text as _text
    try:
        fields = ", ".join(params.keys())
        placeholders = ", ".join(f":{k}" for k in params.keys())
        db.execute(_text(f"INSERT INTO {table} ({fields}) VALUES ({placeholders})"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"create_courtier_relation failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/relation/{row_id}", response_class=HTMLResponse)
async def update_courtier_relation(row_id: str, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    soc_id_raw = form.get("societe_id")
    try:
        societe_id = int(soc_id_raw) if soc_id_raw not in (None, "") else None
    except Exception:
        societe_id = None
    table = _resolve_table_name(db, ["DER_courtier_relation_commerciale"]) or "DER_courtier_relation_commerciale"
    id_col, fk_soc_col, fk_cour_col, creation_col, modification_col = _courtier_relation_columns(db, table)
    if not fk_soc_col or not id_col:
        return _redirect_back(request, "courtierDetails")
    params: dict = {fk_soc_col: societe_id, "__pk__": row_id}
    set_parts = [f"{fk_soc_col} = :{fk_soc_col}"]
    if modification_col:
        params["__now__"] = datetime.now().isoformat(sep=" ", timespec="seconds")
        set_parts.append(f"{modification_col} = :__now__")
    from sqlalchemy import text as _text
    try:
        db.execute(_text(f"UPDATE {table} SET {', '.join(set_parts)} WHERE {id_col} = :__pk__"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"update_courtier_relation failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/relation/{row_id}/delete", response_class=HTMLResponse)
async def delete_courtier_relation(row_id: str, request: Request, db: Session = Depends(get_db)):
    table = _resolve_table_name(db, ["DER_courtier_relation_commerciale"]) or "DER_courtier_relation_commerciale"
    id_col, *_rest = _courtier_relation_columns(db, table)
    if not id_col:
        return _redirect_back(request, "courtierDetails")
    from sqlalchemy import text as _text
    try:
        db.execute(_text(f"DELETE FROM {table} WHERE {id_col} = :__pk__"), {"__pk__": row_id})
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"delete_courtier_relation failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


# ---- Courtier > Activité ----
@router.post("/parametres/courtier/activite", response_class=HTMLResponse)
async def create_courtier_activite(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    ref_id_raw = form.get("ref_id")
    try:
        ref_id = int(ref_id_raw) if ref_id_raw not in (None, "") else None
    except Exception:
        ref_id = None
    table = _resolve_table_name(db, ["DER_courtier_activite"]) or "DER_courtier_activite"
    id_col, fk_ref_col, fk_cour_col, creation_col, modification_col = _courtier_activite_columns(db, table)
    if not fk_ref_col:
        return _redirect_back(request, "courtierDetails")
    params: dict = {fk_ref_col: ref_id}
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    if creation_col:
        params[creation_col] = now
    if modification_col:
        params[modification_col] = now
    # attach current courtier id if column exists
    try:
        row = db.execute(text("SELECT id FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if fk_cour_col and row and row[0] is not None:
            params[fk_cour_col] = row[0]
    except Exception:
        pass
    # Satisfaire les colonnes NOT NULL sans valeur par défaut
    try:
        cols = _sqlite_table_columns(db, table)
        for c in cols:
            name = c.get("name")
            if not name or name in params or name == id_col:
                continue
            if c.get("notnull"):
                if name == creation_col or name == modification_col:
                    params[name] = now
                    continue
                if name == fk_ref_col or name == fk_cour_col:
                    # déjà gérées si possibles
                    continue
                ctype = (c.get("type") or "").upper()
                uname = name.upper()
                if "STATUT" in uname:
                    # Respecter le domaine ('exercee','non_exercee') si texte
                    if any(x in ctype for x in ["CHAR", "TEXT"]):
                        params[name] = "exercee"
                    else:
                        params[name] = 1
                elif any(x in ctype for x in ["INT", "REAL", "FLOA", "DOUB", "DEC", "NUM"]):
                    params[name] = 0
                else:
                    params[name] = ""
    except Exception:
        pass
    from sqlalchemy import text as _text
    try:
        fields = ", ".join(params.keys())
        placeholders = ", ".join(f":{k}" for k in params.keys())
        db.execute(_text(f"INSERT INTO {table} ({fields}) VALUES ({placeholders})"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"create_courtier_activite failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/activite/{row_id}", response_class=HTMLResponse)
async def update_courtier_activite(row_id: str, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    ref_id_raw = form.get("ref_id")
    try:
        ref_id = int(ref_id_raw) if ref_id_raw not in (None, "") else None
    except Exception:
        ref_id = None
    table = _resolve_table_name(db, ["DER_courtier_activite"]) or "DER_courtier_activite"
    id_col, fk_ref_col, fk_cour_col, creation_col, modification_col = _courtier_activite_columns(db, table)
    if not fk_ref_col or not id_col:
        return _redirect_back(request, "courtierDetails")
    params: dict = {fk_ref_col: ref_id, "__pk__": row_id}
    from sqlalchemy import text as _text
    set_parts = [f"{fk_ref_col} = :{fk_ref_col}"]
    # statut update if column exists
    try:
        cols = _sqlite_table_columns(db, table)
        for c in cols:
            nm = (c.get("name") or "").lower()
            if nm == "statut":
                statut_val = (form.get("statut") or "").strip().lower().replace("é", "e")
                if statut_val not in ("exercee", "non_exercee"):
                    statut_val = "exercee"
                params[c.get("name")] = statut_val
                set_parts.append(f"{c.get('name')} = :{c.get('name')}")
                break
    except Exception:
        pass
    if modification_col:
        params["__now__"] = datetime.now().isoformat(sep=" ", timespec="seconds")
        set_parts.append(f"{modification_col} = :__now__")
    try:
        db.execute(_text(f"UPDATE {table} SET {', '.join(set_parts)} WHERE {id_col} = :__pk__"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"update_courtier_activite failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/activite/{row_id}/delete", response_class=HTMLResponse)
async def delete_courtier_activite(row_id: str, request: Request, db: Session = Depends(get_db)):
    table = _resolve_table_name(db, ["DER_courtier_activite"]) or "DER_courtier_activite"
    id_col, *_rest = _courtier_activite_columns(db, table)
    if not id_col:
        return _redirect_back(request, "courtierDetails")
    from sqlalchemy import text as _text
    try:
        db.execute(_text(f"DELETE FROM {table} WHERE {id_col} = :__pk__"), {"__pk__": row_id})
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"delete_courtier_activite failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


def _digits_only(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(ch for ch in str(value) if ch.isdigit()) or None


def _int_from_thousands(value: str | None) -> int | None:
    if value is None:
        return None
    s = str(value).replace(" ", "").replace("\u00A0", "")
    s = "".join(ch for ch in s if ch.isdigit())
    return int(s) if s else None


# ---- Courtier > Rémunération (mode de facturation) ----
@router.post("/parametres/courtier/remuneration", response_class=HTMLResponse)
async def create_courtier_remuneration(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    ref_raw = form.get("ref_id")
    try:
        ref_id = int(ref_raw) if ref_raw not in (None, "") else None
    except Exception:
        ref_id = None
    table = _resolve_table_name(db, ["DER_courtier_mode_facturation"]) or "DER_courtier_mode_facturation"
    cols = _sqlite_table_columns(db, table)
    colnames = [c.get("name") for c in cols]
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
    # infer fk columns
    import unicodedata
    def norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s)
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()
    norm_map = {name: norm(name) for name in colnames}
    inv_map = {v: k for k, v in norm_map.items()}
    fk_ref_col = None
    for cand in ["id_ref","id_mode_facturation_ref","id_courtier_mode_facturation_ref","mode_ref","mode","id_mode_ref"]:
        if cand in inv_map:
            fk_ref_col = inv_map[cand]
            break
    if not fk_ref_col:
        for name in colnames:
            if name != id_col:
                fk_ref_col = name
                break
    fk_cour_col = None
    for cand in ["id_courtier","courtier_id","id_der_courtier","id_cabinet"]:
        if cand in inv_map:
            fk_cour_col = inv_map[cand]
            break
    params: dict = {fk_ref_col: ref_id}
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    # attach courtier id if column exists
    try:
        row = db.execute(text("SELECT id FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if fk_cour_col and row and row[0] is not None:
            params[fk_cour_col] = row[0]
    except Exception:
        pass
    # include extra fields from form
    try:
        for c in cols:
            nm = c.get("name")
            if nm in (id_col, fk_ref_col, fk_cour_col):
                continue
            _ = request.form  # placeholder to satisfy type checker
        form_map = await request.form()
        # Convenience normalization for known fields
        def _as_percent(raw: str | None):
            if raw is None:
                return None
            s = str(raw).strip().replace('%','').replace(' ','').replace(',', '.')
            try:
                return float(s)
            except Exception:
                return None
        def _as_amount(raw: str | None):
            if raw is None:
                return None
            import re
            s = re.sub(r"[^0-9,\.\-]", '', str(raw))
            s = s.replace(',', '.')
            try:
                return float(s)
            except Exception:
                return None
        for c in cols:
            nm = c.get("name")
            if nm in (id_col, fk_ref_col, fk_cour_col):
                continue
            if (nm or '').lower() == 'type':
                continue
            if nm in ("date_creation", "date_création", "dateCreation", "date_obsolescence", "date_modification", "date_modif", "date_maj", "dateMiseAJour"):
                continue
            raw = form_map.get(nm)
            if raw is None:
                continue
            # Field-specific coercions
            if nm.lower() in ("pourcentage", "taux"):
                val = _as_percent(raw)
            elif nm.lower() in ("montant", "valeur"):
                val = _as_amount(raw)
            else:
                val = _coerce_value(str(raw), (c.get("type") or ""))
            params[nm] = val
    except Exception:
        pass

    # If a 'type' column exists with CHECK constraint, derive it from ref libelle when absent/empty
    try:
        if "type" in colnames:
            form_map = form_map if 'form_map' in locals() else await request.form()
            provided = form_map.get("type")
            def _map_type(val: str | None) -> str | None:
                if not val:
                    return None
                t = str(val).strip().upper().replace(' ', '_')
                if ("HONOR" in t) or ("PONCT" in t):
                    return "HONORAIRES"
                if "ENTREE" in t:
                    return "FRAIS_ENTREE"
                if "GESTION" in t:
                    return "FRAIS_GESTION"
                return None
            type_val = _map_type(provided)
            if not type_val:
                # derive from ref libelle
                refs = _fetch_ref_list(db, ["DER_courtier_mode_facturation_ref"]) or []
                label = None
                for r in refs:
                    if str(r.get("id")) == str(ref_id):
                        label = r.get("libelle")
                        break
                type_val = _map_type(label)
            if not type_val:
                type_val = "HONORAIRES"
            params["type"] = type_val
    except Exception:
        pass
    # fill NOT NULL generically
    try:
        for c in cols:
            name = c.get("name")
            if not name or name in params or name == id_col:
                continue
            if c.get("notnull"):
                ctype = (c.get("type") or "").upper()
                if any(x in ctype for x in ["INT","REAL","FLOA","DOUB","DEC","NUM"]):
                    params[name] = 0
                else:
                    params[name] = ""
    except Exception:
        pass
    try:
        from sqlalchemy import text as _text
        fields = ", ".join(params.keys())
        placeholders = ", ".join(f":{k}" for k in params.keys())
        db.execute(_text(f"INSERT INTO {table} ({fields}) VALUES ({placeholders})"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"create remuneration failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/remuneration/{row_id}", response_class=HTMLResponse)
async def update_courtier_remuneration(row_id: str, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    ref_raw = form.get("ref_id")
    try:
        ref_id = int(ref_raw) if ref_raw not in (None, "") else None
    except Exception:
        ref_id = None
    table = _resolve_table_name(db, ["DER_courtier_mode_facturation"]) or "DER_courtier_mode_facturation"
    cols = _sqlite_table_columns(db, table)
    colnames = [c.get("name") for c in cols]
    pk_cols = [c.get("name") for c in cols if c.get("pk")]
    id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else (colnames[0] if colnames else None))
    if not id_col:
        return _redirect_back(request, "courtierDetails")
    # infer fk ref column name
    import unicodedata, re
    def norm(s: str) -> str:
        s2 = unicodedata.normalize('NFKD', s or '')
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        return s2.lower()
    inv_map = {norm(name): name for name in colnames}
    fk_ref_col = None
    for cand in ["id_ref","id_mode_facturation_ref","id_courtier_mode_facturation_ref","mode_ref","mode","id_mode_ref"]:
        if cand in inv_map:
            fk_ref_col = inv_map[cand]
            break
    # Build params whitelisting montant/pourcentage only (avoid unexpected fields)
    params: dict = {"__pk__": row_id}
    set_parts: list[str] = []
    if fk_ref_col is not None:
        params[fk_ref_col] = ref_id
        set_parts.append(f"{fk_ref_col} = :{fk_ref_col}")
    else:
        # Pas de fk → mettre à jour la colonne 'type' si elle existe
        if 'type' in colnames and ref_id is not None:
            # Charger le libellé du ref et le mapper vers domaine autorisé
            refs = _fetch_ref_list(db, ["DER_courtier_mode_facturation_ref"]) or []
            label = None
            for r in refs:
                if str(r.get('id')) == str(ref_id):
                    label = r.get('libelle')
                    break
            import unicodedata, re
            def _norm_upper(val: str | None) -> str:
                s = '' if val is None else str(val)
                s = unicodedata.normalize('NFKD', s)
                s = ''.join(ch for ch in s if not unicodedata.combining(ch))
                s = s.replace("'", " ").replace("’", " ")
                s = re.sub(r"\s+", " ", s)
                return s.upper()
            def _map_type(val: str | None) -> str | None:
                v = _norm_upper(val)
                if 'HONOR' in v or 'PONCT' in v:
                    return 'HONORAIRES'
                if 'ENTREE' in v:
                    return 'FRAIS_ENTREE'
                if 'GESTION' in v:
                    return 'FRAIS_GESTION'
                return None
            tdom = _map_type(label)
            if not tdom:
                tdom = 'HONORAIRES'
            params['type'] = tdom
            set_parts.append("type = :type")
    # montant
    for mkey in ("montant", "valeur"):
        if mkey in colnames:
            raw = form.get(mkey)
            if raw is not None:
                sval = re.sub(r"[^0-9,\.\-]", '', str(raw)).replace(',', '.')
                try:
                    params[mkey] = float(sval) if sval != '' else None
                except Exception:
                    params[mkey] = None
                set_parts.append(f"{mkey} = :{mkey}")
            break
    # pourcentage
    for pkey in ("pourcentage", "taux"):
        if pkey in colnames:
            raw = form.get(pkey)
            if raw is not None:
                sval = str(raw).strip().replace('%','').replace(' ','').replace(',', '.')
                try:
                    params[pkey] = float(sval) if sval != '' else None
                except Exception:
                    params[pkey] = None
                set_parts.append(f"{pkey} = :{pkey}")
            break
    try:
        from sqlalchemy import text as _text
        clause = ', '.join(set_parts) or ''
        if not clause:
            return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
        db.execute(_text(f"UPDATE {table} SET {clause} WHERE {id_col} = :__pk__"), params)
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"update remuneration failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/courtier/remuneration/{row_id}/delete", response_class=HTMLResponse)
async def delete_courtier_remuneration(row_id: str, request: Request, db: Session = Depends(get_db)):
    table = _resolve_table_name(db, ["DER_courtier_mode_facturation"]) or "DER_courtier_mode_facturation"
    id_col = _table_primary_key(db, table)
    if not id_col:
        return _redirect_back(request, "courtierDetails")
    try:
        from sqlalchemy import text as _text
        db.execute(_text(f"DELETE FROM {table} WHERE {id_col} = :__pk__"), {"__pk__": row_id})
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=courtierDetails&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        logger.error(f"delete remuneration failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=courtierDetails&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/der_courtier", response_class=HTMLResponse)
async def save_der_courtier(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    # Sanitize / normalize
    nom_cabinet = (form.get("nom_cabinet") or "").strip()
    nom_responsable = (form.get("nom_responsable") or "").strip()
    statut_social = _as_int(form.get("statut_social"))
    capital_social = _int_from_thousands(form.get("capital_social"))
    siren = _digits_only(form.get("siren"))
    rcs = (form.get("rcs") or "").strip()
    numero_orias = _digits_only(form.get("numero_orias")) or ""
    adresse_rue = (form.get("adresse_rue") or "").strip()
    adresse_cp = _digits_only(form.get("adresse_cp")) or ""
    adresse_ville = (form.get("adresse_ville") or "").strip()
    courriel = (form.get("courriel") or "").strip()
    association_prof = _as_int(form.get("association_prof"))
    num_adh_assoc = (form.get("num_adh_assoc") or "").strip()
    categorie_courtage = (form.get("categorie_courtage") or "").strip().upper() or None
    if categorie_courtage not in (None, "A", "B", "C"):
        categorie_courtage = None
    responsable_dpo = (form.get("responsable_dpo") or "").strip()
    centre_mediation = _as_int(form.get("centre_mediation"))
    mediators = (form.get("mediators") or "").strip()
    mail_mediators = (form.get("mail_mediators") or "").strip()

    # Validations strictes
    errors: dict[str, str] = {}
    def is_email(s: str) -> bool:
        return bool(__import__("re").match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", s))

    if not nom_cabinet:
        errors["nom_cabinet"] = "Le nom du cabinet est requis."
    # Contrôle SIREN retiré sur demande; on conserve la normalisation uniquement
    if numero_orias and len(numero_orias) != 9:
        errors["numero_orias"] = "Le numéro ORIAS doit comporter 9 chiffres."
    if adresse_cp and len(adresse_cp) != 5:
        errors["adresse_cp"] = "Le code postal doit comporter 5 chiffres."
    if courriel:
        if not is_email(courriel):
            errors["courriel"] = "Veuillez saisir une adresse courriel valide."
    # Emails médiateurs: séparés par virgule/point-virgule/espace
    if mail_mediators:
        import re
        parts = [p.strip() for p in re.split(r"[;,\s]+", mail_mediators) if p.strip()]
        bad = [p for p in parts if not is_email(p)]
        if bad:
            errors["mail_mediators"] = "Courriels des médiateurs invalides: " + ", ".join(bad)
    # Champs référentiels requis si la base les impose (statut_social non null)
    if statut_social is None:
        errors["statut_social"] = "Le statut social est requis."
    # Champs requis supplémentaires
    if capital_social is None:
        errors["capital_social"] = "Le capital social est requis."
    if not rcs:
        errors["rcs"] = "Le RCS est requis."
    if not numero_orias:
        errors["numero_orias"] = "Le numéro ORIAS est requis."

    from sqlalchemy import text as _text
    # Existe-t-il déjà une ligne ?
    row = None
    try:
        row = db.execute(_text("SELECT id FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
    except Exception:
        row = None

    params = {
        "nom_cabinet": nom_cabinet,
        "nom_responsable": nom_responsable,
        "statut_social": statut_social,
        "capital_social": capital_social,
        "siren": siren,
        "rcs": rcs,
        "numero_orias": numero_orias or None,
        "adresse_rue": adresse_rue,
        "adresse_cp": adresse_cp or None,
        "adresse_ville": adresse_ville,
        "courriel": courriel,
        "association_prof": association_prof,
        "num_adh_assoc": num_adh_assoc,
        "categorie_courtage": categorie_courtage,
        "responsable_dpo": responsable_dpo,
        "centre_mediation": centre_mediation,
        "mediators": mediators,
        "mail_mediators": mail_mediators,
    }

    # Si erreurs, re-afficher la page avec messages et valeurs saisies
    if errors:
        # Recharger les listes de référence et renvoyer le template
        try:
            statut_sociaux = _fetch_ref_list(db, ["DER_statut_social", "DER_statuts_sociaux"])
        except Exception:
            statut_sociaux = []
        try:
            associations_prof = _fetch_ref_list(db, ["DER_association_professionnelle", "DER_association_prof"])
        except Exception:
            associations_prof = []
        try:
            autorites_mediation = _fetch_ref_list(db, ["DER_courtier_ref_autorite", "DER_autorite_mediation", "DER_courtier_autorite"])
        except Exception:
            autorites_mediation = []
        return templates.TemplateResponse(
            "dashboard_parametres.html",
            {
                "request": request,
                "open_section": "courtier",
                # Garder le contexte principal pour les autres panneaux
                "contrats_generiques": rows_to_dicts(db.execute(_text("SELECT g.id, g.nom_contrat, g.id_societe, g.id_ctg, g.frais_gestion_assureur, g.frais_gestion_courtier, s.nom AS societe_nom FROM mariadb_affaires_generique g LEFT JOIN mariadb_societe s ON s.id = g.id_societe WHERE COALESCE(g.actif, 1) = 1 ORDER BY s.nom, g.nom_contrat")).fetchall()) if db else [],
                "societes": rows_to_dicts(db.execute(_text("SELECT id, nom, id_ctg, contact, telephone, email, commentaire FROM mariadb_societe ORDER BY nom")).fetchall()) if db else [],
                "contrat_categories": rows_to_dicts(db.execute(_text("SELECT id, libelle, description FROM mariadb_affaires_generique_ctg ORDER BY libelle")).fetchall()) if db else [],
                "societe_categories": rows_to_dicts(db.execute(_text("SELECT id, libelle, description FROM mariadb_societe_ctg ORDER BY libelle")).fetchall()) if db else [],
                "contrat_supports": rows_to_dicts(db.execute(_text("SELECT cs.id, cs.id_affaire_generique, cs.id_support, cs.taux_retro, s.nom AS support_nom, s.code_isin, s.cat_gene, s.cat_geo, s.promoteur, g.nom_contrat AS contrat_nom, so.nom AS societe_nom FROM mariadb_contrat_supports cs JOIN mariadb_support s ON s.id = cs.id_support JOIN mariadb_affaires_generique g ON g.id = cs.id_affaire_generique LEFT JOIN mariadb_societe so ON so.id = g.id_societe ORDER BY g.nom_contrat, s.nom")).fetchall()) if db else [],
                "supports": rows_to_dicts(db.execute(_text("SELECT id, nom, code_isin FROM mariadb_support ORDER BY nom")).fetchall()) if db else [],
                "admin_types": _load_admin_types_list(),
                "admin_intervenants": _load_admin_intervenants_list(),
                "groupes_details": _load_groupes_details(db),
                "admin_rh": rows_to_dicts(db.execute(_text("SELECT id, nom, prenom, telephone, mail, niveau_poste, commentaire FROM administration_RH ORDER BY nom, prenom")).fetchall()) if db else [],
                # Contexte courtier avec erreur
                "courtier": None,
                "form_courtier": params,
                "form_errors": errors,
                "statut_sociaux": statut_sociaux,
                "associations_prof": associations_prof,
                "autorites_mediation": autorites_mediation,
            },
        )

    # Construction dynamique selon colonnes existantes
    cols = _sqlite_table_columns(db, "DER_courtier")
    colnames = [c.get("name") for c in cols]
    present = {k: v for k, v in params.items() if k in colnames}
    # Gestion colonnes dates si présentes
    creation_candidates = ["date_creation", "date_création", "dateCreation"]
    modification_candidates = ["date_obsolescence", "date_modification", "date_modif", "date_maj", "dateMiseAJour"]
    creation_col = next((x for x in creation_candidates if x in colnames), None)
    modification_col = next((x for x in modification_candidates if x in colnames), None)

    try:
        if row is None:
            # Création
            if creation_col:
                present[creation_col] = datetime.now().isoformat(sep=" ", timespec="seconds")
            if modification_col:
                # Certaines bases imposent NOT NULL sur la date de modification
                present[modification_col] = present.get(creation_col) or datetime.now().isoformat(sep=" ", timespec="seconds")
            if not present:
                raise RuntimeError("Aucune colonne correspondante pour l'insertion dans DER_courtier")
            fields = ", ".join(present.keys())
            placeholders = ", ".join(f":{k}" for k in present.keys())
            db.execute(_text(f"INSERT INTO DER_courtier ({fields}) VALUES ({placeholders})"), present)
        else:
            # Mise à jour
            set_parts = []
            upd_params = dict(present)
            if modification_col:
                set_parts.append(f"{modification_col} = :__now__")
                upd_params["__now__"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            for k in present.keys():
                set_parts.append(f"{k} = :{k}")
            upd_params["__id__"] = row[0]
            set_clause = ", ".join(set_parts)
            db.execute(_text(f"UPDATE DER_courtier SET {set_clause} WHERE id = :__id__"), upd_params)
        db.commit()
    except Exception as e:
        db.rollback()
        errors["__global__"] = f"Erreur lors de l'enregistrement: {e}"
        # Afficher de nouveau le formulaire avec erreurs
        try:
            statut_sociaux = _fetch_ref_list(db, ["DER_statut_social", "DER_statuts_sociaux"])
        except Exception:
            statut_sociaux = []
        try:
            associations_prof = _fetch_ref_list(db, ["DER_association_professionnelle", "DER_association_prof"])
        except Exception:
            associations_prof = []
        try:
            autorites_mediation = _fetch_ref_list(db, ["DER_courtier_ref_autorite", "DER_autorite_mediation", "DER_courtier_autorite"])
        except Exception:
            autorites_mediation = []
        return templates.TemplateResponse(
            "dashboard_parametres.html",
            {
                "request": request,
                "open_section": "courtier",
                "contrats_generiques": rows_to_dicts(db.execute(_text("SELECT g.id, g.nom_contrat, g.id_societe, g.id_ctg, g.frais_gestion_assureur, g.frais_gestion_courtier, s.nom AS societe_nom FROM mariadb_affaires_generique g LEFT JOIN mariadb_societe s ON s.id = g.id_societe WHERE COALESCE(g.actif, 1) = 1 ORDER BY s.nom, g.nom_contrat")).fetchall()) if db else [],
                "societes": rows_to_dicts(db.execute(_text("SELECT id, nom, id_ctg, contact, telephone, email, commentaire FROM mariadb_societe ORDER BY nom")).fetchall()) if db else [],
                "contrat_categories": rows_to_dicts(db.execute(_text("SELECT id, libelle, description FROM mariadb_affaires_generique_ctg ORDER BY libelle")).fetchall()) if db else [],
                "societe_categories": rows_to_dicts(db.execute(_text("SELECT id, libelle, description FROM mariadb_societe_ctg ORDER BY libelle")).fetchall()) if db else [],
                "contrat_supports": rows_to_dicts(db.execute(_text("SELECT cs.id, cs.id_affaire_generique, cs.id_support, cs.taux_retro, s.nom AS support_nom, s.code_isin, s.cat_gene, s.cat_geo, s.promoteur, g.nom_contrat AS contrat_nom, so.nom AS societe_nom FROM mariadb_contrat_supports cs JOIN mariadb_support s ON s.id = cs.id_support JOIN mariadb_affaires_generique g ON g.id = cs.id_affaire_generique LEFT JOIN mariadb_societe so ON so.id = g.id_societe ORDER BY g.nom_contrat, s.nom")).fetchall()) if db else [],
                "supports": rows_to_dicts(db.execute(_text("SELECT id, nom, code_isin FROM mariadb_support ORDER BY nom")).fetchall()) if db else [],
                "admin_types": _load_admin_types_list(),
                "admin_intervenants": _load_admin_intervenants_list(),
                "groupes_details": _load_groupes_details(db),
                "admin_rh": rows_to_dicts(db.execute(_text("SELECT id, nom, prenom, telephone, mail, niveau_poste, commentaire FROM administration_RH ORDER BY nom, prenom")).fetchall()) if db else [],
                "courtier": None,
                "form_courtier": params,
                "form_errors": errors,
                "statut_sociaux": statut_sociaux,
                "associations_prof": associations_prof,
                "autorites_mediation": autorites_mediation,
            },
        )
    # Succès → toast "Enregistré"
    return RedirectResponse(url="/dashboard/parametres?open=courtier&saved=1", status_code=303)


# ---------------- KYC index (sélection client) ----------------
@router.get("/client/kyc", response_class=HTMLResponse)
@router.get("/clients/kyc", response_class=HTMLResponse)
def dashboard_kyc_index(request: Request):
    raw_id = (request.query_params.get("id") or request.query_params.get("client_id") or "").strip()
    if raw_id.isdigit():
        return RedirectResponse(url=f"/dashboard/clients/kyc/{int(raw_id)}", status_code=303)
    return templates.TemplateResponse(
        "dashboard_kyc_index.html",
        {"request": request},
    )


def _as_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        import re
        s = str(value).strip().replace(",", ".")
        # Remove spaces, non-numeric and non dot/minus characters (thousand separators, currency, etc.)
        s = re.sub(r"[^0-9.\-]", "", s)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _as_int(value: str | None, default: int | None = None) -> int | None:
    """Parse an int from form values, treating '', None and 'None'/'null' as None.
    Returns default on failure.
    """
    if value is None:
        return default
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null"):
        return default
    try:
        return int(s)
    except Exception:
        return default

def _redirect_back(request: Request, fallback_open: str) -> RedirectResponse:
    target_open = request.query_params.get("open") or fallback_open
    return RedirectResponse(url=f"/dashboard/parametres?open={target_open}", status_code=303)


def _normalize_group_type(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()
    if low in ("personnes", "personne", "client", "clients"):
        return "client"
    if low in ("affaires", "affaire", "contrat", "contrats"):
        return "affaire"
    return low


def _as_flag(value: str | None, default: int = 1) -> int:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("0", "false", "non", "no", "off"):
        return 0
    return 1


def _group_success_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/parametres?open=administration_groupes&saved=1", status_code=303)


def _group_error_redirect(message: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/dashboard/parametres?open=administration_groupes&error=1&errmsg={quote(message)}",
        status_code=303,
    )


# ---- Group tasks (logs + run/simulate) ----
@router.get("/parametres/administration/groupes/{group_id}/tasks/logs", response_class=JSONResponse)
def group_tasks_logs(group_id: int):
    # Aucun stockage de log implémenté : retourne vide
    return {"logs": [], "active": False}


@router.post("/parametres/administration/groupes/{group_id}/tasks/{mode}", response_class=JSONResponse)
def group_tasks_run(
    group_id: int,
    mode: str,
    payload: dict = None,
    db: Session = Depends(get_db),
):
    """
    Crée (mode=run) ou simule (mode=simulate) des tâches pour les membres du groupe.
    Payload JSON: {categorie, type_libelle, statut_id, commentaire, actifs_only, rh_id}
    """
    if mode not in ("run", "simulate"):
        raise HTTPException(status_code=400, detail="Mode invalide (run|simulate).")
    payload = payload or {}
    cat = (payload.get("categorie") or "").strip() or None
    typ = (payload.get("type_libelle") or "").strip() or None
    statut_id = payload.get("statut_id", None)
    try:
        statut_id = int(statut_id) if statut_id not in (None, "") else None
    except Exception:
        statut_id = None
    comm = (payload.get("commentaire") or "").strip() or None
    actifs_only = bool(payload.get("actifs_only", False))
    rh_id_payload = payload.get("rh_id", None)
    try:
        rh_id_payload = int(rh_id_payload) if rh_id_payload not in (None, "") else None
    except Exception:
        rh_id_payload = None

    # Infos groupe (responsable + type)
    detail = db.execute(
        text(
            """
            SELECT id, type_groupe, responsable_id
            FROM administration_groupe_detail
            WHERE id = :gid
            """
        ),
        {"gid": group_id},
    ).fetchone()
    if not detail:
        raise HTTPException(status_code=404, detail="Groupe introuvable")
    type_grp = (detail._mapping.get("type_groupe") or "").lower()
    resp_id = detail._mapping.get("responsable_id")

    members = _list_group_members(db, group_id, actifs_only=actifs_only)
    total = len(members)
    created = 0
    ignored = 0

    for m in members:
        client_id = m.get("client_id")
        affaire_id = m.get("affaire_id")
        # Si type groupe ne correspond pas à l'entité, ignorer
        if type_grp in ("client", "clients", "personne", "personnes") and client_id is None:
            ignored += 1
            continue
        if type_grp in ("affaire", "affaires", "contrat", "contrats") and affaire_id is None:
            ignored += 1
            continue
        # Résoudre client depuis affaire le cas échéant
        aff_ref = None
        if client_id is None and affaire_id is not None:
            client_id, aff_ref = _resolve_affaire_client(db, affaire_id)
        # RH
        rh_id = _resolve_rh_for_member(db, client_id, resp_id, rh_id_payload)
        if mode == "simulate":
            created += 1
            continue
        # Création
        try:
            t_payload = TacheCreateSchema(
                type_libelle=typ or "tâche",
                categorie=cat or "tache",
                client_id=client_id,
                affaire_id=affaire_id,
                commentaire=comm,
                utilisateur_responsable=None,
                rh_id=rh_id,
            )
            ev = create_tache(db, t_payload)
            if statut_id:
                try:
                    add_statut_to_evenement(
                        db,
                        ev.id,
                        EvenementStatutCreateSchema(
                            statut_id=statut_id,
                            commentaire=comm or None,
                            utilisateur_responsable=None,
                        ),
                    )
                except Exception:
                    pass
            created += 1
        except Exception:
            ignored += 1

    return {"total": total, "created": created, "ignored": ignored, "mode": mode}

def _resolve_rh_for_member(db: Session, client_id: int | None, group_resp_id: int | None, payload_rh: int | None) -> int | None:
    """Détermine le RH à affecter (priorité: RH choisi > responsable de groupe > commercial du client)."""
    if payload_rh:
        return payload_rh
    if group_resp_id:
        return group_resp_id
    if client_id is None:
        return None
    try:
        cli = db.query(Client).filter(Client.id == client_id).first()
        if cli and getattr(cli, "commercial_id", None) is not None:
            return int(cli.commercial_id)
    except Exception:
        return None
    return None


def _list_group_members(db: Session, group_id: int, actifs_only: bool = False) -> list[dict]:
    """Retourne les membres d'un groupe (clients/affaires) avec leur statut actif."""
    rows = db.execute(
        text(
            """
            SELECT id, client_id, affaire_id, actif
            FROM administration_groupe
            WHERE groupe_id = :gid
            """
        ),
        {"gid": group_id},
    ).fetchall()
    members = []
    for r in rows:
        m = r._mapping
        if actifs_only and not (m.get("actif") in (1, True, "1", "true")):
            continue
        members.append(
            {
                "id": m.get("id"),
                "client_id": m.get("client_id"),
                "affaire_id": m.get("affaire_id"),
                "actif": m.get("actif"),
            }
        )
    return members


def _resolve_affaire_client(db: Session, affaire_id: int | None) -> tuple[int | None, str | None]:
    """Pour une affaire, retourne (client_id, ref) si possible."""
    if affaire_id is None:
        return None, None
    try:
        aff = db.query(Affaire).filter(Affaire.id == affaire_id).first()
        if not aff:
            return None, None
        return getattr(aff, "id_personne", None), getattr(aff, "ref", None)
    except Exception:
        return None, None


# ---- Administration : Groupes ----
@router.post("/parametres/administration/groupes", response_class=HTMLResponse)
async def create_administration_group(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    type_value = _normalize_group_type(form.get("type_groupe"))
    nom = (form.get("nom") or "").strip()
    responsable_id = _as_int(form.get("responsable_id"))
    if not type_value:
        return _group_error_redirect("Type de groupe requis.")
    if not nom:
        return _group_error_redirect("Nom du groupe requis.")
    if responsable_id is None:
        return _group_error_redirect("Responsable obligatoire.")
    actif = _as_flag(form.get("actif"), 1)
    date_creation = _parse_date_safe(form.get("date_creation"))
    date_fin = _parse_date_safe(form.get("date_fin"))
    motif = (form.get("motif") or "").strip() or None
    try:
        next_id: int | None = None
        try:
            max_id = db.query(func.max(AdministrationGroupeDetail.id)).scalar()
            next_id = (max_id or 0) + 1
        except Exception:
            next_id = None
        payload = {
            "type_groupe": type_value,
            "nom": nom,
            "date_creation": date_creation,
            "date_fin": date_fin,
            "responsable_id": responsable_id,
            "motif": motif,
            "actif": actif,
        }
        if next_id is not None:
            payload["id"] = next_id
        group = AdministrationGroupeDetail(
            **payload,
        )
        db.add(group)
        db.commit()
        return _group_success_redirect()
    except Exception as e:
        db.rollback()
        logger.error(f"create_administration_group failed: {e}")
        return _group_error_redirect(f"Création impossible : {e}")


@router.post("/parametres/administration/groupes/{group_key}", response_class=HTMLResponse)
async def update_administration_group(group_key: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    try:
        row = db.execute(
            text(
                f"""
                SELECT id, type_groupe, nom, responsable_id, actif
                FROM administration_groupe_detail
                WHERE id = :gid
                LIMIT 1
                """
            ),
            {"gid": group_key},
        ).fetchone()
    except Exception as e:
        logger.error(f"update_administration_group lookup failed: {e}")
        return _group_error_redirect("Table administration_groupe_detail inaccessible.")
    if not row:
        return _group_error_redirect("Groupe introuvable.")
    data = row._mapping
    type_value = _normalize_group_type(form.get("type_groupe")) or data.get("type_groupe")
    nom = (form.get("nom") or "").strip() or data.get("nom")
    responsable_id = _as_int(form.get("responsable_id"), data.get("responsable_id"))
    current_actif = 0
    try:
        current_actif = 0 if int(data.get("actif") or 0) == 0 else 1
    except Exception:
        current_actif = 1
    actif = _as_flag(form.get("actif"), current_actif)
    if not type_value:
        return _group_error_redirect("Type de groupe requis.")
    if not nom:
        return _group_error_redirect("Nom du groupe requis.")
    if responsable_id is None:
        return _group_error_redirect("Responsable obligatoire.")
    params = {
        "type_groupe": type_value,
        "nom": nom,
        "responsable_id": responsable_id,
        "actif": actif,
        "gid": group_key,
    }
    try:
        db.execute(
            text(
                f"""
                UPDATE administration_groupe_detail
                SET type_groupe = :type_groupe,
                    nom = :nom,
                    responsable_id = :responsable_id,
                    actif = :actif
                WHERE id = :gid
                """
            ),
            params,
        )
        db.commit()
        return _group_success_redirect()
    except Exception as e:
        db.rollback()
        logger.error(f"update_administration_group failed: {e}")
        return _group_error_redirect(f"Mise à jour impossible : {e}")


@router.post("/parametres/administration/groupes/{group_key}/delete", response_class=HTMLResponse)
async def delete_administration_group(group_key: int, request: Request, db: Session = Depends(get_db)):
    try:
        row = db.execute(
            text(
                f"""
                SELECT id
                FROM administration_groupe_detail
                WHERE id = :gid
                LIMIT 1
                """
            ),
            {"gid": group_key},
        ).fetchone()
    except Exception as e:
        logger.error(f"delete_administration_group lookup failed: {e}")
        return _group_error_redirect("Table administration_groupe_detail inaccessible.")
    if not row:
        return _group_error_redirect("Groupe introuvable.")
    mapping = row._mapping
    real_id = mapping.get("id")
    try:
        if real_id is not None:
            db.execute(text("DELETE FROM administration_groupe WHERE groupe_id = :gid"), {"gid": real_id})
        db.execute(
            text("DELETE FROM administration_groupe_detail WHERE id = :gid"),
            {"gid": group_key},
        )
        db.commit()
        return _group_success_redirect()
    except Exception as e:
        db.rollback()
        logger.error(f"delete_administration_group failed: {e}")
        return _group_error_redirect(f"Suppression impossible : {e}")


# ---- Contrats génériques ----
@router.post("/parametres/contrats_generiques", response_class=HTMLResponse)
async def create_contrat_generique(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "nom_contrat": (form.get("nom_contrat") or "").strip(),
        "id_societe": _as_int(form.get("id_societe")),
        "id_ctg": _as_int(form.get("id_ctg")),
        "fga": _as_float(form.get("frais_gestion_assureur")),
        "fgc": _as_float(form.get("frais_gestion_courtier")),
    }
    # Validation côté serveur
    if not params["nom_contrat"]:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Le nom du contrat est requis.')}", status_code=303)
    if params["id_societe"] is None:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Assureur obligatoire. Merci de sélectionner une société.')}", status_code=303)
    if params["id_ctg"] is None:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Catégorie obligatoire. Merci de sélectionner une catégorie.')}", status_code=303)
    from sqlalchemy import text as _text
    try:
        # Forcer un id explicite si la table n'est pas en AUTOINCREMENT côté SQLite
        row = db.execute(_text("SELECT MAX(id) AS max_id FROM mariadb_affaires_generique")).fetchone()
        next_id = ((row[0] if row else 0) or 0) + 1
        db.execute(
            _text(
                """
                INSERT INTO mariadb_affaires_generique
                    (id, nom_contrat, id_societe, id_ctg, frais_gestion_assureur, frais_gestion_courtier, actif)
                VALUES (:id, :nom_contrat, :id_societe, :id_ctg, :fga, :fgc, 1)
                """
            ),
            {"id": next_id, **params},
        )
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=contrats&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/contrats_generiques/{contrat_id}", response_class=HTMLResponse)
async def update_contrat_generique(contrat_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": contrat_id,
        "nom_contrat": (form.get("nom_contrat") or "").strip(),
        "id_societe": _as_int(form.get("id_societe")),
        "id_ctg": _as_int(form.get("id_ctg")),
        "fga": _as_float(form.get("frais_gestion_assureur")),
        "fgc": _as_float(form.get("frais_gestion_courtier")),
    }
    # Validation côté serveur
    if not params["nom_contrat"]:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Le nom du contrat est requis.')}", status_code=303)
    if params["id_societe"] is None:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Assureur obligatoire. Merci de sélectionner une société.')}", status_code=303)
    if params["id_ctg"] is None:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote('Catégorie obligatoire. Merci de sélectionner une catégorie.')}", status_code=303)
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text(
                """
                UPDATE mariadb_affaires_generique
                SET nom_contrat = :nom_contrat,
                    id_societe = :id_societe,
                    id_ctg = :id_ctg,
                    frais_gestion_assureur = :fga,
                    frais_gestion_courtier = :fgc
                WHERE id = :id
                """
            ),
            params,
        )
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=contrats&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/contrats_generiques/{contrat_id}/delete", response_class=HTMLResponse)
async def delete_contrat_generique(contrat_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_affaires_generique WHERE id = :id"), {"id": contrat_id})
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=contrats&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=contrats&error=1&errmsg={quote(str(e))}", status_code=303)


# ---- Catégories de contrats génériques ----
@router.post("/parametres/contrats_generiques_ctg", response_class=HTMLResponse)
async def create_contrat_ctg(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "libelle": (form.get("libelle") or "").strip(),
        "description": (form.get("description") or None),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text("INSERT INTO mariadb_affaires_generique_ctg (libelle, description) VALUES (:libelle, :description)"),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


@router.post("/parametres/contrats_generiques_ctg/{ctg_id}", response_class=HTMLResponse)
async def update_contrat_ctg(ctg_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": ctg_id,
        "libelle": (form.get("libelle") or "").strip(),
        "description": (form.get("description") or None),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text("UPDATE mariadb_affaires_generique_ctg SET libelle = :libelle, description = :description WHERE id = :id"),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


@router.post("/parametres/contrats_generiques_ctg/{ctg_id}/delete", response_class=HTMLResponse)
async def delete_contrat_ctg(ctg_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_affaires_generique_ctg WHERE id = :id"), {"id": ctg_id})
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


# ---- Sociétés (assureurs) ----
@router.post("/parametres/societes", response_class=HTMLResponse)
async def create_societe(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "nom": (form.get("nom") or "").strip(),
        "id_ctg": _as_int(form.get("id_ctg")),
        "contact": (form.get("contact") or None),
        "telephone": (form.get("telephone") or None),
        "email": (form.get("email") or None),
        "commentaire": (form.get("commentaire") or None),
    }
    # Validation
    if not params["nom"]:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=societes&error=1&errmsg={quote('Le nom de la société est requis.')}", status_code=303)
    if params["id_ctg"] is None:
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=societes&error=1&errmsg={quote('La catégorie est obligatoire.')}", status_code=303)
    from sqlalchemy import text as _text
    try:
        # Certaines bases (SQLite) issues d'exports MariaDB n'ont pas id en AUTOINCREMENT.
        # On force un identifiant = MAX(id)+1 pour garantir la création.
        row = db.execute(_text("SELECT MAX(id) AS max_id FROM mariadb_societe")).fetchone()
        next_id = ((row[0] if row else 0) or 0) + 1
        db.execute(
            _text(
                """
                INSERT INTO mariadb_societe (id, nom, id_ctg, contact, telephone, email, commentaire, actif)
                VALUES (:id, :nom, :id_ctg, :contact, :telephone, :email, :commentaire, 1)
                """
            ),
            {"id": next_id, **params},
        )
        db.commit()
        return RedirectResponse(url="/dashboard/parametres?open=societes&saved=1", status_code=303)
    except Exception as e:
        db.rollback()
        from urllib.parse import quote
        return RedirectResponse(url=f"/dashboard/parametres?open=societes&error=1&errmsg={quote(str(e))}", status_code=303)


@router.post("/parametres/societes/{soc_id}", response_class=HTMLResponse)
async def update_societe(soc_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": soc_id,
        "nom": (form.get("nom") or "").strip(),
        "id_ctg": _as_int(form.get("id_ctg")),
        "contact": (form.get("contact") or None),
        "telephone": (form.get("telephone") or None),
        "email": (form.get("email") or None),
        "commentaire": (form.get("commentaire") or None),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text(
                """
                UPDATE mariadb_societe
                SET nom = :nom,
                    id_ctg = :id_ctg,
                    contact = :contact,
                    telephone = :telephone,
                    email = :email,
                    commentaire = :commentaire
                WHERE id = :id
                """
            ),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


@router.post("/parametres/societes/{soc_id}/delete", response_class=HTMLResponse)
async def delete_societe(soc_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_societe WHERE id = :id"), {"id": soc_id})
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


# ---- Catégories de sociétés ----
@router.post("/parametres/societe_ctg", response_class=HTMLResponse)
async def create_societe_ctg(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "libelle": (form.get("libelle") or "").strip(),
        "description": (form.get("description") or None),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text("INSERT INTO mariadb_societe_ctg (libelle, description) VALUES (:libelle, :description)"),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


@router.post("/parametres/societe_ctg/{ctg_id}", response_class=HTMLResponse)
async def update_societe_ctg(ctg_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": ctg_id,
        "libelle": (form.get("libelle") or "").strip(),
        "description": (form.get("description") or None),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text("UPDATE mariadb_societe_ctg SET libelle = :libelle, description = :description WHERE id = :id"),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


@router.post("/parametres/societe_ctg/{ctg_id}/delete", response_class=HTMLResponse)
async def delete_societe_ctg(ctg_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_societe_ctg WHERE id = :id"), {"id": ctg_id})
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


# ---- Supports par contrat (mapping + taux) ----
@router.post("/parametres/contrat_supports", response_class=HTMLResponse)
async def create_contrat_support(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    id_affaire_generique = _as_int(form.get("id_affaire_generique"))
    id_support = _as_int(form.get("id_support"))
    taux_percent = _as_float(form.get("taux_retro"), 0.0) or 0.0
    taux_value = taux_percent / 100.0
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text(
                """
                INSERT INTO mariadb_contrat_supports (id_affaire_generique, id_support, taux_retro)
                VALUES (:id_affaire_generique, :id_support, :taux_retro)
                """
            ),
            {"id_affaire_generique": id_affaire_generique, "id_support": id_support, "taux_retro": taux_value},
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "supports")


@router.post("/parametres/contrat_supports/{row_id}", response_class=HTMLResponse)
async def update_contrat_support(row_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    taux_percent = _as_float(form.get("taux_retro"), 0.0) or 0.0
    taux_value = taux_percent / 100.0
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text("UPDATE mariadb_contrat_supports SET taux_retro = :t WHERE id = :id"),
            {"id": row_id, "t": taux_value},
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "supports")


@router.post("/parametres/contrat_supports/{row_id}/delete", response_class=HTMLResponse)
async def delete_contrat_support(row_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_contrat_supports WHERE id = :id"), {"id": row_id})
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "supports")


# ---------------- Accueil ----------------
@router.get("/", response_class=HTMLResponse)
def dashboard_home(request: Request, db: Session = Depends(get_db)):
    t0 = perf_counter()
    # Pas de cache afin d'assurer le chargement des derniers assets/templates

    tb_markers_visible = request.query_params.get("markers") == "1"
    # Totaux simples
    total_clients = db.query(func.count(Client.id)).scalar() or 0
    total_affaires = db.query(func.count(Affaire.id)).scalar() or 0

    # Dernière valo par client → somme
    sub_cli = (
        db.query(
            HistoriquePersonne.id.label("client_id"),
            func.max(HistoriquePersonne.date).label("last_date")
        )
        .group_by(HistoriquePersonne.id)
        .subquery()
    )
    finance_valo_param = request.query_params.get("finance_valo")
    finance_global_ctx = _build_finance_analysis(
        db=db,
        finance_rh_id=None,
        finance_date_param=None,
        finance_valo_param=finance_valo_param,
    )
    total_valo = finance_global_ctx["finance_total_valo"]
    t_finance = perf_counter()
    logger.info("dashboard_home: finance_global_ctx computed in %.3fs", t_finance - t0)

    # Découpage du nombre de clients par intervalles de détention (basé sur la dernière valo par client)
    last_valos = (
        db.query(HistoriquePersonne.valo)
        .join(sub_cli, sub_cli.c.client_id == HistoriquePersonne.id)
        .filter(HistoriquePersonne.date == sub_cli.c.last_date)
        .all()
    )
    last_vals = [float(v or 0) for (v,) in last_valos]
    buckets = [
        ("0-100 000", 0, 100_000),
        ("100 000 - 250 000", 100_000, 250_000),
        ("250 000 - 500 000", 250_000, 500_000),
        ("500 000 - 1 M", 500_000, 1_000_000),
        ("1M - 5M", 1_000_000, 5_000_000),
        ("> 5M", 5_000_000, None),
    ]
    clients_buckets = []
    for label, lo, hi in buckets:
        if hi is None:
            cnt = sum(1 for v in last_vals if v is not None and v >= lo)
        else:
            cnt = sum(1 for v in last_vals if v is not None and lo <= v < hi)
        clients_buckets.append({"label": label, "nb": cnt})

    # Comptes par SRRI
    srri_clients_count = [
        {"srri": s, "nb": n}
        for s, n in db.query(Client.SRRI, func.count(Client.id)).group_by(Client.SRRI).all()
    ]
    srri_affaires_count = [
        {"srri": s, "nb": n}
        for s, n in db.query(Affaire.SRRI, func.count(Affaire.id)).group_by(Affaire.SRRI).all()
    ]

    # Montants par SRRI (clients)
    srri_clients_amount = (
        db.query(
            Client.SRRI,
            func.coalesce(func.sum(HistoriquePersonne.valo), 0).label("total_valo")
        )
        .join(sub_cli, sub_cli.c.client_id == HistoriquePersonne.id)
        .join(Client, Client.id == sub_cli.c.client_id)
        .filter(HistoriquePersonne.date == sub_cli.c.last_date)
        .group_by(Client.SRRI)
        .all()
    )
    srri_clients_amount = [
        {"srri": s, "total": float(v or 0)} for s, v in srri_clients_amount
    ]

    # Montants par SRRI (affaires)
    sub_aff = (
        db.query(
            HistoriqueAffaire.id.label("affaire_id"),
            func.max(HistoriqueAffaire.date).label("last_date")
        )
        .group_by(HistoriqueAffaire.id)
        .subquery()
    )
    srri_affaires_amount = (
        db.query(
            Affaire.SRRI,
            func.coalesce(func.sum(HistoriqueAffaire.valo), 0).label("total_valo")
        )
        .join(sub_aff, sub_aff.c.affaire_id == HistoriqueAffaire.id)
        .join(Affaire, Affaire.id == sub_aff.c.affaire_id)
        .filter(HistoriqueAffaire.date == sub_aff.c.last_date)
        .group_by(Affaire.SRRI)
        .all()
    )
    srri_affaires_amount = [
        {"srri": s, "total": float(v or 0)} for s, v in srri_affaires_amount
    ]

    # Référentiel RH (pour filtres analyse financière)
    rh_entries = fetch_rh_list(db)
    finance_rh_options: list[dict[str, str | int]] = []
    for rh in rh_entries or []:
        try:
            rid = int(rh.get("id"))
        except Exception:
            continue
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        mail = (rh.get("mail") or "").strip()
        label_parts = [p for p in [prenom, nom] if p]
        if label_parts:
            label = " ".join(label_parts)
        elif mail:
            label = mail
        else:
            label = f"RH #{rid}"
        finance_rh_options.append({"id": rid, "label": label})
    finance_rh_options.sort(key=lambda x: str(x["label"] or "").lower())

    finance_rh_param = request.query_params.get("finance_rh")
    finance_rh_id: int | None = None
    if finance_rh_param not in (None, ""):
        try:
            finance_rh_id = int(finance_rh_param)
        except (TypeError, ValueError):
            finance_rh_id = None

    finance_ctx = _build_finance_analysis(
        db=db,
        finance_rh_id=finance_rh_id,
        finance_date_param=request.query_params.get("finance_date"),
        finance_valo_param=request.query_params.get("finance_valo"),
    )
    finance_supports = finance_ctx["finance_supports"]
    finance_total_valo = finance_ctx["finance_total_valo"]
    finance_total_valo_str = finance_ctx["finance_total_valo_str"]
    finance_date_input = finance_ctx["finance_date_input"]
    finance_effective_date_display = finance_ctx["finance_effective_date_display"]
    finance_effective_date_iso = finance_ctx["finance_effective_date_iso"]
    # Types/statuts pour tâches groupées (dashboard Groupes)
    try:
        evt_types_rows = db.execute(text("SELECT libelle, categorie FROM mariadb_type_evenement ORDER BY categorie, libelle")).fetchall()
        evt_types = rows_to_dicts(evt_types_rows)
        evt_categories = sorted({(r.get("categorie") or "").strip() for r in evt_types if r.get("categorie")})
    except Exception:
        evt_types = []
        evt_categories = []
    try:
        evt_statuts_rows = db.execute(text("SELECT id, libelle FROM mariadb_statut_evenement ORDER BY libelle")).fetchall()
        evt_statuts = rows_to_dicts(evt_statuts_rows)
    except Exception:
        evt_statuts = []

    # Informations Documents (comme sur la page Documents)
    orias_doc_found = False
    orias_doc_valid = False
    orias_doc_expiry_display = ""
    assoc_doc_found = False
    assoc_doc_valid = False
    assoc_doc_expiry_display = ""
    rcpro_doc_count = 0
    rcpro_doc_valid = False
    rcpro_doc_expiry_display = ""
    rcpro_doc_expired_count = 0
    rcpro_doc_valid_count = 0
    conditions_doc_count = 0
    conditions_doc_valid = False
    conditions_doc_expiry_display = ""
    conditions_doc_expired_count = 0
    conditions_doc_valid_count = 0
    role_doc_count = 0
    role_doc_valid = False
    role_doc_expiry_display = ""
    role_doc_expired_count = 0
    role_doc_valid_count = 0
    gouvernance_doc_count = 0
    gouvernance_doc_valid = False
    gouvernance_doc_expiry_display = ""
    gouvernance_doc_expired_count = 0
    gouvernance_doc_valid_count = 0
    conclusion_doc_count = 0
    conclusion_doc_valid = False
    conclusion_doc_expiry_display = ""
    conclusion_doc_expired_count = 0
    conclusion_doc_valid_count = 0
    blanchiment_doc_count = 0
    blanchiment_doc_valid = False
    blanchiment_doc_expiry_display = ""
    blanchiment_doc_expired_count = 0
    blanchiment_doc_valid_count = 0
    doc_today = _date.today()
    distribution_avg_obsolete_pct = 0.0
    distribution_avg_valid_pct = 100.0
    reclam_stats = {"total": 0, "by_status": []}
    reclam_pivot = {"types": [], "statuses": []}
    # Comptages documents (init par défaut)
    total_documents = 0
    der_total = 0
    der_obsolete = 0
    der_total_pct = 0.0
    der_obsolete_pct = 0.0
    cc_total = 0
    cc_obsolete = 0
    cc_total_pct = 0.0
    cc_obsolete_pct = 0.0
    esg_total = 0
    esg_obsolete = 0
    esg_total_pct = 0.0
    esg_obsolete_pct = 0.0
    lm_total = 0
    lm_obsolete = 0
    lm_total_pct = 0.0
    lm_obsolete_pct = 0.0
    la_total = 0
    la_obsolete = 0
    la_total_pct = 0.0
    la_obsolete_pct = 0.0
    try:
        total_documents = db.query(func.count(DocumentClient.id)).scalar() or 0
        der_total = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == 'der')
            .scalar()
            or 0
        )
        der_obsolete = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == 'der')
            .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
            .scalar()
            or 0
        )
        der_total_pct = (der_total / total_clients * 100.0) if total_clients else 0.0
        der_obsolete_pct = (der_obsolete / total_clients * 100.0) if total_clients else 0.0
        cc_total = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "recueil d'informations")
            .scalar()
            or 0
        )
        cc_obsolete = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "recueil d'informations")
            .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
            .scalar()
            or 0
        )
        cc_total_pct = (cc_total / total_clients * 100.0) if total_clients else 0.0
        cc_obsolete_pct = (cc_obsolete / total_clients * 100.0) if total_clients else 0.0
        esg_total = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "questionnaire esg")
            .scalar()
            or 0
        )
        esg_obsolete = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "questionnaire esg")
            .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
            .scalar()
            or 0
        )
        esg_total_pct = (esg_total / total_clients * 100.0) if total_clients else 0.0
        esg_obsolete_pct = (esg_obsolete / total_clients * 100.0) if total_clients else 0.0
        lm_total = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "lettre de mission")
            .scalar()
            or 0
        )
        lm_obsolete = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "lettre de mission")
            .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
            .scalar()
            or 0
        )
        lm_total_pct = (lm_total / total_clients * 100.0) if total_clients else 0.0
        lm_obsolete_pct = (lm_obsolete / total_clients * 100.0) if total_clients else 0.0
        la_total = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "lettre d'adéquation")
            .scalar()
            or 0
        )
        la_obsolete = (
            db.query(func.count(DocumentClient.id))
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(func.lower(Document.documents) == "lettre d'adéquation")
            .filter(func.lower(func.trim(DocumentClient.obsolescence)) == 'oui')
            .scalar()
            or 0
        )
        la_total_pct = (la_total / total_clients * 100.0) if total_clients else 0.0
        la_obsolete_pct = (la_obsolete / total_clients * 100.0) if total_clients else 0.0
        # Obsolescences par niveau
        obs_by_niveau_rows = (
            db.query(
                Document.niveau,
                func.count(DocumentClient.id)
            )
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(DocumentClient.obsolescence.isnot(None))
            .group_by(Document.niveau)
            .all()
        )
        obs_by_niveau = [{"niveau": n, "nb": int(nb)} for n, nb in obs_by_niveau_rows]
        # Obsolescences par risque
        obs_by_risque_rows = (
            db.query(
                Document.risque,
                func.count(DocumentClient.id)
            )
            .join(Document, Document.id_document_base == DocumentClient.id_document_base)
            .filter(DocumentClient.obsolescence.isnot(None))
            .group_by(Document.risque)
            .all()
        )
        obs_by_risque = [{"risque": r, "nb": int(nb)} for r, nb in obs_by_risque_rows]

        doc_table_name = _resolve_table_name(db, ["courtier_documents_officiels"])
        if doc_table_name:
            row = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 1},
            ).fetchone()
            if row:
                orias_doc_found = True
                expiry_raw = row._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    orias_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        orias_doc_valid = True
            row_assoc = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 2},
            ).fetchone()
            if row_assoc:
                assoc_doc_found = True
                expiry_raw = row_assoc._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    assoc_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        assoc_doc_valid = True
            rc_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
            ).fetchone()
            count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                    """
                ),
            ).fetchone()
            expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id IN (3, 10)
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"today": doc_today.isoformat()},
            ).fetchone()
            rcpro_doc_count = int(count_row[0]) if count_row else 0
            rcpro_doc_expired_count = int(expired_row[0]) if expired_row else 0
            rcpro_doc_valid_count = max(rcpro_doc_count - rcpro_doc_expired_count, 0)
            if rc_rows:
                expiry_raw = rc_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    rcpro_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        rcpro_doc_valid = True
            cond_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 5},
            ).fetchone()
            cond_count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    """
                ),
                {"act": 5},
            ).fetchone()
            cond_expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"act": 5, "today": doc_today.isoformat()},
            ).fetchone()
            conditions_doc_count = int(cond_count_row[0]) if cond_count_row else 0
            conditions_doc_expired_count = int(cond_expired_row[0]) if cond_expired_row else 0
            conditions_doc_valid_count = max(conditions_doc_count - conditions_doc_expired_count, 0)
            if cond_rows:
                expiry_raw = cond_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    conditions_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        conditions_doc_valid = True
            role_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 6},
            ).fetchone()
            role_count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    """
                ),
                {"act": 6},
            ).fetchone()
            role_expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"act": 6, "today": doc_today.isoformat()},
            ).fetchone()
            role_doc_count = int(role_count_row[0]) if role_count_row else 0
            role_doc_expired_count = int(role_expired_row[0]) if role_expired_row else 0
            role_doc_valid_count = max(role_doc_count - role_doc_expired_count, 0)
            if role_rows:
                expiry_raw = role_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    role_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        role_doc_valid = True
            gouvernance_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 8},
            ).fetchone()
            gouvernance_count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    """
                ),
                {"act": 8},
            ).fetchone()
            gouvernance_expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"act": 8, "today": doc_today.isoformat()},
            ).fetchone()
            gouvernance_doc_count = int(gouvernance_count_row[0]) if gouvernance_count_row else 0
            gouvernance_doc_expired_count = int(gouvernance_expired_row[0]) if gouvernance_expired_row else 0
            gouvernance_doc_valid_count = max(gouvernance_doc_count - gouvernance_doc_expired_count, 0)
            if gouvernance_rows:
                expiry_raw = gouvernance_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    gouvernance_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        gouvernance_doc_valid = True
            conclusion_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 9},
            ).fetchone()
            conclusion_count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    """
                ),
                {"act": 9},
            ).fetchone()
            conclusion_expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"act": 9, "today": doc_today.isoformat()},
            ).fetchone()
            conclusion_doc_count = int(conclusion_count_row[0]) if conclusion_count_row else 0
            conclusion_doc_expired_count = int(conclusion_expired_row[0]) if conclusion_expired_row else 0
            conclusion_doc_valid_count = max(conclusion_doc_count - conclusion_doc_expired_count, 0)
            if conclusion_rows:
                expiry_raw = conclusion_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    conclusion_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        conclusion_doc_valid = True
            blanch_rows = db.execute(
                text(
                    f"""
                    SELECT id, date_validite
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    ORDER BY
                        CASE WHEN date_validite IS NULL OR TRIM(date_validite) = '' THEN 1 ELSE 0 END,
                        date_validite DESC,
                        id DESC
                    LIMIT 1
                    """
                ),
                {"act": 7},
            ).fetchone()
            blanch_count_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                    """
                ),
                {"act": 7},
            ).fetchone()
            blanch_expired_row = db.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS nb
                    FROM {doc_table_name}
                    WHERE activite_id = :act
                      AND (
                        date_validite IS NULL
                        OR TRIM(date_validite) = ''
                        OR DATE(substr(date_validite, 1, 10)) < DATE(:today)
                      )
                    """
                ),
                {"act": 7, "today": doc_today.isoformat()},
            ).fetchone()
            blanchiment_doc_count = int(blanch_count_row[0]) if blanch_count_row else 0
            blanchiment_doc_expired_count = int(blanch_expired_row[0]) if blanch_expired_row else 0
            blanchiment_doc_valid_count = max(blanchiment_doc_count - blanchiment_doc_expired_count, 0)
            if blanch_rows:
                expiry_raw = blanch_rows._mapping.get("date_validite")
                expiry_date = _parse_date_safe(expiry_raw)
                if expiry_date:
                    blanchiment_doc_expiry_display = expiry_date.strftime("%d/%m/%Y")
                    if expiry_date >= doc_today:
                        blanchiment_doc_valid = True

        distribution_obsolete_values = [
            v for v in [
                der_obsolete_pct,
                cc_obsolete_pct,
                esg_obsolete_pct,
                lm_obsolete_pct,
                la_obsolete_pct,
            ]
            if isinstance(v, (int, float))
        ]
        distribution_avg_obsolete_pct = (sum(distribution_obsolete_values) / len(distribution_obsolete_values)) if distribution_obsolete_values else 0.0
        if distribution_avg_obsolete_pct < 0:
            distribution_avg_obsolete_pct = 0.0
        if distribution_avg_obsolete_pct > 100:
            distribution_avg_obsolete_pct = 100.0
        distribution_avg_valid_pct = 100.0 - distribution_avg_obsolete_pct
    except Exception as exc:
        logger.exception("Failed to compute compliance stats", exc_info=exc)
        obs_by_niveau = []
        obs_by_risque = []
        orias_doc_found = False
        orias_doc_valid = False
        orias_doc_expiry_display = ""
        assoc_doc_found = False
        assoc_doc_valid = False
        assoc_doc_expiry_display = ""
        rcpro_doc_count = 0
        rcpro_doc_valid = False
        rcpro_doc_expiry_display = ""
        blanchiment_doc_count = 0
        blanchiment_doc_valid = False
        blanchiment_doc_expiry_display = ""
        distribution_avg_obsolete_pct = 0.0
        distribution_avg_valid_pct = 100.0
        reclam_stats = {"total": 0, "by_status": []}
        reclam_pivot = {"types": [], "statuses": []}

    reclam_stats, reclam_pivot = _compute_reclamations_data(db)

    orias_detail = locals().get("orias_detail", None)

    # Comparatif contrats: au-dessus / identique / en-dessous du risque (SRRI contrat vs calculé)
    try:
        subq_aff = (
            db.query(
                HistoriqueAffaire.id.label("affaire_id"),
                func.max(HistoriqueAffaire.date).label("last_date")
            )
            .group_by(HistoriqueAffaire.id)
            .subquery()
        )
        rows = (
            db.query(
                Affaire.SRRI,
                HistoriqueAffaire.volat,
                HistoriqueAffaire.valo
            )
            .join(subq_aff, subq_aff.c.affaire_id == Affaire.id)
            .join(
                HistoriqueAffaire,
                (HistoriqueAffaire.id == subq_aff.c.affaire_id) &
                (HistoriqueAffaire.date == subq_aff.c.last_date)
            )
            .all()
        )
        def _srri_from_vol(v):
            if v is None:
                return None
            try:
                x = float(v)
            except Exception:
                return None
            if abs(x) <= 1:
                x *= 100.0
            if x <= 0.5: return 1
            if x <= 2: return 2
            if x <= 5: return 3
            if x <= 10: return 4
            if x <= 15: return 5
            if x <= 25: return 6
            return 7
        compare_counts = {"above": 0, "equal": 0, "below": 0}
        compare_amounts = {"above": 0.0, "equal": 0.0, "below": 0.0}
        for srri_contract, vol, valo in rows:
            calc = _srri_from_vol(vol)
            if srri_contract is None or calc is None:
                continue
            try:
                c = int(srri_contract)
                k = int(calc)
            except Exception:
                continue
            try:
                amount = float(valo or 0.0)
            except Exception:
                amount = 0.0
            # Règle cohérente avec les icônes: Au-dessus = c > k, En-dessous = c < k
            if c > k:
                compare_counts["above"] += 1
                compare_amounts["above"] += amount
            elif c == k:
                compare_counts["equal"] += 1
                compare_amounts["equal"] += amount
            else:
                compare_counts["below"] += 1
                compare_amounts["below"] += amount
    except Exception:
        compare_counts = {"above": 0, "equal": 0, "below": 0}
        compare_amounts = {"above": 0.0, "equal": 0.0, "below": 0.0}

    # Comparatif clients: client vs risque (SRRI client vs SRRI historique courant)
    try:
        # Reuse last-date per client subquery (sub_cli)
        rows_cli = (
            db.query(
                Client.SRRI.label("client_srri"),
                HistoriquePersonne.SRRI.label("hist_srri"),
                HistoriquePersonne.valo.label("valo")
            )
            .join(sub_cli, sub_cli.c.client_id == HistoriquePersonne.id)
            .join(Client, Client.id == sub_cli.c.client_id)
            .filter(HistoriquePersonne.date == sub_cli.c.last_date)
            .all()
        )
        cli_counts = {"above": 0, "equal": 0, "below": 0}
        cli_amounts = {"above": 0.0, "equal": 0.0, "below": 0.0}
        for r in rows_cli:
            cs = getattr(r, "client_srri", None)
            hs = getattr(r, "hist_srri", None)
            valo = getattr(r, "valo", None)
            if cs is None or hs is None:
                continue
            try:
                c = int(cs)
                h = int(hs)
            except Exception:
                continue
            try:
                amount = float(valo or 0.0)
            except Exception:
                amount = 0.0
            # Interprétation: "Au-dessus du risque" = le niveau de risque de l'historique (réalité)
            # est au-dessus du risque cible client (h > c). "Sous le risque" = h < c.
            if h > c:
                cli_counts["above"] += 1
                cli_amounts["above"] += amount
            elif h == c:
                cli_counts["equal"] += 1
                cli_amounts["equal"] += amount
            else:
                cli_counts["below"] += 1
                cli_amounts["below"] += amount
    except Exception:
        cli_counts = {"above": 0, "equal": 0, "below": 0}
        cli_amounts = {"above": 0.0, "equal": 0.0, "below": 0.0}

    def _format_amount(value: float) -> str:
        try:
            return "{:,}".format(round(value)).replace(",", " ")
        except Exception:
            return "0"

    def _format_pct(amount: float, total: float) -> str:
        try:
            if not total:
                return "0.00"
            return f"{(amount / total) * 100:.2f}".replace(".", ",")
        except Exception:
            return "0,00"

    compare_amounts_fmt = {k: _format_amount(v) for k, v in compare_amounts.items()}
    cli_amounts_fmt = {k: _format_amount(v) for k, v in cli_amounts.items()}
    total_valo_safe = 0.0
    try:
        total_valo_safe = float(total_valo or 0.0)
    except Exception:
        total_valo_safe = 0.0

    cli_under_pct = _format_pct(cli_amounts.get("above", 0.0), total_valo_safe)
    aff_under_pct = _format_pct(compare_amounts.get("above", 0.0), total_valo_safe)

    # ------- Tâches / événements (lazy, chargées via /dashboard/tasks/summary) -------
    try:
        range_days = int(request.query_params.get("tasks_range", 14))
        if range_days not in (7, 14, 30):
            range_days = 14
    except Exception:
        range_days = 14
    tasks_total = 0
    open_count = 0
    tasks_statut: list[dict] = []
    tasks_categorie: list[dict] = []
    tasks_days: list[dict] = []
    tasks_avg_by_statut: list[dict] = []
    tasks_avg_close_days = 0.0
    tasks_close_dist: list[dict] = []
    tasks_type_by_rh: list[dict] = []
    tasks_type_total: list[dict] = []

    logger.info("dashboard_home: tasks/veille/remu sections computed in %.3fs", perf_counter() - t_finance)

    veille_ctx = _get_veille_context()

    rem_contracts = []
    rem_rows_full = []
    rem_total_commission = 0.0
    rem_total_valorisation = 0.0
    rem_total_contracts = 0
    rem_error = None
    rem_limit_options = [10, 25, 50, 100]

    try:
        rem_contracts = rows_to_dicts(
            db.execute(
                text(
                    """
                    SELECT id, nom_contrat
                    FROM mariadb_affaires_generique
                    WHERE actif IS NULL OR actif <> 0
                    ORDER BY nom_contrat
                    """
                )
            ).fetchall()
        )
    except Exception:
        rem_contracts = []

    today = _date.today()
    default_start = today - timedelta(days=84)
    default_end = today

    rem_start_input = request.query_params.get("rem_start") or default_start.isoformat()
    rem_end_input = request.query_params.get("rem_end") or default_end.isoformat()

    parsed_start = _parse_date_safe(rem_start_input) or default_start
    parsed_end = _parse_date_safe(rem_end_input) or default_end
    if parsed_start > parsed_end:
        parsed_start, parsed_end = parsed_end, parsed_start

    rem_start_effective = _align_to_friday(parsed_start)
    rem_end_effective = _align_to_friday(parsed_end)
    if (
        rem_start_effective is not None
        and rem_end_effective is not None
        and rem_end_effective < rem_start_effective
    ):
        rem_end_effective = rem_start_effective

    try:
        rem_limit = int(request.query_params.get("rem_limit", rem_limit_options[0]))
    except Exception:
        rem_limit = rem_limit_options[0]
    if rem_limit not in rem_limit_options:
        rem_limit = rem_limit_options[0]

    try:
        rem_page = int(request.query_params.get("rem_page", 1))
    except Exception:
        rem_page = 1
    if rem_page < 1:
        rem_page = 1

    raw_contract = request.query_params.get("rem_contract")
    rem_selected_contract = None
    try:
        if raw_contract is not None:
            rem_selected_contract = int(raw_contract)
    except Exception:
        rem_selected_contract = None
    if rem_selected_contract is None and rem_contracts:
        rem_selected_contract = rem_contracts[0]["id"]
    if rem_selected_contract is not None and rem_contracts:
        valid_contract_ids = {c["id"] for c in rem_contracts}
        if rem_selected_contract not in valid_contract_ids:
            rem_selected_contract = rem_contracts[0]["id"]

    if (
        rem_selected_contract is not None
        and rem_start_effective is not None
        and rem_end_effective is not None
    ):
        try:
            rows = db.execute(
                text(
                    """
                    SELECT
                        h.date AS date,
                        SUM(h.valo) AS total_valorisation,
                        SUM(h.valo) * (g.frais_gestion_courtier / 52.0 / 100.0) AS commission_frais_gestion,
                        COUNT(DISTINCT a.id) AS nb_contrats
                    FROM mariadb_historique_affaire_w h
                    JOIN mariadb_affaires a ON h.id = a.id
                    JOIN mariadb_affaires_generique g ON a.id_affaire_generique = g.id
                    WHERE h.date BETWEEN :start AND :end
                      AND g.id = :contract_id
                    GROUP BY h.date, g.frais_gestion_courtier
                    ORDER BY h.date
                    """
                ),
                {
                    "start": rem_start_effective.isoformat(),
                    "end": rem_end_effective.isoformat(),
                    "contract_id": rem_selected_contract,
                },
            ).fetchall()

            for row in rows:
                data = row._mapping
                week_date = _parse_date_safe(data.get("date"))
                total_valo_week = float(data.get("total_valorisation") or 0)
                commission_week = float(data.get("commission_frais_gestion") or 0)
                contracts_week = int(data.get("nb_contrats") or 0)
                rem_rows_full.append(
                    {
                        "date": week_date,
                        "total_valorisation": total_valo_week,
                        "commission": commission_week,
                        "contracts_count": contracts_week,
                    }
                )
                rem_total_valorisation += total_valo_week
                rem_total_commission += commission_week
                rem_total_contracts += contracts_week
        except Exception:
            rem_error = "Impossible de calculer les commissions pour la période demandée."

    rem_rows_count_total = len(rem_rows_full)
    if rem_rows_count_total == 0:
        rem_page = 1

    rem_total_pages = max(1, (rem_rows_count_total + rem_limit - 1) // rem_limit)
    if rem_page > rem_total_pages:
        rem_page = rem_total_pages

    page_start_idx = (rem_page - 1) * rem_limit
    page_end_idx = page_start_idx + rem_limit
    rem_rows = rem_rows_full[page_start_idx:page_end_idx]

    rem_page_start = page_start_idx + 1 if rem_rows else 0
    rem_page_end = page_start_idx + len(rem_rows)
    rem_has_prev = rem_page > 1
    rem_has_next = rem_page < rem_total_pages

    base_params = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if not key.startswith("rem_")
    ]
    if rem_selected_contract is not None:
        base_params.append(("rem_contract", str(rem_selected_contract)))
    if rem_start_input:
        base_params.append(("rem_start", rem_start_input))
    if rem_end_input:
        base_params.append(("rem_end", rem_end_input))
    base_params.append(("rem_limit", str(rem_limit)))

    rem_prev_url = None
    rem_next_url = None
    if rem_has_prev:
        rem_prev_url = f"{request.url.path}?{urlencode(base_params + [('rem_page', str(rem_page - 1))], doseq=True)}"
    if rem_has_next:
        rem_next_url = f"{request.url.path}?{urlencode(base_params + [('rem_page', str(rem_page + 1))], doseq=True)}"

    retro_contracts = []
    retro_error = None
    retro_weeks: list[dict] = []
    retro_supports: list[dict] = []
    retro_total_week = 0.0
    retro_total_support = 0.0
    retro_selected_contract = None
    retro_week_limit_options = [10, 25, 50]
    retro_support_limit_options = [10, 25, 50, 100]

    try:
        retro_contracts = rows_to_dicts(
            db.execute(
                text(
                    """
                    SELECT id,
                           COALESCE(nom_contrat, 'Contrat ' || id) AS nom_contrat
                    FROM mariadb_affaires_generique
                    WHERE COALESCE(actif, 1) = 1
                    ORDER BY nom_contrat
                    """
                )
            ).fetchall()
        )
    except Exception as exc:
        retro_error = "Impossible de récupérer la liste des contrats génériques."
        logger.debug("Dashboard rétrocessions: erreur lors de la récupération des contrats: %s", exc, exc_info=True)
        retro_contracts = []

    retro_sort = request.query_params.get("ret_sort") or "date_desc"
    allowed_sort = {"date_desc", "date_asc", "retrocession_desc", "retrocession_asc"}
    if retro_sort not in allowed_sort:
        retro_sort = "date_desc"
    retro_order_week = {
        "date_desc": "ORDER BY date DESC",
        "date_asc": "ORDER BY date ASC",
        "retrocession_desc": "ORDER BY retrocession DESC",
        "retrocession_asc": "ORDER BY retrocession ASC",
    }[retro_sort]
    retro_order_support = {
        "date_desc": "ORDER BY retrocession DESC",
        "date_asc": "ORDER BY retrocession DESC",
        "retrocession_desc": "ORDER BY retrocession DESC",
        "retrocession_asc": "ORDER BY retrocession ASC",
    }[retro_sort]

    try:
        retro_week_limit = int(request.query_params.get("ret_week_limit", retro_week_limit_options[0]))
    except Exception:
        retro_week_limit = retro_week_limit_options[0]
    if retro_week_limit not in retro_week_limit_options:
        retro_week_limit = retro_week_limit_options[0]

    try:
        retro_support_limit = int(request.query_params.get("ret_support_limit", retro_support_limit_options[0]))
    except Exception:
        retro_support_limit = retro_support_limit_options[0]
    if retro_support_limit not in retro_support_limit_options:
        retro_support_limit = retro_support_limit_options[0]

    retro_start_input = request.query_params.get("ret_start")
    retro_end_input = request.query_params.get("ret_end")
    retro_promoteur = (request.query_params.get("ret_promoteur") or "").strip()

    retro_default_start = today - timedelta(days=180)
    retro_default_end = today
    parsed_retro_start = _parse_date_safe(retro_start_input) or retro_default_start
    parsed_retro_end = _parse_date_safe(retro_end_input) or retro_default_end
    if parsed_retro_start > parsed_retro_end:
        parsed_retro_start, parsed_retro_end = parsed_retro_end, parsed_retro_start

    retro_start_effective = _align_to_friday(parsed_retro_start)
    retro_end_effective = _align_to_friday(parsed_retro_end)
    if (
        retro_start_effective is not None
        and retro_end_effective is not None
        and retro_end_effective < retro_start_effective
    ):
        retro_end_effective = retro_start_effective

    try:
        raw_ret_contract = request.query_params.get("ret_contract")
        if raw_ret_contract is not None:
            retro_selected_contract = int(raw_ret_contract)
    except Exception:
        retro_selected_contract = None
    if retro_selected_contract is None and retro_contracts:
        retro_selected_contract = retro_contracts[0]["id"]
    if retro_selected_contract is not None and retro_contracts:
        valid_ret_ids = {c["id"] for c in retro_contracts}
        if retro_selected_contract not in valid_ret_ids:
            retro_selected_contract = retro_contracts[0]["id"]

    logger.debug(
        "Dashboard rétrocessions paramètres: contrat=%s, start=%s, end=%s, week_limit=%s, support_limit=%s, promoteur=%s, sort=%s",
        retro_selected_contract,
        retro_start_effective,
        retro_end_effective,
        retro_week_limit,
        retro_support_limit,
        retro_promoteur,
        retro_sort,
    )

    if (
        retro_selected_contract is not None
        and retro_start_effective is not None
        and retro_end_effective is not None
        and not retro_error
    ):
        try:
            params = {
                "start": retro_start_effective.isoformat(),
                "end": retro_end_effective.isoformat(),
                "contract_id": retro_selected_contract,
                "week_limit": retro_week_limit,
            }
            week_query = text(
                f"""
                SELECT date,
                       valo_total,
                       retrocession,
                       nb_contrats
                FROM (
                    SELECT h.date AS date,
                           SUM(h.valo) AS valo_total,
                           SUM(h.valo * COALESCE(cs.taux_retro, 0) / 52.0) AS retrocession,
                           COUNT(DISTINCT a.id) AS nb_contrats
                    FROM mariadb_historique_support_w h
                    JOIN mariadb_affaires a ON a.id = h.id_source
                    JOIN mariadb_affaires_generique g ON g.id = a.id_affaire_generique
                    LEFT JOIN mariadb_contrat_supports cs
                        ON cs.id_affaire_generique = g.id AND cs.id_support = h.id_support
                    WHERE h.date BETWEEN :start AND :end
                      AND g.id = :contract_id
                    GROUP BY h.date
                ) base
                {retro_order_week}
                LIMIT :week_limit
                """
            )
            week_rows = db.execute(week_query, params).fetchall()
            retro_weeks = []
            for row in week_rows:
                data = row._mapping
                week_date = _parse_date_safe(data.get("date"))
                retro_val = float(data.get("retrocession") or 0)
                retro_weeks.append(
                    {
                        "date": week_date,
                        "date_str": week_date.strftime("%d/%m/%Y") if week_date else (data.get("date") or ""),
                        "retrocession": retro_val,
                        "retrocession_str": "{:,.2f}".format(retro_val).replace(",", " "),
                        "valo_total": float(data.get("valo_total") or 0),
                        "valo_total_str": "{:,.0f}".format(float(data.get("valo_total") or 0)).replace(",", " "),
                        "nb_contrats": int(data.get("nb_contrats") or 0),
                    }
                )
                retro_total_week += retro_val
            logger.debug(
                "Dashboard rétrocessions: %s lignes hebdo récupérées pour le contrat %s",
                len(retro_weeks),
                retro_selected_contract,
            )

            support_params = {
                "start": retro_start_effective.isoformat(),
                "end": retro_end_effective.isoformat(),
                "contract_id": retro_selected_contract,
                "support_limit": retro_support_limit,
                "promoteur": retro_promoteur,
                "promoteur_pattern": f"%{retro_promoteur.lower()}%",
            }
            support_query = text(
                f"""
                SELECT promoteur,
                       support_nom,
                       code_isin,
                       retrocession,
                       valo_total
                FROM (
                    SELECT COALESCE(LOWER(s.promoteur), '') AS promoteur_key,
                           COALESCE(s.promoteur, 'N/A') AS promoteur,
                           s.nom AS support_nom,
                           s.code_isin AS code_isin,
                           SUM(h.valo) AS valo_total,
                           SUM(h.valo * COALESCE(cs.taux_retro, 0) / 52.0) AS retrocession
                    FROM mariadb_historique_support_w h
                    JOIN mariadb_support s ON s.id = h.id_support
                    JOIN mariadb_affaires a ON a.id = h.id_source
                    JOIN mariadb_affaires_generique g ON g.id = a.id_affaire_generique
                    LEFT JOIN mariadb_contrat_supports cs
                        ON cs.id_affaire_generique = g.id AND cs.id_support = h.id_support
                    WHERE h.date BETWEEN :start AND :end
                      AND g.id = :contract_id
                    GROUP BY promoteur_key, promoteur, s.nom, s.code_isin
                ) base
                WHERE (:promoteur = '' OR promoteur_key LIKE :promoteur_pattern)
                {retro_order_support}
                LIMIT :support_limit
                """
            )
            support_rows = db.execute(support_query, support_params).fetchall()
            retro_supports = []
            for row in support_rows:
                data = row._mapping
                retro_val = float(data.get("retrocession") or 0)
                retro_supports.append(
                    {
                        "promoteur": data.get("promoteur"),
                        "support_nom": data.get("support_nom"),
                        "code_isin": data.get("code_isin"),
                        "retrocession": retro_val,
                        "retrocession_str": "{:,.2f}".format(retro_val).replace(",", " "),
                        "valo_total": float(data.get("valo_total") or 0),
                        "valo_total_str": "{:,.0f}".format(float(data.get("valo_total") or 0)).replace(",", " "),
                    }
                )
                retro_total_support += retro_val
            logger.debug(
                "Dashboard rétrocessions: %s lignes support récupérées pour le contrat %s",
                len(retro_supports),
                retro_selected_contract,
            )
        except Exception as exc:
            retro_error = "Impossible de calculer les rétrocessions pour la période demandée."
            logger.debug("Dashboard rétrocessions: erreur de calcul: %s", exc, exc_info=True)

    # --- ESG (global) UI context: allocation names + ESG field labels (cached) ---
    try:
        alloc_names_dash = [r[0] for r in db.query(Allocation.nom).filter(Allocation.nom.isnot(None)).distinct().order_by(Allocation.nom.asc()).all()]
    except Exception:
        alloc_names_dash = []
    esg_fields_dash, esg_fields_debug_dash = _get_esg_fields(db, debug=True)
    esg_field_labels_dash = {it["col"]: it["label"] for it in esg_fields_dash}

    group_rows_dash = _load_groupes_details(db)

    def _format_dash_date(value):
        if not value:
            return "—"
        try:
            return value.strftime("%d/%m/%Y")
        except Exception:
            return str(value)

    dashboard_groups_personnes = []
    dashboard_groups_affaires = []
    for row in group_rows_dash or []:
        gid = row.get("id")
        if not gid:
            continue
        type_raw = (row.get("type_groupe") or "").lower()
        resp_label = " ".join(
            p for p in [
                (row.get("resp_prenom") or "").strip(),
                (row.get("resp_nom") or "").strip()
            ] if p
        ) or (f"RH #{row.get('responsable_id')}" if row.get("responsable_id") else "—")
        entry = {
            "id": gid,
            "nom": row.get("nom") or f"Groupe #{gid}",
            "nb_membres": row.get("nb_membres") or 0,
            "responsable": resp_label,
            "date_creation": _format_dash_date(row.get("date_creation")),
            "date_fin": _format_dash_date(row.get("date_fin")),
        }
        if type_raw in ("client", "clients", "personne", "personnes"):
            entry["link"] = f"/dashboard/clients?group_id={gid}&group_type=client"
            entry["link_label"] = "Voir les personnes"
            dashboard_groups_personnes.append(entry)
        elif type_raw in ("affaire", "affaires", "contrat", "contrats"):
            entry["link"] = f"/dashboard/affaires?group_id={gid}&group_type=affaire"
            entry["link_label"] = "Voir les affaires"
            dashboard_groups_affaires.append(entry)

    ctx = {
        "total_valo": total_valo,
        "total_clients": total_clients,
        "total_affaires": total_affaires,
        "clients_buckets": clients_buckets,
        "srri_clients_count": srri_clients_count,
        "srri_affaires_count": srri_affaires_count,
        "srri_clients_amount": srri_clients_amount,
        "srri_affaires_amount": srri_affaires_amount,
        # Infos Documents pour la carte Risque Documents
        "docs_total": total_documents,
        "docs_der_total": der_total,
        "docs_der_obsolete": der_obsolete,
        "docs_der_total_pct": der_total_pct,
        "docs_der_obsolete_pct": der_obsolete_pct,
        "docs_cc_total": cc_total,
        "docs_cc_obsolete": cc_obsolete,
        "docs_cc_total_pct": cc_total_pct,
        "docs_cc_obsolete_pct": cc_obsolete_pct,
        "docs_esg_total": esg_total,
        "docs_esg_obsolete": esg_obsolete,
        "docs_esg_total_pct": esg_total_pct,
        "docs_esg_obsolete_pct": esg_obsolete_pct,
        "docs_lm_total": lm_total,
        "docs_lm_obsolete": lm_obsolete,
        "docs_lm_total_pct": lm_total_pct,
        "docs_lm_obsolete_pct": lm_obsolete_pct,
        "docs_la_total": la_total,
        "docs_la_obsolete": la_obsolete,
        "docs_la_total_pct": la_total_pct,
        "docs_la_obsolete_pct": la_obsolete_pct,
        "orias_doc_found": orias_doc_found,
        "orias_doc_valid": orias_doc_valid,
        "orias_doc_expiry_display": orias_doc_expiry_display,
        "assoc_doc_found": assoc_doc_found,
        "assoc_doc_valid": assoc_doc_valid,
        "assoc_doc_expiry_display": assoc_doc_expiry_display,
        "rcpro_doc_count": rcpro_doc_count,
        "rcpro_doc_valid": rcpro_doc_valid,
        "rcpro_doc_expiry_display": rcpro_doc_expiry_display,
        "rcpro_doc_expired_count": rcpro_doc_expired_count,
        "rcpro_doc_valid_count": rcpro_doc_valid_count,
        "conditions_doc_count": conditions_doc_count,
        "conditions_doc_valid": conditions_doc_valid,
        "conditions_doc_expiry_display": conditions_doc_expiry_display,
        "conditions_doc_expired_count": conditions_doc_expired_count,
        "conditions_doc_valid_count": conditions_doc_valid_count,
        "role_doc_count": role_doc_count,
        "role_doc_valid": role_doc_valid,
        "role_doc_expiry_display": role_doc_expiry_display,
        "role_doc_expired_count": role_doc_expired_count,
        "role_doc_valid_count": role_doc_valid_count,
        "gouvernance_doc_count": gouvernance_doc_count,
        "gouvernance_doc_valid": gouvernance_doc_valid,
        "gouvernance_doc_expiry_display": gouvernance_doc_expiry_display,
        "gouvernance_doc_expired_count": gouvernance_doc_expired_count,
        "gouvernance_doc_valid_count": gouvernance_doc_valid_count,
        "conclusion_doc_count": conclusion_doc_count,
        "conclusion_doc_valid": conclusion_doc_valid,
        "conclusion_doc_expiry_display": conclusion_doc_expiry_display,
        "conclusion_doc_expired_count": conclusion_doc_expired_count,
        "conclusion_doc_valid_count": conclusion_doc_valid_count,
        "blanchiment_doc_count": blanchiment_doc_count,
        "blanchiment_doc_valid": blanchiment_doc_valid,
        "blanchiment_doc_expiry_display": blanchiment_doc_expiry_display,
        "blanchiment_doc_expired_count": blanchiment_doc_expired_count,
        "blanchiment_doc_valid_count": blanchiment_doc_valid_count,
        "distribution_avg_obsolete_pct": distribution_avg_obsolete_pct,
        "distribution_avg_valid_pct": distribution_avg_valid_pct,
        "orias_detail": orias_detail,
        "reclam_stats": reclam_stats,
        "reclam_pivot": reclam_pivot,
        "docs_obs_by_niveau": obs_by_niveau,
        "docs_obs_by_risque": obs_by_risque,
        # Infos contrats vs risque
        "aff_risk_counts": compare_counts,
        "aff_risk_amounts": compare_amounts,
        "aff_risk_amounts_fmt": compare_amounts_fmt,
        "aff_risk_under_pct": aff_under_pct,
        # Infos clients vs risque
        "cli_risk_counts": cli_counts,
        "cli_risk_amounts": cli_amounts,
        "cli_risk_amounts_fmt": cli_amounts_fmt,
        "cli_risk_under_pct": cli_under_pct,
        # Analyse financière (supports)
        "finance_supports": finance_supports,
        "finance_total_valo": finance_total_valo,
        "finance_total_valo_str": finance_total_valo_str,
        "finance_date_input": finance_date_input,
        "finance_effective_date_display": finance_effective_date_display,
        "finance_effective_date_iso": finance_effective_date_iso,
        "finance_rh_options": finance_rh_options,
        "finance_rh_selected": finance_rh_id,
        "finance_valo_input": finance_ctx.get("finance_valo_input"),
        # Tâches / événements
        "tasks_total": tasks_total,
        "tasks_open": open_count,
        "tasks_statut": tasks_statut,
        "tasks_categorie": tasks_categorie,
        "tasks_days": tasks_days,
        "tasks_avg_by_statut": tasks_avg_by_statut,
        "tasks_avg_close_days": tasks_avg_close_days,
        "tasks_close_dist": tasks_close_dist,
        "tasks_range": range_days,
        "tasks_type_by_rh": tasks_type_by_rh,
        "tasks_type_total": tasks_type_total,
        # Veille
        "veille_items": veille_ctx.get("items"),
        "veille_recent": veille_ctx.get("recent"),
        "veille_grouped": veille_ctx.get("grouped"),
        "veille_markets": veille_ctx.get("markets"),
        "veille_markets_meta": veille_ctx.get("markets_meta"),
        "rem_contracts": rem_contracts,
        "rem_selected_contract": rem_selected_contract,
        "rem_limit": rem_limit,
        "rem_limit_options": rem_limit_options,
        "rem_start_input": rem_start_input,
        "rem_end_input": rem_end_input,
        "rem_start_effective": rem_start_effective,
        "rem_end_effective": rem_end_effective,
        "rem_rows": rem_rows,
        "rem_total_commission": rem_total_commission,
        "rem_total_valorisation": rem_total_valorisation,
        "rem_rows_count": rem_rows_count_total,
        "rem_total_contracts": rem_total_contracts,
        "rem_error": rem_error,
        "rem_page": rem_page,
        "rem_total_pages": rem_total_pages,
        "rem_has_prev": rem_has_prev,
        "rem_has_next": rem_has_next,
        "rem_prev_url": rem_prev_url,
        "rem_next_url": rem_next_url,
        "rem_page_start": rem_page_start,
        "rem_page_end": rem_page_end,
        "retro_contracts": retro_contracts,
        "retro_selected_contract": retro_selected_contract,
        "retro_sort": retro_sort,
        "retro_week_limit": retro_week_limit,
        "retro_week_limit_options": retro_week_limit_options,
        "retro_support_limit": retro_support_limit,
        "retro_support_limit_options": retro_support_limit_options,
        "retro_start_input": parsed_retro_start.isoformat(),
        "retro_end_input": parsed_retro_end.isoformat(),
        "retro_start_effective": retro_start_effective,
        "retro_end_effective": retro_end_effective,
        "retro_promoteur": retro_promoteur,
        "retro_weeks": retro_weeks,
        "retro_supports": retro_supports,
        "retro_total_week": retro_total_week,
        "retro_total_support": retro_total_support,
        "retro_error": retro_error,
        # ESG (global) UI context
        "alloc_names": alloc_names_dash,
        "esg_fields": esg_fields_dash,            # backwards compatibility
        "esg_field_labels": esg_field_labels_dash, # backwards compatibility
        "esg_fields_list": esg_fields_dash,        # explicit list of {col,label}
        "esg_fields_debug": esg_fields_debug_dash,
        "tb_markers_visible": tb_markers_visible,
        "tb_markers_param": "markers=1" if tb_markers_visible else "",
        "dashboard_groups_personnes": dashboard_groups_personnes,
        "dashboard_groups_affaires": dashboard_groups_affaires,
        "evt_types": evt_types if 'evt_types' in locals() else [],
        "evt_categories": evt_categories if 'evt_categories' in locals() else [],
        "evt_statuts": evt_statuts if 'evt_statuts' in locals() else [],
    }
    total_time = perf_counter() - t0
    logger.info("dashboard_home: total context built in %.3fs", total_time)
    if not request.query_params:
        HOME_CACHE["default"] = {"ts": perf_counter(), "data": ctx}
    ctx["request"] = request
    return templates.TemplateResponse("dashboard.html", ctx)


@router.get("/clients/kyc/{client_id}", response_class=HTMLResponse)
@router.post("/clients/kyc/{client_id}", response_class=HTMLResponse)
async def dashboard_client_kyc(
    client_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return templates.TemplateResponse(
            "dashboard_client_kyc.html",
            {"request": request, "error": "Client introuvable."},
        )

    etat_success: str | None = None
    etat_error: str | None = None
    adresse_success: str | None = None
    adresse_error: str | None = None
    matrimonial_success: str | None = None
    matrimonial_error: str | None = None
    professionnel_success: str | None = None
    professionnel_error: str | None = None
    passif_success: str | None = None
    passif_error: str | None = None
    revenu_success: str | None = None
    revenu_error: str | None = None
    charge_success: str | None = None
    charge_error: str | None = None
    contrat_success: str | None = None
    contrat_error: str | None = None
    actif_success: str | None = None
    actif_error: str | None = None
    objectifs_success: str | None = None
    objectifs_error: str | None = None
    active_objectif_id: int | None = None
    active_section: str = "etat_civil"
    ui_focus_section: str | None = None
    ui_focus_panel: str | None = None
    esg_success: str | None = None
    esg_error: str | None = None
    risque_commentaire: str | None = None
    risque_snapshot: dict | None = None

    def _fmt_amount(v):
        if v is None:
            return "-"
        try:
            return "{:,.2f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)

    def _safe_text(value):
        if value is None:
            return ""
        return str(value)

    def _snapshot_synthese():
        """Recompute totals and upsert today's snapshot in KYC_Client_Synthese.
        Update if a row for today's date exists; otherwise insert.
        Be resilient to column naming variations (ex: total_revenu vs total_revenus).
        """
        try:
            # Detect column names once per call
            cols = set()
            try:
                col_rows = db.execute(text("PRAGMA table_info('KYC_Client_Synthese')")).fetchall()
                cols = {str(r[1]).lower() for r in col_rows}
            except Exception:
                cols = set()
            col_rev = 'total_revenus' if 'total_revenus' in cols else ('total_revenu' if 'total_revenu' in cols else 'total_revenus')
            col_chg = 'total_charges' if 'total_charges' in cols else ('total_charge' if 'total_charge' in cols else 'total_charges')
            col_act = 'total_actif'
            col_pas = 'total_passif'

            sums = {}
            for key, query in (
                ("ta", "SELECT COALESCE(SUM(valeur),0) FROM KYC_Client_Actif WHERE client_id = :cid"),
                ("tp", "SELECT COALESCE(SUM(montant_rest_du),0) FROM KYC_Client_Passif WHERE client_id = :cid"),
                ("tr", "SELECT COALESCE(SUM(montant_annuel),0) FROM KYC_Client_Revenus WHERE client_id = :cid"),
                ("tc", "SELECT COALESCE(SUM(montant_annuel),0) FROM KYC_Client_Charges WHERE client_id = :cid"),
            ):
                row = db.execute(text(query), {"cid": client_id}).fetchone()
                sums[key] = float(row[0] or 0)

            today_str = _date.today().isoformat()
            row = db.execute(
                text(
                    """
                    SELECT id, date_saisie
                    FROM KYC_Client_Synthese
                    WHERE client_id = :cid
                      AND substr(COALESCE(date_saisie,''), 1, 10) = :today
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {"cid": client_id, "today": today_str},
            ).fetchone()
            params_s = {"cid": client_id} | sums
            if row and str(row._mapping.get("date_saisie") or "")[:10] == today_str:
                params_s["id"] = row._mapping.get("id")
                db.execute(
                    text(
                        f"""
                        UPDATE KYC_Client_Synthese
                        SET {col_rev} = :tr,
                            {col_chg} = :tc,
                            {col_act} = :ta,
                            {col_pas} = :tp
                        WHERE id = :id
                        """
                    ),
                    params_s,
                )
            else:
                db.execute(
                    text(
                        f"""
                        INSERT INTO KYC_Client_Synthese (
                            client_id, {col_rev}, {col_chg}, {col_act}, {col_pas}
                        ) VALUES (
                            :cid, :tr, :tc, :ta, :tp
                        )
                        """
                    ),
                    params_s,
                )
            db.commit()
        except Exception as _exc_synth:
            try:
                db.rollback()
            except SystemExit:
                pass
            except Exception:
                pass
            logger.debug("Dashboard KYC client: erreur snapshot synthese: %s", _exc_synth, exc_info=True)

    def _fmt_date(value):
        if value is None:
            return None
        try:
            if hasattr(value, "strftime"):
                return value.strftime("%Y-%m-%d")
        except Exception:
            pass
        return str(value)

    if request.method == "POST":
        form = await request.form()
        action = (form.get("form_action") or "").strip().lower()

        if action == "etat_civil":
            payload = {k: (form.get(k) or None) for k in [
                "civilite",
                "date_naissance",
                "lieu_naissance",
                "nationalite",
                "commentaire",
                "email",
                "telephone",
                "nif",
            ]}
            rec_id = form.get("id") or None
            try:
                existing = db.execute(
                    text("SELECT id FROM etat_civil_client WHERE id_client = :cid ORDER BY id LIMIT 1"),
                    {"cid": client_id},
                ).fetchone()
                record_id = rec_id or (existing[0] if existing else None)

                if record_id:
                    params = payload | {"id": record_id}
                    db.execute(
                        text(
                            """
                            UPDATE etat_civil_client
                            SET civilite = :civilite,
                                date_naissance = :date_naissance,
                                lieu_naissance = :lieu_naissance,
                                nationalite = :nationalite,
                                commentaire = :commentaire
                            WHERE id = :id
                            """
                        ),
                        params,
                    )
                else:
                    db.execute(
                        text(
                            """
                            INSERT INTO etat_civil_client (
                                id_client,
                                civilite,
                                date_naissance,
                                lieu_naissance,
                                nationalite,
                                commentaire
                            ) VALUES (
                                :cid,
                                :civilite,
                                :date_naissance,
                                :lieu_naissance,
                                :nationalite,
                                :commentaire
                            )
                            """
                        ),
                        payload | {"cid": client_id},
                    )
                # Mettre à jour email / téléphone / NIF dans mariadb_clients
                try:
                    db.execute(
                        text(
                            """
                            UPDATE mariadb_clients
                            SET email = :email,
                                telephone = :telephone,
                                nif = :nif
                            WHERE id = :cid
                            """
                        ),
                        {
                            "email": payload.get("email"),
                            "telephone": payload.get("telephone"),
                            "nif": payload.get("nif"),
                            "cid": client_id,
                        },
                    )
                except Exception:
                    db.rollback()
                    raise
                db.commit()
                etat_success = "Etat civil sauvegardé avec succès."
            except Exception as exc:
                db.rollback()
                etat_error = "Impossible d'enregistrer les informations d'état civil."
                logger.debug("Dashboard KYC client: erreur état civil: %s", exc, exc_info=True)
            active_section = "etat_civil"

        elif action == "adresse_save":
            adresse_id = form.get("adresse_id") or None
            type_id = form.get("type_adresse_id") or None
            rue = (form.get("rue") or "").strip()
            complement = (form.get("complement") or "").strip() or None
            code_postal = (form.get("code_postal") or "").strip()
            ville = (form.get("ville") or "").strip()
            pays = (form.get("pays") or "").strip()
            date_saisie = (form.get("date_saisie") or None) or None
            date_expiration = (form.get("date_expiration") or None) or None

            if not type_id or not rue or not code_postal or not ville or not pays:
                adresse_error = "Veuillez renseigner le type d'adresse et les champs obligatoires."
            else:
                try:
                    params = {
                        "cid": client_id,
                        "type_id": int(type_id),
                        "rue": rue,
                        "complement": complement,
                        "code_postal": code_postal,
                        "ville": ville,
                        "pays": pays,
                        "date_saisie": date_saisie,
                        "date_expiration": date_expiration,
                    }
                    if adresse_id:
                        params["id"] = int(adresse_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Adresse
                                SET type_adresse_id = :type_id,
                                    rue = :rue,
                                    complement = :complement,
                                    code_postal = :code_postal,
                                    ville = :ville,
                                    pays = :pays,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Adresse (
                                    client_id,
                                    type_adresse_id,
                                    rue,
                                    complement,
                                    code_postal,
                                    ville,
                                    pays,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :type_id,
                                    :rue,
                                    :complement,
                                    :code_postal,
                                    :ville,
                                    :pays,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                    db.commit()
                    _snapshot_synthese()
                    adresse_success = "Adresse enregistrée."
                except Exception as exc:
                    db.rollback()
                    adresse_error = "Impossible d'enregistrer l'adresse."
                    logger.debug("Dashboard KYC client: erreur adresse save: %s", exc, exc_info=True)
            active_section = "adresse"

        elif action == "adresse_delete":
            adresse_id = form.get("adresse_id") or None
            if not adresse_id:
                adresse_error = "Adresse introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Adresse WHERE id = :id AND client_id = :cid"),
                        {"id": int(adresse_id), "cid": client_id},
                    )
                    db.commit()
                    _snapshot_synthese()
                    adresse_success = "Adresse supprimée."
                except Exception as exc:
                    db.rollback()
                    adresse_error = "Impossible de supprimer l'adresse."
                    logger.debug("Dashboard KYC client: erreur adresse delete: %s", exc, exc_info=True)
            active_section = "adresse"

        elif action == "matrimonial_save":
            matrimonial_id = form.get("matrimonial_id") or None
            situation_id = form.get("situation_id") or None
            convention_id = form.get("convention_id") or None
            nb_enfants_raw = form.get("nb_enfants") or "0"
            date_saisie = form.get("date_saisie") or None
            date_expiration = form.get("date_expiration") or None

            try:
                nb_enfants = int(nb_enfants_raw or 0)
                if nb_enfants < 0:
                    nb_enfants = 0
            except Exception:
                nb_enfants = 0

            if not situation_id:
                matrimonial_error = "Veuillez sélectionner une situation matrimoniale."
            else:
                try:
                    params = {
                        "cid": client_id,
                        "situation_id": int(situation_id),
                        "nb_enfants": nb_enfants,
                        "convention_id": int(convention_id) if convention_id else None,
                        "date_saisie": date_saisie,
                        "date_expiration": date_expiration,
                    }
                    if matrimonial_id:
                        params["id"] = int(matrimonial_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Situation_Matrimoniale
                                SET situation_id = :situation_id,
                                    nb_enfants = :nb_enfants,
                                    convention_id = :convention_id,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Situation_Matrimoniale (
                                    client_id,
                                    situation_id,
                                    nb_enfants,
                                    convention_id,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :situation_id,
                                    :nb_enfants,
                                    :convention_id,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                    db.commit()
                    _snapshot_synthese()
                    matrimonial_success = "Situation matrimoniale enregistrée."
                except Exception as exc:
                    db.rollback()
                    matrimonial_error = "Impossible d'enregistrer la situation matrimoniale."
                    logger.debug("Dashboard KYC client: erreur situation matrimoniale save: %s", exc, exc_info=True)
            active_section = "matrimonial"

        elif action == "matrimonial_delete":
            matrimonial_id = form.get("matrimonial_id") or None
            if not matrimonial_id:
                matrimonial_error = "Situation matrimoniale introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Situation_Matrimoniale WHERE id = :id AND client_id = :cid"),
                        {"id": int(matrimonial_id), "cid": client_id},
                    )
                    db.commit()
                    _snapshot_synthese()
                    matrimonial_success = "Situation matrimoniale supprimée."
                except Exception as exc:
                    db.rollback()
                    matrimonial_error = "Impossible de supprimer la situation matrimoniale."
                    logger.debug("Dashboard KYC client: erreur situation matrimoniale delete: %s", exc, exc_info=True)
            active_section = "matrimonial"

        elif action == "professionnel_save":
            professionnel_id = form.get("professionnel_id") or None
            profession = (form.get("profession") or "").strip()
            secteur_id = form.get("secteur_id") or None
            statut_id = form.get("statut_id") or None
            employeur = (form.get("employeur") or "").strip() or None
            anciennete_raw = form.get("anciennete_annees") or ""
            date_saisie = form.get("date_saisie") or None
            date_expiration = form.get("date_expiration") or None

            try:
                anciennete = int(anciennete_raw or 0)
                if anciennete < 0:
                    anciennete = 0
            except Exception:
                anciennete = 0

            if not profession or not secteur_id or not statut_id:
                professionnel_error = "Veuillez renseigner la profession, le secteur et le statut professionnel."
            else:
                try:
                    params = {
                        "cid": client_id,
                        "profession": profession,
                        "secteur_id": int(secteur_id),
                        "employeur": employeur,
                        "anciennete": anciennete,
                        "statut_id": int(statut_id),
                        "date_saisie": date_saisie,
                        "date_expiration": date_expiration,
                    }
                    if professionnel_id:
                        params["id"] = int(professionnel_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Situation_Professionnelle
                                SET profession = :profession,
                                    secteur_id = :secteur_id,
                                    employeur = :employeur,
                                    anciennete_annees = :anciennete,
                                    statut_id = :statut_id,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Situation_Professionnelle (
                                    client_id,
                                    profession,
                                    secteur_id,
                                    employeur,
                                    anciennete_annees,
                                    statut_id,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :profession,
                                    :secteur_id,
                                    :employeur,
                                    :anciennete,
                                    :statut_id,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                    db.commit()
                    _snapshot_synthese()
                    professionnel_success = "Situation professionnelle enregistrée."
                except Exception as exc:
                    db.rollback()
                    professionnel_error = "Impossible d'enregistrer la situation professionnelle."
                    logger.debug("Dashboard KYC client: erreur situation professionnelle save: %s", exc, exc_info=True)
            active_section = "professionnel"

        elif action == "professionnel_delete":
            professionnel_id = form.get("professionnel_id") or None
            if not professionnel_id:
                professionnel_error = "Situation professionnelle introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Situation_Professionnelle WHERE id = :id AND client_id = :cid"),
                        {"id": int(professionnel_id), "cid": client_id},
                    )
                    db.commit()
                    _snapshot_synthese()
                    professionnel_success = "Situation professionnelle supprimée."
                except Exception as exc:
                    db.rollback()
                    professionnel_error = "Impossible de supprimer la situation professionnelle."
                    logger.debug("Dashboard KYC client: erreur situation professionnelle delete: %s", exc, exc_info=True)
            active_section = "professionnel"

        elif action == "actif_save":
            actif_id = form.get("id") or None
            type_id = form.get("type_actif_id") or None
            description = (form.get("description") or "").strip() or None
            valeur_raw = form.get("valeur")
            date_expiration = (form.get("date_expiration") or None) or None

            if not type_id:
                actif_error = "Veuillez sélectionner un type d'actif."
            valeur_decimal: Decimal | None = None
            if not actif_error:
                if valeur_raw in (None, ""):
                    actif_error = "Veuillez renseigner la valeur de l'actif."
                else:
                    try:
                        valeur_decimal = Decimal(str(valeur_raw).replace(",", "."))
                        if valeur_decimal < 0:
                            actif_error = "La valeur de l'actif doit être positive."
                    except (InvalidOperation, ValueError):
                        actif_error = "Valeur d'actif invalide."

            type_id_int: int | None = None
            if not actif_error and type_id:
                try:
                    type_id_int = int(type_id)
                except (TypeError, ValueError):
                    actif_error = "Type d'actif invalide."

            valeur_float: float | None = None
            if not actif_error:
                today_str = datetime.utcnow().date().isoformat()
                if valeur_decimal is not None:
                    try:
                        valeur_float = float(valeur_decimal)
                    except (TypeError, ValueError):
                        actif_error = "Valeur d'actif invalide."

            if not actif_error:
                today_str = datetime.utcnow().date().isoformat()
                params = {
                    "cid": client_id,
                    "type_id": type_id_int,
                    "description": description,
                    "valeur": valeur_float,
                    "date_saisie": today_str,
                    "date_expiration": date_expiration,
                }
                try:
                    if actif_id:
                        params["id"] = int(actif_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Actif
                                SET type_actif_id = :type_id,
                                    description = :description,
                                    valeur = :valeur,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                        actif_success = "Actif mis à jour."
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Actif (
                                    client_id,
                                    type_actif_id,
                                    description,
                                    valeur,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :type_id,
                                    :description,
                                    :valeur,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                        actif_success = "Actif enregistré."
                    db.commit()
                    _snapshot_synthese()
                except Exception as exc:
                    db.rollback()
                    actif_error = "Impossible d'enregistrer l'actif."
                    logger.debug("Dashboard KYC client: erreur actif save: %s", exc, exc_info=True)
            active_section = "patrimoine"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "actifsPanel"

        elif action == "actif_delete":
            actif_id = form.get("actif_id") or form.get("id") or None
            if not actif_id:
                actif_error = "Actif introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Actif WHERE id = :id AND client_id = :cid"),
                        {"id": int(actif_id), "cid": client_id},
                    )
                    db.commit()
                    _snapshot_synthese()
                    actif_success = "Actif supprimé."
                except Exception as exc:
                    db.rollback()
                    actif_error = "Impossible de supprimer l'actif."
                    logger.debug("Dashboard KYC client: erreur actif delete: %s", exc, exc_info=True)
            active_section = "patrimoine"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "actifsPanel"

        elif action == "passif_save":
            passif_id = form.get("id") or None
            type_id = form.get("type_passif_id") or None
            description = (form.get("description") or "").strip() or None
            montant_raw = form.get("montant")
            date_expiration = (form.get("date_expiration") or None) or None

            if not type_id:
                passif_error = "Veuillez sélectionner un type de passif."

            montant_decimal: Decimal | None = None
            if not passif_error:
                if montant_raw in (None, ""):
                    passif_error = "Veuillez renseigner le montant restant dû."
                else:
                    try:
                        montant_decimal = Decimal(str(montant_raw).replace(",", "."))
                        if montant_decimal < 0:
                            passif_error = "Le montant de passif doit être positif."
                    except (InvalidOperation, ValueError):
                        passif_error = "Montant de passif invalide."

            type_id_int: int | None = None
            if not passif_error and type_id:
                try:
                    type_id_int = int(type_id)
                except (TypeError, ValueError):
                    passif_error = "Type de passif invalide."

            montant_float: float | None = None
            if not passif_error and montant_decimal is not None:
                try:
                    montant_float = float(montant_decimal)
                except (TypeError, ValueError):
                    passif_error = "Montant de passif invalide."

            if not passif_error:
                today_str = datetime.utcnow().date().isoformat()
                params = {
                    "cid": client_id,
                    "type_id": type_id_int,
                    "description": description,
                    "montant": montant_float,
                    "date_saisie": today_str,
                    "date_expiration": date_expiration,
                }
                try:
                    if passif_id:
                        params["id"] = int(passif_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Passif
                                SET type_passif_id = :type_id,
                                    description = :description,
                                    montant_rest_du = :montant,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                        passif_success = "Passif mis à jour."
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Passif (
                                    client_id,
                                    type_passif_id,
                                    description,
                                    montant_rest_du,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :type_id,
                                    :description,
                                    :montant,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                        passif_success = "Passif enregistré."
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    passif_error = "Impossible d'enregistrer le passif."
                    logger.debug("Dashboard KYC client: erreur passif save: %s", exc, exc_info=True)
            active_section = "passif"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "passifPanel"

        elif action == "passif_delete":
            passif_id = form.get("passif_id") or form.get("id") or None
            if not passif_id:
                passif_error = "Passif introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Passif WHERE id = :id AND client_id = :cid"),
                        {"id": int(passif_id), "cid": client_id},
                    )
                    db.commit()
                    passif_success = "Passif supprimé."
                except Exception as exc:
                    db.rollback()
                    passif_error = "Impossible de supprimer le passif."
                    logger.debug("Dashboard KYC client: erreur passif delete: %s", exc, exc_info=True)
            active_section = "passif"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "passifPanel"

        elif action == "revenu_save":
            revenu_id = form.get("id") or None
            type_id = form.get("type_revenu_id") or None
            montant_raw = form.get("montant")
            date_expiration = (form.get("date_expiration") or None) or None

            if not type_id:
                revenu_error = "Veuillez sélectionner un type de revenu."

            montant_decimal: Decimal | None = None
            if not revenu_error:
                if montant_raw in (None, ""):
                    revenu_error = "Veuillez renseigner le montant annuel."
                else:
                    try:
                        montant_decimal = Decimal(str(montant_raw).replace(",", "."))
                        if montant_decimal < 0:
                            revenu_error = "Le montant doit être positif."
                    except (InvalidOperation, ValueError):
                        revenu_error = "Montant invalide."

            type_id_int: int | None = None
            if not revenu_error and type_id:
                try:
                    type_id_int = int(type_id)
                except (TypeError, ValueError):
                    revenu_error = "Type de revenu invalide."

            montant_float: float | None = None
            if not revenu_error and montant_decimal is not None:
                try:
                    montant_float = float(montant_decimal)
                except (TypeError, ValueError):
                    revenu_error = "Montant invalide."

            if not revenu_error:
                today_str = datetime.utcnow().date().isoformat()
                params = {
                    "cid": client_id,
                    "type_id": type_id_int,
                    "montant": montant_float,
                    "date_saisie": today_str,
                    "date_expiration": date_expiration,
                }
                try:
                    if revenu_id:
                        params["id"] = int(revenu_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Revenus
                                SET type_revenu_id = :type_id,
                                    montant_annuel = :montant,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                        revenu_success = "Revenu mis à jour."
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Revenus (
                                    client_id,
                                    type_revenu_id,
                                    montant_annuel,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :type_id,
                                    :montant,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                        revenu_success = "Revenu enregistré."
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    revenu_error = "Impossible d'enregistrer le revenu."
                    logger.debug("Dashboard KYC client: erreur revenu save: %s", exc, exc_info=True)
            active_section = "recettes"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "recettesPanel"

        elif action == "revenu_delete":
            revenu_id = form.get("revenu_id") or form.get("id") or None
            if not revenu_id:
                revenu_error = "Revenu introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Revenus WHERE id = :id AND client_id = :cid"),
                        {"id": int(revenu_id), "cid": client_id},
                    )
                    db.commit()
                    revenu_success = "Revenu supprimé."
                except Exception as exc:
                    db.rollback()
                    revenu_error = "Impossible de supprimer le revenu."
                    logger.debug("Dashboard KYC client: erreur revenu delete: %s", exc, exc_info=True)
            active_section = "recettes"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "recettesPanel"

        elif action == "charge_save":
            charge_id = form.get("id") or None
            type_id = form.get("type_charge_id") or None
            montant_raw = form.get("montant")
            date_expiration = (form.get("date_expiration") or None) or None

            if not type_id:
                charge_error = "Veuillez sélectionner un type de charge."

            montant_decimal: Decimal | None = None
            if not charge_error:
                if montant_raw in (None, ""):
                    charge_error = "Veuillez renseigner le montant annuel."
                else:
                    try:
                        montant_decimal = Decimal(str(montant_raw).replace(",", "."))
                        if montant_decimal < 0:
                            charge_error = "Le montant doit être positif."
                    except (InvalidOperation, ValueError):
                        charge_error = "Montant invalide."

            type_id_int: int | None = None
            if not charge_error and type_id:
                try:
                    type_id_int = int(type_id)
                except (TypeError, ValueError):
                    charge_error = "Type de charge invalide."

            montant_float: float | None = None
            if not charge_error and montant_decimal is not None:
                try:
                    montant_float = float(montant_decimal)
                except (TypeError, ValueError):
                    charge_error = "Montant invalide."

            if not charge_error:
                today_str = datetime.utcnow().date().isoformat()
                params = {
                    "cid": client_id,
                    "type_id": type_id_int,
                    "montant": montant_float,
                    "date_saisie": today_str,
                    "date_expiration": date_expiration,
                }
                try:
                    if charge_id:
                        params["id"] = int(charge_id)
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Charges
                                SET type_charge_id = :type_id,
                                    montant_annuel = :montant,
                                    date_saisie = :date_saisie,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                        charge_success = "Charge mise à jour."
                    else:
                        db.execute(
                            text(
                                """
                                INSERT INTO KYC_Client_Charges (
                                    client_id,
                                    type_charge_id,
                                    montant_annuel,
                                    date_saisie,
                                    date_expiration
                                ) VALUES (
                                    :cid,
                                    :type_id,
                                    :montant,
                                    :date_saisie,
                                    :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                        charge_success = "Charge enregistrée."
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    charge_error = "Impossible d'enregistrer la charge."
                    logger.debug("Dashboard KYC client: erreur charge save: %s", exc, exc_info=True)
            active_section = "charges"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "chargesPanel"

        elif action == "charge_delete":
            charge_id = form.get("charge_id") or form.get("id") or None
            if not charge_id:
                charge_error = "Charge introuvable."
            else:
                try:
                    db.execute(
                        text("DELETE FROM KYC_Client_Charges WHERE id = :id AND client_id = :cid"),
                        {"id": int(charge_id), "cid": client_id},
                    )
                    db.commit()
                    charge_success = "Charge supprimée."
                except Exception as exc:
                    db.rollback()
                    charge_error = "Impossible de supprimer la charge."
                    logger.debug("Dashboard KYC client: erreur charge delete: %s", exc, exc_info=True)
            active_section = "charges"
            ui_focus_section = "patrimoine"
            ui_focus_panel = "chargesPanel"

        elif action == "contrat_choisir":
            # Sélection unique d'un contrat pour le client
            sel_id_raw = form.get("id_contrat")
            try:
                sel_id = int(sel_id_raw) if sel_id_raw and str(sel_id_raw).isdigit() else None
            except Exception:
                sel_id = None
            if sel_id is None:
                contrat_error = "Veuillez sélectionner un contrat."
            else:
                try:
                    # Table (si absente)
                    db.execute(text(
                        """
                        CREATE TABLE IF NOT EXISTS KYC_contrat_choisi (
                          id_client INTEGER NOT NULL,
                          id_contrat INTEGER NOT NULL,
                          PRIMARY KEY (id_client)
                        )
                        """
                    ))
                    # Remplace l'existant
                    db.execute(text("DELETE FROM KYC_contrat_choisi WHERE id_client = :cid"), {"cid": client_id})
                    db.execute(text("INSERT INTO KYC_contrat_choisi (id_client, id_contrat) VALUES (:cid, :kid)"), {"cid": client_id, "kid": sel_id})
                    db.commit()
                    contrat_success = "Contrat enregistré."
                except Exception as exc:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    contrat_error = "Impossible d'enregistrer le contrat."
            # Revenir sur le panneau Contrats
            ui_focus_section = "contrats"
            ui_focus_panel = "contratsPanel"

        elif action == "objectifs_save":
            active_section = "objectifs"
            ui_focus_section = "objectifs"
            ui_focus_panel = "objectifsPanel"
            objectif_id_raw = form.get("objectif_id")
            link_id_raw = form.get("link_id")
            horizon = (form.get("horizon_investissement") or "").strip() or None
            niveau_id_raw = form.get("niveau_id")
            montant_raw = (form.get("montant") or "").strip()
            commentaire = (form.get("commentaire") or "").strip() or None
            date_expiration = (form.get("date_expiration") or "").strip() or None

            objectif_id: int | None = None
            try:
                if objectif_id_raw:
                    objectif_id = int(objectif_id_raw)
                    active_objectif_id = objectif_id
                else:
                    objectifs_error = "Veuillez sélectionner un objectif."
            except (TypeError, ValueError):
                objectifs_error = "Identifiant d'objectif invalide."

            niveau_id: int | None = None
            if not objectifs_error:
                if not niveau_id_raw:
                    objectifs_error = "Veuillez renseigner le niveau de priorité."
                else:
                    try:
                        niveau_id = int(niveau_id_raw)
                    except (TypeError, ValueError):
                        objectifs_error = "Niveau de priorité invalide."

            if not objectifs_error and objectif_id is not None and niveau_id is not None:
                # Parse montant (accept comma as decimal separator)
                try:
                    montant_val = float(montant_raw.replace(',', '.')) if montant_raw != '' else 0.0
                except Exception:
                    montant_val = 0.0
                params = {
                    "cid": client_id,
                    "objectif_id": objectif_id,
                    "horizon": horizon,
                    "niveau_id": niveau_id,
                    "montant": montant_val,
                    "commentaire": commentaire,
                    "date_expiration": date_expiration or None,
                }
                try:
                    link_id: int | None = None
                    if link_id_raw:
                        try:
                            link_id = int(link_id_raw)
                        except (TypeError, ValueError):
                            link_id = None

                    if link_id:
                        params["id"] = link_id
                        db.execute(
                            text(
                                """
                                UPDATE KYC_Client_Objectifs
                                SET horizon_investissement = :horizon,
                                    niveau_id = :niveau_id,
                                    montant = :montant,
                                    commentaire = :commentaire,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params,
                        )
                        db.commit()
                        objectifs_success = "Objectif mis à jour."
                    else:
                        duplicate = db.execute(
                            text(
                                "SELECT id FROM KYC_Client_Objectifs WHERE client_id = :cid AND objectif_id = :objectif_id"
                            ),
                            {"cid": client_id, "objectif_id": objectif_id},
                        ).fetchone()
                        if duplicate:
                            objectifs_error = "Cet objectif est déjà enregistré pour ce client."
                        else:
                            db.execute(
                                text(
                                    """
                                    INSERT INTO KYC_Client_Objectifs (
                                        client_id,
                                        objectif_id,
                                        horizon_investissement,
                                        niveau_id,
                                        montant,
                                        commentaire,
                                        date_expiration
                                    ) VALUES (
                                        :cid,
                                        :objectif_id,
                                        :horizon,
                                        :niveau_id,
                                        :montant,
                                        :commentaire,
                                        :date_expiration
                                    )
                                    """
                                ),
                                params,
                            )
                            db.commit()
                            objectifs_success = "Objectif enregistré."
                except Exception as exc:
                    db.rollback()
                    objectifs_error = "Impossible d'enregistrer l'objectif."
                    logger.debug("Dashboard KYC client: erreur objectif save: %s", exc, exc_info=True)

        elif action == "objectifs_save_all":
            active_section = "objectifs"
            ui_focus_section = "objectifs"
            ui_focus_panel = "objectifsPanel"
            from sqlalchemy import text as _text
            raw = form.get("objectifs_all")
            saved = 0
            skipped = 0
            try:
                import json as _json
                items = _json.loads(raw or "[]")
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = []
            for it in items:
                try:
                    objectif_id = int(it.get("objectif_id")) if it.get("objectif_id") is not None else None
                    if not objectif_id:
                        skipped += 1
                        continue
                    niveau_id = it.get("niveau_id")
                    try:
                        niveau_id = int(niveau_id) if niveau_id is not None else None
                    except Exception:
                        niveau_id = None
                    horizon = (it.get("horizon_investissement") or None)
                    commentaire = (it.get("commentaire") or None)
                    date_expiration = (it.get("date_expiration") or None)
                    montant = it.get("montant")
                    try:
                        montant = float(montant) if montant is not None else 0.0
                    except Exception:
                        montant = 0.0
                    if niveau_id is None:
                        skipped += 1
                        continue
                    # insert/update unique by (client_id, objectif_id)
                    row = db.execute(
                        _text("SELECT id FROM KYC_Client_Objectifs WHERE client_id = :cid AND objectif_id = :oid"),
                        {"cid": client_id, "oid": objectif_id},
                    ).fetchone()
                    params = {
                        "cid": client_id,
                        "objectif_id": objectif_id,
                        "horizon": horizon,
                        "niveau_id": niveau_id,
                        "montant": montant,
                        "commentaire": commentaire,
                        "date_expiration": date_expiration,
                    }
                    if row:
                        db.execute(
                            _text(
                                """
                                UPDATE KYC_Client_Objectifs
                                SET horizon_investissement = :horizon,
                                    niveau_id = :niveau_id,
                                    montant = :montant,
                                    commentaire = :commentaire,
                                    date_expiration = :date_expiration
                                WHERE id = :id AND client_id = :cid
                                """
                            ),
                            params | {"id": int(row[0])},
                        )
                    else:
                        db.execute(
                            _text(
                                """
                                INSERT INTO KYC_Client_Objectifs (
                                    client_id, objectif_id, horizon_investissement, niveau_id, montant, commentaire, date_expiration
                                ) VALUES (
                                    :cid, :objectif_id, :horizon, :niveau_id, :montant, :commentaire, :date_expiration
                                )
                                """
                            ),
                            params,
                        )
                    saved += 1
                except Exception:
                    skipped += 1
            try:
                db.commit()
            except Exception:
                db.rollback()
            objectifs_success = f"{saved} objectif(s) enregistré(s)." if saved else None
            if skipped and not objectifs_error:
                objectifs_error = f"{skipped} objectif(s) ignoré(s) (incomplets)."

        elif action == "objectifs_delete":
            active_section = "objectifs"
            ui_focus_section = "objectifs"
            ui_focus_panel = "objectifsPanel"
            link_id_raw = form.get("link_id")
            objectif_id_raw = form.get("objectif_id")
            if objectif_id_raw:
                try:
                    active_objectif_id = int(objectif_id_raw)
                except (TypeError, ValueError):
                    active_objectif_id = None
            if not link_id_raw:
                objectifs_error = "Objectif introuvable."
            else:
                try:
                    link_id = int(link_id_raw)
                    db.execute(
                        text("DELETE FROM KYC_Client_Objectifs WHERE id = :id AND client_id = :cid"),
                        {"id": link_id, "cid": client_id},
                    )
                    db.commit()
                    objectifs_success = "Objectif supprimé."
                    active_objectif_id = None
                except Exception as exc:
                    db.rollback()
                    objectifs_error = "Impossible de supprimer l'objectif."
                    logger.debug("Dashboard KYC client: erreur objectif delete: %s", exc, exc_info=True)

        elif action == "esg_save":
            # Sauvegarde du questionnaire ESG
            from sqlalchemy import text as _text
            active_section = "esg"
            ui_focus_section = "esg"
            ui_focus_panel = "esgPanel"
            allowed = {"oui", "non", "indifférent"}

            def _pick(name: str):
                v = (form.get(name) or "").strip().lower()
                # normaliser au jeu autorisé avec accent
                if v == "indifferent":
                    v = "indifférent"
                if v in allowed:
                    return v
                return None

            env_importance = _pick("env_importance")
            env_ges_reduc = _pick("env_ges_reduc")
            soc_droits_humains = _pick("soc_droits_humains")
            soc_parite = _pick("soc_parite")
            gov_transparence = _pick("gov_transparence")
            gov_controle_ethique = _pick("gov_controle_ethique")

            def _extract_multi_ids(field: str) -> list[int]:
                """Extract repeated form values regardless of the backend form type."""
                raw_values: list[str] = []
                if hasattr(form, "getlist"):
                    try:
                        raw_values.extend(form.getlist(field))
                    except Exception:
                        pass
                if not raw_values and hasattr(form, "multi_items"):
                    try:
                        for key, value in form.multi_items():
                            if key == field:
                                raw_values.append(value)
                    except Exception:
                        pass
                if not raw_values and hasattr(form, "lists"):
                    try:
                        values_map = form.lists()
                        if field in values_map:
                            raw_values.extend(values_map[field])
                    except Exception:
                        pass
                # fallback: scan plain items in case framework flattens duplicates
                if not raw_values and hasattr(form, "items"):
                    try:
                        raw_values.extend(value for key, value in form.items() if key == field)
                    except Exception:
                        pass
                cleaned: list[int] = []
                for value in raw_values:
                    try:
                        value_str = str(value).strip()
                        if not value_str:
                            continue
                        cleaned.append(int(value_str))
                    except Exception:
                        continue
                # de-duplicate while preserving order
                seen: set[int] = set()
                unique: list[int] = []
                for vid in cleaned:
                    if vid in seen:
                        continue
                    seen.add(vid)
                    unique.append(vid)
                return unique

            excl_ids = _extract_multi_ids("exclusions")
            ind_ids = _extract_multi_ids("indicators")

            if not all([env_importance, env_ges_reduc, soc_droits_humains, soc_parite, gov_transparence, gov_controle_ethique]):
                esg_error = "Veuillez renseigner toutes les réponses ESG."
            else:
                from datetime import datetime as _dt, timedelta as _td
                now = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                # Obsolescence à 2 ans
                obso = (_dt.utcnow() + _td(days=730)).strftime("%Y-%m-%d %H:%M:%S")
                base_params = {
                    "client_ref": str(client_id),
                    "saisie_at": now,
                    "obsolescence_at": obso,
                    "env_importance": env_importance,
                    "env_ges_reduc": env_ges_reduc,
                    "soc_droits_humains": soc_droits_humains,
                    "soc_parite": soc_parite,
                    "gov_transparence": gov_transparence,
                    "gov_controle_ethique": gov_controle_ethique,
                }
                try:
                    row = db.execute(
                        _text("SELECT id FROM esg_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"),
                        {"r": str(client_id)},
                    ).fetchone()
                    qid = row[0] if row else None
                    if qid:
                        params = base_params | {"id": qid}
                        db.execute(
                            _text(
                                """
                                UPDATE esg_questionnaire
                                SET saisie_at = :saisie_at,
                                    obsolescence_at = :obsolescence_at,
                                    env_importance = :env_importance,
                                    env_ges_reduc = :env_ges_reduc,
                                    soc_droits_humains = :soc_droits_humains,
                                    soc_parite = :soc_parite,
                                    gov_transparence = :gov_transparence,
                                    gov_controle_ethique = :gov_controle_ethique
                                WHERE id = :id
                                """
                            ),
                            params,
                        )
                        db.execute(_text("DELETE FROM esg_questionnaire_exclusion WHERE questionnaire_id = :id"), {"id": qid})
                        db.execute(_text("DELETE FROM esg_questionnaire_indicator WHERE questionnaire_id = :id"), {"id": qid})
                    else:
                        db.execute(
                            _text(
                                """
                                INSERT INTO esg_questionnaire (
                                  client_ref, saisie_at, obsolescence_at,
                                  env_importance, env_ges_reduc,
                                  soc_droits_humains, soc_parite,
                                  gov_transparence, gov_controle_ethique
                                ) VALUES (
                                  :client_ref, :saisie_at, :obsolescence_at,
                                  :env_importance, :env_ges_reduc,
                                  :soc_droits_humains, :soc_parite,
                                  :gov_transparence, :gov_controle_ethique
                                )
                                """
                            ),
                            base_params,
                        )
                        qid = _last_insert_id(db)
                    # (ré)insérer les associations
                    for oid in excl_ids:
                        db.execute(
                            _text("INSERT OR IGNORE INTO esg_questionnaire_exclusion (questionnaire_id, option_id) VALUES (:q, :o)"),
                            {"q": qid, "o": oid},
                        )
                    for oid in ind_ids:
                        db.execute(
                            _text("INSERT OR IGNORE INTO esg_questionnaire_indicator (questionnaire_id, option_id) VALUES (:q, :o)"),
                            {"q": qid, "o": oid},
                        )
                    db.commit()
                    esg_success = "Préférences ESG enregistrées."
                except Exception as exc:
                    db.rollback()
                    esg_error = "Impossible d'enregistrer les préférences ESG."
                    logger.debug("Dashboard KYC client: erreur esg_save: %s", exc, exc_info=True)

        elif action == "risque_save":
            # Sauvegarde du questionnaire Connaissance financière
            from sqlalchemy import text as _text
            active_section = "knowledge"
            ui_focus_section = "knowledge"
            ui_focus_panel = "knowledgePanel"
            try:
                # Charger les référentiels nécessaires localement (pour éviter toute dépendance d'ordre)
                risque_opts_local = {}
                for name, query in [
                    ("niveaux", "SELECT id, code, label FROM risque_connaissance_niveau_option ORDER BY id"),
                    ("perte", "SELECT id, code, label FROM risque_perte_option ORDER BY id"),
                    ("patrimoine_part", "SELECT id, code, label FROM risque_patrimoine_part_option ORDER BY id"),
                    ("disponibilite", "SELECT id, code, label FROM risque_disponibilite_option ORDER BY id"),
                    ("duree", "SELECT id, code, label FROM risque_duree_option ORDER BY id"),
                    ("objectifs", "SELECT id, code, label FROM risque_objectif_option ORDER BY id"),
                ]:
                    try:
                        rows = db.execute(_text(query)).fetchall()
                        risque_opts_local[name] = [dict(r._mapping) for r in rows]
                    except Exception:
                        risque_opts_local[name] = []
                conso = (form.get("connaissance_adequate") or "").strip().lower()  # 'oui'/'non'
                # map produits -> niveaux
                prod_levels: dict[int,int] = {}
                for k, v in form.multi_items() if hasattr(form, 'multi_items') else form.items():
                    if k.startswith("connaissance_") and k.endswith("_niveau_id"):
                        try:
                            pid = int(k.split("_")[1])
                            nid = int(v)
                            prod_levels[pid] = nid
                        except Exception:
                            pass
                perte_id = int(form.get("perte_option_id") or 0) or None
                patr_id = int(form.get("patrimoine_part_option_id") or 0) or None
                disp_id = int(form.get("disponibilite_option_id") or 0) or None
                duree_id = int(form.get("duree_option_id") or 0) or None
                obj_ids = []
                if hasattr(form, 'getlist'):
                    obj_ids = [int(x) for x in form.getlist("objectif_ids") if str(x).isdigit()]
                autre_detail = (form.get("objectif_autre_detail") or "").strip() or None
                revenus_ct_accept = (form.get("revenus_ct_accept") or "").strip().lower()  # 'oui'/'non'
                offre_personnelle = form.get("offre_personnelle_niveau_id")
                accept_offre_calculee = (form.get("accept_offre_calculee") or "").strip().lower()  # 'oui'/'non'
                motivation_refus = (form.get("motivation_refus") or "").strip() or None
                try:
                    offre_personnelle_id = int(offre_personnelle) if offre_personnelle else None
                except Exception:
                    offre_personnelle_id = None

                # Compute base offer
                def clamp(n, lo, hi):
                    return max(lo, min(hi, n))

                OFFRE = {"court_terme":1, "prudente":2, "equilibree":3, "dynamique":4, "offensif":5}
                offre_calc = 1
                if conso == "non":
                    offre_calc = OFFRE["court_terme"]
                else:
                    # counts from niveaux id → need codes; map nid->code
                    niveaux_map = {int(x["id"]): x["code"] for x in risque_opts_local.get("niveaux", [])}
                    c_f, c_m, c_i = 0,0,0
                    for nid in prod_levels.values():
                        code = niveaux_map.get(int(nid))
                        if code == "faible": c_f += 1
                        elif code == "moyen": c_m += 1
                        elif code == "important": c_i += 1
                    if c_f == 4:
                        offre_calc = OFFRE["court_terme"]
                    elif c_i >= 3:
                        offre_calc = OFFRE["offensif"]
                    elif c_m == 4 and c_f == 0:
                        offre_calc = OFFRE["equilibree"]
                    elif c_f == 2:
                        offre_calc = OFFRE["prudente"]
                    elif c_f == 1:
                        offre_calc = OFFRE["equilibree"]
                    elif c_m == 2 and c_i == 2:
                        offre_calc = OFFRE["dynamique"]
                    else:
                        offre_calc = OFFRE["equilibree"]

                # Adjustments
                # Perte
                if perte_id is not None:
                    perte_code = next((x["code"] for x in risque_opts_local.get("perte", []) if int(x["id"])==perte_id), None)
                    if perte_code == "p5":
                        offre_calc = OFFRE["prudente"]
                    elif perte_code == "p5_10":
                        offre_calc = OFFRE["equilibree"]
                    elif perte_code == "p10_15":
                        pass  # no change

                # Patrimoine
                if patr_id is not None:
                    patr_code = next((x["code"] for x in risque_opts_local.get("patrimoine_part", []) if int(x["id"])==patr_id), None)
                    if patr_code == "m25":
                        pass
                    elif patr_code == "25_50":
                        offre_calc = max(OFFRE["prudente"], offre_calc - 1)
                    elif patr_code == "50_75":
                        offre_calc = max(OFFRE["prudente"], offre_calc - 2)
                    elif patr_code == "p75":
                        offre_calc = OFFRE["prudente"]

                # Disponibilité
                if disp_id is not None:
                    disp_code = next((x["code"] for x in risque_opts_local.get("disponibilite", []) if int(x["id"])==disp_id), None)
                    if disp_code in ("court_terme", "tres_liquide"):
                        # Cap maximum = prudent
                        offre_calc = min(offre_calc, OFFRE["prudente"])
                    # "autres_economies": aucun changement

                # Durée (cap maximum)
                if duree_id is not None:
                    duree_code = next((x["code"] for x in risque_opts_local.get("duree", []) if int(x["id"])==duree_id), None)
                    caps = {
                        "1_3": OFFRE["court_terme"],
                        "3_5": OFFRE["prudente"],
                        "5_8": OFFRE["equilibree"],
                    }
                    if duree_code in caps:
                        offre_calc = min(offre_calc, caps[duree_code])

                # Objectifs (cap prudent if epargne de précaution)
                if any(int(x)==obj_id for x in obj_ids for obj_id in [
                    next((o["id"] for o in risque_opts_local.get("objectifs", []) if o["code"]=="epargne_precaution"), None)
                ]):
                    offre_calc = min(offre_calc, OFFRE["prudente"])

                # Revenus court-terme objective handling
                rev_ct_id = next((int(o["id"]) for o in risque_opts_local.get("objectifs", []) if o.get("code") in ("revenus_court_terme","revenus")), None)

                # Persist
                from datetime import datetime as _dt, timedelta as _td
                now = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                obso = (_dt.utcnow() + _td(days=730)).strftime("%Y-%m-%d %H:%M:%S")
                # Toujours créer un nouveau questionnaire (historisation)
                # On lit éventuellement l'ancien pour information, mais on n'update plus.
                row = db.execute(
                    _text("SELECT id FROM risque_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"),
                    {"r": str(client_id)},
                ).fetchone()
                last_rqid = row[0] if row else None
                # Déterminer l'offre finale selon acceptation et cas revenus CT
                final_offer = int(offre_calc)
                rev_ct_id = next((int(o["id"]) for o in risque_opts_local.get("objectifs", []) if o.get("code") in ("revenus_court_terme", "revenus")), None)
                if rev_ct_id and rev_ct_id in (obj_ids or []):
                    if revenus_ct_accept == "oui":
                        final_offer = 1  # Court Terme
                    elif revenus_ct_accept == "non" and offre_personnelle_id in (1,2,3,4,5):
                        final_offer = int(offre_personnelle_id)

                # Acceptation générale de l'offre calculée
                if accept_offre_calculee == "oui":
                    final_offer = int(offre_calc)
                elif accept_offre_calculee == "non" and offre_personnelle_id in (1,2,3,4,5):
                    final_offer = int(offre_personnelle_id)

                params_main = {
                    "client_ref": str(client_id),
                    "saisie_at": now,
                    "obsolescence_at": obso,
                    "connaissance_adequate": conso if conso in ("oui","non") else "non",
                    "decharge_responsabilite": 0,
                    "perte_option_id": perte_id,
                    "patrimoine_part_option_id": patr_id,
                    "disponibilite_option_id": disp_id,
                    "duree_option_id": duree_id,
                    "offre_calculee_niveau_id": int(offre_calc),
                    "offre_finale_niveau_id": final_offer,
                    "objectif_autre_detail": autre_detail,
                }
                # Décharge si l'offre finale diffère de l'offre calculée suite à un refus
                if accept_offre_calculee == "non" and params_main["offre_finale_niveau_id"] != int(offre_calc):
                    params_main["decharge_responsabilite"] = 1
                # SRRI mappé sur l'offre finale
                srri_map = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
                srri_val = srri_map.get(int(params_main["offre_finale_niveau_id"]), None)
                # Insertion systématique d'un nouveau questionnaire
                db.execute(
                    _text(
                        """
                        INSERT INTO risque_questionnaire (
                          client_ref, saisie_at, obsolescence_at,
                          connaissance_adequate, decharge_responsabilite,
                          perte_option_id, patrimoine_part_option_id,
                          disponibilite_option_id, duree_option_id,
                          offre_calculee_niveau_id, offre_finale_niveau_id,
                          objectif_autre_detail
                        ) VALUES (
                          :client_ref, :saisie_at, :obsolescence_at,
                          :connaissance_adequate, :decharge_responsabilite,
                          :perte_option_id, :patrimoine_part_option_id,
                          :disponibilite_option_id, :duree_option_id,
                          :offre_calculee_niveau_id, :offre_finale_niveau_id,
                          :objectif_autre_detail
                        )
                        """
                    ),
                    params_main,
                )
                rqid = _last_insert_id(db)
                # insert children
                for pid, nid in prod_levels.items():
                    db.execute(
                        _text("INSERT INTO risque_questionnaire_connaissance (questionnaire_id, produit_id, niveau_id) VALUES (:q,:p,:n)"),
                        {"q": rqid, "p": int(pid), "n": int(nid)},
                    )
                for oid in obj_ids:
                    db.execute(
                        _text("INSERT OR IGNORE INTO risque_questionnaire_objectif (questionnaire_id, option_id) VALUES (:q,:o)"),
                        {"q": rqid, "o": int(oid)},
                    )
                # Upsert risque_decision_client
                try:
                    dec_row = db.execute(_text("SELECT id FROM risque_decision_client WHERE questionnaire_id = :q"), {"q": rqid}).fetchone()
                    decision = 'accepte' if accept_offre_calculee == 'oui' else 'refuse'
                    dec_params = {
                        'questionnaire_id': rqid,
                        'saisie_at': now,
                        'obsolescence_at': obso,
                        'decision': decision,
                        'message': 'Notre proposition est validée' if decision == 'accepte' else 'Le client refuse le risque proposé',
                        'niveau_client_id': None,
                        'motivation_refus': motivation_refus if decision == 'refuse' else None,
                    }
                    if decision == 'refuse' and offre_personnelle_id in (1,2,3,4,5):
                        dec_params['niveau_client_id'] = int(offre_personnelle_id)
                    if dec_row:
                        db.execute(
                            _text(
                                """
                                UPDATE risque_decision_client
                                SET saisie_at=:saisie_at,
                                    obsolescence_at=:obsolescence_at,
                                    decision=:decision,
                                    message=:message,
                                    niveau_client_id=:niveau_client_id,
                                    motivation_refus=:motivation_refus
                                WHERE questionnaire_id=:questionnaire_id
                                """
                            ),
                            dec_params,
                        )
                    else:
                        db.execute(
                            _text(
                                """
                                INSERT INTO risque_decision_client (
                                  questionnaire_id, saisie_at, obsolescence_at,
                                  decision, message, niveau_client_id, motivation_refus
                                ) VALUES (
                                  :questionnaire_id, :saisie_at, :obsolescence_at,
                                  :decision, :message, :niveau_client_id, :motivation_refus
                                )
                                """
                            ),
                            dec_params,
                        )
                except Exception as exc:
                    logger.debug("risque_decision_client upsert error: %s", exc, exc_info=True)
                # Upsert KYC_Client_Risque (synthèse risque client du jour)
                try:
                    # Helper to fetch option label by id
                    def _label(options: list[dict], oid: int | None):
                        if oid is None:
                            return None
                        try:
                            for x in options or []:
                                if int(x.get("id")) == int(oid):
                                    return x.get("label") or x.get("libelle") or None
                        except Exception:
                            return None
                        return None

                    niveau_id = int(final_offer)
                    connaissance_txt = (conso if conso in ("oui", "non") else None) or None
                    patrimoine_txt = _label(risque_opts_local.get("patrimoine_part", []), patr_id)
                    duree_txt = _label(risque_opts_local.get("duree", []), duree_id)
                    contraintes_txt = _label(risque_opts_local.get("perte", []), perte_id)
                    confirmation_txt = (accept_offre_calculee if accept_offre_calculee in ("oui", "non") else None) or None
                    # Libellé de l'offre finale (table risque_niveau)
                    offre_libelle_txt = None
                    try:
                        rlab = db.execute(
                            _text("SELECT label FROM risque_niveau WHERE id = :id"),
                            {"id": niveau_id},
                        ).fetchone()
                        if rlab:
                            offre_libelle_txt = rlab[0]
                    except Exception:
                        offre_libelle_txt = None
                    # Commentaire = Motivation, mais on peut y adjoindre le libellé de l'offre pour traçabilité
                    commentaire_txt = (motivation_refus or None)
                    if offre_libelle_txt:
                        if commentaire_txt:
                            commentaire_txt = f"{commentaire_txt} | Offre finale: {offre_libelle_txt}"
                        else:
                            commentaire_txt = f"Offre finale: {offre_libelle_txt}"

                    from datetime import date as _dt_date
                    today_only = _dt_date.today().isoformat()
                    row_kr = db.execute(
                        _text(
                            """
                            SELECT id FROM KYC_Client_Risque
                            WHERE client_id = :cid
                              AND substr(COALESCE(date_saisie,''), 1, 10) = :today
                            ORDER BY id DESC
                            LIMIT 1
                            """
                        ),
                        {"cid": client_id, "today": today_only},
                    ).fetchone()
                    # Experience: libellé de ref_niveau_risque correspondant à niveau_id
                    exp_txt = None
                    try:
                        row_exp = db.execute(
                            _text("SELECT libelle FROM ref_niveau_risque WHERE id = :id"),
                            {"id": niveau_id},
                        ).fetchone()
                        if row_exp:
                            exp_txt = row_exp[0]
                    except Exception:
                        exp_txt = None

                    payload = {
                        "cid": client_id,
                        "niv": niveau_id,
                        "exp": exp_txt,
                        "connaissance": connaissance_txt,
                        "patrimoine": patrimoine_txt,
                        "duree": duree_txt,
                        "contraintes": contraintes_txt,
                        "confirmation": confirmation_txt,
                        "commentaire": commentaire_txt,
                    }
                    risque_id: int | None = None
                    if row_kr:
                        db.execute(
                            _text(
                                """
                                UPDATE KYC_Client_Risque
                                SET niveau_id = :niv,
                                    experience = :exp,
                                    connaissance = :connaissance,
                                    patrimoine = :patrimoine,
                                    duree = :duree,
                                    contraintes = :contraintes,
                                    confirmation_client = :confirmation,
                                    commentaire = :commentaire
                                WHERE id = :id
                                """
                            ),
                            payload | {"id": int(row_kr[0])},
                        )
                        risque_id = int(row_kr[0])
                    else:
                        db.execute(
                            _text(
                                """
                                INSERT INTO KYC_Client_Risque (
                                    client_id, niveau_id, experience, connaissance, patrimoine, duree, contraintes, confirmation_client, commentaire
                                ) VALUES (
                                    :cid, :niv, :exp, :connaissance, :patrimoine, :duree, :contraintes, :confirmation, :commentaire
                                )
                                """
                            ),
                            payload,
                        )
                        try:
                            last_id = _last_insert_id(db)
                            risque_id = int(last_id) if last_id is not None else None
                        except Exception:
                            risque_id = None
                    # Enregistrer le détail "Niveau par produit" dans une table enfant normalisée
                    try:
                        # Créer la table si absente
                        db.execute(_text(
                            """
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
                        ))
                        if risque_id is not None:
                            # Purge puis insert des paires produit->niveau
                            db.execute(_text("DELETE FROM KYC_Client_Risque_Connaissance WHERE risque_id = :rid"), {"rid": risque_id})
                            # utilitaires libellés
                            produits_map = {int(x.get("id")): (x.get("label") or x.get("libelle") or str(x.get("id"))) for x in (risque_opts_local.get("produits") or [])}
                            niveaux_map = {int(x.get("id")): (x.get("label") or x.get("libelle") or str(x.get("id"))) for x in (risque_opts_local.get("niveaux") or [])}
                            for pid, nid in (prod_levels or {}).items():
                                try:
                                    pid_i = int(pid); nid_i = int(nid)
                                except Exception:
                                    continue
                                db.execute(
                                    _text(
                                        """
                                        INSERT OR REPLACE INTO KYC_Client_Risque_Connaissance (
                                          risque_id, produit_id, niveau_id, produit_label, niveau_label
                                        ) VALUES (
                                          :rid, :pid, :nid, :plabel, :nlabel
                                        )
                                        """
                                    ),
                                    {
                                        "rid": risque_id,
                                        "pid": pid_i,
                                        "nid": nid_i,
                                        "plabel": produits_map.get(pid_i),
                                        "nlabel": niveaux_map.get(nid_i),
                                    },
                                )
                    except Exception as _exc_kcr:
                        logger.debug("KYC_Client_Risque_Connaissance persist error: %s", _exc_kcr, exc_info=True)
                    # pour affichage UI
                    risque_commentaire = payload.get("commentaire")
                    # Récupérer allocation liée au niveau de risque
                    allocation_nom = None
                    allocation_md = None
                    allocation_id_client = None
                    try:
                        row_alloc = db.execute(
                            _text(
                                """
                                SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom,
                                       a.id AS allocation_id,
                                       ar.texte AS allocation_texte
                                FROM allocation_risque ar
                                LEFT JOIN allocations a ON a.nom = ar.allocation_name
                                WHERE ar.risque_id = :rid
                                ORDER BY ar.date_attribution DESC, ar.id DESC
                                LIMIT 1
                                """
                            ),
                            {"rid": niveau_id},
                        ).fetchone()
                        if row_alloc:
                            allocation_nom = row_alloc[0]
                            allocation_id_client = row_alloc[1] if hasattr(row_alloc, "_mapping") and row_alloc._mapping.get("allocation_id") is not None else None
                            allocation_md = row_alloc[2] if hasattr(row_alloc, "_mapping") else row_alloc[1]
                    except Exception:
                        allocation_nom = None
                        allocation_md = None
                        allocation_id_client = None

                    # Si le client accepte le risque calculé, mettre à jour mariadb_clients.allocation_id
                    try:
                        if accept_offre_calculee == "oui":
                            if allocation_id_client is None and allocation_nom:
                                try:
                                    row_alloc_id = db.execute(
                                        _text("SELECT id FROM allocations WHERE nom = :nom LIMIT 1"),
                                        {"nom": allocation_nom},
                                    ).fetchone()
                                    if row_alloc_id:
                                        allocation_id_client = row_alloc_id[0] if not hasattr(row_alloc_id, "_mapping") else (row_alloc_id._mapping.get("id") or row_alloc_id[0])
                                except Exception:
                                    allocation_id_client = None
                            if allocation_id_client is not None:
                                db.execute(
                                    _text("UPDATE mariadb_clients SET allocation_id = :aid WHERE id = :cid"),
                                    {"aid": int(allocation_id_client), "cid": client_id},
                                )
                    except Exception as exc:
                        logger.debug("KYC_Client_Risque: mise à jour allocation_id client impossible: %s", exc, exc_info=True)
                    # Mettre à jour le SRRI dans mariadb_clients selon la correspondance
                    try:
                        if srri_val is not None:
                            db.execute(
                                _text("UPDATE mariadb_clients SET SRRI = :s WHERE id = :cid"),
                                {"s": int(srri_val), "cid": client_id},
                            )
                    except Exception as exc:
                        logger.debug("KYC_Client_Risque: mise à jour SRRI client impossible: %s", exc, exc_info=True)

                    # Charger la série de performance/volatilité pour cette allocation
                    alloc_chart = None
                    try:
                        if allocation_nom:
                            rows_series = (
                                db.query(Allocation.date, Allocation.sicav, Allocation.volat)
                                .filter(Allocation.nom == allocation_nom)
                                .order_by(Allocation.date.asc())
                                .all()
                            )
                            labels: list[str] = []
                            sicav_vals: list[float] = []
                            vol_vals: list[float] = []
                            for d, sicav_v, vol_v in rows_series:
                                try:
                                    dstr = d.strftime("%Y-%m-%d") if d else None
                                except Exception:
                                    dstr = str(d)[:10] if d else None
                                if not dstr:
                                    continue
                                labels.append(dstr)
                                try:
                                    sicav_vals.append(float(sicav_v or 0))
                                except Exception:
                                    sicav_vals.append(0.0)
                                try:
                                    vol_vals.append(float(vol_v or 0))
                                except Exception:
                                    vol_vals.append(0.0)
                            if labels:
                                alloc_chart = {"labels": labels, "sicav": sicav_vals, "vol": vol_vals}
                    except Exception:
                        alloc_chart = None

                    # Convertir markdown en HTML simple
                    def _md_to_html(md: str | None) -> str | None:
                        if not md:
                            return None
                        try:
                            import re, html as _html
                            text = str(md)
                            # Protect HTML
                            text = _html.escape(text)
                            # Headings
                            text = re.sub(r"^###\s+(.*)$", r"<h5>\1</h5>", text, flags=re.M)
                            text = re.sub(r"^##\s+(.*)$", r"<h4>\1</h4>", text, flags=re.M)
                            text = re.sub(r"^#\s+(.*)$", r"<h3>\1</h3>", text, flags=re.M)
                            # Bold / Italic
                            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                            text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
                            # Lists: convert lines starting with -
                            lines = text.split("\n")
                            out = []
                            in_ul = False
                            for ln in lines:
                                if re.match(r"^\s*-\s+", ln):
                                    if not in_ul:
                                        out.append("<ul>")
                                        in_ul = True
                                    out.append("<li>" + re.sub(r"^\s*-\s+", "", ln) + "</li>")
                                else:
                                    if in_ul:
                                        out.append("</ul>")
                                        in_ul = False
                                    # Paragraphs and line breaks
                                    if ln.strip():
                                        out.append("<p>" + ln + "</p>")
                            if in_ul:
                                out.append("</ul>")
                            return "\n".join(out)
                        except Exception:
                            return md

                    risque_snapshot = {
                        "niveau_id": niveau_id,
                        "niveau_label": offre_libelle_txt,
                        "experience": exp_txt,
                        "connaissance": connaissance_txt,
                        "patrimoine": patrimoine_txt,
                        "duree": duree_txt,
                        "contraintes": contraintes_txt,
                        "confirmation_client": confirmation_txt,
                        "commentaire": risque_commentaire,
                        "allocation_nom": allocation_nom,
                        "allocation_chart": alloc_chart,
                        "allocation_html": _md_to_html(allocation_md),
                    }
                    # Ajouter connaissance par produit pour l'affichage
                    try:
                        if risque_id is not None:
                            rows_c = db.execute(
                                _text(
                                    "SELECT produit_id, produit_label, niveau_id, niveau_label FROM KYC_Client_Risque_Connaissance WHERE risque_id = :rid ORDER BY produit_label, produit_id"
                                ),
                                {"rid": risque_id},
                            ).fetchall()
                            risque_snapshot["connaissance_produits"] = [dict(r._mapping) for r in rows_c]
                    except Exception:
                        pass
                except Exception as exc:
                    logger.debug("KYC_Client_Risque upsert error: %s", exc, exc_info=True)
                db.commit()
                # set to show panel
                active_section = "knowledge"
            except Exception as exc:
                db.rollback()
                logger.debug("Dashboard KYC client: erreur risque_save: %s", exc, exc_info=True)

        elif action == "lcbft_save":
            from sqlalchemy import text as _text
            active_section = "lcbft"
            ui_focus_section = "lcbft"
            ui_focus_panel = "lcbftPanel"
            lcbft_success = None
            lcbft_error = None
            try:
                form = await request.form()
                def _i(name, default: int | None = 0):
                    v = form.get(name)
                    if v is None:
                        return default
                    s = str(v).strip().lower()
                    if s == "":
                        return default
                    if s in {"1", "true", "on", "oui", "yes"}:
                        return 1
                    if s in {"0", "false", "off", "non", "no"}:
                        return 0
                    try:
                        return int(float(s))
                    except Exception:
                        return default
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                from datetime import timedelta as _td
                obso = (datetime.utcnow() + _td(days=730)).strftime("%Y-%m-%d %H:%M:%S")
                params = {
                    "client_ref": str(client_id),
                    "created_at": now,
                    "updated_at": obso,
                    "relation_mode": (form.get("relation_mode") or None),
                    "relation_since": (form.get("relation_since") or None),
                    "has_existing_contracts": _i("has_existing_contracts"),
                    "existing_with_our_insurer": _i("existing_with_our_insurer"),
                    "existing_contract_ref": (form.get("existing_contract_ref") or None),
                    "reason_new_contract": (form.get("reason_new_contract") or None),
                    "ppe_self": _i("ppe_self"),
                    "ppe_self_fonction": (form.get("ppe_self_fonction") or None),
                    "ppe_self_pays": (form.get("ppe_self_pays") or None),
                    "ppe_family": _i("ppe_family"),
                    "ppe_family_fonction": (form.get("ppe_family_fonction") or None),
                    "ppe_family_pays": (form.get("ppe_family_pays") or None),
                    # Flags comportement/opération/modalités (oui=1/non=0)
                    "flag_1a": _i("flag_1a"),
                    "flag_1b": _i("flag_1b"),
                    "flag_1c": _i("flag_1c"),
                    "flag_1d": _i("flag_1d"),
                    "flag_2a": _i("flag_2a"),
                    "flag_2b": _i("flag_2b"),
                    "flag_3a": _i("flag_3a"),
                    # Profession / exposition
                    "prof_profession": (form.get("prof_profession") or None),
                    "prof_statut_professionnel_id": _i("prof_statut_professionnel_id"),
                    "prof_secteur_id": _i("prof_secteur_id"),
                    "prof_self_ppe": _i("prof_self_ppe"),
                    "prof_self_ppe_fonction": (form.get("prof_self_ppe_fonction") or None),
                    "prof_self_ppe_pays": (form.get("prof_self_ppe_pays") or None),
                    "prof_family_ppe": _i("prof_family_ppe"),
                    "prof_family_ppe_fonction": (form.get("prof_family_ppe_fonction") or None),
                    "prof_family_ppe_pays": (form.get("prof_family_ppe_pays") or None),
                    # Objet / montants
                    "operation_objet": (form.get("operation_objet") or None),
                    "montant": (float(form.get("montant") or 0) or None),
                    "patrimoine_pct": (float(form.get("patrimoine_pct") or 0) or None),
                    # Justificatifs textes
                    "just_fonds": (form.get("just_fonds") or None),
                    "just_destination": (form.get("just_destination") or None),
                    "just_finalite": (form.get("just_finalite") or None),
                    "just_produits": (form.get("just_produits") or None),
                }
                # Calcul niveau de risque (1..4)
                risk = 1
                f = lambda k: (params.get(k) or 0) == 1
                if f("flag_1a") or f("flag_1c") or f("flag_1d") or f("flag_2b") or f("flag_3a"):
                    risk = 4
                elif f("flag_2a"):
                    risk = 3
                elif f("flag_1b") or (params.get("ppe_self") == 1) or (params.get("ppe_family") == 1):
                    risk = 2
                else:
                    risk = 1
                params["computed_risk_level"] = risk
                # Ensure DB NOT NULL client_id is satisfied
                try:
                    params["client_id"] = int(client_id)
                except Exception:
                    params["client_id"] = None
                existing_q = db.execute(
                    _text(
                        """
                        SELECT id, created_at
                        FROM LCBFT_questionnaire
                        WHERE client_ref = :cr
                        ORDER BY COALESCE(updated_at, created_at) DESC, id DESC
                        LIMIT 1
                        """
                    ),
                    {"cr": str(client_id)},
                ).fetchone()
                if existing_q:
                    qid = existing_q[0]
                    update_params = params.copy()
                    update_params["__id"] = qid
                    update_params["updated_at"] = obso
                    set_clause = ", ".join(
                        [
                            "client_ref = :client_ref",
                            "client_id = :client_id",
                            "updated_at = :updated_at",
                            "relation_mode = :relation_mode",
                            "relation_since = :relation_since",
                            "has_existing_contracts = :has_existing_contracts",
                            "existing_with_our_insurer = :existing_with_our_insurer",
                            "existing_contract_ref = :existing_contract_ref",
                            "reason_new_contract = :reason_new_contract",
                            "ppe_self = :ppe_self",
                            "ppe_self_fonction = :ppe_self_fonction",
                            "ppe_self_pays = :ppe_self_pays",
                            "ppe_family = :ppe_family",
                            "ppe_family_fonction = :ppe_family_fonction",
                            "ppe_family_pays = :ppe_family_pays",
                            "flag_1a = :flag_1a",
                            "flag_1b = :flag_1b",
                            "flag_1c = :flag_1c",
                            "flag_1d = :flag_1d",
                            "flag_2a = :flag_2a",
                            "flag_2b = :flag_2b",
                            "flag_3a = :flag_3a",
                            "computed_risk_level = :computed_risk_level",
                            "prof_profession = :prof_profession",
                            "prof_statut_professionnel_id = :prof_statut_professionnel_id",
                            "prof_secteur_id = :prof_secteur_id",
                            "prof_self_ppe = :prof_self_ppe",
                            "prof_self_ppe_fonction = :prof_self_ppe_fonction",
                            "prof_self_ppe_pays = :prof_self_ppe_pays",
                            "prof_family_ppe = :prof_family_ppe",
                            "prof_family_ppe_fonction = :prof_family_ppe_fonction",
                            "prof_family_ppe_pays = :prof_family_ppe_pays",
                            "operation_objet = :operation_objet",
                            "montant = :montant",
                            "patrimoine_pct = :patrimoine_pct",
                            "just_fonds = :just_fonds",
                            "just_destination = :just_destination",
                            "just_finalite = :just_finalite",
                            "just_produits = :just_produits",
                        ]
                    )
                    db.execute(
                        _text(f"UPDATE LCBFT_questionnaire SET {set_clause} WHERE id = :__id"),
                        update_params,
                    )
                else:
                    db.execute(
                        _text(
                            """
                            INSERT INTO LCBFT_questionnaire (
                              client_ref, client_id, created_at, updated_at,
                              relation_mode, relation_since,
                              has_existing_contracts, existing_with_our_insurer,
                              existing_contract_ref, reason_new_contract,
                              ppe_self, ppe_self_fonction, ppe_self_pays,
                              ppe_family, ppe_family_fonction, ppe_family_pays,
                              flag_1a, flag_1b, flag_1c, flag_1d,
                              flag_2a, flag_2b, flag_3a,
                              computed_risk_level,
                              prof_profession, prof_statut_professionnel_id, prof_secteur_id,
                              prof_self_ppe, prof_self_ppe_fonction, prof_self_ppe_pays,
                              prof_family_ppe, prof_family_ppe_fonction, prof_family_ppe_pays,
                              operation_objet, montant, patrimoine_pct,
                              just_fonds, just_destination, just_finalite, just_produits
                            ) VALUES (
                              :client_ref, :client_id, :created_at, :updated_at,
                              :relation_mode, :relation_since,
                              :has_existing_contracts, :existing_with_our_insurer,
                              :existing_contract_ref, :reason_new_contract,
                              :ppe_self, :ppe_self_fonction, :ppe_self_pays,
                              :ppe_family, :ppe_family_fonction, :ppe_family_pays,
                              :flag_1a, :flag_1b, :flag_1c, :flag_1d,
                              :flag_2a, :flag_2b, :flag_3a,
                              :computed_risk_level,
                              :prof_profession, :prof_statut_professionnel_id, :prof_secteur_id,
                              :prof_self_ppe, :prof_self_ppe_fonction, :prof_self_ppe_pays,
                              :prof_family_ppe, :prof_family_ppe_fonction, :prof_family_ppe_pays,
                              :operation_objet, :montant, :patrimoine_pct,
                              :just_fonds, :just_destination, :just_finalite, :just_produits
                            )
                            """
                        ),
                        params,
                    )
                    qid = _last_insert_id(db)
                # Vigilance options
                try:
                    if not existing_q and qid is None:
                        qid = _last_insert_id(db)
                except Exception:
                    pass
                try:
                    if existing_q:
                        qid = existing_q[0]
                    # Persist FATCA fields linked to this questionnaire
                    try:
                        # Read form values
                        fatca_contrat_id = _i("fatca_contrat_id")
                        fatca_pays_residence = (form.get("fatca_pays_residence") or None)
                        fatca_nif = (form.get("fatca_nif") or None)
                        fatca_date_operation = (form.get("fatca_date_operation") or None)
                        # Resolve societe_nom from DB if possible
                        societe_nom = (form.get("fatca_societe") or None)
                        if fatca_contrat_id and (not societe_nom or str(societe_nom).strip() == ""):
                            try:
                                row_soc = db.execute(
                                    _text(
                                        """
                                        SELECT COALESCE(s.nom, '') AS societe_nom
                                        FROM mariadb_affaires_generique g
                                        LEFT JOIN mariadb_societe s ON s.id = g.id_societe
                                        WHERE g.id = :gid
                                        """
                                    ),
                                    {"gid": fatca_contrat_id},
                                ).fetchone()
                                if row_soc:
                                    societe_nom = row_soc[0]
                            except Exception:
                                pass
                        if not fatca_date_operation:
                            from datetime import date as _dt_date
                            fatca_date_operation = _dt_date.today().isoformat()
                        # Ensure table exists
                        db.execute(
                            _text(
                                """
                                CREATE TABLE IF NOT EXISTS LCBFT_fatca (
                                  id INTEGER PRIMARY KEY,
                                  questionnaire_id INTEGER UNIQUE,
                                  contrat_id INTEGER NULL,
                                  societe_nom TEXT NULL,
                                  date_operation TEXT NULL,
                                  pays_residence TEXT NULL,
                                  nif TEXT NULL
                                )
                                """
                            )
                        )
                        # Upsert-fatca for this questionnaire
                        fatca_params = {
                            "qid": qid,
                            "contrat_id": fatca_contrat_id,
                            "societe_nom": societe_nom,
                            "date_operation": fatca_date_operation,
                            "pays_residence": fatca_pays_residence,
                            "nif": fatca_nif,
                        }
                        res = db.execute(
                            _text(
                                """
                                UPDATE LCBFT_fatca
                                SET contrat_id=:contrat_id,
                                    societe_nom=:societe_nom,
                                    date_operation=:date_operation,
                                    pays_residence=:pays_residence,
                                    nif=:nif
                                WHERE questionnaire_id=:qid
                                """
                            ),
                            fatca_params,
                        )
                        if (getattr(res, 'rowcount', None) or 0) == 0:
                            db.execute(
                                _text(
                                    """
                                    INSERT INTO LCBFT_fatca (
                                      questionnaire_id, contrat_id, societe_nom, date_operation, pays_residence, nif
                                    ) VALUES (
                                      :qid, :contrat_id, :societe_nom, :date_operation, :pays_residence, :nif
                                    )
                                    """
                                ),
                                fatca_params,
                            )
                        # Optionally sync NIF to mariadb_clients.nif if provided
                        try:
                            if fatca_nif and str(fatca_nif).strip() != "":
                                db.execute(
                                    _text("UPDATE mariadb_clients SET nif = :nif WHERE id = :cid"),
                                    {"nif": fatca_nif, "cid": client_id},
                                )
                        except Exception:
                            # Column may not exist or different name; ignore
                            pass
                        # Optionally sync pays de résidence fiscale back to client/adresse
                        try:
                            if fatca_pays_residence and str(fatca_pays_residence).strip() != "":
                                # Try common column on mariadb_clients
                                try:
                                    db.execute(
                                        _text("UPDATE mariadb_clients SET adresse_pays = :p WHERE id = :cid"),
                                        {"p": fatca_pays_residence, "cid": client_id},
                                    )
                                except Exception:
                                    # ignore if column doesn't exist
                                    pass
                                # Update latest KYC_Client_Adresse row for this client
                                try:
                                    row_addr = db.execute(
                                        _text(
                                            """
                                            SELECT id FROM KYC_Client_Adresse
                                            WHERE client_id = :cid
                                            ORDER BY (date_saisie IS NULL), date_saisie DESC, id DESC
                                            LIMIT 1
                                            """
                                        ),
                                        {"cid": client_id},
                                    ).fetchone()
                                    if row_addr and row_addr[0]:
                                        db.execute(
                                            _text("UPDATE KYC_Client_Adresse SET pays = :p WHERE id = :id"),
                                            {"p": fatca_pays_residence, "id": row_addr[0]},
                                        )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception as _exc_f:
                        logger.debug("LCBFT_fatca persist error: %s", _exc_f, exc_info=True)
                    if hasattr(form, 'getlist'):
                        vids = [int(x) for x in form.getlist('vigilance_ids') if str(x).isdigit()]
                        if qid:
                            db.execute(
                                _text("DELETE FROM LCBFT_questionnaire_vigilance WHERE questionnaire_id = :q"),
                                {"q": qid},
                            )
                        for oid in vids:
                            db.execute(
                                _text("INSERT OR IGNORE INTO LCBFT_questionnaire_vigilance (questionnaire_id, option_id) VALUES (:q,:o)"),
                                {"q": qid, "o": oid},
                            )
                except Exception:
                    pass
                db.commit()
                try:
                    row = db.execute(
                        _text("SELECT * FROM LCBFT_questionnaire WHERE id = :id"),
                        {"id": qid},
                    ).fetchone()
                    if row:
                        lcbft_current = dict(row._mapping)
                except Exception:
                    pass
                try:
                    if hasattr(form, 'getlist'):
                        lcbft_raison_selected_ids = [int(x) for x in form.getlist('vigilance_ids') if str(x).isdigit()]
                except Exception:
                    pass
                lcbft_success = "Questionnaire LCB-FT enregistré."
            except Exception as exc:
                db.rollback()
                lcbft_error = "Impossible d'enregistrer le questionnaire LCB-FT."
                logger.debug("Dashboard KYC client: erreur lcbft_save: %s", exc, exc_info=True)


    etat_civil_row = None
    try:
        row = db.execute(
            text(
                """
                SELECT id,
                       civilite,
                       date_naissance,
                       lieu_naissance,
                       nationalite,
                       situation_familiale,
                       profession,
                       commentaire
                FROM etat_civil_client
                WHERE id_client = :cid
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id},
        ).fetchone()
        if row:
            data = row._mapping
            etat_civil_row = {
                "id": data.get("id"),
                "civilite": _safe_text(data.get("civilite")),
                "date_naissance": _safe_text(data.get("date_naissance")),
                "lieu_naissance": _safe_text(data.get("lieu_naissance")),
                "nationalite": _safe_text(data.get("nationalite")),
                "situation_familiale": _safe_text(data.get("situation_familiale")),
                "profession": _safe_text(data.get("profession")),
                "commentaire": _safe_text(data.get("commentaire")),
            }
    except Exception:
        etat_civil_row = None

    actifs: list[dict] = []
    actifs_total = Decimal("0")
    try:
        rows_actifs = db.execute(
            text(
                """
                SELECT a.id,
                       a.type_actif_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       a.description,
                       a.valeur,
                       a.date_saisie,
                       a.date_expiration
                FROM KYC_Client_Actif a
                LEFT JOIN ref_type_actif t ON t.id = a.type_actif_id
                WHERE a.client_id = :cid
                ORDER BY (a.date_saisie IS NULL), a.date_saisie DESC, a.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_actifs:
            data = row._mapping
            valeur_num = data.get("valeur")
            try:
                if valeur_num is not None:
                    actifs_total += Decimal(str(valeur_num))
            except (InvalidOperation, ValueError):
                pass
            actifs.append(
                {
                    "id": data.get("id"),
                    "type_actif_id": data.get("type_actif_id"),
                    "type_libelle": _safe_text(data.get("type_libelle")),
                    "description": _safe_text(data.get("description")),
                    "valeur": data.get("valeur"),
                    "valeur_str": _fmt_amount(data.get("valeur")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        actifs = []
        actifs_total = Decimal("0")

    actifs_total_str = _fmt_amount(actifs_total)

    passifs: list[dict] = []
    passifs_total = Decimal("0")
    try:
        rows_passifs = db.execute(
            text(
                """
                SELECT p.id,
                       p.type_passif_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       p.description,
                       p.montant_rest_du,
                       p.date_saisie,
                       p.date_expiration
                FROM KYC_Client_Passif p
                LEFT JOIN ref_type_passif t ON t.id = p.type_passif_id
                WHERE p.client_id = :cid
                ORDER BY (p.date_saisie IS NULL), p.date_saisie DESC, p.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_passifs:
            data = row._mapping
            montant_num = data.get("montant_rest_du")
            try:
                if montant_num is not None:
                    passifs_total += Decimal(str(montant_num))
            except (InvalidOperation, ValueError):
                pass
            passifs.append(
                {
                    "id": data.get("id"),
                    "type_passif_id": data.get("type_passif_id"),
                    "type_libelle": _safe_text(data.get("type_libelle")),
                    "description": _safe_text(data.get("description")),
                    "montant": data.get("montant_rest_du"),
                    "montant_str": _fmt_amount(data.get("montant_rest_du")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        passifs = []
        passifs_total = Decimal("0")

    passifs_total_str = _fmt_amount(passifs_total)

    revenus: list[dict] = []
    revenus_total = Decimal("0")
    try:
        rows_revenus = db.execute(
            text(
                """
                SELECT r.id,
                       r.type_revenu_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       r.montant_annuel,
                       r.date_saisie,
                       r.date_expiration
                FROM KYC_Client_Revenus r
                LEFT JOIN ref_type_revenu t ON t.id = r.type_revenu_id
                WHERE r.client_id = :cid
                ORDER BY (r.date_saisie IS NULL), r.date_saisie DESC, r.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_revenus:
            data = row._mapping
            montant_num = data.get("montant_annuel")
            try:
                if montant_num is not None:
                    revenus_total += Decimal(str(montant_num))
            except (InvalidOperation, ValueError):
                pass
            revenus.append(
                {
                    "id": data.get("id"),
                    "type_revenu_id": data.get("type_revenu_id"),
                    "type_libelle": _safe_text(data.get("type_libelle")),
                    "montant": data.get("montant_annuel"),
                    "montant_str": _fmt_amount(data.get("montant_annuel")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        revenus = []
        revenus_total = Decimal("0")

    revenus_total_str = _fmt_amount(revenus_total)

    charges: list[dict] = []
    charges_total = Decimal("0")
    try:
        rows_charges = db.execute(
            text(
                """
                SELECT c.id,
                       c.type_charge_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       c.montant_annuel,
                       c.date_saisie,
                       c.date_expiration
                FROM KYC_Client_Charges c
                LEFT JOIN ref_type_charge t ON t.id = c.type_charge_id
                WHERE c.client_id = :cid
                ORDER BY (c.date_saisie IS NULL), c.date_saisie DESC, c.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_charges:
            data = row._mapping
            montant_num = data.get("montant_annuel")
            try:
                if montant_num is not None:
                    charges_total += Decimal(str(montant_num))
            except (InvalidOperation, ValueError):
                pass
            charges.append(
                {
                    "id": data.get("id"),
                    "type_charge_id": data.get("type_charge_id"),
                    "type_libelle": _safe_text(data.get("type_libelle")),
                    "montant": data.get("montant_annuel"),
                    "montant_str": _fmt_amount(data.get("montant_annuel")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        charges = []
        charges_total = Decimal("0")

    charges_total_str = _fmt_amount(charges_total)

    def _rows_for_chart(rows, label_key, amount_key):
        bucket: dict[str, Decimal] = defaultdict(Decimal)
        for row in rows:
            raw_label = row.get(label_key)
            label = _safe_text(raw_label) or "Non renseigné"
            value = row.get(amount_key)
            if value is None:
                continue
            try:
                bucket[label] += Decimal(str(value))
            except (InvalidOperation, ValueError):
                continue
        dataset: list[dict] = []
        for label, amount in bucket.items():
            if amount <= 0:
                continue
            try:
                dataset.append({
                    "label": label,
                    "value": float(amount),
                    "display": f"{amount:,.0f}".replace(",", " ")
                })
            except (TypeError, ValueError):
                continue
        return dataset

    synth_actifs = _rows_for_chart(actifs, "type_libelle", "valeur")
    synth_passifs = _rows_for_chart(passifs, "type_libelle", "montant")
    synth_revenus = _rows_for_chart(revenus, "type_libelle", "montant")
    synth_charges = _rows_for_chart(charges, "type_libelle", "montant")

    patrimoine_net = actifs_total - passifs_total
    budget_net = revenus_total - charges_total
    patrimoine_net_str = _fmt_amount(patrimoine_net)
    budget_net_str = _fmt_amount(budget_net)

    # Snapshot des synthèses déjà effectué après chaque mutation via _snapshot_synthese()

    # Planifier l'écriture de la synthèse (affichage en bas de page)
    synth_today = _date.today().isoformat()
    try:
        row_today = db.execute(
            text(
                """
                SELECT id FROM KYC_Client_Synthese
                WHERE client_id = :cid
                  AND substr(COALESCE(date_saisie,''), 1, 10) = :today
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"cid": client_id, "today": synth_today},
        ).fetchone()
        synthese_push_action = "update" if row_today else "insert"
    except Exception:
        synthese_push_action = "insert"

    # Dernière synthèse patrimoniale (totaux)
    synthese_last = None
    try:
        row = db.execute(text(
            "SELECT * FROM KYC_Client_Synthese WHERE client_id = :cid ORDER BY date_saisie DESC, id DESC LIMIT 1"
        ), {"cid": client_id}).fetchone()
        if row:
            m = row._mapping
            def _first(keys):
                for k in keys:
                    if k in m and m.get(k) is not None:
                        return m.get(k)
                return None
            synthese_last = {
                "total_revenus": _first(["total_revenus", "total_revenu", "revenus_total"]) or 0,
                "total_charges": _first(["total_charges", "charges_total"]) or 0,
                "total_actif": _first(["total_actif", "actifs_total", "total_actifs"]) or 0,
                "total_passif": _first(["total_passif", "passifs_total", "total_passifs"]) or 0,
                "commentaire": m.get("commentaire"),
            }
    except Exception:
        synthese_last = None


    adresses: list[dict] = []
    try:
        rows_adresses = db.execute(
            text(
                """
                SELECT a.id,
                       a.type_adresse_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       a.rue,
                       a.complement,
                       a.code_postal,
                       a.ville,
                       a.pays,
                       a.date_saisie,
                       a.date_expiration
                FROM KYC_Client_Adresse a
                LEFT JOIN ref_type_adresse t ON t.id = a.type_adresse_id
                WHERE a.client_id = :cid
                ORDER BY (a.date_saisie IS NULL), a.date_saisie DESC, a.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_adresses:
            data = row._mapping
            date_saisie = data.get("date_saisie")
            date_expiration = data.get("date_expiration")
            libelle = _safe_text(data.get("type_libelle"))
            libelle_lower = libelle.lower()
            is_primary = "princip" in libelle_lower
            is_secondary = (not is_primary) and "second" in libelle_lower
            adresses.append(
                {
                    "id": data.get("id"),
                    "type_adresse_id": data.get("type_adresse_id"),
                    "type_libelle": libelle,
                    "is_primary": is_primary,
                    "is_secondary": is_secondary,
                    "rue": _safe_text(data.get("rue")),
                    "complement": _safe_text(data.get("complement")),
                    "code_postal": _safe_text(data.get("code_postal")),
                    "ville": _safe_text(data.get("ville")),
                    "pays": _safe_text(data.get("pays")),
                    "date_saisie": _fmt_date(date_saisie),
                    "date_expiration": _fmt_date(date_expiration),
                }
            )
    except Exception:
        adresses = []

    situations_matrimoniales: list[dict] = []
    try:
        rows_matrimonial = db.execute(
            text(
                """
                SELECT m.id,
                       m.situation_id,
                       sm.libelle AS situation_libelle,
                       m.nb_enfants,
                       m.convention_id,
                       sc.libelle AS convention_libelle,
                       m.date_saisie,
                       m.date_expiration
                FROM KYC_Client_Situation_Matrimoniale m
                LEFT JOIN ref_situation_matrimoniale sm ON sm.id = m.situation_id
                LEFT JOIN ref_situation_matrimoniale_convention sc ON sc.id = m.convention_id
                WHERE m.client_id = :cid
                ORDER BY (m.date_saisie IS NULL), m.date_saisie DESC, m.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_matrimonial:
            data = row._mapping
            situations_matrimoniales.append(
                {
                    "id": data.get("id"),
                    "situation_id": data.get("situation_id"),
                    "situation_libelle": _safe_text(data.get("situation_libelle")),
                    "nb_enfants": data.get("nb_enfants") or 0,
                    "convention_id": data.get("convention_id"),
                    "convention_libelle": _safe_text(data.get("convention_libelle")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        situations_matrimoniales = []

    situations_professionnelles: list[dict] = []
    try:
        rows_professionnelles = db.execute(
            text(
                """
                SELECT p.id,
                       p.profession,
                       p.secteur_id,
                       ps.libelle AS secteur_libelle,
                       p.employeur,
                       p.anciennete_annees,
                       p.statut_id,
                       st.libelle AS statut_libelle,
                       p.date_saisie,
                       p.date_expiration
                FROM KYC_Client_Situation_Professionnelle p
                LEFT JOIN ref_profession_secteur ps ON ps.id = p.secteur_id
                LEFT JOIN ref_statut_professionnel st ON st.id = p.statut_id
                WHERE p.client_id = :cid
                ORDER BY (p.date_saisie IS NULL), p.date_saisie DESC, p.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows_professionnelles:
            data = row._mapping
            situations_professionnelles.append(
                {
                    "id": data.get("id"),
                    "profession": _safe_text(data.get("profession")),
                    "secteur_id": data.get("secteur_id"),
                    "secteur_libelle": _safe_text(data.get("secteur_libelle")),
                    "employeur": _safe_text(data.get("employeur")),
                    "anciennete_annees": data.get("anciennete_annees") or 0,
                    "statut_id": data.get("statut_id"),
                    "statut_libelle": _safe_text(data.get("statut_libelle")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
    except Exception:
        situations_professionnelles = []

    try:
        ref_type_actif_rows = db.execute(
            text("SELECT id, libelle FROM ref_type_actif ORDER BY libelle")
        ).fetchall()
        ref_type_actif = [dict(row._mapping) for row in ref_type_actif_rows]
    except Exception:
        ref_type_actif = []

    try:
        ref_type_passif_rows = db.execute(
            text("SELECT id, libelle FROM ref_type_passif ORDER BY libelle")
        ).fetchall()
        ref_type_passif = [dict(row._mapping) for row in ref_type_passif_rows]
    except Exception:
        ref_type_passif = []

    try:
        ref_type_revenu_rows = db.execute(
            text("SELECT id, libelle FROM ref_type_revenu ORDER BY libelle")
        ).fetchall()
        ref_type_revenu = [dict(row._mapping) for row in ref_type_revenu_rows]
    except Exception:
        ref_type_revenu = []

    try:
        ref_type_charge_rows = db.execute(
            text("SELECT id, libelle FROM ref_type_charge ORDER BY libelle")
        ).fetchall()
        ref_type_charge = [dict(row._mapping) for row in ref_type_charge_rows]
    except Exception:
        ref_type_charge = []

    # Contrats disponibles (pour choix du contrat)
    kyc_contracts: list[dict] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT id, nom_contrat, description, frais_gestion_assureur, frais_gestion_courtier
                FROM mariadb_affaires_generique
                WHERE COALESCE(actif,1)=1
                ORDER BY nom_contrat
                """
            )
        ).fetchall()
        kyc_contracts = [dict(r._mapping) for r in rows]
    except Exception:
        kyc_contracts = []

    # Contrat sélectionné (si existant)
    kyc_contrat_selected_id: int | None = None
    try:
        row = db.execute(text("SELECT id_contrat FROM KYC_contrat_choisi WHERE id_client = :cid LIMIT 1"), {"cid": client_id}).fetchone()
        if row:
            try:
                kyc_contrat_selected_id = int(row[0]) if row[0] is not None else None
            except Exception:
                kyc_contrat_selected_id = None
    except Exception:
        kyc_contrat_selected_id = None

    # ESG: options et questionnaire courant
    esg_exclusion_options = []
    esg_indicator_options = []
    esg_selected_exclusions: list[int] = []
    esg_selected_indicators: list[int] = []
    esg_current: dict | None = None
    try:
        rows = db.execute(text("SELECT id, code, label FROM esg_exclusion_option ORDER BY label")).fetchall()
        esg_exclusion_options = [dict(r._mapping) for r in rows]
    except Exception:
        esg_exclusion_options = []
    try:
        rows = db.execute(text("SELECT id, code, label FROM esg_indicator_option ORDER BY label")).fetchall()
        esg_indicator_options = [dict(r._mapping) for r in rows]
    except Exception:
        esg_indicator_options = []
    try:
        row = db.execute(
            text("SELECT * FROM esg_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"),
            {"r": str(client_id)},
        ).fetchone()
        if row:
            m = row._mapping
            qid = m.get("id")
            esg_current = {
                "id": qid,
                "saisie_at": _fmt_date(m.get("saisie_at")),
                "obsolescence_at": _fmt_date(m.get("obsolescence_at")),
                "env_importance": m.get("env_importance"),
                "env_ges_reduc": m.get("env_ges_reduc"),
                "soc_droits_humains": m.get("soc_droits_humains"),
                "soc_parite": m.get("soc_parite"),
                "gov_transparence": m.get("gov_transparence"),
                "gov_controle_ethique": m.get("gov_controle_ethique"),
            }
            try:
                ids = db.execute(text("SELECT option_id FROM esg_questionnaire_exclusion WHERE questionnaire_id = :q"), {"q": qid}).fetchall()
                esg_selected_exclusions = [int(x[0]) for x in ids]
            except Exception:
                esg_selected_exclusions = []
            try:
                ids = db.execute(text("SELECT option_id FROM esg_questionnaire_indicator WHERE questionnaire_id = :q"), {"q": qid}).fetchall()
                esg_selected_indicators = [int(x[0]) for x in ids]
            except Exception:
                esg_selected_indicators = []
    except Exception:
        esg_current = None

    # Dates d'affichage ESG: si aucune saisie, proposer date du jour et obsolescence à 2 ans
    from datetime import date as _dt_date2, timedelta as _td2
    if esg_current and esg_current.get("saisie_at") and esg_current.get("obsolescence_at"):
        esg_display_saisie = esg_current.get("saisie_at")
        esg_display_obsolescence = esg_current.get("obsolescence_at")
    else:
        today = _dt_date2.today().isoformat()
        in2y = (_dt_date2.today() + _td2(days=730)).isoformat()
        esg_display_saisie = today
        esg_display_obsolescence = in2y

    # -------- Bloc Connaissance financière (risque) --------
    risque_opts = {
        "produits": [],
        "niveaux": [],
        "perte": [],
        "patrimoine_part": [],
        "disponibilite": [],
        "duree": [],
        "objectifs": [],
        "niveaux_offre": [],
    }
    try:
        for name, query in [
            ("produits", "SELECT id, code, label FROM risque_connaissance_produit_option ORDER BY id"),
            ("niveaux", "SELECT id, code, label FROM risque_connaissance_niveau_option ORDER BY id"),
            ("perte", "SELECT id, code, label FROM risque_perte_option ORDER BY id"),
            ("patrimoine_part", "SELECT id, code, label FROM risque_patrimoine_part_option ORDER BY id"),
            ("disponibilite", "SELECT id, code, label FROM risque_disponibilite_option ORDER BY id"),
            ("duree", "SELECT id, code, label FROM risque_duree_option ORDER BY id"),
            ("objectifs", "SELECT id, code, label FROM risque_objectif_option ORDER BY id"),
            ("niveaux_offre", "SELECT id, code, label FROM risque_niveau ORDER BY id"),
        ]:
            rows = db.execute(text(query)).fetchall()
            risque_opts[name] = [dict(r._mapping) for r in rows]
    except Exception:
        pass

    risque_current = None
    risque_decision = None
    risque_connaissance_map = {}
    risque_objectifs_ids: list[int] = []
    try:
        row = db.execute(
            text("SELECT * FROM risque_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"),
            {"r": str(client_id)},
        ).fetchone()
        if row:
            m = row._mapping
            rqid = m.get("id")
            risque_current = {
                "id": rqid,
                "saisie_at": _fmt_date(m.get("saisie_at")),
                "obsolescence_at": _fmt_date(m.get("obsolescence_at")),
                "connaissance_adequate": m.get("connaissance_adequate"),
                "decharge_responsabilite": m.get("decharge_responsabilite"),
                "perte_option_id": m.get("perte_option_id"),
                "patrimoine_part_option_id": m.get("patrimoine_part_option_id"),
                "disponibilite_option_id": m.get("disponibilite_option_id"),
                "duree_option_id": m.get("duree_option_id"),
                "offre_calculee_niveau_id": m.get("offre_calculee_niveau_id"),
                "offre_finale_niveau_id": m.get("offre_finale_niveau_id"),
                "objectif_autre_detail": m.get("objectif_autre_detail"),
            }
            try:
                rows = db.execute(text("SELECT produit_id, niveau_id FROM risque_questionnaire_connaissance WHERE questionnaire_id = :q"), {"q": rqid}).fetchall()
                risque_connaissance_map = {int(r.produit_id): int(r.niveau_id) for r in rows}
            except Exception:
                risque_connaissance_map = {}
            try:
                rows = db.execute(text("SELECT option_id FROM risque_questionnaire_objectif WHERE questionnaire_id = :q"), {"q": rqid}).fetchall()
                risque_objectifs_ids = [int(x[0]) for x in rows]
            except Exception:
                risque_objectifs_ids = []
            # Décision client (acceptation/refus) liée au questionnaire
            try:
                drow = db.execute(
                    text("SELECT decision, niveau_client_id, motivation_refus FROM risque_decision_client WHERE questionnaire_id = :q"),
                    {"q": rqid},
                ).fetchone()
                if drow:
                    dm = drow._mapping
                    risque_decision = {
                        "decision": dm.get("decision"),
                        "niveau_client_id": dm.get("niveau_client_id"),
                        "motivation_refus": dm.get("motivation_refus"),
                    }
            except Exception:
                risque_decision = None
    except Exception:
        risque_current = None

    # Calculer affichage dates
    if risque_current and risque_current.get("saisie_at") and risque_current.get("obsolescence_at"):
        risque_display_saisie = risque_current.get("saisie_at")
        risque_display_obsolescence = risque_current.get("obsolescence_at")
    else:
        today = _date.today().isoformat()
        in2y = (_date.today() + timedelta(days=730)).isoformat()
        risque_display_saisie = today
        risque_display_obsolescence = in2y

    # Charger les snapshots KYC_Client_Risque et gérer la sélection via ?kr=<id>
    risque_snapshots: list[dict] = []
    risque_selected_id: int | None = None
    try:
        rows_kr = db.execute(
            text(
                """
                SELECT id, date_saisie, date_expiration, niveau_id, experience, connaissance,
                       patrimoine, duree, contraintes, confirmation_client, commentaire
                FROM KYC_Client_Risque
                WHERE client_id = :cid
                ORDER BY date_saisie DESC, id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        risque_snapshots = [dict(r._mapping) for r in rows_kr]
    except Exception:
        risque_snapshots = []

    # Déterminer la sélection: ?kr=<id> sinon le plus récent
    try:
        sel_kr_raw = request.query_params.get("kr")
        if sel_kr_raw:
            risque_selected_id = int(sel_kr_raw)
    except Exception:
        risque_selected_id = None
    risque_history_mode = bool(request.query_params.get("kr"))
    if risque_history_mode:
        ui_focus_section = ui_focus_section or "knowledge"
        ui_focus_panel = ui_focus_panel or "knowledgePanel"
    if not risque_selected_id and risque_snapshots:
        try:
            risque_selected_id = int(risque_snapshots[0]["id"])  # le plus récent
        except Exception:
            risque_selected_id = None

    # Construire risque_snapshot pour l'affichage si une ligne sélectionnée existe
    if risque_selected_id and risque_snapshots:
        sel = next((s for s in risque_snapshots if int(s.get("id")) == int(risque_selected_id)), None)
        if sel:
            # Résoudre libellé de niveau (risque_niveau)
            try:
                rlab = db.execute(text("SELECT label FROM risque_niveau WHERE id = :id"), {"id": sel.get("niveau_id")}).fetchone()
                niveau_label = rlab[0] if rlab else None
            except Exception:
                niveau_label = None
            # Récupérer allocation correspondant à ce niveau
            allocation_nom = None
            allocation_md = None
            try:
                row_alloc = db.execute(
                    text(
                        """
                        SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom,
                               ar.texte AS allocation_texte
                        FROM allocation_risque ar
                        LEFT JOIN allocations a ON a.nom = ar.allocation_name
                        WHERE ar.risque_id = :rid
                        ORDER BY ar.date_attribution DESC, ar.id DESC
                        LIMIT 1
                        """
                    ),
                    {"rid": sel.get("niveau_id")},
                ).fetchone()
                if row_alloc:
                    allocation_nom = row_alloc[0]
                    allocation_md = row_alloc[1]
            except Exception:
                allocation_nom = None
                allocation_md = None

            # Charger série perf/vol pour l'allocation sélectionnée
            alloc_chart = None
            try:
                if allocation_nom:
                    rows_series = (
                        db.query(Allocation.date, Allocation.sicav, Allocation.volat)
                        .filter(Allocation.nom == allocation_nom)
                        .order_by(Allocation.date.asc())
                        .all()
                    )
                    labels: list[str] = []
                    sicav_vals: list[float] = []
                    vol_vals: list[float] = []
                    for d, sicav_v, vol_v in rows_series:
                        try:
                            dstr = d.strftime("%Y-%m-%d") if d else None
                        except Exception:
                            dstr = str(d)[:10] if d else None
                        if not dstr:
                            continue
                        labels.append(dstr)
                        try:
                            sicav_vals.append(float(sicav_v or 0))
                        except Exception:
                            sicav_vals.append(0.0)
                        try:
                            vol_vals.append(float(vol_v or 0))
                        except Exception:
                            vol_vals.append(0.0)
                    if labels:
                        alloc_chart = {"labels": labels, "sicav": sicav_vals, "vol": vol_vals}
            except Exception:
                alloc_chart = None

            # Convert markdown to HTML
            def _md_to_html(md: str | None) -> str | None:
                if not md:
                    return None
                try:
                    import re, html as _html
                    text = str(md)
                    text = _html.escape(text)
                    text = re.sub(r"^###\s+(.*)$", r"<h5>\1</h5>", text, flags=re.M)
                    text = re.sub(r"^##\s+(.*)$", r"<h4>\1</h4>", text, flags=re.M)
                    text = re.sub(r"^#\s+(.*)$", r"<h3>\1</h3>", text, flags=re.M)
                    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
                    lines = text.split("\n")
                    out = []
                    in_ul = False
                    for ln in lines:
                        if re.match(r"^\s*-\s+", ln):
                            if not in_ul:
                                out.append("<ul>")
                                in_ul = True
                            out.append("<li>" + re.sub(r"^\s*-\s+", "", ln) + "</li>")
                        else:
                            if in_ul:
                                out.append("</ul>")
                                in_ul = False
                            if ln.strip():
                                out.append("<p>" + ln + "</p>")
                    if in_ul:
                        out.append("</ul>")
                    return "\n".join(out)
                except Exception:
                    return md

            risque_snapshot = {
                "niveau_id": sel.get("niveau_id"),
                "niveau_label": niveau_label,
                "experience": sel.get("experience"),
                "connaissance": sel.get("connaissance"),
                "patrimoine": sel.get("patrimoine"),
                "duree": sel.get("duree"),
                "contraintes": sel.get("contraintes"),
                "confirmation_client": sel.get("confirmation_client"),
                "commentaire": sel.get("commentaire"),
                "allocation_nom": allocation_nom,
                "allocation_chart": alloc_chart,
                "allocation_html": _md_to_html(allocation_md),
            }
            # Utiliser les dates du snapshot pour l'entête
            risque_display_saisie = sel.get("date_saisie") or risque_display_saisie
            risque_display_obsolescence = sel.get("date_expiration") or risque_display_obsolescence
            # Charger les connaissances par produit reliées à ce snapshot
            try:
                rows_c = db.execute(
                    text(
                        "SELECT produit_id, produit_label, niveau_id, niveau_label FROM KYC_Client_Risque_Connaissance WHERE risque_id = :rid ORDER BY produit_label, produit_id"
                    ),
                    {"rid": int(sel.get("id"))},
                ).fetchall()
                risque_snapshot["connaissance_produits"] = [dict(r._mapping) for r in rows_c]
            except Exception:
                pass

    try:
        ref_niveau_rows = db.execute(
            text("SELECT id, libelle FROM ref_niveau_risque ORDER BY id")
        ).fetchall()
        ref_niveau_risque = [dict(row._mapping) for row in ref_niveau_rows]
    except Exception:
        ref_niveau_risque = []

    preselected_objectifs: list[dict] = []
    preselected_objectifs_ids: list[int] = []
    try:
        objectifs_rows = db.execute(
            text(
                """
                SELECT o.id AS link_id,
                       o.objectif_id,
                       o.horizon_investissement,
                       o.niveau_id,
                       o.montant,
                       o.commentaire,
                       o.date_saisie,
                       o.date_expiration,
                       ro.libelle AS libelle,
                       nr.libelle AS niveau_libelle
                FROM KYC_Client_Objectifs o
                LEFT JOIN ref_objectif ro ON ro.id = o.objectif_id
                LEFT JOIN ref_niveau_risque nr ON nr.id = o.niveau_id
                WHERE o.client_id = :cid
                ORDER BY COALESCE(o.niveau_id, 9999), ro.libelle, o.id
                """
            ),
            {"cid": client_id},
        ).fetchall()

        for row in objectifs_rows:
            data = row._mapping
            objectif_id = data.get("objectif_id")
            if objectif_id is None:
                continue
            objectif_id = int(objectif_id)
            libelle = _safe_text(data.get("libelle")) or f"Objectif {objectif_id}"
            niveau_source = data.get("niveau_id")
            niveau_value = int(niveau_source) if niveau_source is not None else None
            preselected_objectifs.append(
                {
                    "id": objectif_id,
                    "libelle": libelle,
                    "link_id": data.get("link_id"),
                    "horizon_investissement": _safe_text(data.get("horizon_investissement")),
                    "niveau_id": niveau_value,
                    "montant": float(data.get("montant") or 0.0),
                    "niveau_libelle": _safe_text(data.get("niveau_libelle")),
                    "commentaire": _safe_text(data.get("commentaire")),
                    "date_saisie": _fmt_date(data.get("date_saisie")),
                    "date_expiration": _fmt_date(data.get("date_expiration")),
                }
            )
            preselected_objectifs_ids.append(objectif_id)
    except Exception as exc:
        preselected_objectifs = []
        preselected_objectifs_ids = []
        logger.debug(
            "Dashboard KYC client: erreur lecture objectifs: %s", exc, exc_info=True
        )

    try:
        ref_objectifs_rows = db.execute(
            text("SELECT id, libelle FROM ref_objectif ORDER BY libelle")
        ).fetchall()
        ref_objectifs = [
            {"id": int(row.id), "libelle": _safe_text(row.libelle)}
            for row in ref_objectifs_rows
        ]
    except Exception as exc:
        ref_objectifs = []
        logger.debug(
            "Dashboard KYC client: erreur lecture ref_objectif: %s", exc, exc_info=True
        )

    try:
        ref_type_adresse_rows = db.execute(
            text("SELECT id, libelle FROM ref_type_adresse ORDER BY libelle")
        ).fetchall()
        ref_type_adresse = [dict(row._mapping) for row in ref_type_adresse_rows]
    except Exception:
        ref_type_adresse = []

    try:
        ref_situation_rows = db.execute(
            text("SELECT id, libelle FROM ref_situation_matrimoniale ORDER BY libelle")
        ).fetchall()
        ref_situation_matrimoniale = [dict(row._mapping) for row in ref_situation_rows]
    except Exception:
        ref_situation_matrimoniale = []

    try:
        ref_convention_rows = db.execute(
            text("SELECT id, libelle FROM ref_situation_matrimoniale_convention ORDER BY libelle")
        ).fetchall()
        ref_situation_convention = [dict(row._mapping) for row in ref_convention_rows]
    except Exception:
        ref_situation_convention = []

    try:
        ref_secteur_rows = db.execute(
            text("SELECT id, libelle FROM ref_profession_secteur ORDER BY libelle")
        ).fetchall()
        ref_profession_secteur = [dict(row._mapping) for row in ref_secteur_rows]
    except Exception:
        ref_profession_secteur = []

    try:
        ref_statut_rows = db.execute(
            text("SELECT id, libelle FROM ref_statut_professionnel ORDER BY libelle")
        ).fetchall()
        ref_statut_professionnel = [dict(row._mapping) for row in ref_statut_rows]
    except Exception:
        ref_statut_professionnel = []

    # --- LCBFT: lecture du dernier questionnaire + options vigilance ---
    lcbft_current: dict | None = None
    lcbft_vigilance_ids: list[int] = []
    lcbft_vigilance_options: list[dict] = []
    lcbft_ppe_options: list[dict] = []
    lcbft_operation_types: list[dict] = []
    lcbft_operation_selected_ids: list[int] = []
    lcbft_revenue_total: float | None = None
    lcbft_revenue_tranches: list[dict] = []
    lcbft_revenue_tranche_id: int | None = None
    lcbft_patrimoine_total: float | None = None
    lcbft_patrimoine_tranches: list[dict] = []
    lcbft_patrimoine_tranche_id: int | None = None
    lcbft_raison_options: list[dict] = []
    lcbft_raison_selected_ids: list[int] = []
    lcbft_raison_forced_ids: set[int] = set()
    lcbft_raison_disabled_ids: set[int] = set()
    try:
        rows = db.execute(text("SELECT id, code, label FROM LCBFT_vigilance_option ORDER BY label")).fetchall()
        lcbft_vigilance_options = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_vigilance_options = []
    try:
        rows = db.execute(text("SELECT id, code, lib FROM LCBFT_ref_operation_type ORDER BY lib")).fetchall()
        lcbft_operation_types = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_operation_types = []
    try:
        row = db.execute(
            text("SELECT * FROM LCBFT_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"),
            {"r": str(client_id)},
        ).fetchone()
        if row:
            m = row._mapping
            qid = m.get("id")
            lcbft_current = {k: m.get(k) for k in m.keys()}
            try:
                ids = db.execute(text("SELECT option_id FROM LCBFT_questionnaire_vigilance WHERE questionnaire_id = :q"), {"q": qid}).fetchall()
                lcbft_vigilance_ids = [int(x[0]) for x in ids]
            except Exception:
                lcbft_vigilance_ids = []
            if qid is not None:
                try:
                    rows = db.execute(
                        text("SELECT operation_type_id FROM LCBFT_questionnaire_operation_type WHERE questionnaire_id = :q"),
                        {"q": qid},
                    ).fetchall()
                    lcbft_operation_selected_ids = [int(r[0]) for r in rows if r and r[0] is not None]
                except Exception:
                    lcbft_operation_selected_ids = []
        if lcbft_current and not lcbft_operation_selected_ids:
            raw_ops = None
            for key in ("operation_type_ids", "operation_types", "operation_type_codes"):
                val = lcbft_current.get(key)
                if val not in (None, "", []):
                    raw_ops = val
                    break
            if raw_ops is not None:
                if isinstance(raw_ops, (list, tuple, set)):
                    candidates = raw_ops
                else:
                    candidates = str(raw_ops).replace(";", ",").split(",")
                for item in candidates:
                    try:
                        lcbft_operation_selected_ids.append(int(str(item).strip()))
                    except (TypeError, ValueError):
                        continue
    except Exception:
        lcbft_current = None
    try:
        total_row = db.execute(
            text("SELECT SUM(montant_annuel) FROM KYC_Client_Revenus WHERE client_id = :cid"),
            {"cid": client_id},
        ).scalar()
        if total_row not in (None, ''):
            lcbft_revenue_total = float(total_row)
    except Exception:
        lcbft_revenue_total = None
    try:
        tranche_rows = db.execute(
            text("SELECT id, lib, min_eur, max_eur FROM LCBFT_ref_tranche_revenu ORDER BY COALESCE(min_eur, 0) ASC")
        ).fetchall()
        converted_tranches: list[dict] = []
        for row in tranche_rows:
            data = dict(row._mapping)
            raw_min = data.get("min_eur")
            raw_max = data.get("max_eur")
            min_eff = float(raw_min) if raw_min is not None else None
            max_eff = float(raw_max) if raw_max is not None else None
            data["min_effective"] = min_eff
            data["max_effective"] = max_eff
            converted_tranches.append(data)
        lcbft_revenue_tranches = converted_tranches
        if lcbft_revenue_total is not None:
            total_val = lcbft_revenue_total
            for tr in lcbft_revenue_tranches:
                min_eff = tr.get("min_effective")
                max_eff = tr.get("max_effective")
                meets_min = (min_eff is None) or (total_val >= min_eff)
                meets_max = (max_eff is None) or (total_val <= max_eff)
                if meets_min and meets_max:
                    lcbft_revenue_tranche_id = tr.get("id")
                    break
    except Exception:
        lcbft_revenue_tranches = []
    try:
        patrimoine_row = db.execute(
            text("SELECT SUM(valeur) FROM KYC_Client_Actif WHERE client_id = :cid"),
            {"cid": client_id},
        ).scalar()
        if patrimoine_row not in (None, ''):
            lcbft_patrimoine_total = float(patrimoine_row)
    except Exception:
        lcbft_patrimoine_total = None
    try:
        patrimoine_tr_rows = db.execute(
            text("SELECT id, lib, min_eur, max_eur FROM LCBFT_ref_tranche_patrimoine ORDER BY COALESCE(min_eur, 0) ASC")
        ).fetchall()
        converted_patr_tranches: list[dict] = []
        for row in patrimoine_tr_rows:
            data = dict(row._mapping)
            raw_min = data.get("min_eur")
            raw_max = data.get("max_eur")
            min_eff = float(raw_min) if raw_min is not None else None
            max_eff = float(raw_max) if raw_max is not None else None
            data["min_effective"] = min_eff
            data["max_effective"] = max_eff
            converted_patr_tranches.append(data)
        lcbft_patrimoine_tranches = converted_patr_tranches
        if lcbft_patrimoine_total is not None:
            total_val = lcbft_patrimoine_total
            for tr in lcbft_patrimoine_tranches:
                min_eff = tr.get("min_effective")
                max_eff = tr.get("max_effective")
                meets_min = (min_eff is None) or (total_val >= min_eff)
                meets_max = (max_eff is None) or (total_val <= max_eff)
                if meets_min and meets_max:
                    lcbft_patrimoine_tranche_id = tr.get("id")
                    break
    except Exception:
        lcbft_patrimoine_tranches = []
    try:
        raison_rows = db.execute(
            text("SELECT id, lib FROM LCBFT_ref_raison_vigilance ORDER BY lib"),
        ).fetchall()
        lcbft_raison_options = [dict(r._mapping) for r in raison_rows]
        if qid is not None:
            rows = db.execute(
                text(
                    """
                    SELECT DISTINCT rv.raison_id
                    FROM LCBFT_operation op
                    JOIN LCBFT_operation_raison_vigilance rv ON rv.operation_id = op.id
                    WHERE op.questionnaire_id = :qid
                    """
                ),
                {"qid": qid},
            ).fetchall()
            lcbft_raison_selected_ids = [int(r[0]) for r in rows if r and r[0] is not None]
    except Exception:
        lcbft_raison_options = []
    try:
        invest_total = float(lcbft_invest_total) if lcbft_invest_total not in (None, "") else None
    except Exception:
        invest_total = None
    patrimoine_total = lcbft_patrimoine_total if isinstance(lcbft_patrimoine_total, (int, float)) else None
    # Montant > 100k (raison id 1) => toujours inclus, lecture seule
    if invest_total is not None and invest_total > 100000:
        lcbft_raison_forced_ids.add(1)
    if invest_total is not None and patrimoine_total not in (None, 0) and invest_total > float(patrimoine_total):
        lcbft_raison_forced_ids.add(2)
    for rid in lcbft_raison_forced_ids:
        if rid not in lcbft_raison_selected_ids:
            lcbft_raison_selected_ids.append(rid)
    # PPE options (fonctions) for selects
    try:
        rows = db.execute(text("SELECT id, lib FROM LCBFT_ref_ppe_fonction ORDER BY lib"), {}).fetchall()
        lcbft_ppe_options = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_ppe_options = []

    # FATCA: contrats disponibles et infos client (pays fiscal, NIF)
    fatca_contracts: list[dict] = []
    fatca_client_country: str | None = None
    fatca_client_nif: str | None = None
    fatca_today = _date.today().isoformat()

    # --- DER data for Conformité modal (client detail) ---
    DER_courtier = None
    DER_statut_social = None
    DER_courtier_garanties_normes: list[dict] = []
    lm_remunerations: list[dict] = []
    DER_courtier_activite: list[dict] = []
    DER_sql_activite: str | None = None
    DER_sql_mediation: str | None = None
    DER_sql_params_activite: dict | None = {":cid": None}
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            DER_courtier = dict(row._mapping)
            # Resolve statut social via reference table if value is an id
            ss_label = None
            raw_val = None
            for key in ("statut_social", "statut", "statut_soc", "statut_social_lib"):
                if key in DER_courtier and DER_courtier.get(key) not in (None, ""):
                    raw_val = DER_courtier.get(key)
                    break
            if raw_val is not None:
                try:
                    ss_id = int(str(raw_val).strip())
                    r2 = db.execute(text("SELECT lib FROM DER_statut_social WHERE id = :i"), {"i": ss_id}).fetchone()
                    if r2 and r2[0]:
                        ss_label = r2[0]
                except Exception:
                    ss_label = str(raw_val)
            DER_statut_social = {"lib": ss_label} if ss_label is not None else None
    except Exception:
        DER_courtier = None
        DER_statut_social = None

    # Lettre de mission — Rémunération (point H): requête directe, schema connu
    try:
        cid = DER_courtier.get("id") if DER_courtier else None
        if cid is not None:
            q = text(
                """
                SELECT type, montant, pourcentage
                FROM DER_courtier_mode_facturation
                WHERE courtier_id = :cid
                """
            )
            rows = db.execute(q, {"cid": cid}).fetchall()
        else:
            rows = db.execute(text("SELECT type, montant, pourcentage FROM DER_courtier_mode_facturation"))
        # Build ref mode map for human-friendly labels
        ref_modes = []
        try:
            ref_modes = db.execute(text("SELECT id, mode FROM DER_courtier_mode_facturation_ref")).fetchall()
        except Exception:
            ref_modes = []
        import unicodedata, re
        def _normtxt(s: str | None) -> str:
            if s is None:
                return ""
            t = unicodedata.normalize('NFKD', str(s))
            t = ''.join(ch for ch in t if not unicodedata.combining(ch))
            t = t.lower().replace("_", " ").replace("'", " ")
            t = re.sub(r"[^a-z0-9]+", " ", t)
            return re.sub(r"\s+", " ", t).strip()
        ref_map = {}
        for rm in ref_modes:
            try:
                key = _normtxt(rm._mapping.get('mode') if hasattr(rm, '_mapping') else rm[1])
                ref_map[key] = (rm._mapping.get('mode') if hasattr(rm, '_mapping') else rm[1])
            except Exception:
                continue
        def _label_for_type(t: str | None) -> str | None:
            n = _normtxt(t)
            if not n:
                return None
            # Heuristics
            if 'honor' in n:
                return ref_map.get('honoraires') or 'Honoraires'
            if 'entree' in n:
                return ref_map.get('frais d entree') or "Frais d'entrée"
            if 'gestion' in n:
                return ref_map.get('frais de gestion') or 'Frais de gestion'
            # direct map
            return ref_map.get(n)
        for r in rows:
            m = r._mapping if hasattr(r, "_mapping") else {"type": r[0], "montant": r[1], "pourcentage": r[2]}
            t = m.get("type")
            lm_remunerations.append({
                "type": t,
                "mode": _label_for_type(t) or (str(t).replace('_', ' ').title() if t else None),
                "montant": m.get("montant"),
                "pourcentage": m.get("pourcentage"),
            })
        # Tri: HONORAIRES d'abord, puis alpha
        import unicodedata
        def _normv(v: str) -> str:
            s2 = unicodedata.normalize('NFKD', v or '')
            s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
            return s2.upper()
        lm_remunerations.sort(key=lambda r: (0 if 'HONOR' in _normv(str(r.get('type') or '')) else 1, _normv(str(r.get('type') or ''))))
    except Exception:
        lm_remunerations = []
    # SQL affichée pour le bloc Médiation (point 8)
    try:
        DER_sql_mediation = "SELECT centre_mediation, mediators, mail_mediators FROM DER_courtier ORDER BY id LIMIT 1"
    except Exception:
        DER_sql_mediation = None

    try:
        gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
        if gar_table:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(n): n for n in colnames }
            c_type = inv.get('type_garantie') or inv.get('type') or inv.get('garantie') or (colnames[0] if colnames else None)
            c_ias = inv.get('ias') or 'IAS'
            c_iobsp = inv.get('iobsp') or 'IOBSP'
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            for r in rows:
                m = r._mapping
                DER_courtier_garanties_normes.append({
                    'type_garantie': m.get(c_type) if c_type else None,
                    'IAS': m.get(c_ias),
                    'IOBSP': m.get(c_iobsp),
                })
    except Exception:
        DER_courtier_garanties_normes = []

    # Load DER activities (courtier_id/activite_id) to feed section 2 — joined with reference for labels
    DER_activites: list[dict] = []
    try:
        cid_val = DER_courtier.get("id") if DER_courtier else None
        DER_sql_params_activite = {":cid": cid_val}
        rows_der_act = []
        if cid_val is not None:
            rows_der_act = db.execute(
                text(
                    """
                    SELECT a.activite_id, a.statut,
                           r.code, r.libelle, r.domaine, r.sous_categorie, r.description
                    FROM DER_courtier_activite a
                    JOIN DER_courtier_activite_ref r ON r.id = a.activite_id
                    WHERE a.courtier_id = :cid
                    ORDER BY r.domaine, r.libelle
                    """
                ),
                {"cid": cid_val},
            ).fetchall()
        for rr in rows_der_act or []:
            mm = rr._mapping
            DER_activites.append({
                "activite_id": mm.get("activite_id"),
                "statut": mm.get("statut"),
                "code": mm.get("code"),
                "libelle": mm.get("libelle"),
                "domaine": mm.get("domaine"),
                "sous_categorie": mm.get("sous_categorie"),
                "description": mm.get("description"),
            })
    except Exception:
        DER_activites = []
        DER_sql_params_activite = {":cid": None}

    # Ensure DER variables exist in this path
    try:
        DER_activites
    except NameError:
        DER_activites = []
    try:
        DER_sql_activite
    except NameError:
        DER_sql_activite = None
    try:
        DER_sql_mediation
    except NameError:
        DER_sql_mediation = None
    try:
        DER_sql_params_activite
    except NameError:
        DER_sql_params_activite = {":cid": None}

    # DER: activités + SQL (section 2) et SQL médiation (section 8)
    DER_courtier_activite: list[dict] = []
    DER_sql_activite: str | None = None
    DER_sql_mediation: str | None = None
    try:
        DER_sql_mediation = "SELECT centre_mediation, mediators, mail_mediators FROM DER_courtier ORDER BY id LIMIT 1"
    except Exception:
        DER_sql_mediation = None
    try:
        act_table = _resolve_table_name(db, ["DER_courtier_activite"]) or None
        ref_table = _resolve_table_name(db, ["DER_courtier_activite_ref"]) or None
        if act_table:
            id_col, fk_ref_col, fk_cour_col, _c1, _c2 = _courtier_activite_columns(db, act_table)
            where_clause = ""
            params: dict[str, object] = {}
            if fk_cour_col and DER_courtier and DER_courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = DER_courtier.get("id")
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {act_table}{where_clause}"), params).fetchall()
            # Map ref id -> libellé
            ref_map: dict[str, str] = {}
            if ref_table:
                try:
                    refs = _fetch_ref_list(db, [ref_table]) or []
                    for r in refs:
                        ref_map[str(r.get("id"))] = str(r.get("libelle") or "")
                except Exception:
                    ref_map = {}
            # Colonnes facultatives
            cols = _sqlite_table_columns(db, act_table)
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(c.get("name") or ""): c.get("name") for c in cols }
            c_statut = inv.get('statut')
            c_domaine = inv.get('domaine') or inv.get('domaine_exercice') or inv.get('domaineactivite')
            for r in rows:
                m = r._mapping
                ref_id_val = m.get(fk_ref_col) if fk_ref_col else None
                lib = ref_map.get(str(ref_id_val)) if ref_id_val is not None else None
                DER_courtier_activite.append({
                    'libelle': lib,
                    'statut': m.get(c_statut) if c_statut else None,
                    'domaine': m.get(c_domaine) if c_domaine else None,
                })
            # SQL join affichable
            try:
                if ref_table and fk_ref_col:
                    ref_cols = _sqlite_table_columns(db, ref_table)
                    ref_pk = next((c.get("name") for c in ref_cols if c.get("pk")), None) or 'id'
                    name_map = {_norm(c.get("name") or ''): c.get("name") for c in ref_cols}
                    ref_label_col = name_map.get('libelle') or name_map.get('label') or name_map.get('nom') or name_map.get('name') or name_map.get('intitule') or name_map.get('intitulé') or 'libelle'
                    where_sql = f" WHERE a.{fk_cour_col} = :cid" if (fk_cour_col and where_clause) else ""
                    DER_sql_activite = f"SELECT a.*, r.{ref_label_col} AS ref_label FROM {act_table} a LEFT JOIN {ref_table} r ON r.{ref_pk} = a.{fk_ref_col}{where_sql} ORDER BY r.{ref_label_col}"
                else:
                    DER_sql_activite = f"SELECT * FROM {act_table}"
            except Exception:
                DER_sql_activite = f"SELECT * FROM {act_table}"
    except Exception:
        DER_courtier_activite = []
        DER_sql_activite = None

    # DER: activités + SQL affichage (section 2) et SQL médiation (section 8)
    DER_courtier_activite: list[dict] = []
    DER_sql_activite: str | None = None
    DER_sql_mediation: str | None = None
    try:
        DER_sql_mediation = "SELECT centre_mediation, mediators, mail_mediators FROM DER_courtier ORDER BY id LIMIT 1"
    except Exception:
        DER_sql_mediation = None
    try:
        act_table = _resolve_table_name(db, ["DER_courtier_activite"]) or None
        ref_table = _resolve_table_name(db, ["DER_courtier_activite_ref"]) or None
        if act_table:
            id_col, fk_ref_col, fk_cour_col, _c1, _c2 = _courtier_activite_columns(db, act_table)
            where_clause = ""
            params: dict[str, object] = {}
            if fk_cour_col and DER_courtier and DER_courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = DER_courtier.get("id")
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {act_table}{where_clause}"), params).fetchall()
            # Map ref id -> libellé
            ref_map: dict[str, str] = {}
            if ref_table:
                try:
                    refs = _fetch_ref_list(db, [ref_table]) or []
                    for r in refs:
                        ref_map[str(r.get("id"))] = str(r.get("libelle") or "")
                except Exception:
                    ref_map = {}
            # Detect facultatives
            cols = _sqlite_table_columns(db, act_table)
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(c.get("name") or ""): c.get("name") for c in cols }
            c_statut = inv.get('statut')
            c_domaine = inv.get('domaine') or inv.get('domaine_exercice') or inv.get('domaineactivite')
            for r in rows:
                m = r._mapping
                ref_id_val = m.get(fk_ref_col) if fk_ref_col else None
                lib = ref_map.get(str(ref_id_val)) if ref_id_val is not None else None
                DER_courtier_activite.append({
                    'libelle': lib,
                    'statut': m.get(c_statut) if c_statut else None,
                    'domaine': m.get(c_domaine) if c_domaine else None,
                })
            # SQL join affichable
            try:
                if ref_table and fk_ref_col:
                    ref_cols = _sqlite_table_columns(db, ref_table)
                    ref_pk = next((c.get("name") for c in ref_cols if c.get("pk")), None) or 'id'
                    name_map = {_norm(c.get("name") or ''): c.get("name") for c in ref_cols}
                    ref_label_col = name_map.get('libelle') or name_map.get('label') or name_map.get('nom') or name_map.get('name') or name_map.get('intitule') or name_map.get('intitulé') or 'libelle'
                    where_sql = f" WHERE a.{fk_cour_col} = :cid" if (fk_cour_col and where_clause) else ""
                    DER_sql_activite = f"SELECT a.*, r.{ref_label_col} AS ref_label FROM {act_table} a LEFT JOIN {ref_table} r ON r.{ref_pk} = a.{fk_ref_col}{where_sql} ORDER BY r.{ref_label_col}"
                else:
                    DER_sql_activite = f"SELECT * FROM {act_table}"
            except Exception:
                DER_sql_activite = f"SELECT * FROM {act_table}"
    except Exception:
        DER_courtier_activite = []
        DER_sql_activite = None

    # Activités du courtier (pour DER §2)
    try:
        act_table = _resolve_table_name(db, ["DER_courtier_activite"]) or None
        ref_table = _resolve_table_name(db, ["DER_courtier_activite_ref"]) or None
        if act_table:
            id_col, fk_ref_col, fk_cour_col, _c1, _c2 = _courtier_activite_columns(db, act_table)
            # Charger activités
            where_clause = ""
            params: dict[str, object] = {}
            if fk_cour_col and DER_courtier and DER_courtier.get("id") is not None:
                where_clause = f" WHERE {fk_cour_col} = :cid"
                params["cid"] = DER_courtier.get("id")
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {act_table}{where_clause}"), params).fetchall()
            # Map ref id -> libellé
            ref_map: dict[str, str] = {}
            if ref_table:
                try:
                    refs = _fetch_ref_list(db, [ref_table]) or []
                    for r in refs:
                        ref_map[str(r.get("id"))] = str(r.get("libelle") or "")
                except Exception:
                    ref_map = {}
            # Detect colonnes statut/domaine si présentes
            cols = _sqlite_table_columns(db, act_table)
            norm = lambda s: ''.join(ch for ch in __import__('unicodedata').normalize('NFKD', (s or '')) if not __import__('unicodedata').combining(ch)).lower()
            inv = { norm(c.get("name") or ""): c.get("name") for c in cols }
            c_statut = inv.get('statut')
            c_domaine = inv.get('domaine') or inv.get('domaine_exercice') or inv.get('domaineactivite')
            for r in rows:
                m = r._mapping
                ref_id_val = m.get(fk_ref_col) if fk_ref_col else None
                lib = ref_map.get(str(ref_id_val)) if ref_id_val is not None else None
                DER_courtier_activite.append({
                    'libelle': lib,
                    'statut': m.get(c_statut) if c_statut else None,
                    'domaine': m.get(c_domaine) if c_domaine else None,
                })
            # Construire une requête SQL affichable (JOIN explicite si possible)
            try:
                if ref_table and fk_ref_col:
                    # Détection colonnes id/label de la ref
                    ref_cols = _sqlite_table_columns(db, ref_table)
                    ref_pk = next((c.get("name") for c in ref_cols if c.get("pk")), None) or 'id'
                    # label heuristique
                    import unicodedata
                    def _norm(s: str) -> str:
                        s2 = unicodedata.normalize('NFKD', s or '')
                        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                        return s2.lower()
                    name_map = {_norm(c.get("name") or ''): c.get("name") for c in ref_cols}
                    ref_label_col = name_map.get('libelle') or name_map.get('label') or name_map.get('nom') or name_map.get('name') or name_map.get('intitule') or name_map.get('intitulé') or 'libelle'
                    where_sql = f" WHERE a.{fk_cour_col} = :cid" if (fk_cour_col and where_clause) else ""
                    DER_sql_activite = f"SELECT a.*, r.{ref_label_col} AS ref_label FROM {act_table} a LEFT JOIN {ref_table} r ON r.{ref_pk} = a.{fk_ref_col}{where_sql} ORDER BY r.{ref_label_col}"
                else:
                    DER_sql_activite = f"SELECT * FROM {act_table}"
            except Exception:
                DER_sql_activite = f"SELECT * FROM {act_table}"
    except Exception:
        DER_courtier_activite = []
        DER_sql_activite = None

    # --- DER data for Conformité modal (client detail) ---
    DER_courtier = None
    DER_statut_social = None
    DER_courtier_garanties_normes: list[dict] = []
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            DER_courtier = dict(row._mapping)
            # Try to expose statut social label directly if present
            ss = None
            for key in ("statut_social", "statut", "statut_soc", "statut_social_lib"):
                if key in DER_courtier and DER_courtier.get(key):
                    ss = DER_courtier.get(key)
                    break
            DER_statut_social = {"lib": ss} if ss is not None else None
    except Exception:
        DER_courtier = None
        DER_statut_social = None

    try:
        gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
        if gar_table:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(n): n for n in colnames }
            c_type = inv.get('type_garantie') or inv.get('type') or inv.get('garantie') or (colnames[0] if colnames else None)
            c_ias = inv.get('ias') or 'IAS'
            c_iobsp = inv.get('iobsp') or 'IOBSP'
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            for r in rows:
                m = r._mapping
                DER_courtier_garanties_normes.append({
                    'type_garantie': m.get(c_type) if c_type else None,
                    'IAS': m.get(c_ias),
                    'IOBSP': m.get(c_iobsp),
                })
    except Exception:
        DER_courtier_garanties_normes = []

    # --- DER data for Conformité modal (client detail) ---
    DER_courtier = None
    DER_statut_social = None
    DER_courtier_garanties_normes: list[dict] = []
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            DER_courtier = dict(row._mapping)
            # Expose statut social label if present as plain text
            ss = None
            for key in ("statut_social", "statut", "statut_soc", "statut_social_lib"):
                if key in DER_courtier and DER_courtier.get(key):
                    ss = DER_courtier.get(key)
                    break
            DER_statut_social = {"lib": ss} if ss is not None else None
    except Exception:
        DER_courtier = None
        DER_statut_social = None

    try:
        gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
        if gar_table:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(n): n for n in colnames }
            c_type = inv.get('type_garantie') or inv.get('type') or inv.get('garantie') or (colnames[0] if colnames else None)
            c_ias = inv.get('ias') or 'IAS'
            c_iobsp = inv.get('iobsp') or 'IOBSP'
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            for r in rows:
                m = r._mapping
                DER_courtier_garanties_normes.append({
                    'type_garantie': m.get(c_type) if c_type else None,
                    'IAS': m.get(c_ias),
                    'IOBSP': m.get(c_iobsp),
                })
    except Exception:
        DER_courtier_garanties_normes = []

    # --- DER data for Conformité modal (client detail) ---
    DER_courtier = None
    DER_statut_social = None
    DER_courtier_garanties_normes: list[dict] = []
    try:
        row = db.execute(text("SELECT * FROM DER_courtier ORDER BY id LIMIT 1")).fetchone()
        if row:
            DER_courtier = dict(row._mapping)
            # Try to expose statut social label directly if present as text
            ss = None
            for key in ("statut_social", "statut", "statut_soc", "statut_social_lib"):
                if key in DER_courtier and DER_courtier.get(key):
                    ss = DER_courtier.get(key)
                    break
            DER_statut_social = {"lib": ss} if ss is not None else None
    except Exception:
        DER_courtier = None
        DER_statut_social = None

    # Load guarantees norms (robust to column names)
    try:
        gar_table = _resolve_table_name(db, ["DER_courtier_garanties_normes"])
        if gar_table:
            cols = _sqlite_table_columns(db, gar_table)
            colnames = [c.get("name") for c in cols]
            import unicodedata
            def _norm(s: str) -> str:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                return s2.lower()
            inv = { _norm(n): n for n in colnames }
            c_type = inv.get('type_garantie') or inv.get('type') or inv.get('garantie') or (colnames[0] if colnames else None)
            c_ias = inv.get('ias') or 'IAS'
            c_iobsp = inv.get('iobsp') or 'IOBSP'
            pk_cols = [c.get("name") for c in cols if c.get("pk")]
            id_col = pk_cols[0] if pk_cols else ("id" if "id" in colnames else None)
            select_pk = f"{id_col} AS __pk, " if id_col else ""
            rows = db.execute(text(f"SELECT {select_pk}* FROM {gar_table}")).fetchall()
            for r in rows:
                m = r._mapping
                DER_courtier_garanties_normes.append({
                    'type_garantie': m.get(c_type) if c_type else None,
                    'IAS': m.get(c_ias),
                    'IOBSP': m.get(c_iobsp),
                })
    except Exception:
        DER_courtier_garanties_normes = []
    try:
        rows = db.execute(
            text(
                """
                SELECT g.id, g.nom_contrat, COALESCE(s.nom, '') AS societe_nom
                FROM mariadb_affaires_generique g
                LEFT JOIN mariadb_societe s ON s.id = g.id_societe
                WHERE COALESCE(g.actif, 1) = 1
                ORDER BY s.nom, g.nom_contrat
                """
            )
        ).fetchall()
        fatca_contracts = [dict(r._mapping) for r in rows]
    except Exception:
        fatca_contracts = []
    # Load saved FATCA for latest questionnaire
    fatca_saved: dict | None = None
    try:
        if lcbft_current and lcbft_current.get("id"):
            qid = lcbft_current.get("id")
            row = db.execute(
                text("SELECT contrat_id, societe_nom, date_operation, pays_residence, nif FROM LCBFT_fatca WHERE questionnaire_id = :q"),
                {"q": qid},
            ).fetchone()
            if row:
                fatca_saved = dict(row._mapping)
    except Exception:
        fatca_saved = None
    # Pays de résidence fiscale: priorité à l'adresse KYC principale, sinon dernière adresse,
    # sinon tentatives depuis mariadb_clients (colonnes variables selon environnement).
    try:
        primary_addr = None
        if 'adresses' in locals() and adresses:
            for a in adresses:
                if a.get('is_primary'):
                    primary_addr = a
                    break
            if not primary_addr:
                primary_addr = adresses[0]
        if primary_addr:
            fatca_client_country = primary_addr.get('pays') or ''
    except Exception:
        pass
    try:
        crow = db.execute(text("SELECT * FROM mariadb_clients WHERE id = :cid"), {"cid": client_id}).fetchone()
        if crow:
            m = crow._mapping
            # Recherche souple des clés potentielles
            lower_map = { (k.lower() if isinstance(k, str) else k): v for k, v in m.items() }
            if not fatca_client_country:
                for key in ("adresse_pays", "pays_fiscal", "residence_fiscale", "pays"):
                    if key in lower_map and lower_map.get(key):
                        fatca_client_country = lower_map.get(key) or ''
                        break
            for key in ("nif", "num_fiscal", "numero_fiscal", "tin"):
                if key in lower_map and lower_map.get(key):
                    fatca_client_nif = lower_map.get(key) or ''
                    break
    except Exception:
        pass

    # LCBFT – montant total objectifs et détail (pour KYC)
    try:
        inv = db.execute(text("SELECT COALESCE(SUM(montant),0) FROM KYC_Client_Objectifs WHERE client_id = :cid"), {"cid": client_id}).scalar()
        lcbft_invest_total = float(inv or 0.0)
    except Exception:
        lcbft_invest_total = 0.0
    try:
        rows = db.execute(
            text(
                """
                SELECT id, objectif_id, horizon_investissement, niveau_id, commentaire,
                       date_saisie, date_expiration, montant
                FROM KYC_Client_Objectifs
                WHERE client_id = :cid
                ORDER BY (date_saisie IS NULL), date_saisie DESC, id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        lcbft_objectifs = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_objectifs = []

    nb_contrats_actifs = 0
    nb_supports_references = 0
    try:
        nb_contrats_actifs = (
            db.query(func.count(Affaire.id))
            .filter(Affaire.id_personne == client_id)
            .filter(Affaire.date_cle.is_(None))
            .scalar()
        ) or 0
        affaire_ids = [
            rid for (rid,) in db.query(Affaire.id).filter(Affaire.id_personne == client_id).all()
        ]
        if affaire_ids:
            nb_supports_references = (
                db.query(func.count(func.distinct(HistoriqueSupport.id_support)))
                .filter(HistoriqueSupport.id_source.in_(affaire_ids))
                .filter(
                    or_(
                        HistoriqueSupport.valo > 0,
                        HistoriqueSupport.nbuc > 0,
                    )
                )
                .scalar()
            ) or 0
            if not nb_supports_references:
                nb_supports_references = (
                    db.query(func.count(func.distinct(HistoriqueSupport.id_support)))
                    .filter(HistoriqueSupport.id_source.in_(affaire_ids))
                    .scalar()
                ) or 0
    except Exception:
        nb_contrats_actifs = 0
        nb_supports_references = 0

    return templates.TemplateResponse(
        "dashboard_client_kyc.html",
        {
            "request": request,
            "client": client,
            "client_id": client_id,
            "ui_focus_section": ui_focus_section,
            "ui_focus_panel": ui_focus_panel,
            "synthese_push_action": synthese_push_action,
            "synthese_push_date": synth_today,
            "synthese_totaux": {
                "actif": float(actifs_total),
                "passif": float(passifs_total),
                "revenus": float(revenus_total),
                "charges": float(charges_total),
            },
            "synthese_totaux_str": {
                "actif": _fmt_amount(actifs_total),
                "passif": _fmt_amount(passifs_total),
                "revenus": _fmt_amount(revenus_total),
                "charges": _fmt_amount(charges_total),
            },
            "kyc_actifs": actifs,
            "kyc_actifs_total": actifs_total_str,
            "actifs_total_value": float(actifs_total),
            "kyc_passifs": passifs,
            "kyc_passifs_total": passifs_total_str,
            "kyc_revenus": revenus,
            "kyc_revenus_total": revenus_total_str,
            "kyc_charges": charges,
            "kyc_charges_total": charges_total_str,
            "kyc_adresses": adresses,
            "kyc_situations_matrimoniales": situations_matrimoniales,
            "kyc_situations_professionnelles": locals().get("situations_professionnelles", []),
            "kyc_etat_civil": etat_civil_row,
            "etat_success": etat_success,
            "etat_error": etat_error,
            "adresse_success": adresse_success,
            "adresse_error": adresse_error,
            "matrimonial_success": matrimonial_success,
            "matrimonial_error": matrimonial_error,
            "professionnel_success": professionnel_success,
            "professionnel_error": professionnel_error,
            "passif_success": passif_success,
            "passif_error": passif_error,
            "revenu_success": revenu_success,
            "revenu_error": revenu_error,
            "charge_success": charge_success,
            "charge_error": charge_error,
            "actif_success": actif_success,
            "actif_error": actif_error,
            "ref_type_actif": ref_type_actif,
            "ref_type_passif": ref_type_passif,
            "ref_type_revenu": ref_type_revenu,
            "ref_type_charge": ref_type_charge,
            "kyc_contracts": kyc_contracts,
            "kyc_contrat_selected_id": kyc_contrat_selected_id,
            "contrat_success": contrat_success,
            "contrat_error": contrat_error,
            "ref_objectifs": ref_objectifs,
            "preselected_objectifs": preselected_objectifs,
            "preselected_objectifs_ids": preselected_objectifs_ids,
            "ref_type_adresse": ref_type_adresse,
            "ref_situation_matrimoniale": ref_situation_matrimoniale,
            "ref_situation_convention": ref_situation_convention,
            "ref_profession_secteur": ref_profession_secteur,
            "ref_statut_professionnel": ref_statut_professionnel,
            "active_section": active_section,
            # LCBFT
            "lcbft_current": lcbft_current,
            "lcbft_vigilance_options": lcbft_vigilance_options,
            "lcbft_vigilance_ids": lcbft_vigilance_ids,
            "lcbft_ppe_options": lcbft_ppe_options,
            "lcbft_operation_types": lcbft_operation_types,
            "lcbft_operation_selected_ids": lcbft_operation_selected_ids,
            "lcbft_revenue_total": lcbft_revenue_total,
            "lcbft_revenue_tranches": lcbft_revenue_tranches,
            "lcbft_revenue_tranche_id": lcbft_revenue_tranche_id,
            "lcbft_patrimoine_total": lcbft_patrimoine_total,
            "lcbft_patrimoine_tranches": lcbft_patrimoine_tranches,
            "lcbft_patrimoine_tranche_id": lcbft_patrimoine_tranche_id,
            "lcbft_raison_options": lcbft_raison_options,
            "lcbft_raison_selected_ids": lcbft_raison_selected_ids,
            "lcbft_raison_forced_ids": list(lcbft_raison_forced_ids),
            "lcbft_raison_disabled_ids": list(lcbft_raison_disabled_ids),
            # RISK (connaissance financière)
            "risque_opts": risque_opts,
            "risque_current": risque_current,
            "risque_connaissance_map": risque_connaissance_map,
            "risque_objectifs_ids": risque_objectifs_ids,
            "risque_decision": risque_decision,
            "risque_commentaire": risque_commentaire,
            "risque_snapshot": risque_snapshot,
            "risque_snapshots": risque_snapshots,
            "risque_selected_id": risque_selected_id,
            "risque_history_mode": risque_history_mode,
            "risque_display_saisie": risque_display_saisie,
            "risque_display_obsolescence": risque_display_obsolescence,
            # ESG
            "esg_exclusion_options": esg_exclusion_options,
            "esg_indicator_options": esg_indicator_options,
            "esg_selected_exclusions": esg_selected_exclusions,
            "esg_selected_indicators": esg_selected_indicators,
            "esg_current": esg_current,
            "esg_success": esg_success,
            "esg_error": esg_error,
            "esg_display_saisie": esg_display_saisie,
            "esg_display_obsolescence": esg_display_obsolescence,
            # FATCA block
            "fatca_contracts": fatca_contracts,
            "fatca_saved": fatca_saved,
            "fatca_client_country": fatca_client_country or "",
            "fatca_client_nif": fatca_client_nif or "",
            "fatca_today": fatca_today,
            # LCBFT objectifs (KYC)
            "lcbft_invest_total": locals().get('lcbft_invest_total', 0.0),
            "lcbft_objectifs": locals().get('lcbft_objectifs', []),
            "summary_data": {
                "actifs": synth_actifs,
                "passifs": synth_passifs,
                "revenus": synth_revenus,
                "charges": synth_charges,
            },
            "patrimoine_net_str": patrimoine_net_str,
            "budget_net_str": budget_net_str,
            "patrimoine_net_value": float(patrimoine_net) if patrimoine_net is not None else 0.0,
            "budget_net_value": float(budget_net) if budget_net is not None else 0.0,
            "nb_contrats_actifs": nb_contrats_actifs,
            "nb_supports_references": nb_supports_references,
        },
    )


# ---------------- Clients ----------------
@router.get("/clients", response_class=HTMLResponse)
def dashboard_clients(request: Request, db: Session = Depends(get_db)):
    tb_markers_visible = request.query_params.get("markers") == "1"
    total_clients = db.query(func.count(Client.id)).scalar() or 0
    group_filter_param = request.query_params.get("group_id")
    group_filter_ids: set[int] | None = None
    group_filter_label: str | None = None

    # Données SRRI pour le graphique
    srri_data = (
        db.query(Client.SRRI, func.count(Client.id).label("nb"))
        .group_by(Client.SRRI)
        .all()
    )
    srri_chart = [{"srri": s.SRRI, "nb": s.nb} for s in srri_data]

    # Utilise le service pour la liste des clients enrichie et calcule l'icône de risque (comme Affaires)
    rows = get_clients(db)
    if group_filter_param:
        try:
            gid = int(group_filter_param)
            meta = db.execute(
                text("SELECT nom, type_groupe FROM administration_groupe_detail WHERE id = :gid"),
                {"gid": gid},
            ).fetchone()
            if meta:
                meta_map = meta._mapping
                type_db = (meta_map.get("type_groupe") or "").lower()
                if type_db in ("client", "clients", "personne", "personnes"):
                    member_rows = db.execute(
                        text(
                            """
                            SELECT client_id FROM administration_groupe
                            WHERE groupe_id = :gid
                              AND client_id IS NOT NULL
                              AND (COALESCE(actif,1) != 0)
                            """
                        ),
                        {"gid": gid},
                    ).fetchall()
                    ids = {int(r[0]) for r in member_rows if r[0] is not None}
                    group_filter_ids = ids
                    group_filter_label = meta_map.get("nom") or f"Groupe #{gid}"
        except Exception:
            group_filter_ids = set()
    if group_filter_ids is not None:
        rows = [r for r in rows if getattr(r, "id", None) in group_filter_ids]

    # Référentiel RH pour associer le commercial (responsable)
    rh_entries = fetch_rh_list(db)
    rh_label_map: dict[int, str] = {}
    rh_options: list[dict[str, str | int]] = []
    for rh in rh_entries or []:
        try:
            rid_raw = rh.get("id")
            rid = int(rid_raw)
        except Exception:
            continue
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        mail = (rh.get("mail") or "").strip()
        parts = [p for p in [prenom, nom] if p]
        if parts:
            label = " ".join(parts)
        elif mail:
            label = mail
        else:
            label = f"RH #{rid}"
        rh_label_map[rid] = label
        rh_options.append({"id": rid, "label": label})
    rh_options.sort(key=lambda x: str(x["label"] or "").lower())

    def icon_for_compare(client_srri, hist_srri):
        if client_srri is None or hist_srri is None:
            return None
        try:
            c = int(client_srri)
            h = int(hist_srri)
        except Exception:
            return None
        # Mapping: Au‑dessus = 🔥, Identique = 🙏, En‑dessous = ❄️
        if c > h:
            return "fire"           # supérieur → 🔥
        if c == h:
            return "hands-praying" # identique → 🙏
        return "snowflake"         # inférieur → ❄️

    # Helpers pour parsing des filtres numériques (valeurs et pourcentages)
    def _num_token(raw: str | None) -> float | None:
        if raw is None:
            return None
        s = str(raw).strip().lower().replace(" ", "").replace(",", ".")
        m = func_re_num.match(s)  # type: ignore[name-defined]
        if not m:
            return None
        n = float(m.group(1))
        suf = m.group(2) or ""
        if suf == "k":
            n *= 1_000
        elif suf == "m":
            n *= 1_000_000
        return n

    def _as_float(raw):
        try:
            return float(raw)
        except Exception:
            return None

    func_re_num = re.compile(r"^([-+]?\d*\.?\d+)([km])?$")

    def build_sql_range(expr: str | None, column):
        """Parse a simple numeric expression into SQLAlchemy filters."""
        if not expr:
            return []
        expr = expr.replace("%", "").strip()
        # Range a-b
        range_match = re.match(r"^\s*([^\-]+)\s*-\s*([^\-]+)\s*$", expr)
        if range_match:
            a = _num_token(range_match.group(1))
            b = _num_token(range_match.group(2))
            if a is not None and b is not None:
                lo, hi = (a, b) if a <= b else (b, a)
                return [column >= lo, column <= hi]
        filters = []
        for op, num, suf in re.findall(r"(<=|>=|==|=|!=|<|>)\s*([-+]?\d*[\.,]?\d+)\s*([km]?)", expr, flags=re.I):
            n = _num_token(num + suf)
            if n is None:
                continue
            if op == ">":
                filters.append(column > n)
            elif op == ">=":
                filters.append(column >= n)
            elif op == "<":
                filters.append(column < n)
            elif op == "<=":
                filters.append(column <= n)
            elif op in ("=", "=="):
                filters.append(column == n)
            elif op == "!=":
                filters.append(column != n)
        if filters:
            return filters
        single = _num_token(expr)
        if single is not None:
            return [column == single]
        return []

    # Params filtres
    params = request.query_params
    q_nom = params.get("nom", "").strip()
    q_prenom = params.get("prenom", "").strip()
    q_resp = params.get("resp_id", "").strip()
    q_srri_calc = params.get("srri_calc", "").strip()
    q_valo = params.get("valo", "").strip()
    q_perf = params.get("perf", "").strip()
    q_vol = params.get("vol", "").strip()
    # Pagination côté serveur
    page_size = 50
    try:
        page = int(params.get("page") or 1)
    except Exception:
        page = 1
    if page < 1:
        page = 1

    def fmt_money(val):
        try:
            n = float(val)
        except Exception:
            return ""
        return f"{n:,.2f}".replace(",", " ").replace(".", ",")

    def fmt_pct(frac):
        try:
            n = float(frac)
        except Exception:
            return ""
        return f"{n*100:.2f} %"

    # Requête clients avec filtres SQL (limite/pagination)
    subquery = (
        db.query(
            HistoriquePersonne.id.label("client_id"),
            func.max(HistoriquePersonne.date).label("last_date"),
        )
        .group_by(HistoriquePersonne.id)
        .subquery()
    )
    base_query = (
        db.query(
            Client.id.label("id"),
            Client.nom,
            Client.prenom,
            Client.commercial_id,
            Client.SRRI,
            HistoriquePersonne.SRRI.label("srri_hist"),
            HistoriquePersonne.valo.label("total_valo"),
            HistoriquePersonne.perf_sicav_52.label("perf_52_sem"),
            HistoriquePersonne.volat.label("volatilite"),
        )
        .join(subquery, subquery.c.client_id == Client.id)
        .join(
            HistoriquePersonne,
            (HistoriquePersonne.id == subquery.c.client_id)
            & (HistoriquePersonne.date == subquery.c.last_date),
        )
    )

    # Filtres SQL
    if group_filter_ids is not None:
        if group_filter_ids:
            base_query = base_query.filter(Client.id.in_(group_filter_ids))
        else:
            base_query = base_query.filter(text("0=1"))
    if q_nom:
        base_query = base_query.filter(Client.nom.ilike(f"%{q_nom}%"))
    if q_prenom:
        base_query = base_query.filter(Client.prenom.ilike(f"%{q_prenom}%"))
    if q_resp:
        try:
            resp_id = int(q_resp)
            base_query = base_query.filter(Client.commercial_id == resp_id)
        except Exception:
            pass
    if q_srri_calc:
        try:
            srri_val = int(q_srri_calc)
            base_query = base_query.filter(HistoriquePersonne.SRRI == srri_val)
        except Exception:
            pass
    for cond in build_sql_range(q_valo, HistoriquePersonne.valo):
        base_query = base_query.filter(cond)
    for cond in build_sql_range(q_perf, HistoriquePersonne.perf_sicav_52):
        base_query = base_query.filter(cond)
    for cond in build_sql_range(q_vol, HistoriquePersonne.volat):
        base_query = base_query.filter(cond)

    total_filtered = base_query.count()
    base_query = base_query.order_by(HistoriquePersonne.valo.desc(), Client.nom.asc())
    offset = (page - 1) * page_size
    rows = base_query.offset(offset).limit(page_size).all()
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)

    clients = []
    for r in rows:
        comm_id = getattr(r, "commercial_id", None)
        responsable_label = None
        if comm_id is not None:
            try:
                responsable_label = rh_label_map.get(int(comm_id))
            except Exception:
                responsable_label = None
        total_valo_num = _as_float(getattr(r, "total_valo", None))
        perf_num = _as_float(getattr(r, "perf_52_sem", None))
        vol_num = _as_float(getattr(r, "volatilite", None))
        clients.append({
            "id": getattr(r, "id", None),
            "nom": getattr(r, "nom", None) or "",
            "prenom": getattr(r, "prenom", None) or "",
            "SRRI": getattr(r, "SRRI", None),
            "srri_hist": getattr(r, "srri_hist", None),
            "srri_icon": icon_for_compare(getattr(r, "SRRI", None), getattr(r, "srri_hist", None)),
            "total_valo": total_valo_num,
            "perf_52_sem": perf_num,
            "volatilite": vol_num,
            "total_valo_str": fmt_money(total_valo_num),
            "perf_52_sem_str": fmt_pct(perf_num if perf_num is not None else None),
            "volatilite_str": fmt_pct(vol_num if vol_num is not None else None),
            "commercial_id": comm_id,
            "responsable": responsable_label,
        })

    # Options de filtrage SRRI calculé (srri_hist) à partir du jeu paginé
    srri_options = sorted({c["srri_hist"] for c in clients if c["srri_hist"] is not None})

    # ---------------- Analyse financière (supports) ----------------
    finance_rh_param = request.query_params.get("finance_rh")
    finance_rh_id: int | None = None
    if finance_rh_param not in (None, ""):
        try:
            finance_rh_id = int(finance_rh_param)
        except (TypeError, ValueError):
            finance_rh_id = None

    return templates.TemplateResponse(
        "dashboard_clients.html",
        {
            "request": request,
            "total_clients": total_clients,
            "total_clients_filtered": total_filtered,
            "srri_chart": srri_chart,
            "clients": clients,
            "page": page,
            "total_pages": total_pages,
            "filters": {
                "nom": q_nom,
                "prenom": q_prenom,
                "resp_id": q_resp,
                "srri_calc": q_srri_calc,
                "valo": q_valo,
                "perf": q_perf,
                "vol": q_vol,
            },
            "srri_options": srri_options,
            # Analyse financière supprimée pour accélérer le chargement
            "finance_supports": [],
            "finance_total_valo": 0,
            "finance_total_valo_str": "0",
            "finance_date_input": None,
            "finance_effective_date_display": None,
            "finance_rh_options": [],
            "finance_rh_selected": None,
            "finance_effective_date_iso": None,
            "finance_valo_input": None,
            "tb_markers_visible": tb_markers_visible,
            "tb_markers_param": "markers=1" if tb_markers_visible else "",
            "group_filter_label": group_filter_label if group_filter_ids is not None else None,
            "responsables": rh_options,
        }
    )


# ---------------- Affaires ----------------
@router.get("/affaires", response_class=HTMLResponse)
def dashboard_affaires(request: Request, db: Session = Depends(get_db)):
    total_affaires = db.query(func.count(Affaire.id)).scalar() or 0
    group_filter_param = request.query_params.get("group_id")
    group_filter_label: str | None = None
    page_size = 50
    try:
        page = int(request.query_params.get("page") or 1)
    except Exception:
        page = 1
    if page < 1:
        page = 1

    srri_data = (
        db.query(Affaire.SRRI, func.count(Affaire.id).label("nb"))
        .group_by(Affaire.SRRI)
        .all()
    )
    srri_chart = [{"srri": s.SRRI, "nb": s.nb} for s in srri_data]

    # dernière ligne d'historique par affaire (valo, perf 52s, volat)
    subq = (
        db.query(
            HistoriqueAffaire.id.label("affaire_id"),
            func.max(HistoriqueAffaire.date).label("last_date")
        )
        .group_by(HistoriqueAffaire.id)
        .subquery()
    )
    affaires_query = db.query(
        Affaire.id.label("id"),
        Affaire.ref,
        Affaire.SRRI,
        Affaire.date_debut,
        Affaire.date_cle,
        Affaire.frais_negocies,
        Client.nom.label("client_nom"),
        Client.prenom.label("client_prenom"),
        HistoriqueAffaire.valo.label("last_valo"),
        HistoriqueAffaire.perf_sicav_52.label("last_perf"),
        HistoriqueAffaire.volat.label("last_volat"),
    ).join(subq, subq.c.affaire_id == Affaire.id)
    affaires_query = affaires_query.join(
        HistoriqueAffaire,
        (HistoriqueAffaire.id == subq.c.affaire_id) &
        (HistoriqueAffaire.date == subq.c.last_date)
    ).outerjoin(Client, Client.id == Affaire.id_personne)
    filter_ids: set[int] | None = None
    if group_filter_param:
        try:
            gid = int(group_filter_param)
            meta = db.execute(
                text("SELECT nom, type_groupe FROM administration_groupe_detail WHERE id = :gid"),
                {"gid": gid},
            ).fetchone()
            if meta:
                meta_map = meta._mapping
                type_db = (meta_map.get("type_groupe") or "").lower()
                if type_db in ("affaire", "affaires", "contrat", "contrats"):
                    member_rows = db.execute(
                        text(
                            """
                            SELECT affaire_id FROM administration_groupe
                            WHERE groupe_id = :gid
                              AND affaire_id IS NOT NULL
                              AND (COALESCE(actif,1) != 0)
                            """
                        ),
                        {"gid": gid},
                    ).fetchall()
                    ids = {int(r[0]) for r in member_rows if r[0] is not None}
                    filter_ids = ids
                    group_filter_label = meta_map.get("nom") or f"Groupe #{gid}"
        except Exception:
            filter_ids = set()

    if filter_ids is not None:
        if filter_ids:
            affaires_query = affaires_query.filter(Affaire.id.in_(filter_ids))
        else:
            affaires_query = affaires_query.filter(text("0=1"))

    total_filtered = affaires_query.count()
    affaires_query = affaires_query.order_by(desc(HistoriqueAffaire.valo), Affaire.id)
    offset = (page - 1) * page_size
    affaires_rows = affaires_query.offset(offset).limit(page_size).all()
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)

    # SRRI calculé selon bandes standard à partir de la volat (valeurs <=1 interprétées comme fraction → %)
    def srri_from_vol(v: float | None) -> int | None:
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if abs(x) <= 1:
            x *= 100.0
        # bandes officielles
        if x <= 0.5:
            return 1
        if x <= 2:
            return 2
        if x <= 5:
            return 3
        if x <= 10:
            return 4
        if x <= 15:
            return 5
        if x <= 25:
            return 6
        return 7

    def icon_for_compare(contract_srri, calc_srri):
        if contract_srri is None or calc_srri is None:
            return None
        try:
            c = int(contract_srri)
            k = int(calc_srri)
        except Exception:
            return None
        # Mapping: Au‑dessus = 🔥, Identique = 🙏, En‑dessous = ❄️
        if c > k:
            return "fire"           # supérieur
        if c == k:
            return "hands-praying"  # identique
        return "snowflake"          # inférieur

    affaires = []
    for r in affaires_rows:
        srri_calc = srri_from_vol(r.last_volat)
        icon = icon_for_compare(r.SRRI, srri_calc)
        affaires.append({
            "id": r.id,
            "ref": r.ref,
            "SRRI": r.SRRI,
            "date_debut": r.date_debut,
            "date_cle": r.date_cle,
            "frais_negocies": getattr(r, 'frais_negocies', None),
            "client_nom": r.client_nom,
            "client_prenom": r.client_prenom,
            "last_valo": r.last_valo,
            "last_perf": r.last_perf,
            "last_volat": r.last_volat,
            "srri_calc": srri_calc,
            "srri_icon": icon,
        })
    # Comptage par comparaison SRRI (contrat vs calculé)
    compare_counts = {"above": 0, "equal": 0, "below": 0}
    for a in affaires:
        if a["srri_icon"] == "fire":
            compare_counts["above"] += 1
        elif a["srri_icon"] == "hands-praying":
            compare_counts["equal"] += 1
        elif a["srri_icon"] == "snowflake":
            compare_counts["below"] += 1

    return templates.TemplateResponse(
        "dashboard_affaires.html",
        {
            "request": request,
            "total_affaires": total_affaires,
            "total_affaires_filtered": total_filtered,
            "page": page,
            "total_pages": total_pages,
            "page_size": page_size,
            "srri_chart": srri_chart,
            "affaires": affaires,
            "srri_compare_counts": compare_counts,
            "group_filter_label": group_filter_label,
        }
    )


# ---------------- Détail Affaire ----------------
@router.get("/affaires/{affaire_id}", response_class=HTMLResponse)
def dashboard_affaire_detail(affaire_id: int, request: Request, db: Session = Depends(get_db)):
    tb_markers_visible = request.query_params.get("markers") == "1"
    affaire = db.query(Affaire).filter(Affaire.id == affaire_id).first()
    if not affaire:
        return templates.TemplateResponse(
            "dashboard_affaire_detail.html",
            {
                "request": request,
                "error": "Affaire introuvable",
                "tb_markers_visible": tb_markers_visible,
                "tb_markers_param": "markers=1" if tb_markers_visible else "",
            },
        )

    # Informations client liées à l'affaire (pour en-tête et liens)
    client_nom = None
    client_prenom = None
    client_id = None
    client_rh_id = None
    try:
        if getattr(affaire, 'id_personne', None) is not None:
            cli = db.query(Client).filter(Client.id == affaire.id_personne).first()
            if cli:
                client_id = cli.id
                client_nom = getattr(cli, 'nom', None)
                client_prenom = getattr(cli, 'prenom', None)
                try:
                    client_rh_id = getattr(cli, "commercial_id", None)
                except Exception:
                    client_rh_id = None
    except Exception:
        pass

    # Historique complet
    hist = (
        db.query(HistoriqueAffaire.date, HistoriqueAffaire.valo, HistoriqueAffaire.mouvement, HistoriqueAffaire.volat, HistoriqueAffaire.perf_sicav_52, HistoriqueAffaire.sicav, HistoriqueAffaire.annee)
        .filter(HistoriqueAffaire.id == affaire_id)
        .order_by(HistoriqueAffaire.date.asc())
        .all()
    )

    # Valorisation actuelle et agrégats mouvements
    last_valo = hist[-1].valo if hist else None
    depots = sum((h.mouvement or 0) for h in hist if (h.mouvement or 0) > 0)
    retraits = sum((h.mouvement or 0) for h in hist if (h.mouvement or 0) < 0)
    solde = sum((h.mouvement or 0) for h in hist)

    # SRRI calculé sur dernière volat
    def srri_from_vol(v):
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if abs(x) <= 1:
            x *= 100.0
        if x <= 0.5: return 1
        if x <= 2: return 2
        if x <= 5: return 3
        if x <= 10: return 4
        if x <= 15: return 5
        if x <= 25: return 6
        return 7
    srri_calc = srri_from_vol(hist[-1].volat) if hist else None

    # Dernières métriques perf/vol (en % si nécessaires)
    def _to_pct_float(x):
        if x is None:
            return None
        try:
            n = float(x)
        except Exception:
            return None
        if abs(n) <= 1:
            n *= 100.0
        return float(n)

    def _fmt_float_2(v):
        if v is None:
            return "-"
        try:
            return "{:,.2f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)
    last_perf_pct_aff = _to_pct_float(hist[-1].perf_sicav_52) if hist else None
    last_vol_pct_aff = _to_pct_float(hist[-1].volat) if hist else None

    # Séries pour graphiques
    labels = []
    serie_valo = []
    serie_cum_mouv = []
    cum = 0.0
    for h in hist:
        try:
            d = h.date.strftime("%Y-%m-%d") if h.date else None
        except Exception:
            d = str(h.date)[:10] if h.date else None
        labels.append(d)
        serie_valo.append(float(h.valo or 0))
        cum += float(h.mouvement or 0)
        serie_cum_mouv.append(cum)

    # Bar annuelles: prendre dernière date par année
    yearly = {}
    for h in hist:
        if h.annee is None:
            continue
        y = int(h.annee)
        if y not in yearly or (h.date and yearly[y]['date'] and h.date > yearly[y]['date']):
            yearly[y] = { 'date': h.date, 'perf': h.perf_sicav_52, 'vol': h.volat }
    years = sorted(yearly.keys())
    ann_perf = [ yearly[y]['perf'] for y in years ]
    ann_vol = [ yearly[y]['vol'] for y in years ]

    # Reportings pluriannuels pour l'affaire: agrégats annuels + cumul
    # Regrouper par année
    yearly_rows_aff: dict[int, list] = {}
    for h in hist:
        y = None
        try:
            y = int(getattr(h, 'annee', None)) if getattr(h, 'annee', None) is not None else None
        except Exception:
            y = None
        if y is None:
            try:
                y = int(getattr(h, 'date', None).year) if getattr(h, 'date', None) else None
            except Exception:
                y = None
        if y is None:
            continue
        yearly_rows_aff.setdefault(y, []).append(h)

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _to_pct_num(x):
        v = _to_float(x)
        if v is None:
            return None
        if abs(v) <= 1:
            v = v * 100.0
        return v

    def _to_return_decimal(x):
        v = _to_float(x)
        if v is None:
            return 0.0
        return v if abs(v) <= 1 else (v / 100.0)

    def _fmt_thousand(v):
        if v is None:
            return "-"
        try:
            return "{:,.0f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)

    reporting_years_aff = []
    cum_solde_aff = 0.0
    cum_factor_aff = 1.0
    n_years = 0
    for y in sorted(yearly_rows_aff.keys()):
        rows = yearly_rows_aff[y]
        pos = 0.0
        neg = 0.0
        total = 0.0
        for r in rows:
            m = _to_float(getattr(r, 'mouvement', None)) or 0.0
            total += m
            if m > 0:
                pos += m
            elif m < 0:
                neg += m
        # dernière ligne de l'année
        last_r = None
        for r in rows:
            if last_r is None:
                last_r = r
            else:
                try:
                    if getattr(r, 'date', None) and getattr(last_r, 'date', None) and r.date > last_r.date:
                        last_r = r
                except Exception:
                    pass
        last_valo = _to_float(getattr(last_r, 'valo', None)) if last_r else None
        last_perf_pct = _to_pct_num(getattr(last_r, 'perf_sicav_52', None)) if last_r else None
        last_vol_pct = _to_pct_num(getattr(last_r, 'volat', None)) if last_r else None

        vols = [ _to_float(getattr(r, 'volat', None)) for r in rows ]
        vols = [ v for v in vols if v is not None ]
        avg_vol_pct = _to_pct_num(sum(vols)/len(vols)) if vols else None

        # cumul solde et perf
        cum_solde_aff += total
        ann_return = _to_return_decimal(getattr(last_r, 'perf_sicav_52', None)) if last_r else 0.0
        try:
            cum_factor_aff *= (1.0 + float(ann_return or 0.0))
        except Exception:
            pass
        cum_perf_pct = (cum_factor_aff - 1.0) * 100.0
        n_years += 1
        try:
            ann_perf_pct = ((cum_factor_aff ** (1.0 / max(1, n_years))) - 1.0) * 100.0
        except Exception:
            ann_perf_pct = None

        reporting_years_aff.append({
            'year': y,
            'versements': pos,
            'versements_str': _fmt_thousand(pos),
            'retraits': neg,
            'retraits_str': _fmt_thousand(neg),
            'solde': total,
            'solde_str': _fmt_thousand(total),
            'solde_cum': cum_solde_aff,
            'solde_cum_str': _fmt_thousand(cum_solde_aff),
            'last_valo': last_valo,
            'last_valo_str': _fmt_thousand(last_valo),
            'last_perf_pct': last_perf_pct,
            'cum_perf_pct': cum_perf_pct,
            'ann_perf_pct': ann_perf_pct,
            'last_vol_pct': last_vol_pct,
            'avg_vol_pct': avg_vol_pct,
        })

    # Allocations: séries sicav par nom (pour comparaison)
    alloc_rows = (
        db.query(Allocation.nom, Allocation.date, Allocation.sicav)
        .order_by(Allocation.nom.asc(), Allocation.date.asc())
        .all()
    )
    alloc_series = {}
    for nom, d, s in alloc_rows:
        arr = alloc_series.setdefault(nom, [])
        try:
            dd = d.strftime("%Y-%m-%d") if d else None
        except Exception:
            dd = str(d)[:10] if d else None
        arr.append({"date": dd, "sicav": float(s or 0)})
    # Allocation par défaut suggérée via SRRI -> niveau risque -> allocation_risque
    default_alloc_name = None
    try:
        srri_to_level = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        srri_aff = getattr(affaire, "SRRI", None)
        if srri_aff is not None:
            level = srri_to_level.get(int(srri_aff))
            if level:
                row_alloc = db.execute(
                    text(
                        """
                        SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom
                        FROM allocation_risque ar
                        LEFT JOIN allocations a ON a.nom = ar.allocation_name
                        WHERE ar.risque_id = :rid
                        ORDER BY ar.date_attribution DESC, ar.id DESC
                        LIMIT 1
                        """
                    ),
                    {"rid": int(level)},
                ).fetchone()
                if row_alloc:
                    default_alloc_name = row_alloc[0]
    except Exception:
        default_alloc_name = None

    # Série SICAV affaire
    affaire_sicav = [ {"date": labels[i], "sicav": float(hist[i].sicav or 0)} for i in range(len(hist)) ]

    # Supports financiers: choisir une date effective (vendredi suivant une date choisie) sinon dernière
    supports = []
    try:
        # dates disponibles
        dates_rows = db.execute(
            text("SELECT DISTINCT date FROM mariadb_historique_support_w WHERE id_source = :aid ORDER BY date"),
            {"aid": affaire_id}
        ).fetchall()
        avail = []
        for r in dates_rows:
            d = r[0]
            try:
                ds = d.strftime("%Y-%m-%d")
            except Exception:
                ds = str(d)[:10]
            avail.append(ds)
        raw_as_of = request.query_params.get("as_of")
        from datetime import datetime as _dt, timedelta as _td
        pick = None
        if raw_as_of and avail:
            try:
                base = _dt.fromisoformat(raw_as_of)
                delta = (4 - base.weekday()) % 7
                candidate = (base + _td(days=delta)).strftime("%Y-%m-%d")
                pick = next((ds for ds in avail if ds >= candidate), None)
            except Exception:
                pick = None
        if not pick:
            pick = avail[-1] if avail else None
        last_date = pick
        q = text(
            """
            SELECT s.id AS support_id,
                   s.code_isin AS code_isin,
                   s.nom AS nom,
                   s.SRRI AS srri_support,
                   s.cat_gene AS cat_gene,
                   s.cat_principale AS cat_principale,
                   s.cat_det AS cat_det,
                   s.cat_geo AS cat_geo,
                   h.nbuc AS nbuc,
                   h.vl AS vl,
                   h.prmp AS prmp,
                   h.valo AS valo,
                   esg.noteE AS noteE,
                   esg.noteS AS noteS,
                   esg.noteG AS noteG
            FROM mariadb_historique_support_w h
            JOIN mariadb_support s ON s.id = h.id_support
            LEFT JOIN donnees_esg_etendu esg ON esg.isin = s.code_isin
            WHERE h.id_source = :aid AND h.date = :d
            """
        )
        rows = db.execute(q, {"aid": affaire_id, "d": last_date}).fetchall()
        for r in rows:
            try:
                support_id = int(getattr(r, "support_id", None)) if getattr(r, "support_id", None) is not None else None
            except Exception:
                support_id = None
            movements: list[dict] = []
            if support_id is not None:
                try:
                    mov_rows = db.execute(
                        text(
                            """
                            SELECT COALESCE(m.vl_date, m.date_sp) AS mouvement_date,
                                   m.montant_ope,
                                   m.vl,
                                   m.nb_uc,
                                   mr.sens,
                                   mr.titre AS regle_label
                            FROM mouvement m
                            LEFT JOIN mouvement_regle mr ON mr.id = m.id_mouvement_regle
                            WHERE m.id_affaire = :aid AND m.id_support = :sid
                            ORDER BY COALESCE(m.vl_date, m.date_sp) DESC, m.id DESC
                            """
                        ),
                        {"aid": affaire_id, "sid": support_id},
                    ).fetchall()
                except Exception:
                    mov_rows = []
                for mv in mov_rows or []:
                    dt_raw = getattr(mv, "mouvement_date", None)
                    try:
                        dt_str = dt_raw.strftime("%Y-%m-%d") if dt_raw else None
                    except Exception:
                        dt_str = str(dt_raw)[:10] if dt_raw else None
                    sens_raw = getattr(mv, "sens", None)
                    try:
                        sens_int = int(sens_raw) if sens_raw is not None else None
                    except Exception:
                        sens_int = None

                    def _sign_for_value(val: float | None) -> str:
                        if sens_int is not None:
                            if sens_int > 0:
                                return "+"
                            if sens_int < 0:
                                return "-"
                        if val is None:
                            return ""
                        try:
                            num = float(val)
                        except Exception:
                            return ""
                        if num > 0:
                            return "+"
                        if num < 0:
                            return "-"
                        return ""

                    def _fmt_signed(val) -> str:
                        if val is None:
                            return "-"
                        try:
                            num = float(val or 0.0)
                        except Exception:
                            return "-"
                        sign_char = _sign_for_value(num)
                        base = _fmt_float_2(abs(num))
                        return f"{sign_char}{base}" if sign_char else base

                    movements.append({
                        "date": dt_str or "-",
                        "regle": getattr(mv, "regle_label", None),
                        "montant": _fmt_signed(getattr(mv, "montant_ope", None)),
                        "vl": ("-" if getattr(mv, "vl", None) is None else _fmt_float_2(getattr(mv, "vl", None))),
                        "nb_uc": _fmt_signed(getattr(mv, "nb_uc", None)),
                    })
            supports.append({
                "support_id": support_id,
                "code_isin": r.code_isin,
                "nom": r.nom,
                "nbuc": r.nbuc,
                "vl": r.vl,
                "prmp": r.prmp,
                "valo": r.valo,
                "srri_support": getattr(r, "srri_support", None),
                "noteE": getattr(r, "noteE", None),
                "noteS": getattr(r, "noteS", None),
                "noteG": getattr(r, "noteG", None),
                "cat_gene": getattr(r, "cat_gene", None),
                "cat_principale": getattr(r, "cat_principale", None),
                "cat_det": getattr(r, "cat_det", None),
                "cat_geo": getattr(r, "cat_geo", None),
                "movements": movements,
            })
    except Exception:
        supports = []

    # Icône de comparaison SRRI contrat vs calculé
    def _icon_for_compare_srri(contract_srri, calc_srri):
        if contract_srri is None or calc_srri is None:
            return None
        try:
            c = int(contract_srri)
            k = int(calc_srri)
        except Exception:
            return None
        # Mapping: Au‑dessus = 🔥, Identique = 🙏, En‑dessous = ❄️
        if c > k:
            return "fire"
        if c == k:
            return "hands-praying"
        return "snowflake"
    srri_icon_aff = _icon_for_compare_srri(affaire.SRRI, srri_calc)

    # Durée depuis la première date de l'historique
    from datetime import datetime as _dt
    first_dt_aff = None
    for ds in labels:
        try:
            first_dt_aff = _dt.fromisoformat(ds) if ds else None
        except Exception:
            first_dt_aff = None
        if first_dt_aff:
            break
    last_dt_aff = None
    try:
        last_dt_aff = _dt.fromisoformat(labels[-1]) if labels else None
    except Exception:
        last_dt_aff = None

    def _human_duration(a, b):
        if not a or not b:
            return "-"
        total_months = (b.year - a.year) * 12 + (b.month - a.month)
        if b.day < a.day:
            total_months -= 1
        years = total_months // 12
        months = total_months % 12
        if years <= 0 and months <= 0:
            days = max(0, (b - a).days)
            return f"{days} jours"
        parts = []
        if years > 0:
            parts.append(f"{years} ans")
        if months > 0:
            parts.append(f"{months} mois")
        return ", ".join(parts) if parts else "-"
    duree_historique_aff_str = _human_duration(first_dt_aff, last_dt_aff)

    # Perf annualisée globale sur la durée
    overall_ann_perf_pct_aff = None
    try:
        if first_dt_aff and last_dt_aff and cum_factor_aff and cum_factor_aff > 0:
            years_span = max(1e-6, (last_dt_aff - first_dt_aff).days / 365.25)
            overall_ann_perf_pct_aff = ((float(cum_factor_aff) ** (1.0 / years_span)) - 1.0) * 100.0
    except Exception:
        overall_ann_perf_pct_aff = None

    # Comptages ouvert/fermé pour cette affaire
    nb_contrats_ouverts_aff = 1 if not getattr(affaire, 'date_cle', None) else 0
    nb_contrats_fermes_aff = 1 - nb_contrats_ouverts_aff
    # Données création tâche (accordéon) pour affaire: préremplir client + ref affaire
    from sqlalchemy import text as _text
    types = db.execute(_text("SELECT id, libelle, categorie FROM mariadb_type_evenement ORDER BY categorie, libelle")).fetchall()
    cats = sorted({getattr(t, 'categorie', None) for t in types if getattr(t, 'categorie', None)})
    from src.services.evenements import list_statuts as _list_statuts
    statuts = _list_statuts(db)
    # status ui
    def _norm(s: str | None) -> str | None:
        if not s: return None
        x = s.strip().lower()
        for a,b in [("à","a"),("â","a"),("ä","a"),("é","e"),("è","e"),("ê","e"),("ë","e"),("î","i"),("ï","i"),("ô","o"),("ö","o"),("û","u"),("ü","u"),("ç","c")]: x=x.replace(a,b)
        return x
    stat_ids = {}
    for s in statuts:
        k=_norm(getattr(s,'libelle',None))
        if k and getattr(s,'id',None) is not None: stat_ids[k]=s.id
    status_ui = []
    for label_ui, key in [
        ("à faire", "a faire"),
        ("en cours", "en cours"),
        ("en attente", "en attente"),
        ("terminé", "termine"),
        ("annulé", "annule"),
    ]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")
    reclam_statuses = (
        db.query(EvenementStatutReclamation)
        .order_by(EvenementStatutReclamation.is_cloture.asc(), EvenementStatutReclamation.libelle.asc())
        .all()
    )
    clients_suggest = db.query(Client.id, Client.nom, Client.prenom).order_by(Client.nom.asc(), Client.prenom.asc()).all()
    aff_rows = db.query(Affaire.id, Affaire.ref, Affaire.id_personne).order_by(Affaire.ref.asc()).all()
    _clients_map = {c.id: f"{getattr(c,'nom','') or ''} {getattr(c,'prenom','') or ''}".strip() for c in clients_suggest}
    affaires_suggest = [{"id":a.id, "ref":getattr(a,'ref',''), "client": _clients_map.get(getattr(a,'id_personne',None), '')} for a in aff_rows]
    client_fullname_default = f"{(client_nom or '')} {(client_prenom or '')}".strip() if (client_nom or client_prenom) else None
    affaire_ref_default = getattr(affaire,'ref',None)

    # Messages/alerts for this affaire (open tasks/events)
    from src.models.evenement import Evenement
    from src.models.type_evenement import TypeEvenement
    OPEN_STATES = ("termine", "terminé", "cloture", "clôturé", "cloturé", "clôture", "annule", "annulé")
    try:
        aff_events_open = db.execute(
            text(
                """
                SELECT
                    e.id AS id,
                    e.id AS evenement_id,
                    e.client_id,
                    e.affaire_id,
                    e.date_evenement,
                    e.statut,
                    e.commentaire,
                    e.utilisateur_responsable,
                    e.rh_id,
                    e.statut_reclamation_id,
                    te.libelle AS type_libelle,
                    te.categorie AS type_categorie,
                    esr.libelle AS statut_reclamation_libelle
                FROM mariadb_evenement e
                JOIN mariadb_type_evenement te ON te.id = e.type_id
                LEFT JOIN mariadb_evenement_statut_reclamation esr ON esr.id = e.statut_reclamation_id
                WHERE e.affaire_id = :aid
                  AND (
                    e.statut IS NULL OR LOWER(e.statut) NOT IN ('termine','terminé','cloture','clôturé','cloturé','clôture','annule','annulé')
                  )
                ORDER BY e.date_evenement DESC
                """
            ),
            {"aid": affaire_id},
        ).fetchall()
    except Exception as exc:
        logger.exception("affaire_events_open query failed", exc_info=exc)
        aff_events_open = []
    def _norm_cat(s: str | None) -> str:
        if not s:
            return ""
        x = (s or "").strip().lower()
        for a, b in [("à","a"),("â","a"),("ä","a"),("é","e"),("è","e"),("ê","e"),("ë","e"),("î","i"),("ï","i"),("ô","o"),("ö","o"),("û","u"),("ü","u"),("ç","c")]:
            x = x.replace(a, b)
        return x
    msgs_reg_count = 0
    msgs_nonreg_count = 0
    affaire_events_open: list[dict] = []
    for r in aff_events_open:
        catn = _norm_cat(getattr(r, "type_categorie", None))
        is_reg = (catn == "reglementaire")
        if is_reg:
            msgs_reg_count += 1
        else:
            msgs_nonreg_count += 1
        mp = getattr(r, "_mapping", {})
        eid_raw = mp.get("id") if isinstance(mp, dict) else None
        eid_alt = mp.get("evenement_id") if isinstance(mp, dict) else None
        if eid_raw is None:
            try:
                eid_raw = getattr(r, "id", None)
            except Exception:
                eid_raw = None
        if eid_alt is None:
            try:
                eid_alt = getattr(r, "evenement_id", None)
            except Exception:
                eid_alt = None
        eid_int = eid_raw if eid_raw is not None else eid_alt
        try:
            dstr = r.date_evenement.strftime("%Y-%m-%d %H:%M") if getattr(r, 'date_evenement', None) else None
        except Exception:
            dstr = str(getattr(r, 'date_evenement', None))[:16] if getattr(r, 'date_evenement', None) else None
        affaire_events_open.append({
            "id": eid_raw,
            "evenement_id": eid_alt,
            "id_resolved": eid_int,
            "date_evenement": dstr,
            "statut": getattr(r, 'statut', None),
            "commentaire": getattr(r, 'commentaire', None),
            "type_id": getattr(r, 'type_id', None),
            "type_libelle": getattr(r, 'type_libelle', None),
            "type_categorie": getattr(r, 'type_categorie', None),
            "statut_reclamation_id": getattr(r, 'statut_reclamation_id', None),
            "statut_reclamation_label": getattr(r, 'statut_reclamation_libelle', None),
        })

    # Avis d'opération pour cette affaire (avis + avis_regle)
    try:
        from sqlalchemy import text as _text
        rows_avis = db.execute(
            _text(
                """
                SELECT a.id AS avis_id, a.date AS dt, a.reference AS reference, a.id_etape AS etape_id,
                       ar.nom AS etape_nom, a.entree AS entree, a.sortie AS sortie
                FROM avis a
                LEFT JOIN avis_regle ar ON ar.id = a.id_etape
                LEFT JOIN mariadb_affaires ma ON ma.id = a.id_affaire
                WHERE a.id_affaire = :aid
                ORDER BY a.date DESC
                """
            ),
            {"aid": affaire_id},
        ).fetchall()
    except Exception:
        rows_avis = []

    def _fmt_money2(v):
        try:
            return "{:,.2f}".format(float(v or 0)).replace(",", " ")
        except Exception:
            return v

    avis_affaire = []
    for r in rows_avis:
        # Format date en YYYY-MM-DD
        try:
            dstr = r.dt.strftime("%Y-%m-%d") if getattr(r, 'dt', None) else None
        except Exception:
            dstr = str(getattr(r, 'dt', None))[:10] if getattr(r, 'dt', None) else None
        avis_affaire.append({
            "avis_id": getattr(r, 'avis_id', None),
            "date": dstr,
            "reference": getattr(r, 'reference', None),
            "etape": getattr(r, 'etape_nom', None),
            "entree_str": _fmt_money2(getattr(r, 'entree', None)),
            "sortie_str": _fmt_money2(getattr(r, 'sortie', None)),
        })

    # Noms d'allocations disponibles (distincts)
    try:
        alloc_names = [r[0] for r in db.query(Allocation.nom).filter(Allocation.nom.isnot(None)).distinct().order_by(Allocation.nom.asc()).all()]
    except Exception:
        alloc_names = []

    # Champs ESG disponibles (colonnes de esg_fonds, hors identifiants texte)
    # On renvoie à la fois le nom de colonne et un libellé lisible.
    esg_fields: list[dict] = []
    esg_field_labels: dict[str, str] = {}
    # Fallback multi-SGBD: SHOW COLUMNS (MySQL/MariaDB) -> PRAGMA (SQLite) -> SELECT * LIMIT 1
    try:
        ok = False
        try:
            rows_cols = db.execute(text("SHOW COLUMNS FROM esg_fonds")).fetchall()
            if rows_cols:
                for rc in rows_cols:
                    col = rc[0]
                    if str(col).lower() in ("isin", "company name"):
                        continue
                    esg_fields.append({"col": col, "label": col})
                ok = True
        except Exception:
            ok = False
        if not ok:
            try:
                rows_cols = db.execute(text("PRAGMA table_info(esg_fonds)")).fetchall()
                def _labelize(name: str) -> str:
                    # Transforme camelCase / snake_case en libellé lisible
                    if not name:
                        return name
                    s = str(name)
                    s = s.replace('_', ' ')
                    # insert spaces before capitals
                    import re as _re
                    s = _re.sub(r'(?<!^)([A-Z])', r' \1', s)
                    return s.strip().capitalize()
                for rc in rows_cols or []:
                    col = rc[1]
                    if str(col).lower() in ("isin", "company name"):
                        continue
                    label = _labelize(col)
                    esg_fields.append({"col": col, "label": label})
                ok = True
            except Exception:
                ok = False
        if not ok:
            try:
                row1 = db.execute(text("SELECT * FROM esg_fonds LIMIT 1")).first()
                if row1 is not None:
                    for k in row1._mapping.keys():
                        if str(k).lower() in ("isin", "company name"):
                            continue
                        esg_fields.append({"col": k, "label": k})
                ok = True
            except Exception:
                ok = False
    except Exception:
        pass
    # Déduplique et trie par libellé
    seen = set()
    uniq = []
    for it in esg_fields:
        key = str(it.get("col"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    esg_fields = sorted(uniq, key=lambda x: str(x.get("label","" )).lower())
    esg_field_labels = { it["col"]: it["label"] for it in esg_fields }
    # RH list pour affectation des tâches
    rh_list = fetch_rh_list(db)

    return templates.TemplateResponse(
        "dashboard_affaire_detail.html",
        {
            "request": request,
            "affaire": affaire,
            "client_id": client_id,
            "client_nom": client_nom,
            "client_prenom": client_prenom,
            "client_rh_id": client_rh_id,
            # Tâches: assistance création locale
            "types": types,
            "categories": cats,
            "statuts": statuts,
            "status_ui": status_ui,
            "en_cours_id": en_cours_id,
            "clients_suggest": clients_suggest,
            "affaires_suggest": affaires_suggest,
            "client_fullname_default": client_fullname_default,
            "affaire_ref_default": affaire_ref_default,
            "reclam_statuses": reclam_statuses,
            "rh_list": rh_list,
            "tb_markers_visible": tb_markers_visible,
            "tb_markers_param": "markers=1" if tb_markers_visible else "",
            "avis_affaire": avis_affaire,
            # Messages/alertes en-tête affaire
            "msgs_reg_count": msgs_reg_count,
            "msgs_nonreg_count": msgs_nonreg_count,
            "affaire_events_open": affaire_events_open,
            "last_valo": last_valo,
            "depots": depots,
            "retraits": retraits,
            "solde": solde,
            "srri_client": affaire.SRRI,
            "srri_calc": srri_calc,
            "srri_icon_aff": srri_icon_aff,
            "labels": labels,
            "serie_valo": serie_valo,
            "serie_cum_mouv": serie_cum_mouv,
            "years": years,
            "ann_perf": ann_perf,
            "ann_vol": ann_vol,
            "alloc_series": alloc_series,
            "default_alloc_name": default_alloc_name,
            "affaire_sicav": affaire_sicav,
            "supports": supports,
            "available_dates": avail,
            "as_of_effective": last_date,
            "as_of_input": request.query_params.get("as_of"),
            # Reportings pluriannuels (affaire)
            "reporting_years": reporting_years_aff,
            # Indicateurs synthèse/risque
            "last_perf_pct_aff": last_perf_pct_aff,
            "last_vol_pct_aff": last_vol_pct_aff,
            "overall_ann_perf_pct_aff": overall_ann_perf_pct_aff,
            "duree_historique_aff_str": duree_historique_aff_str,
            "nb_contrats_ouverts_aff": nb_contrats_ouverts_aff,
            "nb_contrats_fermes_aff": nb_contrats_fermes_aff,
            # ESG UI context
            "alloc_names": alloc_names,
            "esg_fields": esg_fields,
            "esg_field_labels": esg_field_labels,
        }
    )

# ---------------- ESG data API (Affaire) ----------------
from fastapi import Query
from fastapi.responses import JSONResponse

@router.get("/affaires/{affaire_id}/esg", response_class=JSONResponse)
def dashboard_affaire_esg(
    affaire_id: int,
    alloc: str = Query(None, description="Nom de l'allocation de référence"),
    alloc_isin: str = Query(None, description="ISIN de l'allocation (prioritaire si fourni)"),
    fields: str = Query(None, description="Champs ESG séparés par des virgules"),
    as_of: str | None = Query(None, description="Date d'analyse YYYY-MM-DD pour l'affaire"),
    alloc_date: str | None = Query(None, description="Date exacte pour l'allocation (YYYY-MM-DD)"),
    debug: int = Query(0, description="Activer la sortie debug"),
    db: Session = Depends(get_db),
):
    # Parse fields
    sel_fields: list[str] = []
    if fields:
        sel_fields = [f.strip() for f in fields.split(',') if f and f.strip()]
    if not sel_fields:
        return {"error": "Aucun champ ESG sélectionné."}

    # Date de référence affaire
    debug_info = {"affaire_query": None, "alloc_query": None, "esg_query": None}

    try:
        if as_of:
            as_of_dt = as_of
        else:
            as_of_dt = db.execute(text("SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :i"), {"i": affaire_id}).scalar()
    except Exception:
        as_of_dt = None

    # Composition affaire (ISIN -> poids) — dernière valeur par support, optionnellement <= as_of
    affaire_weights: dict[str, float] = {}
    try:
        if as_of_dt:
            # Strictement à la date choisie
            q = text(
                """
                SELECT s.code_isin AS isin, SUM(h.valo) AS somme_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_support s ON s.id = h.id_support
                WHERE h.id_source = :aid AND h.date = :d
                GROUP BY s.code_isin
                """
            )
            params = {"aid": affaire_id, "d": as_of_dt}
        else:
            # Fallback: dernière valeur par support
            q = text(
                """
                WITH sub AS (
                  SELECT id_support, MAX(date) AS last_date
                  FROM mariadb_historique_support_w
                  WHERE id_source = :aid
                  GROUP BY id_support
                )
                SELECT s.code_isin AS isin, SUM(h.valo) AS somme_valo
                FROM mariadb_historique_support_w h
                JOIN sub ON sub.id_support = h.id_support AND h.date = sub.last_date
                JOIN mariadb_support s ON s.id = h.id_support
                WHERE h.id_source = :aid
                GROUP BY s.code_isin
                """
            )
            params = {"aid": affaire_id}
        debug_info["affaire_query"] = {"sql": q.text, "params": {k: (str(v) if v is not None else None) for k,v in params.items()}}
        rows = db.execute(q, params).fetchall()
        total = sum(float(r.somme_valo or 0) for r in rows) if rows else 0.0
        if total and total > 0:
            for r in rows:
                isin = getattr(r, 'isin', None)
                v = float(getattr(r, 'somme_valo', 0) or 0)
                if isin and v is not None:
                    affaire_weights[isin] = v / total
    except Exception:
        affaire_weights = {}

    # Composition allocation (ISIN -> poids)
    alloc_weights: dict[str, float] = {}
    alloc_date_val = None
    try:
        # Déterminer si on filtre par ISIN explicite
        import re as _re
        isin_param = alloc_isin or (alloc if alloc and _re.match(r'^[A-Z0-9]{9,12}$', str(alloc).strip()) else None)
        if isin_param:
            # Mode ISIN — forcer la date: dernière <= alloc_date si fournie, sinon dernière
            if alloc_date:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i AND date <= :d"), {"i": isin_param, "d": alloc_date}).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"), {"i": isin_param}).scalar()
            else:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"), {"i": isin_param}).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE ISIN = :i AND date = :d
                """
            )
            params2 = {"i": isin_param, "d": alloc_date_val}
            debug_info["alloc_query"] = {"sql": q2.text, "params": {k: (str(v) if v is not None else None) for k,v in params2.items()}}
            rows2 = db.execute(q2, params2).fetchall()
        elif alloc:
            # Mode nom — forcer la date: dernière <= alloc_date si fournie, sinon dernière
            if alloc_date:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n)) AND date <= :d"), {"n": alloc, "d": alloc_date}).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"), {"n": alloc}).scalar()
            else:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"), {"n": alloc}).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE lower(trim(nom)) = lower(trim(:n)) AND date = :d
                """
            )
            params2 = {"n": alloc, "d": alloc_date_val}
            debug_info["alloc_query"] = {"sql": q2.text, "params": {k: (str(v) if v is not None else None) for k,v in params2.items()}}
            rows2 = db.execute(q2, params2).fetchall()
        total2 = sum(float(getattr(r, 'v', 0) or 0) for r in rows2) if rows2 else 0.0
        if total2 and total2 > 0:
            for r in rows2:
                isin = getattr(r, 'isin', None)
                v = float(getattr(r, 'v', 0) or 0)
                if isin and v is not None:
                    alloc_weights[isin] = v / total2
    except Exception:
        alloc_weights = {}

    # Rassembler tous les ISIN utiles
    all_isins = set(affaire_weights.keys()) | set(alloc_weights.keys())
    if not all_isins:
        return {"fields": sel_fields, "results": []}

    # Charger les valeurs ESG pour ces ISIN
    # Quoter les champs pour compatibilité MySQL (noms avec espaces/traits)
    def quote_col(c: str) -> str:
        c = c.strip()
        if not c:
            return c
        # utiliser backticks
        return f"`{c}`"

    # Construire des alias sûrs pour récupérer les colonnes (évite problèmes d'espaces/traits)
    aliases = [(c, f"c{idx}") for idx, c in enumerate(sel_fields)]
    col_expr = ", ".join(f"{quote_col(c)} AS {al}" for c, al in aliases)
    esg_map: dict[str, dict[str, float]] = {}
    try:
        # Construire une liste de paramètres
        isin_list = list(all_isins)
        # Créer placeholders
        placeholders = ",".join([":i%d" % idx for idx in range(len(isin_list))])
        params = { ("i%d" % idx): val for idx, val in enumerate(isin_list) }
        q_esg = text(f"SELECT ISIN, {col_expr} FROM esg_fonds WHERE ISIN IN ({placeholders})")
        debug_info["esg_query"] = {"sql": q_esg.text, "params": {**params}}
        rows_esg = db.execute(q_esg, params).fetchall()
        for row in rows_esg or []:
            mm = row._mapping
            isin = mm.get("ISIN")
            if not isin:
                continue
            d = {}
            for (f, al) in aliases:
                try:
                    val = mm.get(al)
                except Exception:
                    val = None
                try:
                    d[f] = float(val) if val is not None else None
                except Exception:
                    d[f] = None
            esg_map[isin] = d
    except Exception:
        esg_map = {}

    # Calcul des indicateurs pondérés et normalisés (indice = 100)
    results = []
    for f in sel_fields:
        # affaire
        aff_val = 0.0
        aff_wsum = 0.0
        for isin, w in affaire_weights.items():
            v = (esg_map.get(isin) or {}).get(f)
            if v is None:
                continue
            aff_val += float(w) * float(v)
            aff_wsum += float(w)
        aff_val = (aff_val / aff_wsum) if aff_wsum > 0 else None

        # index
        idx_val = 0.0
        idx_wsum = 0.0
        for isin, w in alloc_weights.items():
            v = (esg_map.get(isin) or {}).get(f)
            if v is None:
                continue
            idx_val += float(w) * float(v)
            idx_wsum += float(w)
        idx_val = (idx_val / idx_wsum) if idx_wsum > 0 else None

        # ajouter compteurs de présence par champ
        aff_present = sum(1 for isin,_w in affaire_weights.items() if (esg_map.get(isin) or {}).get(f) is not None)
        idx_present = sum(1 for isin,_w in alloc_weights.items() if (esg_map.get(isin) or {}).get(f) is not None)
        if aff_val is None or idx_val is None or idx_val == 0:
            results.append({"field": f, "index": None, "affaire": None, "aff_present": aff_present, "idx_present": idx_present})
        else:
            results.append({"field": f, "index": 100.0, "affaire": (aff_val / idx_val) * 100.0, "aff_present": aff_present, "idx_present": idx_present})

    payload = {
        "fields": sel_fields,
        "alloc": alloc,
        "alloc_isin": isin_param if 'isin_param' in locals() else None,
        "as_of": as_of_dt,
        "alloc_date": alloc_date_val,
        "results": results,
    }
    if debug:
        payload["debug"] = {
            "affaire": {
                "weights_count": len(affaire_weights),
                "weights_sum": sum(affaire_weights.values()) if affaire_weights else 0,
                "query": debug_info["affaire_query"],
                "isins": sorted(list(affaire_weights.keys())),
            },
            "allocation": {
                "weights_count": len(alloc_weights),
                "weights_sum": sum(alloc_weights.values()) if alloc_weights else 0,
                "date": alloc_date_val,
                "query": debug_info["alloc_query"],
                "isins": sorted(list(alloc_weights.keys())),
            },
            "esg": {
                "isin_count": len(all_isins),
                "query": debug_info["esg_query"],
            }
        }
    return payload


# ---------------- ESG data API (Client consolidé) ----------------
@router.get("/clients/{client_id}/esg", response_class=JSONResponse)
def dashboard_client_esg(
    client_id: int,
    alloc: str = Query(None, description="Nom de l'allocation de référence"),
    alloc_isin: str = Query(None, description="ISIN de l'allocation (prioritaire si fourni)"),
    fields: str = Query(None, description="Champs ESG séparés par des virgules"),
    as_of: str | None = Query(None, description="Date d'analyse YYYY-MM-DD pour le client (supports consolidés)"),
    alloc_date: str | None = Query(None, description="Date exacte pour l'allocation (YYYY-MM-DD)"),
    debug: int = Query(0, description="Activer la sortie debug"),
    db: Session = Depends(get_db),
):
    sel_fields: list[str] = []
    if fields:
        sel_fields = [f.strip() for f in fields.split(',') if f and f.strip()]
    if not sel_fields:
        return {"error": "Aucun champ ESG sélectionné."}

    payload, debug_payload = _compute_client_esg_comparison(
        db=db,
        client_id=client_id,
        fields=sel_fields,
        alloc=alloc,
        alloc_isin=alloc_isin,
        as_of=as_of,
        alloc_date=alloc_date,
        debug=bool(debug),
    )
    if debug and debug_payload is not None:
        payload["debug"] = debug_payload
    return payload


# ---------------- ESG data API (Global consolidé: tous contrats) ----------------
@router.get("/esg", response_class=JSONResponse)
def dashboard_global_esg(
    alloc: str = Query(None, description="Nom de l'allocation de référence"),
    alloc_isin: str = Query(None, description="ISIN de l'allocation (prioritaire si fourni)"),
    fields: str = Query(None, description="Champs ESG séparés par des virgules"),
    as_of: str | None = Query(None, description="Date d'analyse YYYY-MM-DD (supports consolidés sur toutes les affaires)"),
    alloc_date: str | None = Query(None, description="Date exacte pour l'allocation (YYYY-MM-DD)"),
    debug: int = Query(0, description="Activer la sortie debug"),
    db: Session = Depends(get_db),
):
    sel_fields: list[str] = []
    if fields:
        sel_fields = [f.strip() for f in fields.split(',') if f and f.strip()]
    if not sel_fields:
        return {"error": "Aucun champ ESG sélectionné."}

    # Simplification demandée: calcul Top-10 par valorisation globale et agrégation pondérée des champs
    # On utilise MAX(date) sur mariadb_historique_support_w, prend les 10 plus gros fonds,
    # calcule leur poids relatif et agrège SUM(weight * champ) pour chaque champ demandé.
    try:
        def _quote_ident(c: str) -> str:
            c = str(c).replace('"', '""')
            return f'"{c}"'

        select_expr = ",\n  ".join([f"SUM(w.w * ef.{_quote_ident(col)}) AS {_quote_ident(col)}" for col in sel_fields])

        # 1) Valeurs portefeuille global (Top 10 par valorisation à MAX(date))
        sql_portfolio = f"""
WITH last_date AS (
  SELECT MAX(date) AS d FROM mariadb_historique_support_w
),
agg AS (
  SELECT s.code_isin AS isin, s.nom, SUM(h.valo) AS total_valo
  FROM mariadb_historique_support_w h
  JOIN mariadb_support s ON s.id = h.id_support
  JOIN last_date ld
  WHERE h.date = ld.d
  GROUP BY s.code_isin, s.nom
),
top10 AS (
  SELECT * FROM agg ORDER BY total_valo DESC LIMIT 10
),
sum10 AS (
  SELECT SUM(total_valo) AS total_10 FROM top10
),
weights AS (
  SELECT t.isin, (t.total_valo / s.total_10) AS w
  FROM top10 t
  CROSS JOIN sum10 s
)
SELECT
  {select_expr}
FROM weights w
LEFT JOIN esg_fonds ef ON ef.ISIN = w.isin
"""

        row_port = db.execute(text(sql_portfolio)).fetchone()
        port_map = {}
        if row_port is not None:
            mm = getattr(row_port, "_mapping", row_port)
            for f in sel_fields:
                try:
                    v = mm.get(f)
                except Exception:
                    v = None
                try:
                    port_map[f] = float(v) if v is not None else None
                except Exception:
                    port_map[f] = None

        # 2) Valeurs indice (allocation de référence) — Top 10 de la composition à la date choisie (ou MAX)
        alloc_date_val = None
        # Déterminer la date d'allocation
        if alloc:
            if alloc_date:
                alloc_date_val = db.execute(text(
                    "SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n)) AND date <= :d"
                ), {"n": alloc, "d": alloc_date}).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(text(
                        "SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"
                    ), {"n": alloc}).scalar()
            else:
                alloc_date_val = db.execute(text(
                    "SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"
                ), {"n": alloc}).scalar()

        sql_index = None
        idx_map = {f: None for f in sel_fields}
        if alloc and alloc_date_val:
            sql_index = f"""
WITH last_date AS (
  SELECT MAX(date) AS d FROM allocations WHERE lower(trim(nom)) = lower(trim(:n)) AND date <= :d
),
agg AS (
  SELECT a.isin AS isin, SUM(COALESCE(a.valo, a.sicav)) AS total_valo
  FROM allocations a
  JOIN last_date ld
  WHERE lower(trim(a.nom)) = lower(trim(:n)) AND a.date = ld.d
  GROUP BY a.isin
),
top10 AS (
  SELECT * FROM agg ORDER BY total_valo DESC LIMIT 10
),
sum10 AS (
  SELECT SUM(total_valo) AS total_10 FROM top10
),
weights AS (
  SELECT t.isin, (t.total_valo / s.total_10) AS w
  FROM top10 t
  CROSS JOIN sum10 s
)
SELECT
  {select_expr}
FROM weights w
LEFT JOIN esg_fonds ef ON ef.ISIN = w.isin
"""
            row_idx = db.execute(text(sql_index), {"n": alloc, "d": alloc_date_val}).fetchone()
            if row_idx is not None:
                mm2 = getattr(row_idx, "_mapping", row_idx)
                for f in sel_fields:
                    try:
                        v2 = mm2.get(f)
                    except Exception:
                        v2 = None
                    try:
                        idx_map[f] = float(v2) if v2 is not None else None
                    except Exception:
                        idx_map[f] = None

        # 3) Résultats normalisés: indice=100 et portefeuille = (port/index) * 100
        results_norm = []
        for f in sel_fields:
            p = port_map.get(f)
            q = idx_map.get(f)
            if p is None or q is None or q == 0:
                results_norm.append({
                    "field": f,
                    "index": None,
                    "global": None,
                    "index_raw": q,
                    "global_raw": p,
                })
            else:
                results_norm.append({
                    "field": f,
                    "index": (q / q) * 100.0,
                    "global": (p / q) * 100.0,
                    "index_raw": q,
                    "global_raw": p,
                })

        payload_top = {
            "fields": sel_fields,
            "alloc": alloc,
            "alloc_isin": alloc_isin,
            "as_of": None,
            "alloc_date": alloc_date_val,
            "results": results_norm,
        }
        if debug:
            payload_top["debug"] = {
                "sql_portfolio": sql_portfolio,
                "sql_index": sql_index,
                "alloc_date": str(alloc_date_val) if alloc_date_val is not None else None,
                "portfolio_raw": port_map,
                "index_raw": idx_map,
            }
        return payload_top
    except Exception:
        # En cas d'erreur, on retombe sur le comportement précédent (plus bas)
        pass

    debug_info = {"global_query": None, "alloc_query": None, "esg_query": None}

    # Composition globale (toutes les affaires) — calquée sur Client/detail, sans filtre client
    global_weights: dict[str, float] = {}
    try:
        affaire_ids = [rid for (rid,) in db.query(Affaire.id).all()]
        sums: dict[str, float] = {}
        total_valo = 0.0
        params_list: list[dict] = []
        # Même requête que client/detail: on fixe une date par affaire (as_of sinon dernière)
        weight_sql = (
            "SELECT s.code_isin AS isin, SUM(h.valo) AS somme_valo\n"
            "FROM mariadb_historique_support_w h\n"
            "JOIN mariadb_support s ON s.id = h.id_support\n"
            "WHERE h.id_source = :aid AND h.date = :d\n"
            "GROUP BY s.code_isin"
        )
        for aid in affaire_ids:
            # Déterminer la date de référence pour cette affaire
            if as_of:
                ref_date = as_of
            else:
                ref_date = db.execute(text("SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :aid"), {"aid": aid}).scalar()
                if not ref_date:
                    continue
            q = text(weight_sql)
            params = {"aid": aid, "d": ref_date}
            rows = db.execute(q, params).fetchall()
            params_list.append({"aid": aid, "d": str(ref_date) if ref_date is not None else None})
            for r in rows or []:
                isin = getattr(r, 'isin', None)
                v = float(getattr(r, 'somme_valo', 0) or 0)
                if isin:
                    sums[isin] = sums.get(isin, 0.0) + v
                    total_valo += v
        if total_valo > 0:
            for isin, v in sums.items():
                if v is not None:
                    global_weights[isin] = float(v) / float(total_valo)
        debug_info["global_query"] = {"sql": weight_sql, "params_list": params_list, "weights_count": len(global_weights)}
    except Exception:
        global_weights = {}

    # Allocation (identique aux autres endpoints)
    alloc_weights: dict[str, float] = {}
    alloc_date_val = None
    try:
        import re as _re
        isin_param = alloc_isin or (alloc if alloc and _re.match(r'^[A-Z0-9]{9,12}$', str(alloc).strip()) else None)
        if isin_param:
            if alloc_date:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i AND date <= :d"), {"i": isin_param, "d": alloc_date}).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"), {"i": isin_param}).scalar()
            else:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE ISIN = :i"), {"i": isin_param}).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE ISIN = :i AND date = :d
                """
            )
            params2 = {"i": isin_param, "d": alloc_date_val}
            try:
                sql_txt = q2.text
            except Exception:
                sql_txt = str(q2)
            debug_info["alloc_query"] = {"mode": "isin", "sql": sql_txt, "params": {k: (str(v) if v is not None else None) for k,v in params2.items()}}
            rows2 = db.execute(q2, params2).fetchall()
            total2 = sum(float(getattr(r, 'v', 0) or 0) for r in rows2) if rows2 else 0.0
            if total2 and total2 > 0:
                for r in rows2:
                    isin = getattr(r, 'isin', None)
                    v = float(getattr(r, 'v', 0) or 0)
                    if isin and v is not None:
                        alloc_weights[isin] = v / total2
        elif alloc:
            if alloc_date:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n)) AND date <= :d"), {"n": alloc, "d": alloc_date}).scalar()
                if not alloc_date_val:
                    alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"), {"n": alloc}).scalar()
            else:
                alloc_date_val = db.execute(text("SELECT MAX(date) FROM allocations WHERE lower(trim(nom)) = lower(trim(:n))"), {"n": alloc}).scalar()
            q2 = text(
                """
                SELECT ISIN AS isin, COALESCE(valo, sicav) AS v
                FROM allocations
                WHERE lower(trim(nom)) = lower(trim(:n)) AND date = :d
                """
            )
            params2 = {"n": alloc, "d": alloc_date_val}
            try:
                sql_txt = q2.text
            except Exception:
                sql_txt = str(q2)
            debug_info["alloc_query"] = {"mode": "nom", "sql": sql_txt, "params": {k: (str(v) if v is not None else None) for k,v in params2.items()}}
            rows2 = db.execute(q2, params2).fetchall()
            total2 = sum(float(getattr(r, 'v', 0) or 0) for r in rows2) if rows2 else 0.0
            if total2 and total2 > 0:
                for r in rows2:
                    isin = getattr(r, 'isin', None)
                    v = float(getattr(r, 'v', 0) or 0)
                    if isin and v is not None:
                        alloc_weights[isin] = v / total2
    except Exception:
        alloc_weights = {}

    all_isins = set(global_weights.keys()) | set(alloc_weights.keys())
    if not all_isins:
        payload = {
            "fields": sel_fields,
            "alloc": alloc,
            "alloc_isin": isin_param if 'isin_param' in locals() else None,
            "as_of": as_of,
            "alloc_date": alloc_date_val,
            "results": [],
        }
        if debug:
            payload["debug"] = {
                "global": {
                    "weights_count": len(global_weights),
                    "weights_sum": sum(global_weights.values()) if global_weights else 0,
                    "query": debug_info.get("global_query"),
                    "isins": sorted(list(global_weights.keys())),
                },
                "allocation": {
                    "weights_count": len(alloc_weights),
                    "weights_sum": sum(alloc_weights.values()) if alloc_weights else 0,
                    "date": alloc_date_val,
                    "query": debug_info.get("alloc_query"),
                    "isins": sorted(list(alloc_weights.keys())),
                },
                "esg": {
                    "isin_count": 0,
                    "query": None,
                }
            }
        return payload

    def quote_col(c: str) -> str:
        c = c.strip()
        if not c:
            return c
        return f"`{c}`"

    aliases = [(c, f"c{idx}") for idx, c in enumerate(sel_fields)]
    col_expr = ", ".join(f"{quote_col(c)} AS {al}" for c, al in aliases)
    esg_map: dict[str, dict[str, float]] = {}
    try:
        isin_list = list(all_isins)
        placeholders = ",".join([":i%d" % idx for idx in range(len(isin_list))])
        params = { ("i%d" % idx): val for idx, val in enumerate(isin_list) }
        q_esg = text(f"SELECT ISIN, {col_expr} FROM esg_fonds WHERE ISIN IN ({placeholders})")
        debug_info["esg_query"] = {"sql": q_esg.text, "params": {**params}}
        rows_esg = db.execute(q_esg, params).fetchall()
        for row in rows_esg or []:
            mm = row._mapping
            isin = mm.get("ISIN")
            if not isin:
                continue
            d = {}
            for (f, al) in aliases:
                try:
                    val = mm.get(al)
                except Exception:
                    val = None
                try:
                    d[f] = float(val) if val is not None else None
                except Exception:
                    d[f] = None
            esg_map[isin] = d
    except Exception:
        esg_map = {}

    results = []
    for f in sel_fields:
        g_val = 0.0
        g_wsum = 0.0
        for isin, w in global_weights.items():
            v = (esg_map.get(isin) or {}).get(f)
            if v is None:
                continue
            g_val += float(w) * float(v)
            g_wsum += float(w)
        g_val = (g_val / g_wsum) if g_wsum > 0 else None

        idx_val = 0.0
        idx_wsum = 0.0
        for isin, w in alloc_weights.items():
            v = (esg_map.get(isin) or {}).get(f)
            if v is None:
                continue
            idx_val += float(w) * float(v)
            idx_wsum += float(w)
        idx_val = (idx_val / idx_wsum) if idx_wsum > 0 else None

        g_present = sum(1 for isin,_w in global_weights.items() if (esg_map.get(isin) or {}).get(f) is not None)
        idx_present = sum(1 for isin,_w in alloc_weights.items() if (esg_map.get(isin) or {}).get(f) is not None)

        if g_val is None or idx_val is None or idx_val == 0:
            results.append({"field": f, "index": None, "global": None, "global_present": g_present, "idx_present": idx_present})
        else:
            results.append({"field": f, "index": 100.0, "global": (g_val / idx_val) * 100.0, "global_present": g_present, "idx_present": idx_present})

    payload = {
        "fields": sel_fields,
        "alloc": alloc,
        "alloc_isin": isin_param if 'isin_param' in locals() else None,
        "as_of": as_of,
        "alloc_date": alloc_date_val,
        "results": results,
    }
    if debug:
        payload["debug"] = {
            "global": {
                "weights_count": len(global_weights),
                "weights_sum": sum(global_weights.values()) if global_weights else 0,
                "query": debug_info.get("global_query"),
                "isins": sorted(list(global_weights.keys())),
            },
            "allocation": {
                "weights_count": len(alloc_weights),
                "weights_sum": sum(alloc_weights.values()) if alloc_weights else 0,
                "date": alloc_date_val,
                "query": debug_info.get("alloc_query"),
                "isins": sorted(list(alloc_weights.keys())),
            },
            "esg": {
                "isin_count": len(all_isins),
                "query": debug_info.get("esg_query"),
            }
        }
    return payload


# ---------------- Supports ----------------
from sqlalchemy import text

@router.get("/supports", response_class=HTMLResponse)
def dashboard_supports(request: Request, db: Session = Depends(get_db)):
    # Récupérer la dernière date disponible
    last_date = db.execute(
        text("SELECT MAX(date) FROM mariadb_historique_support_w")
    ).scalar()

    # Formatage robuste de la date pour l'affichage
    if isinstance(last_date, (datetime, _date)):
        last_date_str = last_date.strftime("%Y-%m-%d")
    elif isinstance(last_date, str):
        last_date_str = last_date[:10]
    else:
        last_date_str = None

    # Récupérer les supports avec leur valo à cette date
    results = db.execute(
        text("""
            SELECT
                s.code_isin,
                s.nom,
                s.cat_gene AS categorie,
                s.cat_geo AS zone_geo,
                s.SRRI,
                SUM(h.valo) AS total_valo,
                h.date
            FROM mariadb_historique_support_w h
            JOIN mariadb_support s ON s.id = h.id_support
            WHERE h.date = :last_date
            GROUP BY s.code_isin, s.nom, s.cat_gene, s.cat_geo, s.SRRI, h.date
            HAVING SUM(h.valo) > 0
            ORDER BY total_valo DESC
        """),
        {"last_date": last_date}
    ).fetchall()

    return templates.TemplateResponse(
        "dashboard_supports.html",
        {
            "request": request,
            "supports": results,
            "last_date": last_date_str,
            "total_supports": len(results),
        }
    )


@router.get("/supports/details")
def dashboard_support_details(
    code_isin: str = Query(..., description="Code ISIN du support"),
    format: str = Query("json", description="Format de réponse: json ou csv"),
    db: Session = Depends(get_db),
):
    support_rows = db.execute(
        text(
            """
            SELECT id,
                   code_isin,
                   nom,
                   cat_gene,
                   cat_principale,
                   cat_det,
                   cat_geo,
                   promoteur,
                   SRRI
            FROM mariadb_support
            WHERE code_isin = :code
            """
        ),
        {"code": code_isin},
    ).mappings().all()
    if not support_rows:
        raise HTTPException(status_code=404, detail="Support introuvable.")
    support = support_rows[0]
    support_ids = [row["id"] for row in support_rows]

    last_date = db.execute(
        text("SELECT MAX(date) FROM mariadb_historique_support_w")
    ).scalar()
    last_date_iso = None
    last_date_display = None
    if isinstance(last_date, datetime):
        last_date_iso = last_date.date().isoformat()
        last_date_display = last_date.strftime("%d/%m/%Y")
    elif isinstance(last_date, _date):
        last_date_iso = last_date.isoformat()
        last_date_display = last_date.strftime("%d/%m/%Y")
    elif isinstance(last_date, str):
        last_date_iso = last_date[:10]
        last_date_display = last_date_iso

    clients_rows = []
    total_valo_clients = 0.0
    support_total_valo = 0.0
    if last_date_iso:
        support_total_valo = db.execute(
            text(
                """
                SELECT COALESCE(SUM(h.valo), 0) AS total_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_support s ON s.id = h.id_support
                WHERE s.code_isin = :code
                  AND DATE(h.date) = DATE(:last_date)
                """
            ),
            {"code": code_isin, "last_date": last_date_iso},
        ).scalar() or 0.0

        clients_rows = db.execute(
            text(
                """
                SELECT
                    c.id AS client_id,
                    c.nom AS client_nom,
                    c.prenom AS client_prenom,
                    c.commercial_id AS commercial_id,
                    SUM(h.valo) AS total_valo
                FROM mariadb_historique_support_w h
                JOIN mariadb_affaires a ON a.id = h.id_source
                JOIN mariadb_clients c ON c.id = a.id_personne
                JOIN mariadb_support s ON s.id = h.id_support
                WHERE s.code_isin = :code
                  AND DATE(h.date) = DATE(:last_date)
                GROUP BY c.id, c.nom, c.prenom, c.commercial_id
                ORDER BY total_valo DESC
                """
            ),
            {"code": code_isin, "last_date": last_date_iso},
        ).fetchall()
        total_valo_clients = sum(float(row.total_valo or 0) for row in clients_rows)

    rh_lookup = {}
    for rh in fetch_rh_list(db):
        rid = rh.get("id")
        try:
            rid_int = int(rid) if rid is not None else None
        except Exception:
            rid_int = None
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        mail = (rh.get("mail") or "").strip()
        parts = [p for p in [prenom, nom] if p]
        if rid_int is not None:
            if parts:
                rh_lookup[rid_int] = " ".join(parts)
            elif mail:
                rh_lookup[rid_int] = mail
            else:
                rh_lookup[rid_int] = f"RH #{rid_int}"

    clients_payload = []
    for row in clients_rows:
        mapping = getattr(row, "_mapping", row)
        rid = mapping.get("commercial_id")
        clients_payload.append(
            {
                "client_id": mapping.get("client_id"),
                "nom": mapping.get("client_nom"),
                "prenom": mapping.get("client_prenom"),
                "responsable": rh_lookup.get(rid),
                "total_valo": float(mapping.get("total_valo") or 0),
                "total_valo_str": "{:,.0f}".format(float(mapping.get("total_valo") or 0)).replace(",", " "),
            }
        )

    valuations_payload: list[dict] = []
    valuations_weekly_payload: list[dict] = []
    if support_ids:
        try:
            ids_list = [int(x) for x in support_ids if x is not None]
            if ids_list:
                val_query = text(
                    """
                    SELECT id_support, date, valeur
                    FROM mariadb_support_val
                    WHERE id_support IN :ids
                    ORDER BY date
                    """
                ).bindparams(bindparam("ids", expanding=True))
                val_rows = db.execute(val_query, {"ids": ids_list}).fetchall()
            else:
                val_rows = []
            def _format_val_date(raw):
                if isinstance(raw, datetime):
                    return raw.date().isoformat()
                if isinstance(raw, _date):
                    return raw.isoformat()
                if raw is None:
                    return None
                try:
                    raw_str = str(raw)
                    if not raw_str:
                        return None
                    if len(raw_str) >= 10:
                        return raw_str[:10]
                    return raw_str
                except Exception:
                    return None
            for row in val_rows or []:
                mapping = row._mapping if hasattr(row, "_mapping") else None
                date_raw = mapping.get("date") if mapping else (row[1] if len(row) > 1 else None)
                valeur_raw = mapping.get("valeur") if mapping else (row[2] if len(row) > 2 else None)
                try:
                    valeur_num = float(valeur_raw) if valeur_raw is not None else None
                except Exception:
                    valeur_num = None
                valuations_payload.append(
                    {
                        "date": _format_val_date(date_raw),
                        "valeur": valeur_num,
                    }
                )
            if ids_list:
                db_engine = str(db.bind.engine.name).lower() if db.bind and db.bind.engine else "sqlite"
                if "sqlite" in db_engine:
                    weekly_query = text(
                        """
                        SELECT
                            strftime('%Y-%W', h.date) AS iso_week,
                            date(h.date, '-' || ((strftime('%w', h.date) + 6) % 7) || ' days') AS week_start,
                            SUM(h.valo) AS total_valo
                        FROM mariadb_historique_support_w h
                        WHERE h.id_support IN :ids
                        GROUP BY iso_week, week_start
                        ORDER BY week_start
                        """
                    ).bindparams(bindparam("ids", expanding=True))
                else:
                    weekly_query = text(
                        """
                        SELECT
                            DATE_FORMAT(DATE(h.date), '%x-%v') AS iso_week,
                            DATE(DATE_ADD(DATE(h.date), INTERVAL (1 - DAYOFWEEK(DATE(h.date))) DAY)) AS week_start,
                            SUM(h.valo) AS total_valo
                        FROM mariadb_historique_support_w h
                        WHERE h.id_support IN :ids
                        GROUP BY iso_week, week_start
                        ORDER BY week_start
                        """
                    ).bindparams(bindparam("ids", expanding=True))
                weekly_rows = db.execute(weekly_query, {"ids": ids_list}).fetchall()
            else:
                weekly_rows = []
        except Exception:
            valuations_payload = []
            weekly_rows = []

        for row in weekly_rows or []:
            mapping = row._mapping if hasattr(row, "_mapping") else None
            week_start_raw = mapping.get("week_start") if mapping else (row[1] if len(row) > 1 else None)
            total_valo_raw = mapping.get("total_valo") if mapping else (row[2] if len(row) > 2 else None)
            try:
                total_valo_num = float(total_valo_raw)
            except Exception:
                total_valo_num = None
            def _fmt_week_date(raw):
                if isinstance(raw, datetime):
                    return raw.date().isoformat()
                if isinstance(raw, _date):
                    return raw.isoformat()
                if raw is None:
                    return None
                try:
                    text_val = str(raw)
                    if len(text_val) >= 10:
                        return text_val[:10]
                    return text_val
                except Exception:
                    return None
            valuations_weekly_payload.append(
                {
                    "week_start": _fmt_week_date(week_start_raw),
                    "total_valo": total_valo_num,
                }
            )

    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(["Code ISIN", support["code_isin"]])
        writer.writerow(["Nom", support["nom"]])
        writer.writerow(["Catégorie principale", support["cat_principale"] or ""])
        writer.writerow(["Catégorie détaillée", support["cat_det"] or ""])
        writer.writerow(["Catégorie générique", support["cat_gene"] or ""])
        writer.writerow(["Zone géographique", support["cat_geo"] or ""])
        writer.writerow(["Promoteur", support["promoteur"] or ""])
        writer.writerow(["SRRI", support["SRRI"] if support["SRRI"] is not None else ""])
        writer.writerow(["Date", last_date_display or ""])
        writer.writerow([])
        writer.writerow(["Clients"])
        writer.writerow(["ID Client", "Nom", "Prénom", "Responsable", "Valorisation (€)"])
        for client in clients_payload:
            writer.writerow(
                [
                    client["client_id"],
                    client["nom"],
                    client["prenom"],
                    client["responsable"] or "",
                    "{:,.0f}".format(client["total_valo"]).replace(",", " "),
                ]
            )
        writer.writerow([])
        writer.writerow(["Total support", "", "", "", "{:,.0f}".format(support_total_valo).replace(",", " ")])
        if total_valo_clients:
            writer.writerow(["Total clients", "", "", "", "{:,.0f}".format(total_valo_clients).replace(",", " ")])
        output.seek(0)
        filename = f"support_{support['code_isin']}_{last_date_iso or 'latest'}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    task_types_map = _fetch_task_types(db)
    def _format_cat(cat: str) -> str:
        if not cat:
            return "Autre"
        cat = cat.replace('_', ' ').strip()
        return cat[:1].upper() + cat[1:]
    task_types_payload = [
        {
            "categorie": cat,
            "label": _format_cat(cat),
            "types": [
                {
                    "id": entry.get("id"),
                    "libelle": entry.get("libelle")
                }
                for entry in entries
            ]
        }
        for cat, entries in task_types_map.items()
        if entries
    ]
    default_type_id = None
    for group in task_types_payload:
        for entry in group["types"]:
            if entry.get("id") is not None:
                default_type_id = entry["id"]
                break
        if default_type_id is not None:
            break

    return JSONResponse(
        {
            "support": {
                "code_isin": support["code_isin"],
                "nom": support["nom"],
                "cat_gene": support["cat_gene"],
                "cat_principale": support["cat_principale"],
                "cat_det": support["cat_det"],
                "cat_geo": support["cat_geo"],
                "promoteur": support["promoteur"],
                "SRRI": support["SRRI"],
            },
            "last_date": last_date_display,
            "clients": clients_payload,
            "total_valo": support_total_valo,
            "total_valo_str": "{:,.0f}".format(support_total_valo).replace(",", " "),
            "clients_total_valo": total_valo_clients,
            "clients_total_valo_str": "{:,.0f}".format(total_valo_clients).replace(",", " "),
            "support_ids": support_ids,
            "task_types": task_types_payload,
            "default_type_id": default_type_id,
            "valuations_weekly": valuations_weekly_payload,
            "valuations": valuations_payload,
            "csv_url": f"/dashboard/supports/details?code_isin={code_isin}&format=csv",
        }
    )


@router.post("/supports/tasks")
async def dashboard_supports_create_tasks(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    code_isin = str(data.get("code_isin") or "").strip()
    client_ids_raw = data.get("client_ids") or []
    type_id_raw = data.get("type_id")
    commentaire = (data.get("commentaire") or "").strip()

    if not code_isin:
        raise HTTPException(status_code=400, detail="Code ISIN manquant.")
    if not isinstance(client_ids_raw, list) or not client_ids_raw:
        raise HTTPException(status_code=400, detail="Aucun client sélectionné.")
    try:
        type_id = int(type_id_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Type de tâche invalide.")

    type_row = db.execute(
        text(
            """
            SELECT id, libelle, categorie
            FROM mariadb_type_evenement
            WHERE id = :id
            LIMIT 1
            """
        ),
        {"id": type_id},
    ).mappings().first()
    if not type_row:
        raise HTTPException(status_code=404, detail="Type de tâche introuvable.")

    support_row = db.execute(
        text(
            """
            SELECT id
            FROM mariadb_support
            WHERE code_isin = :code
            ORDER BY id ASC
            LIMIT 1
            """
        ),
        {"code": code_isin},
    ).fetchone()
    support_id = support_row[0] if support_row else None

    try:
        client_ids = [int(cid) for cid in client_ids_raw]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Liste de clients invalide.")
    if not client_ids:
        raise HTTPException(status_code=400, detail="Liste de clients vide.")

    clients = db.query(Client.id, Client.commercial_id, Client.nom, Client.prenom).filter(Client.id.in_(client_ids)).all()
    if not clients:
        raise HTTPException(status_code=404, detail="Clients introuvables.")
    clients_map = {c.id: c for c in clients}

    rh_lookup = {}
    for rh in fetch_rh_list(db):
        rid = rh.get("id")
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            continue
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        mail = (rh.get("mail") or "").strip()
        label_parts = [p for p in [prenom, nom] if p]
        label = " ".join(label_parts).strip()
        if not label:
            label = mail or f"RH #{rid_int}"
        rh_lookup[rid_int] = label

    created = []
    skipped = []
    for cid in client_ids:
        client = clients_map.get(cid)
        if not client:
            skipped.append({"client_id": cid, "reason": "inconnu"})
            continue
        rh_id = None
        try:
            rh_id = int(client.commercial_id) if client.commercial_id is not None else None
        except Exception:
            rh_id = None
        responsable_label = rh_lookup.get(rh_id) if rh_id is not None else None
        if commentaire:
            comment = commentaire if code_isin.upper() in commentaire.upper() else f"{commentaire} (Support {code_isin})"
        else:
            comment = f"Suivi automatique pour le support {code_isin}"
        payload = TacheCreateSchema(
            type_libelle=type_row.get("libelle") or "tâche",
            categorie=(type_row.get("categorie") or "tache"),
            client_id=client.id,
            support_id=support_id,
            statut="à faire",
            commentaire=comment,
            utilisateur_responsable=responsable_label,
            rh_id=rh_id,
        )
        ev = create_tache(db, payload)
        created.append(ev.id)

    return JSONResponse(
        {
            "created": len(created),
            "created_ids": created,
            "skipped": skipped,
            "message": f"{len(created)} tâche(s) créée(s) pour le support {code_isin}."
        }
    )


# ---------------- Tâches / Événements ----------------
@router.get("/taches", response_class=HTMLResponse)
def dashboard_taches(
    request: Request,
    db: Session = Depends(get_db),
    statut: str | None = None,
    categorie: str | None = None,
    client_id: int | None = None,
    affaire_id: int | None = None,
    intervenant: str | None = None,
    type_text: str | None = None,
    today: int | None = None,
    late: int | None = None,
    exclude_statut: str | None = None,
):
    from sqlalchemy import text
    from src.models.evenement import Evenement as EvenementModel
    conds = []
    params: dict = {}
    if statut:
        conds.append("statut = :statut")
        params["statut"] = statut
    if categorie:
        conds.append("categorie = :categorie")
        params["categorie"] = categorie
    if client_id is not None:
        conds.append("client_id = :client_id")
        params["client_id"] = client_id
    if affaire_id is not None:
        conds.append("affaire_id = :affaire_id")
        params["affaire_id"] = affaire_id
    if intervenant:
        conds.append("intervenants LIKE :interv")
        params["interv"] = f"%{intervenant}%"
    if type_text:
        conds.append("type_evenement LIKE :type_txt")
        params["type_txt"] = f"%{type_text}%"
    # Quick filters
    from datetime import date as _dt_date3
    today_str = _dt_date3.today().isoformat()
    if today:
        conds.append("date(date_evenement) = :today")
        params["today"] = today_str
    if late:
        conds.append("date(date_evenement) < :today")
        params["today"] = today_str
        # Not finished
        conds.append("(statut IS NULL OR lower(statut) NOT IN ('terminé','termine','cloturé','cloture','clôturé','annulé','annule'))")
    if exclude_statut:
        conds.append("(statut IS NULL OR lower(statut) != lower(:exclude_statut))")
        params["exclude_statut"] = exclude_statut
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql_exists = text(
        """
        SELECT COUNT(*) AS nb
        FROM information_schema.tables
        WHERE table_schema = database() AND table_name = 'vue_suivi_evenement'
        """
    )
    try:
        exists = db.execute(sql_exists).scalar() or 0
    except Exception:
        exists = 0
    items = []
    if exists:
        try:
            sql = f"SELECT * FROM vue_suivi_evenement{where} ORDER BY date_evenement DESC LIMIT 300"
            items = db.execute(text(sql), params).fetchall()
        except Exception as exc:
            logger.exception("vue_suivi_evenement indisponible", exc_info=exc)
            items = []

    # Enrichir avec noms client, affaire et RH pour l'affichage
    rh_list = fetch_rh_list(db)
    rh_options: list[dict] = []
    rh_lookup: dict[int, str] = {}
    for rh in rh_list:
        rid = rh.get("id")
        try:
            rid_int = int(rid) if rid is not None else None
        except Exception:
            rid_int = None
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        parts = [p for p in [prenom, nom] if p]
        label = " ".join(parts) if parts else (rh.get("mail") or "").strip()
        if not label and rid_int is not None:
            label = f"RH #{rid_int}"
        if rid_int is not None:
            rh_lookup[rid_int] = label
        rh_options.append({"id": rid, "label": label or (f"RH #{rid}" if rid is not None else "RH")})
    rh_options.sort(key=lambda x: (x["label"] or "").lower())
    rh_lookup: dict[int, str] = {}
    for rh in rh_list:
        rid = rh.get("id")
        try:
            rid_int = int(rid) if rid is not None else None
        except Exception:
            rid_int = None
        prenom = (rh.get("prenom") or "").strip()
        nom = (rh.get("nom") or "").strip()
        parts = [p for p in [prenom, nom] if p]
        label = " ".join(parts) if parts else (rh.get("mail") or "").strip()
        if not label and rid_int is not None:
            label = f"RH #{rid_int}"
        if rid_int is not None:
            rh_lookup[rid_int] = label
    client_ids = {getattr(r, 'client_id', None) for r in items if getattr(r, 'client_id', None) is not None}
    affaire_ids = {getattr(r, 'affaire_id', None) for r in items if getattr(r, 'affaire_id', None) is not None}
    clients_map_full = {}
    affaires_map_ref = {}
    if client_ids:
        rows_cli = db.query(Client.id, Client.nom, Client.prenom).filter(Client.id.in_(list(client_ids))).all()
        for cid, nom, prenom in rows_cli:
            full = f"{nom or ''} {prenom or ''}".strip()
            clients_map_full[cid] = full or (nom or prenom) or str(cid)
    if affaire_ids:
        rows_aff = db.query(Affaire.id, Affaire.ref).filter(Affaire.id.in_(list(affaire_ids))).all()
        for aid, ref in rows_aff:
            affaires_map_ref[aid] = ref or str(aid)
    converted_items = []
    possible_rh_keys = ('rh_id', 'rhId', 'responsable_id', 'responsableId', 'responsable_rh_id', 'responsableRhId')
    for r in items:
        base = dict(getattr(r, '_mapping', r))
        # Assure un identifiant numérique pour les liens /dashboard/taches/<id>; skip sinon
        eid_raw = base.get('evenement_id') if 'evenement_id' in base else base.get('id')
        try:
            eid_int = int(eid_raw)
        except Exception:
            eid_int = None
        if eid_int is None:
            continue
        base['evenement_id'] = eid_int
        base['nom_client'] = clients_map_full.get(base.get('client_id'))
        base['affaire_ref'] = affaires_map_ref.get(base.get('affaire_id'))
        rh_id_val = None
        for key in possible_rh_keys:
            raw_rh = base.get(key)
            if raw_rh in (None, ''):
                continue
            try:
                rh_id_val = int(raw_rh)
                break
            except Exception:
                try:
                    rh_id_val = int(float(raw_rh))
                    break
                except Exception:
                    rh_id_val = None
        base['rh_id'] = rh_id_val
        if rh_id_val is not None:
            base['rh_label'] = rh_lookup.get(rh_id_val) or f"RH #{rh_id_val}"
        else:
            base['rh_label'] = ''
        converted_items.append(base)
    items = converted_items

    # Récupération du statut de réclamation associé aux événements listés
    ev_ids = [base.get('evenement_id') for base in items if base.get('evenement_id')]
    reclam_lookup: dict[int, dict] = {}
    if ev_ids:
        rows_reclam = (
            db.query(
                EvenementModel.id,
                EvenementModel.statut_reclamation_id,
                EvenementStatutReclamation.libelle,
            )
            .outerjoin(
                EvenementStatutReclamation,
                EvenementStatutReclamation.id == EvenementModel.statut_reclamation_id,
            )
            .filter(EvenementModel.id.in_(ev_ids))
            .all()
        )
        for row in rows_reclam:
            eid = getattr(row, 'id', None)
            if eid is None:
                continue
            reclam_lookup[eid] = {
                "id": getattr(row, 'statut_reclamation_id', None),
                "label": getattr(row, 'libelle', None),
            }
    for base in items:
        eid = base.get('evenement_id')
        info = reclam_lookup.get(eid, {})
        base['statut_reclamation_id'] = info.get('id')
        base['statut_reclamation_label'] = info.get('label')

    # Options types & catégories pour filtres/creation
    types = db.execute(text("SELECT id, libelle, categorie FROM mariadb_type_evenement ORDER BY categorie, libelle")).fetchall()
    cats = sorted({t.categorie for t in types if getattr(t, 'categorie', None)})

    # Statuts (pour formulaire inline)
    statuts = list_statuts(db)
    def _norm(s: str | None) -> str | None:
        if not s:
            return None
        x = s.strip().lower()
        repl = {
            "à": "a", "â": "a", "ä": "a",
            "é": "e", "è": "e", "ê": "e", "ë": "e",
            "î": "i", "ï": "i",
            "ô": "o", "ö": "o",
            "û": "u", "ü": "u",
            "ç": "c",
        }
        for k, v in repl.items():
            x = x.replace(k, v)
        return x
    stat_ids: dict[str, int] = {}
    for s in statuts:
        key = _norm(getattr(s, 'libelle', None))
        if key and getattr(s, 'id', None) is not None:
            stat_ids[key] = s.id
    # UI order and labels
    status_ui = []
    for label_ui, key in [
        ("à faire", "a faire"),
        ("en cours", "en cours"),
        ("en attente", "en attente"),
        ("terminé", "termine"),
        ("annulé", "annule"),
    ]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")

    reclam_statuses = (
        db.query(EvenementStatutReclamation)
        .order_by(EvenementStatutReclamation.is_cloture.asc(), EvenementStatutReclamation.libelle.asc())
        .all()
    )

    # Suggestions Clients / Affaires (ergonomie création)
    clients_suggest = (
        db.query(Client.id, Client.nom, Client.prenom)
        .order_by(Client.nom.asc(), Client.prenom.asc())
        .all()
    )
    # Affaires avec nom client (pour affichage dans datalist, saisie par référence)
    aff_rows = (
        db.query(Affaire.id, Affaire.ref, Affaire.id_personne)
        .order_by(Affaire.ref.asc())
        .all()
    )
    clients_map = {c.id: f"{getattr(c, 'nom', '') or ''} {getattr(c, 'prenom', '') or ''}".strip() for c in clients_suggest}
    affaires_suggest = [
        {
            "id": a.id,
            "ref": getattr(a, 'ref', ''),
            "client": clients_map.get(getattr(a, 'id_personne', None), ''),
        }
        for a in aff_rows
    ]

    # Badges/compteurs pour quick filters
    def _count(sql_text: str, params_: dict):
        try:
            return int(db.execute(text(sql_text), params_).scalar() or 0)
        except Exception:
            return 0

    today_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE date(date_evenement) = :d",
        {"d": today_str},
    )
    late_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE date(date_evenement) < :d AND (statut IS NULL OR lower(statut) NOT IN ('terminé','termine','cloturé','cloture','clôturé','annulé','annule'))",
        {"d": today_str},
    )
    reclamations_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE categorie = 'reclamation' AND (statut IS NULL OR lower(statut) NOT IN ('terminé','termine','cloturé','cloture','clôturé'))",
        {},
    )
    reclamations_terminees_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE categorie = 'reclamation' AND lower(statut) IN ('terminé','termine','cloturé','cloture','clôturé')",
        {},
    )
    en_attente_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE lower(statut) = 'en attente'",
        {},
    )
    a_faire_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE lower(statut) IN ('à faire','a faire')",
        {},
    )
    en_cours_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE lower(statut) = 'en cours'",
        {},
    )
    termine_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE lower(statut) IN ('terminé','termine')",
        {},
    )
    annule_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement WHERE lower(statut) IN ('annulé','annule')",
        {},
    )
    total_count = _count(
        "SELECT COUNT(1) FROM vue_suivi_evenement",
        {},
    )

    # Livre des réclamations (pour modal dédié)
    reclam_stats, reclam_pivot = _compute_reclamations_data(db)

    return templates.TemplateResponse(
        "dashboard_taches.html",
        {
            "request": request,
            "items": items,
            "types": types,
            "categories": cats,
            "statuts": statuts,
            "status_ui": status_ui,
            "en_cours_id": en_cours_id,
            "clients_suggest": clients_suggest,
            "affaires_suggest": affaires_suggest,
            "counts": {
                "total": total_count,
                "today": today_count,
                "late": late_count,
                "reclamations": reclamations_count,
                "reclamations_terminees": reclamations_terminees_count,
                "en_attente": en_attente_count,
                "a_faire": a_faire_count,
                "en_cours": en_cours_count,
                "termine": termine_count,
                "annule": annule_count,
            },
            "filters": {
                "statut": statut or "",
                "categorie": categorie or "",
                "client_id": client_id,
                "affaire_id": affaire_id,
                "intervenant": intervenant or "",
                "type_text": type_text or "",
                "today": int(today or 0),
                "late": int(late or 0),
                "exclude_statut": exclude_statut or "",
            },
            "rh_list": rh_list,
            "rh_options": rh_options,
            "reclam_statuses": reclam_statuses,
            "reclam_stats": reclam_stats,
            "reclam_pivot": reclam_pivot,
        },
    )


# ---------------- Mouvements ----------------
@router.get("/mouvements", response_class=HTMLResponse)
def dashboard_mouvements(
    request: Request,
    db: Session = Depends(get_db),
    affaire_id: int | None = None,
    avis_id: int | None = None,
):
    from sqlalchemy import text as _text
    conds = []
    params: dict = {}
    if affaire_id is not None:
        conds.append("m.id_affaire = :aid")
        params["aid"] = affaire_id
    if avis_id is not None:
        conds.append("m.id_avis = :vid")
        params["vid"] = avis_id
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT m.id,
               m.modif_quand,
               m.id_affaire,
               m.id_avis,
               m.id_support,
               m.vl_date,
               m.date_sp,
               mr.titre AS regle,
               mr.sens AS sens,
               m.montant_ope,
               m.frais,
               m.vl,
               m.nb_uc,
               s.code_isin AS support_isin,
               s.nom AS support_nom
        FROM mouvement m
        LEFT JOIN mouvement_regle mr ON mr.id = m.id_mouvement_regle
        LEFT JOIN mariadb_support s ON s.id = m.id_support
        {where}
        ORDER BY COALESCE(m.vl_date, m.date_sp) DESC, m.id DESC
    """
    rows = db.execute(_text(sql), params).fetchall()

    def _to_date_str(x):
        if x is None:
            return None
        try:
            return x.strftime("%Y-%m-%d")
        except Exception:
            s = str(x)
            return s[:10]

    def _fmt2(v):
        try:
            return "{:,.2f}".format(float(v or 0)).replace(",", " ")
        except Exception:
            return v

    items_pos, items_neg = [], []
    tot_pos_montant = tot_pos_frais = 0.0
    tot_neg_montant = tot_neg_frais = 0.0
    for r in rows:
        montant_val = float(getattr(r, "montant_ope", 0) or 0)
        frais_val = float(getattr(r, "frais", 0) or 0)
        item = {
            "id": getattr(r, "id", None),
            "date": _to_date_str(getattr(r, "vl_date", None) or getattr(r, "date_sp", None)),
            "regle": getattr(r, "regle", None),
            "support": "{} {}".format(
                (getattr(r, "support_isin", None) or "").strip(),
                (getattr(r, "support_nom", None) or "").strip(),
            ).strip(),
            "montant": _fmt2(montant_val),
            "frais": _fmt2(frais_val),
            "vl": _fmt2(getattr(r, "vl", None)),
            "nb_uc": _fmt2(getattr(r, "nb_uc", None)),
        }
        sens_val = getattr(r, "sens", None)
        try:
            is_pos = (int(sens_val) > 0) if sens_val is not None else (montant_val >= 0)
        except Exception:
            is_pos = (montant_val >= 0)
        if is_pos:
            items_pos.append(item)
            tot_pos_montant += montant_val
            tot_pos_frais += frais_val
        else:
            items_neg.append(item)
            tot_neg_montant += montant_val
            tot_neg_frais += frais_val

    return templates.TemplateResponse(
        "dashboard_mouvements.html",
        {
            "request": request,
            "items_pos": items_pos,
            "items_neg": items_neg,
            "tot_pos_montant": _fmt2(tot_pos_montant),
            "tot_pos_frais": _fmt2(tot_pos_frais),
            "tot_neg_montant": _fmt2(tot_neg_montant),
            "tot_neg_frais": _fmt2(tot_neg_frais),
            "affaire_id": affaire_id,
            "avis_id": avis_id,
        },
    )

@router.get("/taches/{evenement_id}", response_class=HTMLResponse)
def dashboard_tache_edit(
    evenement_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    # Charger l'évènement et ses métadonnées
    from src.models.evenement import Evenement
    from src.models.type_evenement import TypeEvenement
    ev = (
        db.query(
            Evenement.id,
            Evenement.date_evenement,
            Evenement.statut,
            Evenement.commentaire,
            Evenement.client_id,
            Evenement.affaire_id,
            Evenement.rh_id,
            Evenement.statut_reclamation_id,
            TypeEvenement.libelle.label("type_libelle"),
            TypeEvenement.categorie.label("type_categorie"),
            EvenementStatutReclamation.libelle.label("statut_reclamation_libelle"),
        )
        .join(TypeEvenement, TypeEvenement.id == Evenement.type_id)
        .outerjoin(
            EvenementStatutReclamation,
            EvenementStatutReclamation.id == Evenement.statut_reclamation_id,
        )
        .filter(Evenement.id == evenement_id)
        .first()
    )
    if not ev:
        return templates.TemplateResponse(
            "dashboard_tache_edit.html",
            {"request": request, "error": "Tâche introuvable", "evenement_id": evenement_id},
        )
    # Libellés client / affaire
    cli = db.query(Client.id, Client.nom, Client.prenom).filter(Client.id == ev.client_id).first() if getattr(ev, 'client_id', None) else None
    aff = db.query(Affaire.id, Affaire.ref).filter(Affaire.id == ev.affaire_id).first() if getattr(ev, 'affaire_id', None) else None
    nom_client = (f"{getattr(cli, 'nom', '') or ''} {getattr(cli, 'prenom', '') or ''}".strip()) if cli else None
    ref_affaire = getattr(aff, 'ref', None) if aff else None

    # Statuts pour actions
    statuts = list_statuts(db)
    def _norm(s: str | None) -> str | None:
        if not s:
            return None
        x = s.strip().lower()
        for a,b in [("à","a"),("â","a"),("ä","a"),("é","e"),("è","e"),("ê","e"),("ë","e"),("î","i"),("ï","i"),("ô","o"),("ö","o"),("û","u"),("ü","u"),("ç","c")]:
            x = x.replace(a,b)
        return x
    stat_ids: dict[str,int] = {}
    for s in statuts:
        k = _norm(getattr(s,'libelle',None))
        if k and getattr(s,'id',None) is not None:
            stat_ids[k] = s.id
    status_ui = []
    for label_ui, key in [
        ("à faire", "a faire"),
        ("en cours", "en cours"),
        ("en attente", "en attente"),
        ("terminé", "termine"),
        ("annulé", "annule"),
    ]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")

    # Formater commentaires en entrées distinctes (timestamp + texte)
    comment_entries: list[dict] = []
    try:
        raw = getattr(ev, 'commentaire', None) or ''
        if raw:
            lines = raw.splitlines()
            cur = None
            for line in lines:
                if line.strip().startswith('[') and ']' in line:
                    # nouvelle entrée
                    if cur:
                        comment_entries.append(cur)
                    ts = line.strip()[1:line.strip().find(']')]
                    cur = { 'ts': ts, 'text': '' }
                else:
                    if cur is None:
                        # texte sans en-tête → tout dans une seule entrée
                        cur = { 'ts': None, 'text': '' }
                    cur['text'] = (cur['text'] + ('\n' if cur['text'] else '') + line).rstrip()
            if cur:
                comment_entries.append(cur)
    except Exception:
        comment_entries = []

    rh_list = fetch_rh_list(db)
    rh_selected = getattr(ev, "rh_id", None)

    reclam_statuses = (
        db.query(EvenementStatutReclamation)
        .order_by(EvenementStatutReclamation.is_cloture.asc(), EvenementStatutReclamation.libelle.asc())
        .all()
    )

    prefill_comment = request.query_params.get("prefill_comment")

    return templates.TemplateResponse(
        "dashboard_tache_edit.html",
        {
            "request": request,
            "ev": ev,
            "evenement_id": evenement_id,
            "nom_client": nom_client,
            "ref_affaire": ref_affaire,
            "status_ui": status_ui,
            "en_cours_id": en_cours_id,
            "comment_entries": comment_entries,
             "rh_list": rh_list,
             "rh_selected": rh_selected,
             "reclam_statuses": reclam_statuses,
            "prefill_comment": prefill_comment,
        },
    )


@router.post("/taches", response_class=HTMLResponse)
async def dashboard_taches_create(request: Request, db: Session = Depends(get_db)):
    from itertools import zip_longest
    from sqlalchemy import func

    form = await request.form()

    # Récupère des listes (plusieurs lignes)
    types_l = form.getlist("type_libelle")
    cats_l = form.getlist("categorie")
    clients_l = form.getlist("client_fullname")
    affaires_l = form.getlist("affaire_ref")
    responsables_l = form.getlist("utilisateur_responsable")
    rh_ids_l = form.getlist("rh_id")
    commentaires_l = form.getlist("commentaire")
    reclam_statuts_l = form.getlist("statut_reclamation_id")

    def resolve_client(fullname: str):
        if not fullname:
            return None
        s = fullname.strip()
        # Format attendu: "Nom;Prénom" ou "Nom Prénom"
        nom = None
        prenom = None
        if ";" in s:
            parts = [p.strip() for p in s.split(";", 1)]
            nom = parts[0] or None
            prenom = parts[1] or None
        else:
            # tente split dernier espace
            parts = s.rsplit(" ", 1)
            if len(parts) == 2:
                nom, prenom = parts[0].strip() or None, parts[1].strip() or None
            else:
                nom = s
        q = db.query(Client).filter(func.lower(Client.nom) == func.lower(nom)) if nom else None
        if q is None:
            return None
        if prenom:
            q = q.filter(func.lower(Client.prenom) == func.lower(prenom))
        return q.first()

    def resolve_affaire(ref: str, client_id: int | None):
        if not ref:
            return None
        q = db.query(Affaire).filter(func.lower(Affaire.ref) == func.lower(ref))
        if client_id is not None:
            q = q.filter(Affaire.id_personne == client_id)
        return q.first()

    # Crée chaque ligne non vide
    for type_lbl, cat, cli_full, aff_ref, resp, comm, reclam_stat_raw, rh_raw in zip_longest(
        types_l,
        cats_l,
        clients_l,
        affaires_l,
        responsables_l,
        commentaires_l,
        reclam_statuts_l,
        rh_ids_l,
        fillvalue="",
    ):
        has_content = (type_lbl or comm or cli_full or aff_ref)
        if not has_content:
            continue
        cli = resolve_client(cli_full)
        aff = resolve_affaire(aff_ref, getattr(cli, "id", None)) if aff_ref else None
        # Si type non fourni mais catégorie présente, choisir le premier type de la catégorie
        if (not type_lbl) and (cat):
            try:
                from sqlalchemy import text as _text
                row = db.execute(_text("SELECT libelle FROM mariadb_type_evenement WHERE categorie = :c ORDER BY libelle LIMIT 1"), {"c": cat}).fetchone()
                if row and row[0]:
                    type_lbl = row[0]
            except Exception:
                pass
        try:
            rh_id = int(rh_raw) if str(rh_raw).strip() else None
        except Exception:
            rh_id = None
        try:
            statut_reclam_id = int(reclam_stat_raw) if str(reclam_stat_raw).strip() else None
        except Exception:
            statut_reclam_id = None
        cat_lower = (cat or "").strip().lower()
        if cat_lower != "reclamation":
            statut_reclam_id = None

        payload = TacheCreateSchema(
            type_libelle=(type_lbl or "tâche").strip(),
            categorie=(cat or "tache").strip() or "tache",
            client_id=getattr(cli, "id", None),
            affaire_id=getattr(aff, "id", None),
            commentaire=comm or None,
            utilisateur_responsable=(resp or None),
            rh_id=rh_id,
            statut_reclamation_id=statut_reclam_id,
        )
        ev = create_tache(db, payload)
        if not ev:
            continue

        # Statut initial si fourni
        try:
            sid = int(form.get("statut_id")) if form.get("statut_id") else None
        except Exception:
            sid = None
        if sid:
            add_statut_to_evenement(db, ev.id, EvenementStatutCreateSchema(statut_id=sid, commentaire="Création via dashboard", utilisateur_responsable=resp or None))

        # Communication éventuelle
        if form.get("comm_toggle"):
            canal = form.get("comm_canal") or "email"
            dest = form.get("comm_destinataire") or ""
            obj = form.get("comm_objet") or None
            cont = form.get("comm_contenu") or None
            if dest:
                create_envoi(db, ev.id, EvenementEnvoiCreateSchema(canal=canal, destinataire=dest, objet=obj, contenu=cont))
    return RedirectResponse(url="/dashboard/taches", status_code=303)


@router.post("/taches/{evenement_id}/statut", response_class=HTMLResponse)
async def dashboard_taches_add_statut(evenement_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    statut_id = form.get("statut_id")
    commentaire = form.get("commentaire")
    statut_reclamation_raw = form.get("statut_reclamation_id")
    user = form.get("utilisateur_responsable")
    redirect_to = form.get("redirect") or "/dashboard/taches"
    rh_id_raw = form.get("rh_id")
    rh_id_val: int | None = None
    if rh_id_raw is not None:
        try:
            rh_id_val = int(rh_id_raw) if str(rh_id_raw).strip() else None
        except Exception:
            rh_id_val = None
    comment_text = (commentaire or '').strip()

    ev_model = None
    needs_event_model = bool(comment_text) or (rh_id_raw is not None) or (statut_reclamation_raw is not None)
    if needs_event_model:
        try:
            from src.models.evenement import Evenement as _Ev
            ev_model = db.query(_Ev).filter(_Ev.id == evenement_id).first()
        except Exception:
            ev_model = None
    # Mise à jour du commentaire de la tâche: préfixer avec date-heure
    def _append_comment_entry(entry: str | None):
        if not entry or ev_model is None:
            return
        current = getattr(ev_model, "commentaire", None)
        if current:
            ev_model.commentaire = f"{current}\n{entry}"
        else:
            ev_model.commentaire = entry

    prefill_comment = request.query_params.get("prefill_comment")

    if comment_text or prefill_comment:
        try:
            if ev_model:
                from datetime import datetime as _dt
                ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M")
                base_text = comment_text
                if prefill_comment and (not comment_text or comment_text.strip() == prefill_comment.strip()):
                    base_text = prefill_comment
                text_trim = (base_text or "").lstrip()
                if text_trim.startswith("[") and "]" in text_trim.split("\n", 1)[0]:
                    processed = base_text
                else:
                    processed = f"[{ts}]\n{base_text}"
                if processed.strip():
                    _append_comment_entry(processed)
        except Exception:
            pass
    if ev_model is not None and rh_id_raw is not None:
        try:
            ev_model.rh_id = rh_id_val
        except Exception:
            pass
    prefill_comment: str | None = None
    recorded_prefill: str | None = None
    if ev_model is not None and statut_reclamation_raw is not None:
        old_reclam_id = getattr(ev_model, "statut_reclamation_id", None)
        try:
            new_reclam_id = int(statut_reclamation_raw) if str(statut_reclamation_raw).strip() else None
        except Exception:
            new_reclam_id = None
        if new_reclam_id != old_reclam_id:
            old_label = None
            if old_reclam_id is not None:
                try:
                    old_label = (
                        db.query(EvenementStatutReclamation.libelle)
                        .filter(EvenementStatutReclamation.id == old_reclam_id)
                        .scalar()
                    )
                except Exception:
                    old_label = None
            ts_change = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            label_text = old_label or "—"
            prefill_comment = f"[{ts_change}]\nStatut réclamation précédent : {label_text}. motif : "
            recorded_prefill = prefill_comment
        try:
            ev_model.statut_reclamation_id = new_reclam_id
        except Exception:
            ev_model.statut_reclamation_id = None

    if ev_model is not None:
        try:
            db.add(ev_model)
            db.commit()
        except Exception:
            db.rollback()

    try:
        sid = int(statut_id) if statut_id else None
    except Exception:
        sid = None
    if sid is not None:
        payload = EvenementStatutCreateSchema(
            statut_id=sid,
            commentaire=(comment_text or None),
            utilisateur_responsable=user,
        )
        add_statut_to_evenement(db, evenement_id, payload)
    if recorded_prefill and not comment_text:
        try:
            _append_comment_entry(recorded_prefill.rstrip())
        except Exception:
            pass

    if prefill_comment:
        parts = urlsplit(redirect_to)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["prefill_comment"] = prefill_comment
        redirect_to = urlunsplit(parts._replace(query=urlencode(query, doseq=True)))

    return RedirectResponse(url=redirect_to, status_code=303)


@router.post("/taches/{evenement_id}/update", response_class=HTMLResponse)
async def dashboard_taches_update(
    evenement_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    from datetime import datetime as _dt
    form = await request.form()
    ev = db.query(Evenement).filter(Evenement.id == evenement_id).first()
    if not ev:
        return RedirectResponse(url="/dashboard/taches", status_code=303)

    date_raw = form.get("date_evenement")
    type_label = (form.get("type_libelle") or "").strip()
    cat_label = (form.get("categorie") or "").strip()
    client_raw = form.get("client_id")
    affaire_raw = form.get("affaire_id")

    if date_raw:
        try:
            ev.date_evenement = _dt.fromisoformat(date_raw.replace(" ", "T"))
        except Exception:
            pass
    if type_label:
        q = db.query(TypeEvenement).filter(func.lower(TypeEvenement.libelle) == func.lower(type_label))
        if cat_label:
            q = q.filter(func.lower(TypeEvenement.categorie) == func.lower(cat_label))
        type_row = q.first()
        if type_row and getattr(type_row, "id", None):
            ev.type_id = type_row.id
    if client_raw is not None:
        try:
            ev.client_id = int(client_raw) if str(client_raw).strip() else None
        except Exception:
            ev.client_id = None
    if affaire_raw is not None:
        try:
            ev.affaire_id = int(affaire_raw) if str(affaire_raw).strip() else None
        except Exception:
            ev.affaire_id = None

    try:
        db.commit()
    except Exception:
        db.rollback()
    return RedirectResponse(url=f"/dashboard/taches/{evenement_id}", status_code=303)


# ---------------- Création Tâche depuis Détail Client ----------------
@router.post("/clients/{client_id}/taches", response_class=HTMLResponse)
async def dashboard_client_create_tache(client_id: int, request: Request, db: Session = Depends(get_db)):
    # Maintenance temporaire : création de tâche désactivée sur la fiche client
    return HTMLResponse(
        "<h3>Création de tâche indisponible (maintenance)</h3><p>Merci de réessayer plus tard.</p>",
        status_code=200,
    )
    form = await request.form()

    from sqlalchemy import func as _func

    def resolve_client(fullname: str):
        if not fullname:
            return None
        s = fullname.strip()
        nom = None
        prenom = None
        parts = s.rsplit(" ", 1)
        if len(parts) == 2:
            nom, prenom = parts[0].strip() or None, parts[1].strip() or None
        else:
            nom = s
        q = db.query(Client).filter(_func.lower(Client.nom) == _func.lower(nom)) if nom else None
        if q is None:
            return None
        if prenom:
            q = q.filter(_func.lower(Client.prenom) == _func.lower(prenom))
        return q.first()

    def resolve_affaire(ref: str, cid: int | None):
        if not ref:
            return None
        q = db.query(Affaire).filter(_func.lower(Affaire.ref) == _func.lower(ref))
        if cid is not None:
            q = q.filter(Affaire.id_personne == cid)
        return q.first()

    cli = resolve_client(form.get("client_fullname") or "")
    if cli is None:
        try:
            cli = db.get(Client, client_id)
        except Exception:
            cli = None
    aff = resolve_affaire(form.get("affaire_ref") or "", getattr(cli, "id", None))

    rh_id_raw = form.get("rh_id")
    try:
        rh_id_val = int(rh_id_raw) if rh_id_raw not in (None, "") else None
    except Exception:
        rh_id_val = None

    categorie_raw = (form.get("categorie") or "tache").strip() or "tache"
    statut_reclam_raw = form.get("statut_reclamation_id")
    try:
        statut_reclam_id = int(statut_reclam_raw) if statut_reclam_raw not in (None, "") else None
    except Exception:
        statut_reclam_id = None
    if categorie_raw.lower() != "reclamation":
        statut_reclam_id = None

    payload = TacheCreateSchema(
        type_libelle=(form.get("type_libelle") or "").strip(),
        categorie=categorie_raw,
        client_id=getattr(cli, "id", None),
        affaire_id=getattr(aff, "id", None),
        commentaire=form.get("commentaire") or None,
        utilisateur_responsable=form.get("utilisateur_responsable") or None,
        rh_id=rh_id_val,
        statut_reclamation_id=statut_reclam_id,
    )
    # Sélection automatique du type si vide et catégorie fournie
    if (not payload.type_libelle) and payload.categorie:
        try:
            from sqlalchemy import text as _text
            row = db.execute(_text("SELECT libelle FROM mariadb_type_evenement WHERE categorie = :c ORDER BY libelle LIMIT 1"), {"c": payload.categorie}).fetchone()
            if row and row[0]:
                payload.type_libelle = row[0]
        except Exception:
            pass
    if not payload.type_libelle:
        payload.type_libelle = "tâche"
    try:
        ev = create_tache(db, payload)
    except Exception as exc:
        logger.exception("Création tâche (client) échouée", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur création tâche: {exc}")
    if not ev:
        logger.error("Création de tâche impossible (client) payload=%s", payload.dict() if hasattr(payload, "dict") else payload)
        raise HTTPException(status_code=500, detail="Création de tâche impossible")

    # Statut initial
    sid = None
    try:
        sid = int(form.get("statut_id")) if form.get("statut_id") else None
    except Exception:
        sid = None
    if sid:
        try:
            add_statut_to_evenement(db, ev.id, EvenementStatutCreateSchema(statut_id=sid, commentaire="Création via client", utilisateur_responsable=payload.utilisateur_responsable))
            logger.info("Statut initial appliqué (client) ev_id=%s sid=%s", getattr(ev, "id", None), sid)
        except Exception as exc:
            logger.exception("Ajout statut tâche (client) échoué ev_id=%s", getattr(ev, "id", None), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erreur statut tâche: {exc}")

    # Communication éventuelle
    if form.get("comm_toggle") == "1":
        canal = form.get("comm_canal") or "email"
        dest = form.get("comm_destinataire") or ""
        obj = form.get("comm_objet") or None
        cont = form.get("comm_contenu") or None
        if dest:
            create_envoi(db, ev.id, EvenementEnvoiCreateSchema(canal=canal, destinataire=dest, objet=obj, contenu=cont))

    return RedirectResponse(url=f"/dashboard/clients/{client_id}", status_code=303)


# ---------------- Création Tâche depuis Détail Affaire ----------------
@router.post("/affaires/{affaire_id}/taches", response_class=HTMLResponse)
async def dashboard_affaire_create_tache(affaire_id: int, request: Request, db: Session = Depends(get_db)):
    # Maintenance temporaire : création de tâche désactivée sur la fiche affaire
    return HTMLResponse(
        "<h3>Création de tâche indisponible (maintenance)</h3><p>Merci de réessayer plus tard.</p>",
        status_code=200,
    )
    form = await request.form()

    from sqlalchemy import func as _func

    def resolve_client(fullname: str):
        if not fullname:
            return None
        s = fullname.strip()
        nom = None
        prenom = None
        parts = s.rsplit(" ", 1)
        if len(parts) == 2:
            nom, prenom = parts[0].strip() or None, parts[1].strip() or None
        else:
            nom = s
        q = db.query(Client).filter(_func.lower(Client.nom) == _func.lower(nom)) if nom else None
        if q is None:
            return None
        if prenom:
            q = q.filter(_func.lower(Client.prenom) == _func.lower(prenom))
        return q.first()

    def resolve_affaire(ref: str, cid: int | None):
        if not ref:
            return None
        q = db.query(Affaire).filter(_func.lower(Affaire.ref) == _func.lower(ref))
        if cid is not None:
            q = q.filter(Affaire.id_personne == cid)
        return q.first()

    cli = resolve_client(form.get("client_fullname") or "")
    aff = resolve_affaire(form.get("affaire_ref") or "", getattr(cli, "id", None))

    rh_id_raw = form.get("rh_id")
    try:
        rh_id_val = int(rh_id_raw) if rh_id_raw not in (None, "") else None
    except Exception:
        rh_id_val = None

    categorie_raw = (form.get("categorie") or "tache").strip() or "tache"
    statut_reclam_raw = form.get("statut_reclamation_id")
    try:
        statut_reclam_id = int(statut_reclam_raw) if statut_reclam_raw not in (None, "") else None
    except Exception:
        statut_reclam_id = None
    if categorie_raw.lower() != "reclamation":
        statut_reclam_id = None

    payload = TacheCreateSchema(
        type_libelle=(form.get("type_libelle") or "").strip(),
        categorie=categorie_raw,
        client_id=getattr(cli, "id", None),
        affaire_id=getattr(aff, "id", None),
        commentaire=form.get("commentaire") or None,
        utilisateur_responsable=form.get("utilisateur_responsable") or None,
        rh_id=rh_id_val,
        statut_reclamation_id=statut_reclam_id,
    )
    # Sélection automatique du type si vide et catégorie fournie
    if (not payload.type_libelle) and payload.categorie:
        try:
            from sqlalchemy import text as _text
            row = db.execute(_text("SELECT libelle FROM mariadb_type_evenement WHERE categorie = :c ORDER BY libelle LIMIT 1"), {"c": payload.categorie}).fetchone()
            if row and row[0]:
                payload.type_libelle = row[0]
        except Exception:
            pass
    if not payload.type_libelle:
        payload.type_libelle = "tâche"
    try:
        ev = create_tache(db, payload)
    except Exception as exc:
        logger.exception("Création tâche (affaire) échouée", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur création tâche: {exc}")
    if not ev:
        logger.error("Création de tâche impossible (affaire) payload=%s", payload.dict() if hasattr(payload, "dict") else payload)
        raise HTTPException(status_code=500, detail="Création de tâche impossible")

    # Statut initial
    sid = None
    try:
        sid = int(form.get("statut_id")) if form.get("statut_id") else None
    except Exception:
        sid = None
    if sid:
        try:
            add_statut_to_evenement(db, ev.id, EvenementStatutCreateSchema(statut_id=sid, commentaire="Création via affaire", utilisateur_responsable=payload.utilisateur_responsable))
            logger.info("Statut initial appliqué (affaire) ev_id=%s sid=%s", getattr(ev, "id", None), sid)
        except Exception as exc:
            logger.exception("Ajout statut tâche (affaire) échoué ev_id=%s", getattr(ev, "id", None), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erreur statut tâche: {exc}")

    # Communication éventuelle
    if form.get("comm_toggle") == "1":
        canal = form.get("comm_canal") or "email"
        dest = form.get("comm_destinataire") or ""
        obj = form.get("comm_objet") or None
        cont = form.get("comm_contenu") or None
        if dest:
            create_envoi(db, ev.id, EvenementEnvoiCreateSchema(canal=canal, destinataire=dest, objet=obj, contenu=cont))

    return RedirectResponse(url=f"/dashboard/affaires/{affaire_id}", status_code=303)


# ---------------- Mise à jour SRRI Affaire ----------------
@router.post("/affaires/{affaire_id}/srri", response_class=HTMLResponse)
async def dashboard_affaire_update_srri(affaire_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    srri_raw = form.get("srri")
    try:
        srri_val = int(srri_raw) if srri_raw not in (None, "", "None") else None
    except Exception:
        srri_val = None
    if srri_val is not None:
        # clamp 1-7
        if srri_val < 1 or srri_val > 7:
            srri_val = None
    affaire = db.query(Affaire).filter(Affaire.id == affaire_id).first()
    updated = False
    if affaire and srri_val is not None:
        try:
            affaire.SRRI = srri_val
            db.add(affaire)
            db.commit()
            updated = True
        except Exception:
            db.rollback()
            updated = False
    # Si SRRI modifié, forcer l'édition de la synthèse PDF
    if updated:
        return RedirectResponse(url=f"/dashboard/affaires/{affaire_id}/synthese/pdf", status_code=303)
    return RedirectResponse(url=f"/dashboard/affaires/{affaire_id}", status_code=303)


# ---------------- Détail Client ----------------
@router.get("/clients/{client_id}", response_class=HTMLResponse)
def dashboard_client_detail(client_id: int, request: Request, db: Session = Depends(get_db)):
    tb_markers_visible = request.query_params.get("markers") == "1"
    # Ensure DER context variables always exist to avoid NameError in templates
    DER_courtier = None
    DER_statut_social = None
    DER_courtier_garanties_normes: list[dict] = []
    DER_courtier_activite: list[dict] = []
    DER_sql_activite: str | None = None
    DER_sql_mediation: str | None = None

    client = db.query(Client).filter(Client.id == client_id).first()
    client_allocation_id = None
    client_allocation_name = None
    allocations_lookup: dict[int, str] = {}
    try:
        row_cli_alloc = db.execute(text("SELECT allocation_id FROM mariadb_clients WHERE id = :cid"), {"cid": client_id}).fetchone()
        if row_cli_alloc:
            client_allocation_id = row_cli_alloc[0] if not hasattr(row_cli_alloc, "_mapping") else (row_cli_alloc._mapping.get("allocation_id") or row_cli_alloc[0])
        if client_allocation_id:
            row_alloc_name = db.execute(text("SELECT nom FROM allocations WHERE id = :aid"), {"aid": client_allocation_id}).fetchone()
            if row_alloc_name:
                client_allocation_name = row_alloc_name[0] if not hasattr(row_alloc_name, "_mapping") else (row_alloc_name._mapping.get("nom") or row_alloc_name[0])
        # Si pas trouvé dans allocations, tenter allocation_risque -> allocation_name
        if client_allocation_name is None and client_allocation_id:
            row_alloc_risque = db.execute(text("SELECT allocation_name FROM allocation_risque WHERE id = :aid"), {"aid": client_allocation_id}).fetchone()
            if row_alloc_risque:
                client_allocation_name = row_alloc_risque[0] if not hasattr(row_alloc_risque, "_mapping") else (row_alloc_risque._mapping.get("allocation_name") or row_alloc_risque[0])
        try:
            rows_lookup = db.execute(text("SELECT id, nom FROM allocations")).fetchall()
            for r in rows_lookup:
                try:
                    rid = r[0] if not hasattr(r, "_mapping") else (r._mapping.get("id") or r[0])
                    rname = r[1] if not hasattr(r, "_mapping") else (r._mapping.get("nom") or r[1])
                    allocations_lookup[int(rid)] = rname
                except Exception:
                    continue
        except Exception:
            allocations_lookup = {}
    except Exception:
        client_allocation_id = getattr(client, "allocation_id", None) if client else None

    # Historique complet pour la courbe (inclut mouvements pour cumul)
    historique = (
        db.query(
            HistoriquePersonne.date,
            HistoriquePersonne.valo,
            HistoriquePersonne.mouvement,
            HistoriquePersonne.sicav,
            HistoriquePersonne.perf_sicav_52,
            HistoriquePersonne.volat,
            HistoriquePersonne.annee,
            HistoriquePersonne.SRRI.label("srri_actuel"),
        )
        .filter(HistoriquePersonne.id == client_id)
        .order_by(HistoriquePersonne.date)
        .all()
    )

    # Dernière ligne (stats actuelles)
    last_row = None
    if historique:
        last_row = historique[-1]

    # Séries pour le graphique: labels, valorisation, cumul des mouvements et mouvements bruts
    labels: list[str] = []
    serie_valo: list[float] = []
    serie_mov_cum: list[float] = []
    serie_mov_raw: list[float] = []
    cumul = 0.0
    available_dates: list[str] = []
    for h in historique:
        # Date formatée YYYY-MM-DD quand possible
        try:
            d = h.date.strftime("%Y-%m-%d") if h.date else None
        except Exception:
            d = str(h.date)[:10] if h.date else None
        labels.append(d)
        if d and (not available_dates or available_dates[-1] != d):
            available_dates.append(d)
        v = float(h.valo or 0)
        m = float(h.mouvement or 0)
        serie_valo.append(v)
        serie_mov_raw.append(m)
        cumul += m
        serie_mov_cum.append(cumul)

    # (Comparatif SICAV client vs allocations retiré)

    # Séries annuelles (prendre la dernière ligne par année)
    yearly = {}
    for h in historique:
        if getattr(h, 'annee', None) is None:
            continue
        try:
            y = int(h.annee)
        except Exception:
            continue
        cur = yearly.get(y)
        if not cur or ((getattr(h, 'date', None) and cur['date'] and h.date > cur['date'])):
            yearly[y] = { 'date': getattr(h, 'date', None), 'perf': getattr(h, 'perf_sicav_52', None), 'vol': getattr(h, 'volat', None) }
    years_client = sorted(yearly.keys())
    ann_perf_client = [ yearly[y]['perf'] for y in years_client ]
    ann_vol_client = [ yearly[y]['vol'] for y in years_client ]

    # ---- TRACFIN/FATCA: indicateurs financiers synthétiques ----
    lcbft_invest_total = 0.0
    lcbft_objectifs: list[dict] = []
    lcbft_patrimoine_net = None
    lcbft_part_pct = None
    lcbft_ppe_options: list[dict] = []
    try:
        try:
            inv = db.execute(text("SELECT COALESCE(SUM(montant),0) FROM KYC_Client_Objectifs WHERE client_id = :cid"), {"cid": client_id}).scalar()
            lcbft_invest_total = float(inv or 0.0)
        except Exception:
            lcbft_invest_total = 0.0
        # Charger le détail des objectifs (pour affichage/debug)
        try:
            rows = db.execute(
                text(
                    """
                    SELECT id, objectif_id, horizon_investissement, niveau_id, commentaire,
                           date_saisie, date_expiration, montant
                    FROM KYC_Client_Objectifs
                    WHERE client_id = :cid
                    ORDER BY (date_saisie IS NULL), date_saisie DESC, id DESC
                    """
                ),
                {"cid": client_id},
            ).fetchall()
            lcbft_objectifs = [dict(r._mapping) for r in rows]
        except Exception:
            lcbft_objectifs = []
        try:
            ta = db.execute(text("SELECT COALESCE(SUM(valeur),0) FROM KYC_Client_Actif WHERE client_id = :cid"), {"cid": client_id}).scalar()
        except Exception:
            ta = 0.0
        try:
            tp = db.execute(text("SELECT COALESCE(SUM(montant_rest_du),0) FROM KYC_Client_Passif WHERE client_id = :cid"), {"cid": client_id}).scalar()
        except Exception:
            tp = 0.0
        try:
            lcbft_patrimoine_net = float(ta or 0.0) - float(tp or 0.0)
        except Exception:
            lcbft_patrimoine_net = None
        if lcbft_patrimoine_net and lcbft_patrimoine_net != 0:
            try:
                lcbft_part_pct = float(lcbft_invest_total) / float(lcbft_patrimoine_net) * 100.0
            except Exception:
                lcbft_part_pct = None
        try:
            rows = db.execute(text("SELECT id, lib FROM LCBFT_ref_ppe_fonction ORDER BY lib"), {}).fetchall()
            lcbft_ppe_options = [dict(r._mapping) for r in rows]
        except Exception:
            lcbft_ppe_options = []
    except Exception:
        lcbft_invest_total = 0.0
        lcbft_patrimoine_net = None
        lcbft_part_pct = None
        lcbft_ppe_options = []
        lcbft_objectifs = []

    # Reportings pluriannuels: agrégats annuels + cumul des perfs
    # Regrouper l'historique par année
    yearly_rows: dict[int, list] = {}
    for h in historique:
        y = None
        try:
            y = int(getattr(h, 'annee', None)) if getattr(h, 'annee', None) is not None else None
        except Exception:
            y = None
        if y is None:
            # fallback à partir de la date si pas d'année
            try:
                y = int(getattr(h, 'date', None).year) if getattr(h, 'date', None) else None
            except Exception:
                y = None
        if y is None:
            continue
        yearly_rows.setdefault(y, []).append(h)

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _to_pct_num(x):
        """Retourne un float en pourcentage (exprimé en %)"""
        v = _to_float(x)
        if v is None:
            return None
        if abs(v) <= 1:
            v = v * 100.0
        return v

    def _to_return_decimal(x):
        """Retourne un rendement décimal (0.12 pour 12%)"""
        v = _to_float(x)
        if v is None:
            return 0.0
        # si valeur déjà décimale (<=1 en absolu), garder telle quelle, sinon convertir de % -> décimal
        return v if abs(v) <= 1 else (v / 100.0)

    def _fmt_thousand(v):
        if v is None:
            return "-"
        try:
            return "{:,.0f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)

    reporting_years = []
    cum_solde = 0.0
    year_idx = 0  # pour perf annualisée
    cum_factor = 1.0
    for y in sorted(yearly_rows.keys()):
        rows = yearly_rows[y]
        # sommes mouvements
        pos = 0.0
        neg = 0.0
        total = 0.0
        for r in rows:
            m = _to_float(getattr(r, 'mouvement', None)) or 0.0
            total += m
            if m > 0:
                pos += m
            elif m < 0:
                neg += m
        # dernière ligne de l'année (par date)
        last_r = None
        for r in rows:
            if last_r is None:
                last_r = r
            else:
                try:
                    if getattr(r, 'date', None) and getattr(last_r, 'date', None) and r.date > last_r.date:
                        last_r = r
                except Exception:
                    pass
        last_valo = _to_float(getattr(last_r, 'valo', None)) if last_r else None
        last_perf_pct = _to_pct_num(getattr(last_r, 'perf_sicav_52', None)) if last_r else None
        last_vol_pct = _to_pct_num(getattr(last_r, 'volat', None)) if last_r else None

        # moyenne vol annuelle
        vols = [ _to_float(getattr(r, 'volat', None)) for r in rows ]
        vols = [ v for v in vols if v is not None ]
        avg_vol_pct = _to_pct_num(sum(vols)/len(vols)) if vols else None

        # cumul des perfs sur les années
        ann_return = _to_return_decimal(getattr(last_r, 'perf_sicav_52', None)) if last_r else 0.0
        try:
            cum_factor *= (1.0 + float(ann_return or 0.0))
        except Exception:
            pass
        cum_perf_pct = (cum_factor - 1.0) * 100.0

        # performance annualisée (CAGR) sur n années depuis le début
        year_idx += 1
        try:
            ann_perf_pct = ((cum_factor ** (1.0 / max(1, year_idx))) - 1.0) * 100.0
        except Exception:
            ann_perf_pct = None

        cum_solde += total
        reporting_years.append({
            'year': y,
            'versements': pos,
            'versements_str': _fmt_thousand(pos),
            'retraits': neg,
            'retraits_str': _fmt_thousand(neg),
            'solde': total,
            'solde_str': _fmt_thousand(total),
            'solde_cum': cum_solde,
            'solde_cum_str': _fmt_thousand(cum_solde),
            'last_valo': last_valo,
            'last_valo_str': _fmt_thousand(last_valo),
            'last_perf_pct': last_perf_pct,
            'cum_perf_pct': cum_perf_pct,
            'ann_perf_pct': ann_perf_pct,
            'last_vol_pct': last_vol_pct,
            'avg_vol_pct': avg_vol_pct,
        })

    # Date effective pour la section Investissements (via ?as_of=YYYY-MM-DD)
    from datetime import datetime as _dt
    raw_as_of = request.query_params.get("as_of")
    as_of_effective: str | None = None
    if raw_as_of and raw_as_of in available_dates:
        as_of_effective = raw_as_of
    elif available_dates:
        as_of_effective = available_dates[-1]
    selected_dt = None
    try:
        selected_dt = _dt.fromisoformat(as_of_effective) if as_of_effective else None
    except Exception:
        selected_dt = None

    # Graph series limitées à la date sélectionnée (si présente) pour cohérence des valeurs
    if selected_dt:
        cutoff = None
        for i, ds in enumerate(labels):
            try:
                dcur = _dt.fromisoformat(ds)
            except Exception:
                dcur = None
            if dcur and dcur <= selected_dt:
                cutoff = i
            else:
                if cutoff is not None:
                    break
        if cutoff is not None:
            chart_labels = labels[:cutoff+1]
            chart_valo = serie_valo[:cutoff+1]
            chart_mov_cum = serie_mov_cum[:cutoff+1]
            chart_mov_raw = serie_mov_raw[:cutoff+1]
        else:
            chart_labels = labels
            chart_valo = serie_valo
            chart_mov_cum = serie_mov_cum
            chart_mov_raw = serie_mov_raw
    else:
        chart_labels = labels
        chart_valo = serie_valo
        chart_mov_cum = serie_mov_cum
        chart_mov_raw = serie_mov_raw

    # KPIs formatés (valorisation + pourcentages) et SRRI actuel vs SRRI client (à la date sélectionnée)
    # Ligne d'historique retenue = dernière ligne <= selected_dt, sinon dernière globale
    if selected_dt:
        filtered_hist = [h for h in historique if (getattr(h, 'date', None) and h.date <= selected_dt)]
        effective_row = filtered_hist[-1] if filtered_hist else (historique[-1] if historique else None)
    else:
        effective_row = last_row
    last_valo = float(effective_row.valo or 0) if effective_row else None
    def _fmt_valo(v):
        if v is None:
            return "-"
        try:
            return "{:,.0f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)
    last_valo_str = _fmt_valo(last_valo)

    def _to_pct_float(x):
        if x is None:
            return None
        try:
            n = float(x)
        except Exception:
            return None
        if abs(n) <= 1:
            n *= 100.0
        return float(n)

    last_perf_pct = _to_pct_float(effective_row.perf_sicav_52) if effective_row else None
    last_vol_pct = _to_pct_float(effective_row.volat) if effective_row else None

    current_srri = int(effective_row.srri_actuel) if (effective_row and effective_row.srri_actuel is not None) else None
    client_srri = int(client.SRRI) if (client and client.SRRI is not None) else None
    def _icon_for_compare_srri(contract_srri, current_srri):
        if contract_srri is None or current_srri is None:
            return None
        try:
            c = int(contract_srri)
            k = int(current_srri)
        except Exception:
            return None
        # Mapping: Au‑dessus = 🔥, Identique = 🙏, En‑dessous = ❄️
        if c > k:
            return "fire"
        if c == k:
            return "hands-praying"
        return "snowflake"
    header_srri_icon = _icon_for_compare_srri(client_srri, current_srri)

    # Totaux mouvements
    if selected_dt:
        hist_for_totals = [h for h in historique if (getattr(h, 'date', None) and h.date <= selected_dt)]
    else:
        hist_for_totals = historique
    depots_total = sum(float(h.mouvement or 0) for h in hist_for_totals if float(h.mouvement or 0) > 0)
    retraits_total = sum(float(h.mouvement or 0) for h in hist_for_totals if float(h.mouvement or 0) < 0)
    solde_total = depots_total + retraits_total
    depots_str = _fmt_valo(depots_total)
    retraits_str = _fmt_valo(retraits_total)
    solde_str = _fmt_valo(solde_total)
    valo_gt_solde = (last_valo is not None and last_valo > solde_total)

    # Affaires de ce client (ouverts et fermés) à la date effective
    if selected_dt:
        subq_aff = (
            db.query(
                HistoriqueAffaire.id.label("affaire_id"),
                func.max(HistoriqueAffaire.date).label("last_date")
            )
            .filter(HistoriqueAffaire.date <= selected_dt)
            .group_by(HistoriqueAffaire.id)
            .subquery()
        )
    else:
        subq_aff = (
            db.query(
                HistoriqueAffaire.id.label("affaire_id"),
                func.max(HistoriqueAffaire.date).label("last_date")
            )
            .group_by(HistoriqueAffaire.id)
            .subquery()
        )
    affaires_rows = (
        db.query(
            Affaire.id.label("id"),
            Affaire.ref,
            Affaire.SRRI,
            Affaire.date_debut,
            Affaire.date_cle,
            HistoriqueAffaire.valo.label("last_valo"),
            HistoriqueAffaire.perf_sicav_52.label("last_perf"),
            HistoriqueAffaire.volat.label("last_volat"),
        )
        .join(subq_aff, subq_aff.c.affaire_id == Affaire.id)
        .join(
            HistoriqueAffaire,
            (HistoriqueAffaire.id == subq_aff.c.affaire_id) &
            (HistoriqueAffaire.date == subq_aff.c.last_date)
        )
        .filter(Affaire.id_personne == client_id)
        .all()
    )

    def _srri_from_vol(v):
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if abs(x) <= 1:
            x *= 100.0
        if x <= 0.5: return 1
        if x <= 2: return 2
        if x <= 5: return 3
        if x <= 10: return 4
        if x <= 15: return 5
        if x <= 25: return 6
        return 7

    def _icon_for_compare(contract_srri, calc_srri):
        if contract_srri is None or calc_srri is None:
            return None
        try:
            c = int(contract_srri)
            k = int(calc_srri)
        except Exception:
            return None
        # Mapping: Au‑dessus = 🔥, Identique = 🙏, En‑dessous = ❄️
        if c > k:
            return "fire"
        if c == k:
            return "hands-praying"
        return "snowflake"

    try:
        if selected_dt:
            mouvements_rows = db.execute(
                text(
                    """
                    SELECT h.id AS id_affaire,
                           SUM(COALESCE(h.mouvement, 0)) AS total_mvt
                    FROM mariadb_historique_affaire_w h
                    WHERE h.date <= :sel
                    GROUP BY h.id
                    """
                ),
                {"sel": selected_dt},
            ).fetchall()
        else:
            mouvements_rows = db.execute(
                text(
                    """
                    SELECT h.id AS id_affaire,
                           SUM(COALESCE(h.mouvement, 0)) AS total_mvt
                    FROM mariadb_historique_affaire_w h
                    GROUP BY h.id
                    """
                )
            ).fetchall()
    except Exception:
        mouvements_rows = []

    mouvements_map: dict[int, float] = {}
    for row in mouvements_rows:
        mapping = row._mapping if hasattr(row, "_mapping") else None
        rid_raw = mapping.get("id_affaire") if mapping else (row[0] if len(row) > 0 else None)
        total_raw = mapping.get("total_mvt") if mapping else (row[1] if len(row) > 1 else None)
        try:
            rid = int(rid_raw) if rid_raw is not None else None
        except Exception:
            rid = None
        if rid is None:
            continue
        try:
            mouvements_map[rid] = float(total_raw or 0.0)
        except Exception:
            mouvements_map[rid] = 0.0

    client_affaires = []
    for r in affaires_rows:
        srri_calc = _srri_from_vol(r.last_volat)
        icon = _icon_for_compare(r.SRRI, srri_calc)
        # format perf/vol en pourcentage (<=1 => *100)
        def _pct_or_none(x):
            if x is None:
                return None
            try:
                n = float(x)
            except Exception:
                return None
            if abs(n) <= 1:
                n *= 100.0
            return n
        perf_pct = _pct_or_none(r.last_perf)
        vol_pct = _pct_or_none(r.last_volat)
        total_mvt = mouvements_map.get(r.id)
        try:
            last_valo_numeric = float(r.last_valo or 0.0)
        except Exception:
            last_valo_numeric = 0.0
        ecart_valo_mvt = None
        if total_mvt is not None:
            ecart_valo_mvt = last_valo_numeric - total_mvt
        client_affaires.append({
            "id": r.id,
            "ref": r.ref,
            "SRRI": r.SRRI,
            "date_debut": r.date_debut,
            "date_cle": r.date_cle,
            "last_valo": r.last_valo,
            "last_valo_str": _fmt_valo(r.last_valo),
            "last_perf_pct": perf_pct,
            "last_vol_pct": vol_pct,
            "srri_calc": srri_calc,
            "srri_icon": icon,
            "total_mvt": total_mvt,
            "ecart_valo_mvt": ecart_valo_mvt,
            "last_valo_numeric": last_valo_numeric,
        })

    # Comptages contrats ouverts/fermés
    total_contrats = len(affaires_rows)
    nb_contrats_fermes = sum(1 for r in affaires_rows if getattr(r, 'date_cle', None))
    nb_contrats_ouverts = max(0, total_contrats - nb_contrats_fermes)

    # Durée depuis la première date de l'historique jusqu'à la date effective
    first_dt = None
    for ds in labels:
        try:
            first_dt = _dt.fromisoformat(ds) if ds else None
        except Exception:
            first_dt = None
        if first_dt:
            break
    last_dt = selected_dt
    if not last_dt:
        # fallback à partir de la dernière étiquette si pas de selected_dt
        try:
            last_dt = _dt.fromisoformat(labels[-1]) if labels else None
        except Exception:
            last_dt = None

    def _human_duration(a, b):
        if not a or not b:
            return "-"
        # Approximation mois/années sans dépendances externes
        total_months = (b.year - a.year) * 12 + (b.month - a.month)
        if b.day < a.day:
            total_months -= 1
        years = total_months // 12
        months = total_months % 12
        if years <= 0 and months <= 0:
            days = max(0, (b - a).days)
            return f"{days} jours"
        parts = []
        if years > 0:
            parts.append(f"{years} ans")
        if months > 0:
            parts.append(f"{months} mois")
        return ", ".join(parts) if parts else "-"

    duree_historique_str = _human_duration(first_dt, last_dt)

    # Perf annualisée sur la durée depuis la première date
    overall_ann_perf_pct = None
    try:
        if first_dt and last_dt and cum_factor and cum_factor > 0:
            years_span = max(1e-6, (last_dt - first_dt).days / 365.25)
            overall_ann_perf_pct = ((float(cum_factor) ** (1.0 / years_span)) - 1.0) * 100.0
    except Exception:
        overall_ann_perf_pct = None

    # Supports consolidés sur l'ensemble des contrats du client
    def _fmt_float_2(v):
        if v is None:
            return "-"
        try:
            return "{:,.2f}".format(float(v)).replace(",", " ")
        except Exception:
            return str(v)

    client_supports_map: dict[str, dict] = {}
    movements_cache: dict[tuple[int, int], list[dict]] = {}
    nb_supports_actifs = 0
    try:
        # Liste des contrats (affaires) de ce client
        affaire_ids = [rid for (rid,) in db.query(Affaire.id).filter(Affaire.id_personne == client_id).all()]
        for aid in affaire_ids:
            # date de référence: choisie (as_of) sinon dernière disponible
            if as_of_effective:
                ref_date = as_of_effective
            else:
                ref_date = db.execute(text("SELECT MAX(date) FROM mariadb_historique_support_w WHERE id_source = :aid"), {"aid": aid}).scalar()
                if not ref_date:
                    continue
            rows = db.execute(
                text(
                    """
                    SELECT s.id AS support_id,
                           s.code_isin AS code_isin,
                           s.nom AS nom,
                           s.SRRI AS srri_support,
                           s.cat_gene AS cat_gene,
                           s.cat_principale AS cat_principale,
                           s.cat_det AS cat_det,
                           s.cat_geo AS cat_geo,
                           a.id AS affaire_id,
                           a.ref AS affaire_ref,
                           h.nbuc AS nbuc,
                           h.vl AS vl,
                           h.prmp AS prmp,
                           h.valo AS valo,
                           esg.noteE AS noteE,
                           esg.noteS AS noteS,
                           esg.noteG AS noteG
                    FROM mariadb_historique_support_w h
                    JOIN mariadb_support s ON s.id = h.id_support
                    JOIN mariadb_affaires a ON a.id = h.id_source
                    LEFT JOIN donnees_esg_etendu esg ON esg.isin = s.code_isin
                    WHERE h.id_source = :aid AND h.date = :d
                    """
                ),
                {"aid": aid, "d": ref_date}
            ).fetchall()
            for r in rows:
                try:
                    support_id = int(getattr(r, "support_id", None)) if getattr(r, "support_id", None) is not None else None
                except Exception:
                    support_id = None
                key = r.code_isin or r.nom
                it = client_supports_map.get(key)
                if it and it.get("support_id") is None and support_id is not None:
                    it["support_id"] = support_id
                nb = float(r.nbuc or 0)
                valo = float(r.valo or 0)
                vl_val = float(r.vl) if getattr(r, "vl", None) is not None else None
                prmp = float(r.prmp or 0) if r.prmp is not None else None
                if not it:
                    it = {
                        "code_isin": r.code_isin,
                        "nom": r.nom,
                        "srri_support": getattr(r, "srri_support", None),
                        "cat_gene": getattr(r, "cat_gene", None),
                        "cat_principale": getattr(r, "cat_principale", None),
                        "cat_det": getattr(r, "cat_det", None),
                        "cat_geo": getattr(r, "cat_geo", None),
                        "support_id": support_id,
                        "contracts": [],
                        "sum_nbuc": 0.0,
                        "sum_valo": 0.0,
                        "vl_num": 0.0,
                        "vl_den": 0.0,
                        "prmp_num": 0.0,
                        "prmp_den": 0.0,
                        "noteE": getattr(r, "noteE", None),
                        "noteS": getattr(r, "noteS", None),
                        "noteG": getattr(r, "noteG", None),
                    }
                    client_supports_map[key] = it
                it["sum_nbuc"] += nb
                it["sum_valo"] += valo
                if vl_val is not None:
                    weight = nb if nb and nb != 0 else 1.0
                    it["vl_num"] += vl_val * weight
                    it["vl_den"] += weight
                if prmp is not None and nb is not None:
                    it["prmp_num"] += prmp * nb
                    it["prmp_den"] += nb
                contract_movements: list[dict] = []
                cache_key = None
                if aid is not None and support_id is not None:
                    cache_key = (aid, support_id)
                    if cache_key in movements_cache:
                        contract_movements = movements_cache[cache_key]
                    else:
                        contract_movements = []
                        try:
                            mov_rows = db.execute(
                                text(
                                    """
                                    SELECT COALESCE(m.vl_date, m.date_sp) AS mouvement_date,
                                           m.montant_ope,
                                           m.vl,
                                           m.nb_uc,
                                           mr.sens,
                                           mr.titre AS regle_label
                                    FROM mouvement m
                                    LEFT JOIN mouvement_regle mr ON mr.id = m.id_mouvement_regle
                                    WHERE m.id_affaire = :aid AND m.id_support = :sid
                                    ORDER BY COALESCE(m.vl_date, m.date_sp) DESC, m.id DESC
                                    """
                                ),
                                {"aid": aid, "sid": support_id},
                            ).fetchall()
                        except Exception:
                            mov_rows = []
                        for mv in mov_rows:
                            dt_raw = getattr(mv, "mouvement_date", None)
                            try:
                                dt_str = dt_raw.strftime("%Y-%m-%d") if dt_raw else None
                            except Exception:
                                dt_str = str(dt_raw)[:10] if dt_raw else None
                            sens_raw = getattr(mv, "sens", None)
                            sens_int: int | None = None
                            try:
                                sens_int = int(sens_raw) if sens_raw is not None else None
                            except Exception:
                                sens_int = None
                            def _sign_for(val: float | None, default_sign: str = "") -> str:
                                if sens_int is not None:
                                    if sens_int > 0:
                                        return "+"
                                    if sens_int < 0:
                                        return "-"
                                if val is None:
                                    return default_sign
                                try:
                                    num = float(val)
                                except Exception:
                                    return default_sign
                                if num > 0:
                                    return "+"
                                if num < 0:
                                    return "-"
                                return default_sign
                            def _fmt_signed(value) -> str:
                                if value is None:
                                    return "-"
                                try:
                                    num = float(value or 0.0)
                                except Exception:
                                    return "-"
                                sign_char = _sign_for(num)
                                abs_val = abs(num)
                                base = _fmt_float_2(abs_val)
                                if sign_char:
                                    return f"{sign_char}{base}"
                                return base
                            montant_signed = _fmt_signed(getattr(mv, "montant_ope", None))
                            nb_parts_signed = _fmt_signed(getattr(mv, "nb_uc", None))
                            vl_str = ("-" if getattr(mv, "vl", None) is None else _fmt_float_2(getattr(mv, "vl", None)))
                            contract_movements.append({
                                "date": dt_str or "-",
                                "regle": getattr(mv, "regle_label", None),
                                "montant": montant_signed,
                                "vl": vl_str,
                                "nb_uc": nb_parts_signed,
                            })
                        movements_cache[cache_key] = contract_movements
                it["contracts"].append({
                    "affaire_id": getattr(r, "affaire_id", aid),
                    "affaire_ref": getattr(r, "affaire_ref", None) or f"Affaire {aid}",
                    "nb_parts": nb,
                    "nb_parts_str": _fmt_float_2(nb),
                    "prix_revient": prmp,
                    "prix_revient_str": ("-" if prmp is None else _fmt_float_2(prmp)),
                    "vl": vl_val,
                    "vl_str": ("-" if vl_val is None else _fmt_float_2(vl_val)),
                    "valorisation": valo,
                    "valorisation_str": _fmt_valo(valo),
                    "code_isin": r.code_isin,
                    "support_nom": r.nom,
                    "support_id": support_id,
                    "movements": contract_movements,
                })
                # Enrichir notes ESG / catégories si absentes
                for k in ("noteE", "noteS", "noteG"):
                    if it.get(k) is None and getattr(r, k, None) is not None:
                        it[k] = getattr(r, k)
                for k in ("cat_gene", "cat_principale", "cat_det", "cat_geo"):
                    if it.get(k) is None and getattr(r, k, None) is not None:
                        it[k] = getattr(r, k)
        client_supports = []
        for it in client_supports_map.values():
            den = it.get("prmp_den", 0.0) or 0.0
            prmp_val = (it.get("prmp_num", 0.0) / den) if den > 0 else None
            vl_den = it.get("vl_den", 0.0) or 0.0
            vl_val = (it.get("vl_num", 0.0) / vl_den) if vl_den > 0 else None
            contracts_list = [
                {
                    **c,
                    "nb_parts_str": c.get("nb_parts_str"),
                    "prix_revient_str": c.get("prix_revient_str"),
                    "vl_str": c.get("vl_str"),
                    "valorisation_str": c.get("valorisation_str"),
                }
                for c in it.get("contracts", [])
            ]
            client_supports.append({
                "code_isin": it.get("code_isin"),
                "nom": it.get("nom"),
                "srri_support": it.get("srri_support"),
                "cat_gene": it.get("cat_gene"),
                "cat_principale": it.get("cat_principale"),
                "cat_det": it.get("cat_det"),
                "cat_geo": it.get("cat_geo"),
                "nbuc": it.get("sum_nbuc", 0.0),
                "nbuc_str": _fmt_float_2(it.get("sum_nbuc", 0.0)),
                "prmp": prmp_val,
                "prmp_str": ("-" if prmp_val is None else _fmt_float_2(prmp_val)),
                "vl": vl_val,
                "vl_str": ("-" if vl_val is None else _fmt_float_2(vl_val)),
                "valo": it.get("sum_valo", 0.0),
                "valo_str": _fmt_valo(it.get("sum_valo", 0.0)),
                "noteE": it.get("noteE"),
                "noteS": it.get("noteS"),
                "noteG": it.get("noteG"),
                "contracts": contracts_list,
            })
        # Trier par valorisation décroissante
        client_supports.sort(key=lambda x: x.get("valo", 0.0), reverse=True)
        nb_supports_actifs = sum(
            1
            for it in client_supports
            if (it.get("valo") or 0) > 0 or (it.get("nbuc") or 0) > 0
        )
        if not nb_supports_actifs:
            nb_supports_actifs = len(client_supports)
    except Exception:
        client_supports = []
        nb_supports_actifs = 0

    # Documents liés au client, avec nom du document de base
    rows = (
        db.query(
            DocumentClient,
            Document.documents.label("base_name")
        )
        .outerjoin(Document, Document.id_document_base == DocumentClient.id_document_base)
        .filter(DocumentClient.id_client == client_id)
        .order_by(DocumentClient.date_creation.desc())
        .all()
    )

    documents_client = [
        {
            "id": d.id,
            "nom_document": d.nom_document,
            "date_creation": d.date_creation,
            "date_obsolescence": d.date_obsolescence,
            "obsolescence": d.obsolescence,
            "base_name": base_name,
        }
        for d, base_name in rows
    ]

    # Séries d'allocations (SICAV) par nom pour graphique de détail client
    alloc_series: dict[str, list[dict]] = {}
    try:
        series_rows = (
            db.query(Allocation.nom, Allocation.date, Allocation.sicav)
            .order_by(Allocation.nom.asc(), Allocation.date.asc())
            .all()
        )
        for nom, date, sicav in series_rows:
            arr = alloc_series.setdefault(nom, [])
            try:
                dstr = date.strftime("%Y-%m-%d") if date else None
            except Exception:
                dstr = str(date)[:10] if date else None
            arr.append({
                "date": dstr,
                "sicav": float(sicav or 0),
            })
    except Exception:
        alloc_series = {}

    # Série SICAV du client (mariadb_historique_personne_w)
    client_sicav: list[dict] = []
    try:
        for h in historique:
            try:
                ds = h.date.strftime("%Y-%m-%d") if getattr(h, 'date', None) else None
            except Exception:
                ds = str(getattr(h, 'date', None))[:10] if getattr(h, 'date', None) else None
            client_sicav.append({
                "date": ds,
                "sicav": float(getattr(h, 'sicav', 0) or 0)
            })
    except Exception:
        client_sicav = []

    # Données pour création de tâche (accordéon)
    from sqlalchemy import text as _text
    types = db.execute(_text("SELECT id, libelle, categorie FROM mariadb_type_evenement ORDER BY categorie, libelle")).fetchall()
    cats = sorted({getattr(t, 'categorie', None) for t in types if getattr(t, 'categorie', None)})
    from src.services.evenements import list_statuts as _list_statuts
    statuts = _list_statuts(db)
    # statuts UI (ordre et mapping)
    def _norm(s: str | None) -> str | None:
        if not s:
            return None
        x = s.strip().lower()
        for a,b in [("à","a"),("â","a"),("ä","a"),("é","e"),("è","e"),("ê","e"),("ë","e"),("î","i"),("ï","i"),("ô","o"),("ö","o"),("û","u"),("ü","u"),("ç","c")]:
            x = x.replace(a,b)
        return x
    stat_ids: dict[str,int] = {}
    for s in statuts:
        k = _norm(getattr(s, 'libelle', None))
        if k and getattr(s, 'id', None) is not None:
            stat_ids[k] = s.id
    status_ui = []
    for label_ui, key in [
        ("à faire", "a faire"),
        ("en cours", "en cours"),
        ("en attente", "en attente"),
        ("terminé", "termine"),
        ("annulé", "annule"),
    ]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")

    reclam_statuses = (
        db.query(EvenementStatutReclamation)
        .order_by(EvenementStatutReclamation.is_cloture.asc(), EvenementStatutReclamation.libelle.asc())
        .all()
    )

    clients_suggest = db.query(Client.id, Client.nom, Client.prenom).order_by(Client.nom.asc(), Client.prenom.asc()).all()
    aff_rows = db.query(Affaire.id, Affaire.ref, Affaire.id_personne).order_by(Affaire.ref.asc()).all()
    _clients_map = {c.id: f"{getattr(c,'nom','') or ''} {getattr(c,'prenom','') or ''}".strip() for c in clients_suggest}
    affaires_suggest = [{"id": a.id, "ref": getattr(a,'ref',''), "client": _clients_map.get(getattr(a,'id_personne',None), '')} for a in aff_rows]
    client_fullname_default = (f"{getattr(client,'nom','') or ''} {getattr(client,'prenom','') or ''}".strip()) if client else None
    rh_list = fetch_rh_list(db)

    # -------- Messages (tâches) par client: comptages + liste ouverte (pour pop-up) --------
    from src.models.evenement import Evenement
    from src.models.type_evenement import TypeEvenement
    # Statuts ouverts: différent de terminé/annulé/clos
    OPEN_STATES = ("termine", "terminé", "cloture", "clôturé", "cloturé", "clôture", "annule", "annulé")
    try:
        events_open = db.execute(
            text(
                """
                SELECT
                    e.id AS id,
                    e.id AS evenement_id,
                    e.client_id,
                    e.affaire_id,
                    e.date_evenement,
                    e.statut,
                    e.commentaire,
                    e.utilisateur_responsable,
                    e.rh_id,
                    e.statut_reclamation_id,
                    te.libelle AS type_libelle,
                    te.categorie AS type_categorie,
                    esr.libelle AS statut_reclamation_libelle
                FROM mariadb_evenement e
                JOIN mariadb_type_evenement te ON te.id = e.type_id
                LEFT JOIN mariadb_evenement_statut_reclamation esr ON esr.id = e.statut_reclamation_id
                WHERE e.client_id = :cid
                  AND (
                    e.statut IS NULL OR LOWER(e.statut) NOT IN ('termine','terminé','cloture','clôturé','cloturé','clôture','annule','annulé')
                  )
                ORDER BY e.date_evenement DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
    except Exception as exc:
        logger.exception("client_events_open query failed", exc_info=exc)
        events_open = []
    # Map affaire_id -> ref
    _aff_ref = {a.id: getattr(a, 'ref', None) for a in aff_rows}
    def _norm_cat(s: str | None) -> str:
        if not s:
            return ""
        x = (s or "").strip().lower()
        for a, b in [("à","a"),("â","a"),("ä","a"),("é","e"),("è","e"),("ê","e"),("ë","e"),("î","i"),("ï","i"),("ô","o"),("ö","o"),("û","u"),("ü","u"),("ç","c")]:
            x = x.replace(a, b)
        return x
    msgs_reg_count = 0
    msgs_nonreg_count = 0
    client_events_open: list[dict] = []
    for r in events_open:
        catn = _norm_cat(getattr(r, "type_categorie", None))
        is_reg = (catn == "reglementaire")
        if is_reg:
            msgs_reg_count += 1
        else:
            msgs_nonreg_count += 1
        mp = getattr(r, "_mapping", {})
        eid_raw = mp.get("id") if isinstance(mp, dict) else None
        eid_alt = mp.get("evenement_id") if isinstance(mp, dict) else None
        if eid_raw is None:
            try:
                eid_raw = getattr(r, "id", None)
            except Exception:
                eid_raw = None
        if eid_alt is None:
            try:
                eid_alt = getattr(r, "evenement_id", None)
            except Exception:
                eid_alt = None
        eid_int = eid_raw if eid_raw is not None else eid_alt
        # Safe date formatting
        try:
            dstr = r.date_evenement.strftime("%Y-%m-%d %H:%M") if getattr(r, 'date_evenement', None) else None
        except Exception:
            dstr = str(getattr(r, 'date_evenement', None))[:16] if getattr(r, 'date_evenement', None) else None
        client_events_open.append({
            "id": eid_raw,
            "evenement_id": eid_alt,
            "id_resolved": eid_int,
            "date_evenement": dstr,
            "statut": getattr(r, 'statut', None),
            "commentaire": getattr(r, 'commentaire', None),
            "type_id": getattr(r, 'type_id', None),
            "type_libelle": getattr(r, 'type_libelle', None),
            "type_categorie": getattr(r, 'type_categorie', None),
            "affaire_id": getattr(r, 'affaire_id', None),
            "affaire_ref": _aff_ref.get(getattr(r, 'affaire_id', None)),
            "statut_reclamation_id": getattr(r, 'statut_reclamation_id', None),
            "statut_reclamation_label": getattr(r, 'statut_reclamation_libelle', None),
        })

    # KYC: Actifs du client
    try:
        from sqlalchemy import text as _text
        rows_actifs = db.execute(
            _text(
                """
                SELECT a.id, a.id_type_actif, ta.libelle AS type_libelle,
                       a.intitule, a.valeur_initiale, a.date_acquisition,
                       a.valeur, a.devise, a.date_eval, a.commentaire
                FROM actif_client a
                LEFT JOIN type_actif ta ON ta.id = a.id_type_actif
                WHERE a.id_client = :cid
                ORDER BY COALESCE(a.date_eval, a.date_acquisition) DESC, a.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
    except Exception:
        rows_actifs = []
    # Types d'actifs pour formulaire
    try:
        from sqlalchemy import text as _text
        rows_types_actifs = db.execute(_text("SELECT id, libelle FROM type_actif WHERE COALESCE(actif,1)=1 ORDER BY libelle")).fetchall()
    except Exception:
        rows_types_actifs = []

    def _dt_str(x):
        if not x:
            return None
        try:
            return x.strftime("%Y-%m-%d")
        except Exception:
            return str(x)[:10]

    def _fmt2(v):
        try:
            return "{:,.2f}".format(float(v or 0)).replace(",", " ")
        except Exception:
            return v

    kyc_actifs = []
    for r in rows_actifs:
        kyc_actifs.append({
            "id": getattr(r, "id", None),
            "type": getattr(r, "type_libelle", None),
            "intitule": getattr(r, "intitule", None),
            "valeur_initiale": _fmt2(getattr(r, "valeur_initiale", None)),
            "date_acquisition": _dt_str(getattr(r, "date_acquisition", None)),
            "valeur": _fmt2(getattr(r, "valeur", None)),
            "devise": getattr(r, "devise", None),
            "date_eval": _dt_str(getattr(r, "date_eval", None)),
            "commentaire": getattr(r, "commentaire", None),
        })

    # ---- Conformité (TRACFIN / FATCA) data for detail modal ----
    lcbft_current = None
    lcbft_vigilance_options: list[dict] = []
    lcbft_vigilance_ids: list[int] = []
    try:
        rows = db.execute(text("SELECT id, code, label FROM LCBFT_vigilance_option ORDER BY label")).fetchall()
        lcbft_vigilance_options = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_vigilance_options = []
    try:
        row = db.execute(text("SELECT * FROM LCBFT_questionnaire WHERE client_ref = :r ORDER BY updated_at DESC LIMIT 1"), {"r": str(client_id)}).fetchone()
        if row:
            m = row._mapping
            qid = m.get('id')
            lcbft_current = {k: m.get(k) for k in m.keys()}
            try:
                ids = db.execute(text("SELECT option_id FROM LCBFT_questionnaire_vigilance WHERE questionnaire_id = :q"), {"q": qid}).fetchall()
                lcbft_vigilance_ids = [int(x[0]) for x in ids]
            except Exception:
                lcbft_vigilance_ids = []
    except Exception:
        lcbft_current = None

    fatca_contracts: list[dict] = []
    try:
        rows = db.execute(text("""
            SELECT g.id, g.nom_contrat, COALESCE(s.nom, '') AS societe_nom
            FROM mariadb_affaires_generique g
            LEFT JOIN mariadb_societe s ON s.id = g.id_societe
            WHERE COALESCE(g.actif, 1) = 1
            ORDER BY s.nom, g.nom_contrat
        """)).fetchall()
        fatca_contracts = [dict(r._mapping) for r in rows]
    except Exception:
        fatca_contracts = []
    fatca_saved = None
    try:
        if lcbft_current and lcbft_current.get('id'):
            qid = lcbft_current.get('id')
            row = db.execute(text("SELECT contrat_id, societe_nom, date_operation, pays_residence, nif FROM LCBFT_fatca WHERE questionnaire_id = :q"), {"q": qid}).fetchone()
            if row:
                fatca_saved = dict(row._mapping)
    except Exception:
        fatca_saved = None
    fatca_client_country = None
    fatca_client_nif = None
    try:
        crow = db.execute(text("SELECT * FROM mariadb_clients WHERE id = :cid"), {"cid": client_id}).fetchone()
        if crow:
            m = crow._mapping
            lower_map = { (k.lower() if isinstance(k, str) else k): v for k, v in m.items() }
            for key in ("adresse_pays", "pays_fiscal", "residence_fiscale", "pays"):
                if key in lower_map and lower_map.get(key):
                    fatca_client_country = lower_map.get(key) or ''
                    break
            for key in ("nif", "num_fiscal", "numero_fiscal", "tin"):
                if key in lower_map and lower_map.get(key):
                    fatca_client_nif = lower_map.get(key) or ''
                    break
    except Exception:
        pass
    fatca_today = _date.today().isoformat()

    # --- DER data for Conformité modal (client detail) ---
    der_context = _build_der_context(db)
    DER_courtier = der_context.get("DER_courtier")
    DER_statut_social = der_context.get("DER_statut_social")
    DER_courtier_garanties_normes = der_context.get("DER_courtier_garanties_normes") or []
    DER_sql_params_activite = der_context.get("DER_sql_params_activite") or {":cid": None}
    lm_remunerations = der_context.get("lm_remunerations") or []
    centre_mediation_lib = der_context.get("centre_mediation_lib")
    DER_activites = der_context.get("DER_activites") or []

    # Lettre de mission : objectifs client (KYC_Client_Objectifs)
    lm_objectifs: list[dict] = []
    lm_total_invest: float = 0.0
    lm_total_invest_str: str = "0"
    lm_entretien_date: str | None = None
    _lm_latest_date: _date | None = None
    try:
        # Libellés des objectifs
        objectif_labels: dict[int, str] = {}
        try:
            rows_labels = db.execute(text("SELECT id, libelle FROM ref_objectif")).fetchall()
            objectif_labels = {int(r.id): str(r.libelle) for r in rows_labels}
        except Exception:
            objectif_labels = {}
        if not objectif_labels:
            try:
                rows_labels = db.execute(text("SELECT id, label FROM risque_objectif_option")).fetchall()
                objectif_labels = {int(r.id): str(r.label) for r in rows_labels}
            except Exception:
                objectif_labels = {}
        # Libellés des niveaux de risque
        niveau_labels: dict[int, str] = {}
        try:
            rows_niveau = db.execute(text("SELECT id, libelle FROM ref_niveau_risque")).fetchall()
            niveau_labels = {int(r.id): str(r.libelle) for r in rows_niveau}
        except Exception:
            niveau_labels = {}
        rows = db.execute(
            text(
                """
                SELECT id,
                       objectif_id,
                       horizon_investissement,
                       niveau_id,
                       commentaire,
                       date_saisie,
                       date_expiration,
                       montant
                FROM KYC_Client_Objectifs
                WHERE client_id = :cid
                ORDER BY COALESCE(date_saisie, '0000-00-00') DESC, id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows:
            m = row._mapping
            montant_val = float(m.get("montant") or 0.0)
            lm_total_invest += montant_val
            objectif_id = m.get("objectif_id")
            libelle = None
            if objectif_id is not None:
                try:
                    libelle = objectif_labels.get(int(objectif_id))
                except Exception:
                    libelle = None
            niveau_libelle = None
            niveau_id = m.get("niveau_id")
            if niveau_id is not None:
                try:
                    niveau_libelle = niveau_labels.get(int(niveau_id))
                except Exception:
                    niveau_libelle = None
            date_saisie_raw = m.get("date_saisie")
            date_expiration_raw = m.get("date_expiration")
            parsed_date = _parse_date_safe(date_saisie_raw)
            if parsed_date is None and isinstance(date_saisie_raw, datetime):
                parsed_date = date_saisie_raw.date()
            if parsed_date:
                if _lm_latest_date is None or parsed_date > _lm_latest_date:
                    _lm_latest_date = parsed_date
            lm_objectifs.append(
                {
                    "id": m.get("id"),
                    "libelle": libelle or (f"Objectif {objectif_id}" if objectif_id is not None else "Objectif"),
                    "montant": montant_val,
                    "montant_str": "{:,.0f}".format(montant_val).replace(",", " "),
                    "horizon": m.get("horizon_investissement") or "",
                    "niveau": niveau_libelle,
                    "commentaire": m.get("commentaire") or "",
                    "date_saisie": parsed_date.strftime("%d/%m/%Y") if parsed_date else (str(date_saisie_raw) if date_saisie_raw else ""),
                    "date_expiration": (
                        _parse_date_safe(date_expiration_raw).strftime("%d/%m/%Y")
                        if _parse_date_safe(date_expiration_raw)
                        else (str(date_expiration_raw) if date_expiration_raw else "")
                    ),
                }
            )
        if lm_objectifs:
            lm_total_invest_str = "{:,.0f}".format(lm_total_invest).replace(",", " ")
        else:
            lm_total_invest = 0.0
            lm_total_invest_str = "0"
        if _lm_latest_date:
            lm_entretien_date = _lm_latest_date.strftime("%d/%m/%Y")
    except Exception:
        lm_objectifs = []
        lm_total_invest = 0.0
        lm_total_invest_str = "0"
        lm_entretien_date = None
    lm_objectifs_summary = ", ".join(
        [obj["libelle"] for obj in lm_objectifs if obj.get("libelle")]
    ) if lm_objectifs else ""

    # DER — Activités et domaines d'exercice pour le courtier courant
    DER_activites: list[dict] = []
    try:
        cid_val = DER_courtier.get("id") if DER_courtier else None
        if cid_val is not None:
            rows_der_act = db.execute(
                text(
                    """
                    SELECT a.activite_id, a.statut,
                           r.code, r.libelle, r.domaine, r.sous_categorie, r.description
                    FROM DER_courtier_activite a
                    JOIN DER_courtier_activite_ref r ON r.id = a.activite_id
                    WHERE a.courtier_id = :cid
                    ORDER BY r.domaine, r.libelle
                    """
                ),
                {"cid": cid_val},
            ).fetchall()
            for rr in rows_der_act or []:
                mm = rr._mapping
                DER_activites.append({
                    "activite_id": mm.get("activite_id"),
                    "statut": mm.get("statut"),
                    "code": mm.get("code"),
                    "libelle": mm.get("libelle"),
                    "domaine": mm.get("domaine"),
                    "sous_categorie": mm.get("sous_categorie"),
                    "description": mm.get("description"),
                })
    except Exception:
        DER_activites = []
    # ESG UI context for client-level
    try:
        alloc_names_client = [r[0] for r in db.query(Allocation.nom).filter(Allocation.nom.isnot(None)).distinct().order_by(Allocation.nom.asc()).all()]
    except Exception:
        alloc_names_client = []
    esg_fields_client: list[dict] = []
    esg_field_labels_client: dict[str, str] = {}
    esg_comparison_alloc: str | None = None
    esg_comparison_rows: list[dict] = []
    esg_comparison_chart_png = None
    esg_unmatched_fields: list[str] = []
    esg_default_fields: list[str] = []
    try:
        esg_fields_client, esg_fields_debug_tmp = _get_esg_fields(db, debug=False)
    except Exception:
        esg_fields_client = []
    # Texte de l'offre (allocation_risque.texte) selon le niveau de risque
    allocation_reference_name: str | None = None
    adequation_allocation_html = None
    try:
        rid = None
        if risque_latest and (risque_latest.get("niveau_id") is not None):
            rid = int(risque_latest.get("niveau_id"))
        if rid is not None:
            row = db.execute(text(
                """
                SELECT COALESCE(a.nom, ar.allocation_name) AS allocation_nom,
                       ar.texte
                FROM allocation_risque ar
                LEFT JOIN allocations a ON a.nom = ar.allocation_name
                WHERE ar.risque_id = :rid
                ORDER BY ar.date_attribution DESC, ar.id DESC
                LIMIT 1
                """
            ), {"rid": rid}).fetchone()
            if row:
                allocation_reference_name = row[0]
            if row and row[1] is not None:
                md = str(row[1])
                try:
                    import re, html as _html
                    text_md = _html.escape(md)
                    text_md = re.sub(r"^###\s+(.*)$", r"<h5>\1</h5>", text_md, flags=re.M)
                    text_md = re.sub(r"^##\s+(.*)$", r"<h4>\1</h4>", text_md, flags=re.M)
                    text_md = re.sub(r"^#\s+(.*)$", r"<h3>\1</h3>", text_md, flags=re.M)
                    text_md = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text_md)
                    text_md = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text_md)
                    lines = text_md.split('\n')
                    out = []
                    in_ul = False
                    for ln in lines:
                        if re.match(r"^\s*-\s+", ln):
                            if not in_ul:
                                out.append("<ul>"); in_ul=True
                            out.append("<li>" + re.sub(r"^\s*-\s+", "", ln) + "</li>")
                        else:
                            if in_ul:
                                out.append("</ul>"); in_ul=False
                            if ln.strip(): out.append("<p>"+ln+"</p>")
                    if in_ul: out.append("</ul>")
                    adequation_allocation_html = "\n".join(out)
                except Exception:
                    adequation_allocation_html = md.replace('\n','<br/>')
    except Exception:
        adequation_allocation_html = None

    esg_field_labels_client = { it["col"]: it["label"] for it in esg_fields_client }

    if not allocation_reference_name and alloc_names_client:
        allocation_reference_name = alloc_names_client[0]

    esg_comparison_alloc = allocation_reference_name or (alloc_names_client[0] if alloc_names_client else None)
    # Tenter de déterminer des indicateurs ESG pertinents (ceux qui possèdent des données numériques)
    fields_to_try = [it.get("col") for it in esg_fields_client if it.get("col")]
    if len(fields_to_try) > 12:
        fields_to_try = fields_to_try[:12]
    esg_comparison_payload = None
    if esg_comparison_alloc and fields_to_try:
        try:
            esg_comparison_payload, _dbg = _compute_client_esg_comparison(
                db=db,
                client_id=client_id,
                fields=fields_to_try,
                alloc=esg_comparison_alloc,
                alloc_isin=None,
                as_of=None,
                alloc_date=None,
                debug=False,
            )
            all_rows: list[dict] = []
            for row in esg_comparison_payload.get("results", []):
                field_name = row.get("field")
                label = esg_field_labels_client.get(field_name, field_name)
                all_rows.append({
                    "field": field_name,
                    "label": label,
                    "index": row.get("index"),
                    "client": row.get("client"),
                    "cli_present": row.get("cli_present"),
                    "idx_present": row.get("idx_present"),
                })
            rows_with_values = [
                r for r in all_rows if (r.get("index") is not None and r.get("client") is not None)
            ]
            esg_comparison_rows = rows_with_values[:4] if rows_with_values else all_rows[:4]
            esg_default_fields = [r.get("field") for r in esg_comparison_rows if r.get("field")]

            preferred_fields = [r.get("field") for r in rows_with_values]
            if preferred_fields:
                ordered = []
                seen_pf = set()
                for f in preferred_fields:
                    for it in esg_fields_client:
                        col = it.get("col")
                        if col == f and col not in seen_pf:
                            ordered.append(it)
                            seen_pf.add(col)
                for it in esg_fields_client:
                    col = it.get("col")
                    if col not in seen_pf:
                        ordered.append(it)
                esg_fields_client = ordered
                esg_field_labels_client = { it["col"]: it["label"] for it in esg_fields_client }

            if rows_with_values:
                categories = [str(r.get("label") or r.get("field")) for r in rows_with_values]
                index_values = [float(r.get("index") or 0.0) for r in rows_with_values]
                client_values = [float(r.get("client") or 0.0) for r in rows_with_values]
                esg_comparison_chart_png = _build_matplotlib_esg_bar_chart(categories, index_values, client_values)
        except Exception:
            esg_comparison_payload = None

    if not esg_comparison_rows and esg_comparison_alloc:
        try:
            payload_tmp, _ = _compute_client_esg_comparison(
                db=db,
                client_id=client_id,
                fields=[it.get("col") for it in esg_fields_client if it.get("col")][:4],
                alloc=esg_comparison_alloc,
                alloc_isin=None,
                as_of=None,
                alloc_date=None,
                debug=False,
            )
            esg_comparison_rows = payload_tmp.get("results") or []
            esg_unmatched_fields = payload_tmp.get("unmatched_fields") or []
            esg_default_fields = payload_tmp.get("resolved_fields") or []
        except Exception:
            esg_comparison_rows = []

    esg_unmatched_fields = esg_comparison_payload.get("unmatched_fields", []) if esg_comparison_payload else []

    if (esg_comparison_alloc and not esg_comparison_rows):
        fallback_fields = esg_default_fields[:] if esg_default_fields else []
        if not fallback_fields:
            fallback_fields = [
                it.get("col")
                for it in esg_fields_client
                if it.get("col")
            ][:4]
        if not fallback_fields:
            fallback_fields = [
                "Avoiding water scarcity",
                "Board independence",
                "GHG intensity-Value",
                "Gender pay gap",
            ]
        try:
            payload_tmp, _dbg2 = _compute_client_esg_comparison(
                db=db,
                client_id=client_id,
                fields=fallback_fields,
                alloc=esg_comparison_alloc,
                alloc_isin=None,
                as_of=None,
                alloc_date=None,
                debug=False,
            )
            esg_comparison_rows = payload_tmp.get("results") or []
            esg_unmatched_fields = payload_tmp.get("unmatched_fields") or []
            esg_default_fields = payload_tmp.get("resolved_fields") or fallback_fields
        except Exception:
            esg_comparison_rows = []

    # Contrat choisi (si défini via KYC_contrat_choisi)
    contrat_choisi_nom = None
    contrat_choisi_societe = None
    try:
        row = db.execute(text(
            """
            SELECT g.nom_contrat, COALESCE(s.nom,'') AS societe_nom
            FROM KYC_contrat_choisi k
            LEFT JOIN mariadb_affaires_generique g ON g.id = k.id_contrat
            LEFT JOIN mariadb_societe s ON s.id = g.id_societe
            WHERE k.id_client = :cid
            LIMIT 1
            """
        ), {"cid": client_id}).fetchone()
        if row:
            m = row._mapping
            contrat_choisi_nom = m.get('nom_contrat') or row[0]
            contrat_choisi_societe = m.get('societe_nom') if 'societe_nom' in m else (row[1] if len(row) > 1 else None)
    except Exception:
        contrat_choisi_nom = None
        contrat_choisi_societe = None

    # --- Lettre d'adéquation: données consolidées (requêtes directes) ---
    etat_civil_latest = None
    try:
        row = db.execute(text("SELECT civilite, situation_familiale FROM etat_civil_client WHERE id_client = :cid ORDER BY id DESC LIMIT 1"), {"cid": client_id}).fetchone()
        if row:
            m = row._mapping
            etat_civil_latest = {"civilite": m.get("civilite"), "situation_familiale": m.get("situation_familiale")}
    except Exception:
        etat_civil_latest = None

    nb_enfants_latest = None
    try:
        row = db.execute(text("SELECT nb_enfants FROM KYC_Client_Situation_Matrimoniale WHERE client_id = :cid ORDER BY (date_saisie IS NULL), date_saisie DESC, id DESC LIMIT 1"), {"cid": client_id}).fetchone()
        if row is not None:
            nb_enfants_latest = row[0]
    except Exception:
        nb_enfants_latest = None

    synthese_latest = None
    try:
        row = db.execute(text("SELECT total_revenus, total_charges, total_actif, total_passif, commentaire FROM KYC_Client_Synthese WHERE client_id = :cid ORDER BY date_saisie DESC, id DESC LIMIT 1"), {"cid": client_id}).fetchone()
        if row:
            m = row._mapping
            synthese_latest = {k: m.get(k) for k in ("total_revenus","total_charges","total_actif","total_passif","commentaire")}
    except Exception:
        synthese_latest = None

    risque_latest = None
    try:
        row = db.execute(text(
            """
            SELECT
              r.libelle AS niveau_risque,
              k.niveau_id AS niveau_id,
              k.duree AS horizon_placement,
              k.experience,
              k.connaissance,
              k.commentaire,
              k.confirmation_client
            FROM KYC_Client_Risque k
            JOIN ref_niveau_risque r ON k.niveau_id = r.id
            WHERE k.client_id = :cid
            ORDER BY k.date_saisie DESC, k.id DESC
            LIMIT 1
            """
        ), {"cid": client_id}).fetchone()
        if row:
            risque_latest = dict(row._mapping)
    except Exception:
        risque_latest = None

    # Contrats génériques (pour comparatif des offres)
    adequation_contracts = []
    try:
        rows = db.execute(text(
            """
            SELECT g.id,
                   g.nom_contrat,
                   COALESCE(s.nom, '') AS societe_nom,
                   COALESCE(g.frais_gestion_assureur,0) + COALESCE(g.frais_gestion_courtier,0) AS total_frais
            FROM mariadb_affaires_generique g
            LEFT JOIN mariadb_societe s ON s.id = g.id_societe
            WHERE COALESCE(g.actif, 1) = 1
            ORDER BY s.nom, g.nom_contrat
            """
        )).fetchall()
        adequation_contracts = [dict(r._mapping) for r in rows]
    except Exception:
        adequation_contracts = []

    # Objectifs client (pour lettre d'adéquation)
    adequation_objectifs = []
    try:
        rows = db.execute(text(
            """
            SELECT o.objectif_id,
                   ro.libelle AS objectif_libelle,
                   o.horizon_investissement,
                   o.commentaire,
                   COALESCE(o.niveau_id, 9999) AS _niv
            FROM KYC_Client_Objectifs o
            LEFT JOIN ref_objectif ro ON ro.id = o.objectif_id
            WHERE o.client_id = :cid
            ORDER BY _niv, ro.libelle, o.id
            """
        ), {"cid": client_id}).fetchall()
        adequation_objectifs = [dict(r._mapping) for r in rows]
    except Exception:
        adequation_objectifs = []

    report_logo_png = None
    try:
        logo_path = Path(__file__).resolve().parents[1] / "logo" / "logo.png"
        report_logo_png = base64.b64encode(logo_path.read_bytes()).decode("ascii") if logo_path.exists() else None
    except Exception:
        report_logo_png = None

    return templates.TemplateResponse(
        "dashboard_client_detail.html",
        {
            "request": request,
            "client": client,
            "historique": historique,
            "last_row": last_row,
            "report_logo_png": report_logo_png,
            "documents_client": documents_client,
            # séries pour graphiques
            "labels": chart_labels,
            "serie_valo": chart_valo,
            "serie_mov_cum": chart_mov_cum,
            "serie_mov_raw": chart_mov_raw,
            "client_affaires": client_affaires,

            # KPIs
            "last_valo_str": last_valo_str,
            "last_perf_pct": last_perf_pct,
            "last_vol_pct": last_vol_pct,
            "current_srri": current_srri,
            "client_srri": client_srri,
            "header_srri_icon": header_srri_icon,
            # Mouvements
            "depots_total": depots_total,
            "retraits_total": retraits_total,
            "solde_total": solde_total,
            "depots_str": depots_str,
            "retraits_str": retraits_str,
            "solde_str": solde_str,
            "valo_gt_solde": valo_gt_solde,
            # Supports consolidés client
            "client_supports": client_supports,
            "nb_supports_actifs": nb_supports_actifs,
            # (comparatif SICAV retiré)
            # Séries annuelles pour graphiques
            "years_client": years_client,
            "ann_perf_client": ann_perf_client,
            "ann_vol_client": ann_vol_client,
            # Reportings pluriannuels
            "reporting_years": reporting_years,
            # Sélection date Investissements
            "available_dates": available_dates,
            "as_of_effective": as_of_effective,
            # Comptages + durée + perf annualisée depuis début
            "nb_contrats_ouverts": nb_contrats_ouverts,
            "nb_contrats_fermes": nb_contrats_fermes,
            "duree_historique_str": duree_historique_str,
            "overall_ann_perf_pct": overall_ann_perf_pct,
            # Données pour graphique allocations (lignes)
            "alloc_series": alloc_series,
            "client_sicav": client_sicav,
            # ESG UI context
            "alloc_names": alloc_names_client,
            "esg_fields": esg_fields_client,
            "esg_field_labels": esg_field_labels_client,
            "esg_comparison_alloc": esg_comparison_alloc,
            "esg_comparison_rows": esg_comparison_rows,
            "esg_comparison_chart_png": esg_comparison_chart_png,
            "esg_default_fields": esg_default_fields,
            "esg_unmatched_fields": esg_unmatched_fields,
            "contrat_choisi_nom": contrat_choisi_nom,
            "contrat_choisi_societe": contrat_choisi_societe,
            "allocation_reference_name": allocation_reference_name or client_allocation_name,
            "client_allocation_id": client_allocation_id,
            "client_allocation_name": client_allocation_name,
            "allocations_lookup": allocations_lookup,
            # Lettre d'adéquation: infos consolidées
            "etat_civil_latest": etat_civil_latest,
            "nb_enfants_latest": nb_enfants_latest,
            "synthese_latest": synthese_latest,
            "risque_latest": risque_latest,
            # Lettre d'adéquation: infos consolidées
            "etat_civil_latest": etat_civil_latest,
            "nb_enfants_latest": nb_enfants_latest,
            "synthese_latest": synthese_latest,
            "risque_latest": risque_latest,
            "adequation_contracts": adequation_contracts,
            "adequation_objectifs": adequation_objectifs,
            "adequation_allocation_html": adequation_allocation_html,
            # Tâches: assistance création locale
            "types": types,
            "categories": cats,
            "statuts": statuts,
            "status_ui": status_ui,
            "en_cours_id": en_cours_id,
            "clients_suggest": clients_suggest,
            "affaires_suggest": affaires_suggest,
            "client_fullname_default": client_fullname_default,
            "reclam_statuses": reclam_statuses,
            "tb_markers_visible": tb_markers_visible,
            "tb_markers_param": "markers=1" if tb_markers_visible else "",
            # Messages/alertes en-tête
            "msgs_reg_count": msgs_reg_count,
            "msgs_nonreg_count": msgs_nonreg_count,
            "client_events_open": client_events_open,
            "rh_list": rh_list,
            # KYC Actifs
            "kyc_actifs": kyc_actifs,
            "kyc_types_actifs": rows_types_actifs,
            "kyc_situations_professionnelles": locals().get("situations_professionnelles", []),
            # Conformité (LCBFT / FATCA) context for detail page modal
            "lcbft_current": lcbft_current,
            "lcbft_vigilance_options": lcbft_vigilance_options,
            "lcbft_vigilance_ids": lcbft_vigilance_ids,
            "fatca_contracts": fatca_contracts,
            "fatca_saved": fatca_saved,
            "fatca_client_country": fatca_client_country,
            "fatca_client_nif": fatca_client_nif,
            "fatca_today": fatca_today,
            # TRACFIN indicateurs
            "lcbft_invest_total": locals().get('lcbft_invest_total', 0.0),
            "lcbft_patrimoine_net": locals().get('lcbft_patrimoine_net', None),
            "lcbft_part_pct": locals().get('lcbft_part_pct', None),
            "lcbft_ppe_options": locals().get('lcbft_ppe_options', []),
            "lcbft_objectifs": locals().get('lcbft_objectifs', []),
            "lcbft_operation_types": locals().get('lcbft_operation_types', []),
            "lcbft_operation_selected_ids": locals().get('lcbft_operation_selected_ids', []),
            "lcbft_revenue_total": locals().get('lcbft_revenue_total', None),
            "lcbft_revenue_tranches": locals().get('lcbft_revenue_tranches', []),
            "lcbft_revenue_tranche_id": locals().get('lcbft_revenue_tranche_id', None),
            "lcbft_patrimoine_total": locals().get('lcbft_patrimoine_total', None),
            "lcbft_patrimoine_tranches": locals().get('lcbft_patrimoine_tranches', []),
            "lcbft_patrimoine_tranche_id": locals().get('lcbft_patrimoine_tranche_id', None),
            "lcbft_raison_options": locals().get('lcbft_raison_options', []),
            "lcbft_raison_selected_ids": locals().get('lcbft_raison_selected_ids', []),
            "lcbft_raison_forced_ids": locals().get('lcbft_raison_forced_ids', []),
            "lcbft_raison_disabled_ids": locals().get('lcbft_raison_disabled_ids', []),
            "lm_today": _date.today().strftime('%d/%m/%Y'),
            # DER context for modal rendering
            "DER_courtier": DER_courtier,
            "DER_statut_social": DER_statut_social,
            "DER_courtier_garanties_normes": DER_courtier_garanties_normes,
            "lm_remunerations": lm_remunerations,
            "centre_mediation_lib": centre_mediation_lib,
            "DER_activites": DER_activites,
            # points 2 et 8 retirés temporairement
            # Lettre de mission : objectifs & montants
            "lm_objectifs": lm_objectifs,
            "lm_total_invest": lm_total_invest,
            "lm_total_invest_str": lm_total_invest_str,
            "lm_entretien_date": lm_entretien_date,
            # Compatibilité avec anciens placeholders du template
            "montantInvesti": lm_total_invest_str,
            "dateEntretien": lm_entretien_date,
            "objectifsMission": lm_objectifs_summary or "—",
        }
    )


@router.post("/clients/{client_id}/commercial", response_class=HTMLResponse)
async def dashboard_client_update_commercial(client_id: int, request: Request, db: Session = Depends(get_db)):
    """Met à jour le responsable (commercial) associé à un client depuis la fiche détail."""
    form = await request.form()
    raw_id = (form.get("commercial_id") or "").strip()
    new_id: int | None
    if not raw_id:
        new_id = None
    else:
        try:
            new_id = int(raw_id)
        except (TypeError, ValueError):
            new_id = None

    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client introuvable.")

    try:
        client.commercial_id = new_id
        db.add(client)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Update commercial failed for client %s: %s", client_id, exc, exc_info=True)
    return RedirectResponse(url=f"/dashboard/clients/{client_id}", status_code=303)


@router.post("/clients/{client_id}/actifs", response_class=HTMLResponse)
async def dashboard_client_add_actif(client_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    # Récupérer champs
    id_type_actif = form.get("id_type_actif")
    intitule = form.get("intitule")
    valeur_initiale = form.get("valeur_initiale")
    date_acquisition = form.get("date_acquisition")
    valeur = form.get("valeur")
    devise = form.get("devise")
    date_eval = form.get("date_eval")
    commentaire = form.get("commentaire")
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text(
                """
                INSERT INTO actif_client
                  (id_client, id_type_actif, intitule, valeur_initiale, date_acquisition, valeur, devise, date_eval, commentaire)
                VALUES
                  (:cid, :tid, :intitule, :val_init, :dacq, :val, :dev, :deval, :comm)
                """
            ),
            {
                "cid": client_id,
                "tid": int(id_type_actif) if id_type_actif else None,
                "intitule": intitule or None,
                "val_init": float(valeur_initiale.replace(',', '.')) if valeur_initiale else None,
                "dacq": date_acquisition or None,
                "val": float(valeur.replace(',', '.')) if valeur else None,
                "dev": devise or None,
                "deval": date_eval or None,
                "comm": commentaire or None,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
    from starlette.responses import RedirectResponse
    return RedirectResponse(url=f"/dashboard/clients/{client_id}", status_code=303)


@router.post("/clients/{client_id}/actifs/delete", response_class=HTMLResponse)
async def dashboard_client_delete_actif(client_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    actif_id = form.get("actif_id")
    if not actif_id:
        from starlette.responses import RedirectResponse
        return RedirectResponse(url=f"/dashboard/clients/{client_id}", status_code=303)
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM actif_client WHERE id = :id AND id_client = :cid"), {"id": int(actif_id), "cid": client_id})
        db.commit()
    except Exception:
        db.rollback()
    from starlette.responses import RedirectResponse
    return RedirectResponse(url=f"/dashboard/clients/{client_id}", status_code=303)


# ---------------- Allocations ----------------
@router.get("/allocations", response_class=HTMLResponse)
def dashboard_allocations(request: Request, db: Session = Depends(get_db)):
    # Dernière valeur par nom (pour le tableau)
    sub_last = (
        db.query(
            Allocation.nom.label("nom"),
            func.max(Allocation.date).label("last_date")
        )
        .group_by(Allocation.nom)
        .subquery()
    )
    last_rows = (
        db.query(
            Allocation.nom,
            Allocation.date,
            Allocation.perf_sicav_52,
            Allocation.volat,
            Allocation.valo,
        )
        .join(sub_last, (Allocation.nom == sub_last.c.nom) & (Allocation.date == sub_last.c.last_date))
        .order_by(Allocation.nom.asc())
        .all()
    )

    # Total des valorisations basé sur la dernière valeur par nom
    total_allocations = sum([(r.valo or 0) for r in last_rows]) if last_rows else 0

    # Série complète pour graphiques (par nom)
    series_rows = (
        db.query(
            Allocation.nom,
            Allocation.date,
            Allocation.perf_sicav_52,
            Allocation.volat,
            Allocation.sicav,
        )
        .order_by(Allocation.nom.asc(), Allocation.date.asc())
        .all()
    )

    series_data: dict[str, list[dict]] = {}
    for nom, date, perf52, vol, sicav in series_rows:
        arr = series_data.setdefault(nom, [])
        # format date en YYYY-MM-DD si possible
        dstr = None
        try:
            dstr = date.strftime("%Y-%m-%d") if date else None
        except Exception:
            dstr = str(date)[:10] if date else None
        arr.append({
            "date": dstr,
            "perf": float(perf52 or 0),
            "vol": float(vol or 0),
            "sicav": float(sicav or 0),
        })
    return templates.TemplateResponse(
        "dashboard_allocations.html",
        {
            "request": request,
            "total_allocations": total_allocations,
            "allocations": last_rows,
            "series_data": series_data,
        }
    )


# ---------------- Documents ----------------
@router.get("/documents", response_class=HTMLResponse)
def dashboard_documents(request: Request, db: Session = Depends(get_db)):
    # Documents liés aux clients avec metadata de type
    rows = (
        db.query(
            DocumentClient.id.label("id"),
            Client.id.label("client_id"),
            DocumentClient.nom_client.label("nom_client"),
            Client.nom.label("c_nom"),
            Client.prenom.label("c_prenom"),
            DocumentClient.nom_document.label("nom_document"),
            Document.documents.label("type_document"),
            Document.niveau.label("niveau"),
            Document.risque.label("risque"),
            DocumentClient.obsolescence.label("obsolescence"),
        )
        .outerjoin(Client, Client.id == DocumentClient.id_client)
        .outerjoin(Document, Document.id_document_base == DocumentClient.id_document_base)
        .all()
    )
    documents = []
    for r in rows:
        nom = r.c_nom
        prenom = r.c_prenom
        if (not nom and not prenom) and r.nom_client:
            parts = (r.nom_client or "").split()
            if parts:
                nom = parts[0]
                prenom = " ".join(parts[1:]) or None
        documents.append({
            "id": r.id,
            "client_id": getattr(r, "client_id", None),
            "nom": nom or "",
            "prenom": prenom or "",
            "document": r.nom_document or r.type_document,
            "niveau": r.niveau,
            "risque": r.risque,
            "obsolescence": r.obsolescence,
        })
    total_documents = len(documents)

    # Totaux obsolescences par niveau
    obs_by_niveau = (
        db.query(
            Document.niveau,
            func.count(DocumentClient.id)
        )
        .join(Document, Document.id_document_base == DocumentClient.id_document_base)
        .filter(DocumentClient.obsolescence.isnot(None))
        .group_by(Document.niveau)
        .all()
    )
    obs_by_niveau = [{"niveau": n, "nb": int(nb)} for n, nb in obs_by_niveau]

    # Totaux obsolescences par risque
    obs_by_risque = (
        db.query(
            Document.risque,
            func.count(DocumentClient.id)
        )
        .join(Document, Document.id_document_base == DocumentClient.id_document_base)
        .filter(DocumentClient.obsolescence.isnot(None))
        .group_by(Document.risque)
        .all()
    )
    obs_by_risque = [{"risque": r, "nb": int(nb)} for r, nb in obs_by_risque]
    return templates.TemplateResponse(
        "dashboard_documents.html",
        {
            "request": request,
            "total_documents": total_documents,
            "documents": documents,
            "obs_by_niveau": obs_by_niveau,
            "obs_by_risque": obs_by_risque,
        }
    )
# List allocation dates for a given name (distinct)
@router.get("/allocations/dates", response_class=JSONResponse)
def list_allocation_dates(name: str | None = None, isin: str | None = None, db: Session = Depends(get_db)):
    try:
        if isin:
            rows = db.execute(text("SELECT DISTINCT date FROM allocations WHERE ISIN = :i ORDER BY date DESC"), {"i": isin}).fetchall()
        else:
            rows = db.execute(text("SELECT DISTINCT date FROM allocations WHERE nom = :n ORDER BY date DESC"), {"n": name}).fetchall()
        # Return ISO strings
        out = []
        for r in rows or []:
            d = getattr(r, 'date', None)
            try:
                out.append(d.strftime('%Y-%m-%d'))
            except Exception:
                out.append(str(d)[:10] if d else None)
        return {"name": name, "isin": isin, "dates": [x for x in out if x]}
    except Exception as e:
        return {"name": name, "isin": isin, "dates": [], "error": str(e)}
    try:
        admin_intervenants = rows_to_dicts(
            db.execute(
                _text(
                    """
                    SELECT id, nom, type_niveau, type_personne, telephone, mail
                    FROM administration_intervenant
                    ORDER BY nom
                    """
                )
            ).fetchall()
        )
    except Exception:
        admin_intervenants = []


def _list_taches_for_calendar(db: Session, statut: str | None = None, categorie: str | None = None, rh_id: int | None = None, range_days: int | None = None) -> list[dict]:
    """Retourne les tâches avec enrichissements (client, affaire, rh) pour calendrier/export."""
    from sqlalchemy import text
    conds: list[str] = []
    params: dict = {}
    if statut:
        conds.append("statut = :statut"); params["statut"] = statut
    if categorie:
        conds.append("categorie = :categorie"); params["categorie"] = categorie
    if rh_id is not None:
        conds.append("rh_id = :rh_id"); params["rh_id"] = rh_id
    if range_days is not None:
        try:
            rd = max(0, int(range_days))
        except Exception:
            rd = 0
        if rd > 0:
            today = datetime.utcnow().date()
            start_date = today - timedelta(days=rd)
            end_date = today + timedelta(days=rd)
            conds.append("date(date_evenement) BETWEEN :start_date AND :end_date")
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"SELECT * FROM vue_suivi_evenement{where} ORDER BY date_evenement DESC"
    try:
        rows = db.execute(text(sql), params).fetchall()
    except Exception as exc:
        logger.exception("Calendrier: vue_suivi_evenement manquante ou inaccessible", exc_info=exc)
        return []
    # Prépare les libellés RH / clients / affaires pour l'affichage du calendrier
    rh_lookup: dict[int, str] = {}
    try:
        for rh in fetch_rh_list(db):
            rid = rh.get("id")
            try:
                rid_int = int(rid)
            except Exception:
                continue
            label_parts = [(rh.get("prenom") or "").strip(), (rh.get("nom") or "").strip()]
            label = " ".join([p for p in label_parts if p]) or (rh.get("mail") or "").strip() or f"RH #{rid_int}"
            rh_lookup[rid_int] = label
    except Exception:
        rh_lookup = {}
    client_ids = {getattr(r, "client_id", None) for r in rows if getattr(r, "client_id", None) is not None}
    affaire_ids = {getattr(r, "affaire_id", None) for r in rows if getattr(r, "affaire_id", None) is not None}
    clients_map_full: dict[int, str] = {}
    affaires_map_ref: dict[int, str] = {}
    if client_ids:
        try:
            cli_rows = db.query(Client.id, Client.nom, Client.prenom).filter(Client.id.in_(list(client_ids))).all()
            for cid, nom, prenom in cli_rows:
                full = f"{nom or ''} {prenom or ''}".strip()
                clients_map_full[cid] = full or nom or prenom or str(cid)
        except Exception:
            pass
    if affaire_ids:
        try:
            aff_rows = db.query(Affaire.id, Affaire.ref).filter(Affaire.id.in_(list(affaire_ids))).all()
            for aid, ref in aff_rows:
                affaires_map_ref[aid] = ref or str(aid)
        except Exception:
            pass
    def _to_iso_date(val):
        if val is None:
            return None
        try:
            return val.isoformat()
        except Exception:
            try:
                return str(val)[:10]
            except Exception:
                return None
    def _col(row_obj, name: str):
        try:
            if hasattr(row_obj, "_mapping") and name in row_obj._mapping:
                return row_obj._mapping.get(name)
        except Exception:
            pass
        try:
            return getattr(row_obj, name)
        except Exception:
            return None
    items: list[dict] = []
    for row in rows:
        # Identifiant obligatoire pour le clic calendrier : ignorer les entrées sans id
        eid_raw = None
        if hasattr(row, "evenement_id"):
            eid_raw = getattr(row, "evenement_id", None)
        if eid_raw is None and hasattr(row, "id"):
            eid_raw = getattr(row, "id", None)
        try:
            eid_int = int(eid_raw)
        except Exception:
            eid_int = None
        if eid_int is None:
            continue
        rh_raw = getattr(row, "rh_id", None)
        try:
            rh_int = int(rh_raw) if rh_raw is not None else None
        except Exception:
            rh_int = None
        client_id_val = getattr(row, "client_id", None)
        affaire_id_val = getattr(row, "affaire_id", None)
        client_nom = clients_map_full.get(client_id_val) or getattr(row, "nom_client", None)
        affaire_ref = affaires_map_ref.get(affaire_id_val) or getattr(row, "ref_affaire", None)
        items.append({
            "id": eid_int,
            "title": (_col(row, "type_evenement") or _col(row, "commentaire") or "Tâche"),
            "start": _to_iso_date(getattr(row, "date_evenement", None)),
            "end": _to_iso_date(getattr(row, "date_evenement", None)),
            "statut": _col(row, "statut"),
            "categorie": _col(row, "categorie"),
            "type_evenement": _col(row, "type_evenement"),
            "responsable": _col(row, "utilisateur_responsable"),
            "rh_id": rh_int,
            "rh_label": rh_lookup.get(rh_int),
            "client_id": client_id_val,
            "client_nom": client_nom,
            "affaire_id": affaire_id_val,
            "affaire_ref": affaire_ref,
            "commentaire": _col(row, "commentaire"),
        })
    return items


@router.get("/api/taches", response_class=JSONResponse)
def api_taches(db: Session = Depends(get_db), statut: str | None = None, categorie: str | None = None, rh_id: int | None = None):
    """Retourne les tâches (vue_suivi_evenement) au format JSON pour le calendrier."""
    items = _list_taches_for_calendar(db, statut=statut, categorie=categorie, rh_id=rh_id)
    return {"items": items}


@router.get("/api/taches.ics", response_class=PlainTextResponse)
def api_taches_ics(
    db: Session = Depends(get_db),
    statut: str | None = None,
    categorie: str | None = None,
    rh_id: int | None = None,
    range_days: int | None = Query(90, ge=1, le=365*3, description="Nombre de jours autour d'aujourd'hui"),
):
    """Export ICS des tâches visibles dans le calendrier."""
    items = _list_taches_for_calendar(db, statut=statut, categorie=categorie, rh_id=rh_id, range_days=range_days)
    def _ics_escape(val: str | None) -> str:
        if val is None:
            return ""
        return (
            str(val)
            .replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )
    def _parse_date(val: str | None):
        if not val:
            return None
        try:
            return datetime.fromisoformat(val).date()
        except Exception:
            try:
                return datetime.fromisoformat(val[:10]).date()
            except Exception:
                return None
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CRM//Taches//FR", "CALSCALE:GREGORIAN", "METHOD:PUBLISH"]
    for it in items:
        d_start = _parse_date(it.get("start"))
        if not d_start:
            continue
        d_end = d_start + timedelta(days=1)
        uid = f"{it.get('id')}@crm"
        summary = _ics_escape(it.get("title") or "Tâche")
        parts_desc = []
        if it.get("statut"):
            parts_desc.append(f"Statut: {it.get('statut')}")
        if it.get("rh_label"):
            parts_desc.append(f"Responsable: {it.get('rh_label')}")
        elif it.get("responsable"):
            parts_desc.append(f"Responsable: {it.get('responsable')}")
        if it.get("client_nom"):
            parts_desc.append(f"Client: {it.get('client_nom')}")
        if it.get("affaire_ref"):
            parts_desc.append(f"Affaire: {it.get('affaire_ref')}")
        if it.get("commentaire"):
            parts_desc.append(f"Commentaire: {it.get('commentaire')}")
        description = _ics_escape("\\n".join(parts_desc)) if parts_desc else ""
        categories = []
        if it.get("categorie"):
            categories.append(it.get("categorie"))
        if it.get("statut"):
            categories.append(it.get("statut"))
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART;VALUE=DATE:{d_start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{d_end.strftime('%Y%m%d')}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{description}",
            f"CATEGORIES:{_ics_escape(','.join(categories))}" if categories else "",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    # Retire les éventuelles lignes vides dues aux catégories absentes
    ics = "\r\n".join([ln for ln in lines if ln != ""])
    return PlainTextResponse(content=ics, media_type="text/calendar; charset=utf-8")
