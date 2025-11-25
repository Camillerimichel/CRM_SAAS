"""
Collecte simple de veille réglementaire (AMF, ACPR, ESMA/EIOPA, presse pro) avec filtrage par mots-clés.

Usage:
    python scripts/veille_reglementaire.py --out data/veille_reglementaire.json

Notes:
- Le script privilégie les flux RSS/Atom pour limiter le parsing HTML.
- Les URLs peuvent devoir être ajustées si certains flux évoluent.
- Aucun module externe n'est requis: utilisation de urllib + xml.etree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterable
from urllib import request, error as urlerror
from urllib.parse import urljoin
import ssl
from html.parser import HTMLParser


# Mots-clés retenus pour la veille
KEYWORDS = [
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


@dataclass
class Source:
    name: str
    url: str
    tags: List[str]
    html_fallback: Optional[List[str]] = None


SOURCES: List[Source] = [
    # Régulateurs
    Source(
        "AMF - Actualités",
        "https://www.amf-france.org/fr/actualites",
        ["amf", "regulateur"],
        html_fallback=[
          "https://www.amf-france.org/fr/actualites?type=all",
          "https://www.amf-france.org/fr/espace-presse/communiques-de-presse",
          "https://www.amf-france.org/fr",
        ],
    ),
    Source(
        "ACPR - Actualités",
        "https://acpr.banque-france.fr/rss.xml",
        ["acpr", "regulateur"],
        html_fallback=[
          "https://acpr.banque-france.fr/communiques-de-presse",
          "https://acpr.banque-france.fr/actualites",
          "https://acpr.banque-france.fr/",
        ],
    ),
    Source(
        "ESMA - News",
        "https://www.esma.europa.eu/whats-new",
        ["esma", "ue"],
        html_fallback=[
          "https://www.esma.europa.eu/press-news/esma-news",
          "https://www.esma.europa.eu/press-news/press-releases",
          "https://www.esma.europa.eu/",
        ],
    ),
    Source(
        "EIOPA - News",
        "https://www.eiopa.europa.eu/news-events/news",
        ["eiopa", "ue"],
        html_fallback=[
          "https://www.eiopa.europa.eu/news-events/news",
          "https://www.eiopa.europa.eu/media/speeches",
          "https://www.eiopa.europa.eu/",
        ],
    ),
    Source(
        "EBA - News",
        "https://www.eba.europa.eu/rss/press-and-news",
        ["eba", "ue"],
        html_fallback=[
          "https://www.eba.europa.eu/press-and-media/news",
          "https://www.eba.europa.eu/",
        ],
    ),
    Source(
        "CNIL - Actualités",
        "https://www.cnil.fr/fr/rss.xml",
        ["cnil", "rgpd"],
        html_fallback=["https://www.cnil.fr/fr/actualites", "https://www.cnil.fr/"],
    ),
    Source("CJUE - Communiqués", "https://curia.europa.eu/jcms/jcms/Jo2_7026/rss", ["cjue", "ue", "jurisprudence"]),
    # Presse spécialisée (flux publics susceptibles d'évoluer)
    Source("Agefi Actifs", "https://www.agefiactifs.com/", ["presse"], html_fallback=["https://www.agefiactifs.com/"]),
    Source("Revue Banque", "https://www.revue-banque.fr/", ["presse"], html_fallback=[
      "https://www.revue-banque.fr/banque-detail/actualites",
      "https://www.revue-banque.fr/assurance/actualites",
      "https://www.revue-banque.fr/",
    ]),
    Source("L'Argus de l'assurance", "https://www.argusdelassurance.com/", ["presse", "assurance"], html_fallback=["https://www.argusdelassurance.com/"]),
    Source("Option Finance", "https://www.optionfinance.fr/", ["presse"], html_fallback=[
      "https://www.optionfinance.fr/accueil/actualites.html",
      "https://www.optionfinance.fr/asset-management/actualites.html",
      "https://www.optionfinance.fr/",
    ]),
    # Décisions AMF (peut nécessiter ajustement si flux change)
    Source("AMF - Décisions et sanctions", "https://www.amf-france.org/fr/decisions-sanctions/rss", ["amf", "sanctions"], html_fallback=["https://www.amf-france.org/fr/decisions-sanctions"]),
]


def fetch_url(url: str, timeout: int = 15, insecure: bool = False, proxy: Optional[str] = None) -> Optional[str]:
    try:
        ctx = ssl._create_unverified_context() if insecure else None
        handlers = []
        if proxy:
            handlers.append(request.ProxyHandler({"http": proxy, "https": proxy}))
        if ctx:
            handlers.append(request.HTTPSHandler(context=ctx))
        opener = request.build_opener(*handlers) if handlers else request.build_opener()
        req = request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; VeilleReglementaire/1.0)")
        req.add_header("Accept", "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8")
        with opener.open(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urlerror.URLError as exc:
        print(f"[warn] Échec de récupération {url}: {exc}", file=sys.stderr)
        return None


def parse_rss(xml_text: str) -> Iterable[Dict[str, str]]:
    """Parse un flux RSS/Atom minimal : titre, lien, date, description."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    # Atom namespaces fallback
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for item in root.findall(".//item"):
        yield {
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "date": (item.findtext("pubDate") or "").strip(),
            "summary": (item.findtext("description") or "").strip(),
        }
    for entry in root.findall(".//atom:entry", ns):
        yield {
            "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
            "link": (entry.find("atom:link", ns).attrib.get("href", "") if entry.find("atom:link", ns) is not None else "").strip(),
            "date": (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip(),
            "summary": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
        }


class LinkParser(HTMLParser):
    """Parseur HTML simple pour extraire les liens et leur texte."""

    def __init__(self):
        super().__init__()
        self.links: List[Dict[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                self._current_href = href
                self._current_text_parts = []

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._current_href:
            text = "".join(self._current_text_parts).strip()
            self.links.append({"href": self._current_href, "text": text})
            self._current_href = None
            self._current_text_parts = []

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_text_parts.append(data)


def parse_html_links(html_text: str, base_url: str) -> Iterable[Dict[str, str]]:
    parser = LinkParser()
    try:
        parser.feed(html_text)
    except Exception:
        return []
    for link in parser.links:
        href = link.get("href", "")
        text = (link.get("text") or "").strip()
        if not href or not text:
            continue
        full_link = urljoin(base_url, href)
        yield {"title": text, "link": full_link, "date": "", "summary": ""}


def matches_keywords(text: str) -> bool:
    content = text.lower()
    return any(k in content for k in KEYWORDS)


def normalize_date(raw: str) -> str:
    """Essaie de normaliser la date au format ISO ; garde brut en cas d'échec."""
    if not raw:
        return ""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(raw, fmt).isoformat()
        except Exception:
            continue
    return raw


def collect(insecure: bool = False, proxy: Optional[str] = None) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()
    for src in SOURCES:
        xml_text = fetch_url(src.url, insecure=insecure, proxy=proxy)
        entries: List[Dict[str, str]] = []
        if xml_text:
            entries.extend(parse_rss(xml_text))
        if not entries and src.html_fallback:
            for alt in src.html_fallback:
                html_text = fetch_url(alt, insecure=insecure, proxy=proxy)
                if not html_text:
                    continue
                entries.extend(parse_html_links(html_text, alt))
                if entries:
                    break

        for entry in entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")
            blob = " ".join([title, summary])
            if not matches_keywords(blob):
                continue
            key = (title, link)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "source": src.name,
                    "tags": src.tags,
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": normalize_date(entry.get("date", "")),
                    "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
    # tri décroissant par date si dispo
    def sort_key(it):
        try:
            return it.get("published") or ""
        except Exception:
            return ""

    items.sort(key=sort_key, reverse=True)
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collecte veille réglementaire (PRIIPS, MiFID, DDA, ESG...).")
    parser.add_argument("--out", type=Path, default=Path("data/veille_reglementaire.json"), help="Fichier de sortie JSON")
    parser.add_argument("--insecure", action="store_true", help="Désactive la vérification SSL (réseaux avec proxy/certificat auto-signé)")
    parser.add_argument("--proxy", type=str, help="Proxy HTTP/HTTPS (ex: http://user:pass@host:port)")
    args = parser.parse_args(argv)

    items = collect(insecure=args.insecure, proxy=args.proxy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"{len(items)} éléments collectés -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
