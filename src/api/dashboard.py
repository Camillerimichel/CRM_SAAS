import logging

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from sqlalchemy import text
from datetime import datetime, date as _date, timedelta
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from urllib.parse import urlencode


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


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


logger = logging.getLogger("uvicorn.error")


def rows_to_dicts(rows):
    return [dict(row._mapping) for row in rows]


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


# ---------------- Paramètres (référentiels) ----------------
@router.get("/parametres", response_class=HTMLResponse)
def dashboard_parametres(request: Request, db: Session = Depends(get_db)):
    """Affiche la page Paramètres. Données minimales pour l'accès à la page.

    Les actions de création/mise à jour/suppression référencées par le template
    (/dashboard/parametres/...) ne sont pas encore implémentées ici. Cette route
    permet au moins d'accéder à la page, en attendant les endpoints POST.
    """
    open_section = request.query_params.get("open") or None

    # Valeurs par défaut pour garantir le rendu même si certaines tables manquent
    contrats_generiques: list[dict] = []
    societes: list[dict] = []
    contrat_categories: list[dict] = []
    societe_categories: list[dict] = []
    contrat_supports: list[dict] = []
    supports: list[dict] = []

    # Chargement des données si les tables existent
    from sqlalchemy import text as _text
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

    return templates.TemplateResponse(
        "dashboard_parametres.html",
        {
            "request": request,
            "open_section": open_section,
            "contrats_generiques": contrats_generiques,
            "societes": societes,
            "contrat_categories": contrat_categories,
            "societe_categories": societe_categories,
            "contrat_supports": contrat_supports,
            "supports": supports,
        },
    )


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
        s = str(value).strip().replace(",", ".")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _redirect_back(request: Request, fallback_open: str) -> RedirectResponse:
    target_open = request.query_params.get("open") or fallback_open
    return RedirectResponse(url=f"/dashboard/parametres?open={target_open}", status_code=303)


# ---- Contrats génériques ----
@router.post("/parametres/contrats_generiques", response_class=HTMLResponse)
async def create_contrat_generique(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "nom_contrat": (form.get("nom_contrat") or "").strip(),
        "id_societe": int(form.get("id_societe") or 0) or None,
        "id_ctg": int(form.get("id_ctg") or 0) or None,
        "fga": _as_float(form.get("frais_gestion_assureur")),
        "fgc": _as_float(form.get("frais_gestion_courtier")),
    }
    from sqlalchemy import text as _text
    try:
        db.execute(
            _text(
                """
                INSERT INTO mariadb_affaires_generique
                    (nom_contrat, id_societe, id_ctg, frais_gestion_assureur, frais_gestion_courtier, actif)
                VALUES (:nom_contrat, :id_societe, :id_ctg, :fga, :fgc, 1)
                """
            ),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


@router.post("/parametres/contrats_generiques/{contrat_id}", response_class=HTMLResponse)
async def update_contrat_generique(contrat_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": contrat_id,
        "nom_contrat": (form.get("nom_contrat") or "").strip(),
        "id_societe": int(form.get("id_societe") or 0) or None,
        "id_ctg": int(form.get("id_ctg") or 0) or None,
        "fga": _as_float(form.get("frais_gestion_assureur")),
        "fgc": _as_float(form.get("frais_gestion_courtier")),
    }
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
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


@router.post("/parametres/contrats_generiques/{contrat_id}/delete", response_class=HTMLResponse)
async def delete_contrat_generique(contrat_id: int, request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text as _text
    try:
        db.execute(_text("DELETE FROM mariadb_affaires_generique WHERE id = :id"), {"id": contrat_id})
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "contrats")


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
        "id_ctg": int(form.get("id_ctg") or 0) or None,
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
                INSERT INTO mariadb_societe (nom, id_ctg, contact, telephone, email, commentaire)
                VALUES (:nom, :id_ctg, :contact, :telephone, :email, :commentaire)
                """
            ),
            params,
        )
        db.commit()
    except Exception:
        db.rollback()
    return _redirect_back(request, "societes")


@router.post("/parametres/societes/{soc_id}", response_class=HTMLResponse)
async def update_societe(soc_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    params = {
        "id": soc_id,
        "nom": (form.get("nom") or "").strip(),
        "id_ctg": int(form.get("id_ctg") or 0) or None,
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
    id_affaire_generique = int(form.get("id_affaire_generique") or 0) or None
    id_support = int(form.get("id_support") or 0) or None
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
    total_valo = (
        db.query(func.coalesce(func.sum(HistoriquePersonne.valo), 0))
        .join(sub_cli, sub_cli.c.client_id == HistoriquePersonne.id)
        .filter(HistoriquePersonne.date == sub_cli.c.last_date)
        .scalar()
    ) or 0

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

    # Informations Documents (comme sur la page Documents)
    try:
        total_documents = db.query(func.count(DocumentClient.id)).scalar() or 0
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
    except Exception:
        total_documents = 0
        obs_by_niveau = []
        obs_by_risque = []

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
                HistoriqueAffaire.volat
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
        for srri_contract, vol in rows:
            calc = _srri_from_vol(vol)
            if srri_contract is None or calc is None:
                continue
            try:
                c = int(srri_contract)
                k = int(calc)
            except Exception:
                continue
            # Règle cohérente avec les icônes: Au-dessus = c > k, En-dessous = c < k
            if c > k:
                compare_counts["above"] += 1
            elif c == k:
                compare_counts["equal"] += 1
            else:
                compare_counts["below"] += 1
    except Exception:
        compare_counts = {"above": 0, "equal": 0, "below": 0}

    # Comparatif clients: client vs risque (SRRI client vs SRRI historique courant)
    try:
        # Reuse last-date per client subquery (sub_cli)
        rows_cli = (
            db.query(
                Client.SRRI.label("client_srri"),
                HistoriquePersonne.SRRI.label("hist_srri")
            )
            .join(sub_cli, sub_cli.c.client_id == HistoriquePersonne.id)
            .join(Client, Client.id == sub_cli.c.client_id)
            .filter(HistoriquePersonne.date == sub_cli.c.last_date)
            .all()
        )
        cli_counts = {"above": 0, "equal": 0, "below": 0}
        for r in rows_cli:
            cs = getattr(r, "client_srri", None)
            hs = getattr(r, "hist_srri", None)
            if cs is None or hs is None:
                continue
            try:
                c = int(cs)
                h = int(hs)
            except Exception:
                continue
            if c > h:
                cli_counts["above"] += 1
            elif c == h:
                cli_counts["equal"] += 1
            else:
                cli_counts["below"] += 1
    except Exception:
        cli_counts = {"above": 0, "equal": 0, "below": 0}

    # ------- Tâches / événements (vue_suivi_evenement) -------
    try:
        from sqlalchemy import text as _text
        # Période sélectionnée pour la section Tâches
        try:
            range_days = int(request.query_params.get("tasks_range", 14))
            if range_days not in (7, 14, 30):
                range_days = 14
        except Exception:
            range_days = 14
        # Compte total et par statut/catégorie
        tasks_total = db.execute(_text("SELECT COUNT(1) FROM vue_suivi_evenement")).scalar() or 0
        rows_statut = db.execute(_text("SELECT COALESCE(TRIM(LOWER(statut)), '(non défini)') as s, COUNT(1) FROM vue_suivi_evenement GROUP BY s ORDER BY COUNT(1) DESC")).fetchall()
        rows_cat = db.execute(_text("SELECT COALESCE(TRIM(LOWER(categorie)), '(non défini)') as c, COUNT(1) FROM vue_suivi_evenement GROUP BY c ORDER BY COUNT(1) DESC")).fetchall()
        # Ouvertes: non terminé / non annulé
        open_count = db.execute(_text("SELECT COUNT(1) FROM vue_suivi_evenement WHERE statut IS NULL OR LOWER(statut) NOT IN ('termine','terminé','cloture','clôturé','annule','annulé')")).scalar() or 0
        # N derniers jours: créations par jour
        rows_days = db.execute(_text(
            """
            WITH RECURSIVE seq(x) AS (
              SELECT 0
              UNION ALL SELECT x+1 FROM seq WHERE x < :n
            )
            SELECT date(julianday('now') - x) AS day,
                   COALESCE((SELECT COUNT(1) FROM vue_suivi_evenement v WHERE date(v.date_evenement) = date(julianday('now') - x)), 0) AS nb
            FROM seq
            ORDER BY day ASC
            """
        ), {"n": range_days - 1}).fetchall()
        tasks_statut = [ {"statut": r[0], "nb": int(r[1] or 0)} for r in rows_statut ]
        tasks_categorie = [ {"categorie": r[0], "nb": int(r[1] or 0)} for r in rows_cat ]
        tasks_days = [ {"day": r[0], "nb": int(r[1] or 0)} for r in rows_days ]

        # Durée moyenne passée dans chaque statut (en jours), basée sur historique
        rows_avg = db.execute(_text(
            """
            WITH es AS (
                SELECT es.evenement_id, es.statut_id, es.date_statut, se.libelle AS statut
                FROM mariadb_evenement_statut es
                JOIN mariadb_statut_evenement se ON se.id = es.statut_id
            ), nxt AS (
                SELECT e1.evenement_id,
                       e1.statut,
                       e1.date_statut AS start_dt,
                       (
                         SELECT MIN(e2.date_statut)
                         FROM es e2
                         WHERE e2.evenement_id = e1.evenement_id AND e2.date_statut > e1.date_statut
                       ) AS end_dt
                FROM es e1
            )
            SELECT statut,
                   AVG((julianday(end_dt) - julianday(start_dt))) AS avg_days
            FROM nxt
            WHERE end_dt IS NOT NULL
            GROUP BY statut
            ORDER BY avg_days DESC NULLS LAST
            """
        )).fetchall()
        tasks_avg_by_statut = [ {"statut": r[0], "avg_days": float(r[1] or 0)} for r in rows_avg ]

        # Durée moyenne de création -> clôture (terminé/annulé)
        row_close = db.execute(_text(
            """
            WITH close AS (
              SELECT es.evenement_id, MIN(es.date_statut) AS close_dt
              FROM mariadb_evenement_statut es
              JOIN mariadb_statut_evenement se ON se.id = es.statut_id
              WHERE LOWER(se.libelle) IN ('termine','terminé','annule','annulé')
              GROUP BY es.evenement_id
            )
            SELECT AVG(julianday(close.close_dt) - julianday(e.date_evenement))
            FROM close
            JOIN mariadb_evenement e ON e.id = close.evenement_id
            """
        )).scalar()
        tasks_avg_close_days = float(row_close or 0)

        # Distribution des durées (création -> clôture) sur la période sélectionnée (par date de clôture)
        rows_dist = db.execute(_text(
            """
            WITH close AS (
              SELECT es.evenement_id, MIN(es.date_statut) AS close_dt
              FROM mariadb_evenement_statut es
              JOIN mariadb_statut_evenement se ON se.id = es.statut_id
              WHERE LOWER(se.libelle) IN ('termine','terminé','annule','annulé')
              GROUP BY es.evenement_id
            ), durations AS (
              SELECT (julianday(c.close_dt) - julianday(e.date_evenement)) AS d, c.close_dt AS cd
              FROM close c JOIN mariadb_evenement e ON e.id = c.evenement_id
              WHERE date(c.close_dt) >= date('now', '-' || :rng || ' days')
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
        ), {"rng": range_days}).fetchall()
        tasks_close_dist = [ {"bucket": r[0], "nb": int(r[1] or 0)} for r in rows_dist ]
    except Exception:
        tasks_total = 0
        open_count = 0
        tasks_statut = []
        tasks_categorie = []
        tasks_days = []
        tasks_avg_by_statut = []
        tasks_avg_close_days = 0.0
        tasks_close_dist = []

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

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
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
            "docs_obs_by_niveau": obs_by_niveau,
            "docs_obs_by_risque": obs_by_risque,
            # Infos contrats vs risque
            "aff_risk_counts": compare_counts,
            # Infos clients vs risque
            "cli_risk_counts": cli_counts,
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
        }
    )


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
    actif_success: str | None = None
    actif_error: str | None = None
    objectifs_success: str | None = None
    objectifs_error: str | None = None
    active_objectif_id: int | None = None
    active_section: str = "etat_civil"
    esg_success: str | None = None
    esg_error: str | None = None

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
                except Exception as exc:
                    db.rollback()
                    actif_error = "Impossible d'enregistrer l'actif."
                    logger.debug("Dashboard KYC client: erreur actif save: %s", exc, exc_info=True)
            active_section = "patrimoine"

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
                    actif_success = "Actif supprimé."
                except Exception as exc:
                    db.rollback()
                    actif_error = "Impossible de supprimer l'actif."
                    logger.debug("Dashboard KYC client: erreur actif delete: %s", exc, exc_info=True)
            active_section = "patrimoine"

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

        elif action == "objectifs_save":
            active_section = "objectifs"
            objectif_id_raw = form.get("objectif_id")
            link_id_raw = form.get("link_id")
            horizon = (form.get("horizon_investissement") or "").strip() or None
            niveau_id_raw = form.get("niveau_id")
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
                params = {
                    "cid": client_id,
                    "objectif_id": objectif_id,
                    "horizon": horizon,
                    "niveau_id": niveau_id,
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
                                        commentaire,
                                        date_expiration
                                    ) VALUES (
                                        :cid,
                                        :objectif_id,
                                        :horizon,
                                        :niveau_id,
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

        elif action == "objectifs_delete":
            active_section = "objectifs"
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

            excl_ids = []
            ind_ids = []
            try:
                if hasattr(form, "getlist"):
                    excl_ids = [int(x) for x in form.getlist("exclusions") if str(x).isdigit()]
                    ind_ids = [int(x) for x in form.getlist("indicators") if str(x).isdigit()]
            except Exception:
                excl_ids = []
                ind_ids = []

            if not all([env_importance, env_ges_reduc, soc_droits_humains, soc_parite, gov_transparence, gov_controle_ethique]):
                esg_error = "Veuillez renseigner toutes les réponses ESG."
            else:
                from datetime import datetime, timedelta
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                # Obsolescence à 2 ans
                obso = (datetime.utcnow() + timedelta(days=730)).strftime("%Y-%m-%d %H:%M:%S")
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
                        qid = db.execute(_text("SELECT last_insert_rowid()")).fetchone()[0]
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
                from datetime import datetime, timedelta
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                obso = (datetime.utcnow() + timedelta(days=730)).strftime("%Y-%m-%d %H:%M:%S")
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
                rqid = db.execute(_text("SELECT last_insert_rowid()" )).fetchone()[0]
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
                db.commit()
                # set to show panel
                active_section = "knowledge"
            except Exception as exc:
                db.rollback()
                logger.debug("Dashboard KYC client: erreur risque_save: %s", exc, exc_info=True)

        elif action == "lcbft_save":
            from sqlalchemy import text as _text
            active_section = "lcbft"
            lcbft_success = None
            lcbft_error = None
            try:
                form = await request.form()
                def _i(name):
                    v = form.get(name)
                    if v is None or v == "":
                        return None
                    try:
                        return int(v)
                    except Exception:
                        return None
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
                db.execute(
                    _text(
                        """
                        INSERT INTO LCBFT_questionnaire (
                          client_ref, created_at, updated_at,
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
                          :client_ref, :created_at, :updated_at,
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
                # Vigilance options
                try:
                    qid = db.execute(_text("SELECT last_insert_rowid()")).fetchone()[0]
                    # Persist FATCA fields linked to this questionnaire
                    try:
                        # Read form values
                        fatca_contrat_id = _i("fatca_contrat_id")
                        fatca_pays_residence = (form.get("fatca_pays_residence") or None)
                        fatca_nif = (form.get("fatca_nif") or None)
                        fatca_date_operation = (form.get("fatca_date_operation") or None)
                        # Resolve societe_nom from DB if possible
                        societe_nom = (form.get("fatca_societe") or None)
                        if fatca_contrat_id:
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
                                            ORDER BY date_saisie DESC NULLS LAST, id DESC
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
                        for oid in vids:
                            db.execute(
                                _text("INSERT OR IGNORE INTO LCBFT_questionnaire_vigilance (questionnaire_id, option_id) VALUES (:q,:o)"),
                                {"q": qid, "o": oid},
                            )
                except Exception:
                    pass
                db.commit()
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
                ORDER BY a.date_saisie DESC NULLS LAST, a.id DESC
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
                ORDER BY p.date_saisie DESC NULLS LAST, p.id DESC
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
                ORDER BY r.date_saisie DESC NULLS LAST, r.id DESC
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
                ORDER BY c.date_saisie DESC NULLS LAST, c.id DESC
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
                ORDER BY a.date_saisie DESC NULLS LAST, a.id DESC
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
                ORDER BY m.date_saisie DESC NULLS LAST, m.id DESC
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
                ORDER BY p.date_saisie DESC NULLS LAST, p.id DESC
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
    from datetime import date as _date, timedelta as _timedelta
    if esg_current and esg_current.get("saisie_at") and esg_current.get("obsolescence_at"):
        esg_display_saisie = esg_current.get("saisie_at")
        esg_display_obsolescence = esg_current.get("obsolescence_at")
    else:
        today = _date.today().isoformat()
        in2y = (_date.today() + _timedelta(days=730)).isoformat()
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
        in2y = (_date.today() + _timedelta(days=730)).isoformat()
        risque_display_saisie = today
        risque_display_obsolescence = in2y

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
    try:
        rows = db.execute(text("SELECT id, code, label FROM LCBFT_vigilance_option ORDER BY label")).fetchall()
        lcbft_vigilance_options = [dict(r._mapping) for r in rows]
    except Exception:
        lcbft_vigilance_options = []
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
    except Exception:
        lcbft_current = None

    # FATCA: contrats disponibles et infos client (pays fiscal, NIF)
    fatca_contracts: list[dict] = []
    fatca_client_country: str | None = None
    fatca_client_nif: str | None = None
    fatca_today = _date.today().isoformat()
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

    return templates.TemplateResponse(
        "dashboard_client_kyc.html",
        {
            "request": request,
            "client": client,
            "client_id": client_id,
            "kyc_actifs": actifs,
            "kyc_actifs_total": actifs_total_str,
            "kyc_passifs": passifs,
            "kyc_passifs_total": passifs_total_str,
            "kyc_revenus": revenus,
            "kyc_revenus_total": revenus_total_str,
            "kyc_charges": charges,
            "kyc_charges_total": charges_total_str,
            "kyc_adresses": adresses,
            "kyc_situations_matrimoniales": situations_matrimoniales,
            "kyc_situations_professionnelles": situations_professionnelles,
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
            # RISK (connaissance financière)
            "risque_opts": risque_opts,
            "risque_current": risque_current,
            "risque_connaissance_map": risque_connaissance_map,
            "risque_objectifs_ids": risque_objectifs_ids,
            "risque_decision": risque_decision,
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
        },
    )


# ---------------- Clients ----------------
@router.get("/clients", response_class=HTMLResponse)
def dashboard_clients(request: Request, db: Session = Depends(get_db)):
    total_clients = db.query(func.count(Client.id)).scalar() or 0

    # Données SRRI pour le graphique
    srri_data = (
        db.query(Client.SRRI, func.count(Client.id).label("nb"))
        .group_by(Client.SRRI)
        .all()
    )
    srri_chart = [{"srri": s.SRRI, "nb": s.nb} for s in srri_data]

    # Utilise le service pour la liste des clients enrichie et calcule l'icône de risque (comme Affaires)
    rows = get_clients(db)

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

    clients = []
    for r in rows:
        clients.append({
            "id": getattr(r, "id", None),
            "nom": getattr(r, "nom", None),
            "prenom": getattr(r, "prenom", None),
            "SRRI": getattr(r, "SRRI", None),
            "srri_hist": getattr(r, "srri_hist", None),
            "srri_icon": icon_for_compare(getattr(r, "SRRI", None), getattr(r, "srri_hist", None)),
            "total_valo": getattr(r, "total_valo", None),
            "perf_52_sem": getattr(r, "perf_52_sem", None),
            "volatilite": getattr(r, "volatilite", None),
        })

    # (Graphiques SRRI supprimés sur la page Clients — calcul montants par SRRI non nécessaire ici)


    return templates.TemplateResponse(
        "dashboard_clients.html",
        {
            "request": request,
            "total_clients": total_clients,
            "srri_chart": srri_chart,
            "clients": clients,
        }
    )


# ---------------- Affaires ----------------
@router.get("/affaires", response_class=HTMLResponse)
def dashboard_affaires(request: Request, db: Session = Depends(get_db)):
    total_affaires = db.query(func.count(Affaire.id)).scalar() or 0
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
    affaires_rows = (
        db.query(
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
        )
        .join(subq, subq.c.affaire_id == Affaire.id)
        .join(
            HistoriqueAffaire,
            (HistoriqueAffaire.id == subq.c.affaire_id) &
            (HistoriqueAffaire.date == subq.c.last_date)
        )
        .outerjoin(Client, Client.id == Affaire.id_personne)
        .all()
    )

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
            "srri_chart": srri_chart,
            "affaires": affaires,
            "srri_compare_counts": compare_counts,
        }
    )


# ---------------- Détail Affaire ----------------
@router.get("/affaires/{affaire_id}", response_class=HTMLResponse)
def dashboard_affaire_detail(affaire_id: int, request: Request, db: Session = Depends(get_db)):
    affaire = db.query(Affaire).filter(Affaire.id == affaire_id).first()
    if not affaire:
        return templates.TemplateResponse("dashboard_affaire_detail.html", {"request": request, "error": "Affaire introuvable"})

    # Informations client liées à l'affaire (pour en-tête et liens)
    client_nom = None
    client_prenom = None
    client_id = None
    try:
        if getattr(affaire, 'id_personne', None) is not None:
            cli = db.query(Client).filter(Client.id == affaire.id_personne).first()
            if cli:
                client_id = cli.id
                client_nom = getattr(cli, 'nom', None)
                client_prenom = getattr(cli, 'prenom', None)
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
            SELECT s.code_isin AS code_isin,
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
            supports.append({
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
    for label_ui,key in [("à faire","a faire"),("en attente","en attente"),("terminé","termine"),("annulé","annule")]:
        sid = stat_ids.get(key)
        if sid: status_ui.append({"label":label_ui, "id":sid, "key":key})
    en_cours_id = stat_ids.get("en cours")
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
    q = (
        db.query(
            Evenement.id,
            Evenement.date_evenement,
            Evenement.statut,
            Evenement.commentaire,
            Evenement.type_id,
            TypeEvenement.libelle.label("type_libelle"),
            TypeEvenement.categorie.label("type_categorie"),
        )
        .join(TypeEvenement, TypeEvenement.id == Evenement.type_id)
        .filter(Evenement.affaire_id == affaire_id)
        .filter(
            or_(
                Evenement.statut.is_(None),
                func.lower(Evenement.statut).notin_(OPEN_STATES),
            )
        )
        .order_by(Evenement.date_evenement.desc())
    )
    aff_events_open = q.all()
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
        try:
            dstr = r.date_evenement.strftime("%Y-%m-%d %H:%M") if getattr(r, 'date_evenement', None) else None
        except Exception:
            dstr = str(getattr(r, 'date_evenement', None))[:16] if getattr(r, 'date_evenement', None) else None
        affaire_events_open.append({
            "id": getattr(r, 'id', None),
            "date_evenement": dstr,
            "statut": getattr(r, 'statut', None),
            "commentaire": getattr(r, 'commentaire', None),
            "type_id": getattr(r, 'type_id', None),
            "type_libelle": getattr(r, 'type_libelle', None),
            "type_categorie": getattr(r, 'type_categorie', None),
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

    return templates.TemplateResponse(
        "dashboard_affaire_detail.html",
        {
            "request": request,
            "affaire": affaire,
            "client_id": client_id,
            "client_nom": client_nom,
            "client_prenom": client_prenom,
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
        }
    )


# ---------------- Supports ----------------
from sqlalchemy import text

@router.get("/supports", response_class=HTMLResponse)
def dashboard_supports(request: Request, db: Session = Depends(get_db)):
    # Récupérer la dernière date disponible
    last_date = db.execute(
        text("SELECT MAX(date) FROM mariadb_historique_support_w")
    ).scalar()

    print(">>> Dernière date trouvée :", last_date, type(last_date))
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
            SELECT s.code_isin,
                   s.nom,
                   s.cat_gene AS categorie,
                   s.cat_geo AS zone_geo,
                   s.SRRI,
       CAST(SUM(h.valo) AS INTEGER) AS total_valo,
                   h.date
            FROM mariadb_historique_support_w h
            JOIN mariadb_support s ON s.id = h.id_support
            WHERE h.date = :last_date
            GROUP BY s.code_isin, s.nom, s.cat_gene, s.cat_geo, s.SRRI, h.date
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
    from datetime import date as _date
    today_str = _date.today().isoformat()
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
    sql = f"SELECT * FROM vue_suivi_evenement{where} ORDER BY date_evenement DESC LIMIT 300"
    items = db.execute(text(sql), params).fetchall()

    # Enrichir avec noms client et référence affaire pour l'affichage
    try:
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
        # Convertir en dicts avec champs dérivés
        items = [
            {
                **dict(getattr(r, '_mapping', r)),
                'nom_client': clients_map_full.get(getattr(r, 'client_id', None)),
                'affaire_ref': affaires_map_ref.get(getattr(r, 'affaire_id', None)),
            }
            for r in items
        ]
    except Exception:
        # En cas d'échec, garder items bruts
        items = items

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
    for label_ui, key in [("à faire", "a faire"), ("en attente", "en attente"), ("terminé", "termine"), ("annulé", "annule")]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")

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
            TypeEvenement.libelle.label("type_libelle"),
            TypeEvenement.categorie.label("type_categorie"),
        )
        .join(TypeEvenement, TypeEvenement.id == Evenement.type_id)
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
    for label_ui, key in [("à faire","a faire"),("en attente","en attente"),("terminé","termine"),("annulé","annule")]:
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
    commentaires_l = form.getlist("commentaire")

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
    for type_lbl, cat, cli_full, aff_ref, resp, comm in zip_longest(
        types_l, cats_l, clients_l, affaires_l, responsables_l, commentaires_l, fillvalue=""
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
        payload = TacheCreateSchema(
            type_libelle=(type_lbl or "tâche").strip(),
            categorie=(cat or "tache").strip() or "tache",
            client_id=getattr(cli, "id", None),
            affaire_id=getattr(aff, "id", None),
            commentaire=comm or None,
            utilisateur_responsable=(resp or None),
        )
        ev = create_tache(db, payload)

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
    user = form.get("utilisateur_responsable")
    redirect_to = form.get("redirect") or "/dashboard/taches"
    # Mise à jour du commentaire de la tâche: préfixer avec date-heure
    if commentaire and commentaire.strip():
        try:
            from src.models.evenement import Evenement as _Ev
            ev = db.query(_Ev).filter(_Ev.id == evenement_id).first()
            if ev:
                from datetime import datetime as _dt
                ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M")
                # Date-heure sur une ligne, texte sur la ligne suivante
                new_line = f"[{ts}]\n{commentaire.strip()}"
                if getattr(ev, "commentaire", None):
                    ev.commentaire = f"{ev.commentaire}\n{new_line}"
                else:
                    ev.commentaire = new_line
                db.add(ev)
                db.commit()
        except Exception:
            pass
    try:
        sid = int(statut_id) if statut_id else None
    except Exception:
        sid = None
    if sid is not None:
        payload = EvenementStatutCreateSchema(statut_id=sid, commentaire=commentaire, utilisateur_responsable=user)
        add_statut_to_evenement(db, evenement_id, payload)
    return RedirectResponse(url=redirect_to, status_code=303)


# ---------------- Création Tâche depuis Détail Client ----------------
@router.post("/clients/{client_id}/taches", response_class=HTMLResponse)
async def dashboard_client_create_tache(client_id: int, request: Request, db: Session = Depends(get_db)):
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

    payload = TacheCreateSchema(
        type_libelle=(form.get("type_libelle") or "").strip(),
        categorie=(form.get("categorie") or "tache").strip() or "tache",
        client_id=getattr(cli, "id", None),
        affaire_id=getattr(aff, "id", None),
        commentaire=form.get("commentaire") or None,
        utilisateur_responsable=form.get("utilisateur_responsable") or None,
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
    ev = create_tache(db, payload)

    # Statut initial
    sid = None
    try:
        sid = int(form.get("statut_id")) if form.get("statut_id") else None
    except Exception:
        sid = None
    if sid:
        add_statut_to_evenement(db, ev.id, EvenementStatutCreateSchema(statut_id=sid, commentaire="Création via client", utilisateur_responsable=payload.utilisateur_responsable))

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

    payload = TacheCreateSchema(
        type_libelle=(form.get("type_libelle") or "").strip(),
        categorie=(form.get("categorie") or "tache").strip() or "tache",
        client_id=getattr(cli, "id", None),
        affaire_id=getattr(aff, "id", None),
        commentaire=form.get("commentaire") or None,
        utilisateur_responsable=form.get("utilisateur_responsable") or None,
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
    ev = create_tache(db, payload)

    # Statut initial
    sid = None
    try:
        sid = int(form.get("statut_id")) if form.get("statut_id") else None
    except Exception:
        sid = None
    if sid:
        add_statut_to_evenement(db, ev.id, EvenementStatutCreateSchema(statut_id=sid, commentaire="Création via affaire", utilisateur_responsable=payload.utilisateur_responsable))

    # Communication éventuelle
    if form.get("comm_toggle") == "1":
        canal = form.get("comm_canal") or "email"
        dest = form.get("comm_destinataire") or ""
        obj = form.get("comm_objet") or None
        cont = form.get("comm_contenu") or None
        if dest:
            create_envoi(db, ev.id, EvenementEnvoiCreateSchema(canal=canal, destinataire=dest, objet=obj, contenu=cont))

    return RedirectResponse(url=f"/dashboard/affaires/{affaire_id}", status_code=303)


# ---------------- Détail Client ----------------
@router.get("/clients/{client_id}", response_class=HTMLResponse)
def dashboard_client_detail(client_id: int, request: Request, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()

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
                    SELECT s.code_isin AS code_isin,
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
                ),
                {"aid": aid, "d": ref_date}
            ).fetchall()
            for r in rows:
                key = r.code_isin or r.nom
                it = client_supports_map.get(key)
                nb = float(r.nbuc or 0)
                valo = float(r.valo or 0)
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
                        "sum_nbuc": 0.0,
                        "sum_valo": 0.0,
                        "prmp_num": 0.0,
                        "prmp_den": 0.0,
                        "noteE": getattr(r, "noteE", None),
                        "noteS": getattr(r, "noteS", None),
                        "noteG": getattr(r, "noteG", None),
                    }
                    client_supports_map[key] = it
                it["sum_nbuc"] += nb
                it["sum_valo"] += valo
                if prmp is not None and nb is not None:
                    it["prmp_num"] += prmp * nb
                    it["prmp_den"] += nb
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
                "valo": it.get("sum_valo", 0.0),
                "valo_str": _fmt_valo(it.get("sum_valo", 0.0)),
                "noteE": it.get("noteE"),
                "noteS": it.get("noteS"),
                "noteG": it.get("noteG"),
            })
        # Trier par valorisation décroissante
        client_supports.sort(key=lambda x: x.get("valo", 0.0), reverse=True)
    except Exception:
        client_supports = []

    # Documents liés au client, avec nom du document de base
    rows = (
        db.query(
            DocumentClient,
            Document.documents.label("base_name")
        )
        .outerjoin(Document, Document.id_document_base == DocumentClient.id_document_base)
        .filter(DocumentClient.id_client == client_id)
        .order_by(DocumentClient.date_creation.desc().nullslast())
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
    for label_ui, key in [("à faire","a faire"),("en attente","en attente"),("terminé","termine"),("annulé","annule")]:
        sid = stat_ids.get(key)
        if sid:
            status_ui.append({"label": label_ui, "id": sid, "key": key})
    en_cours_id = stat_ids.get("en cours")

    clients_suggest = db.query(Client.id, Client.nom, Client.prenom).order_by(Client.nom.asc(), Client.prenom.asc()).all()
    aff_rows = db.query(Affaire.id, Affaire.ref, Affaire.id_personne).order_by(Affaire.ref.asc()).all()
    _clients_map = {c.id: f"{getattr(c,'nom','') or ''} {getattr(c,'prenom','') or ''}".strip() for c in clients_suggest}
    affaires_suggest = [{"id": a.id, "ref": getattr(a,'ref',''), "client": _clients_map.get(getattr(a,'id_personne',None), '')} for a in aff_rows]
    client_fullname_default = (f"{getattr(client,'nom','') or ''} {getattr(client,'prenom','') or ''}".strip()) if client else None

    # -------- Messages (tâches) par client: comptages + liste ouverte (pour pop-up) --------
    from src.models.evenement import Evenement
    from src.models.type_evenement import TypeEvenement
    # Statuts ouverts: différent de terminé/annulé/clos
    OPEN_STATES = ("termine", "terminé", "cloture", "clôturé", "cloturé", "clôture", "annule", "annulé")
    q = (
        db.query(
            Evenement.id,
            Evenement.date_evenement,
            Evenement.statut,
            Evenement.commentaire,
            Evenement.type_id,
            Evenement.affaire_id,
            TypeEvenement.libelle.label("type_libelle"),
            TypeEvenement.categorie.label("type_categorie"),
        )
        .join(TypeEvenement, TypeEvenement.id == Evenement.type_id)
        .filter(Evenement.client_id == client_id)
        .filter(
            or_(
                Evenement.statut.is_(None),
                func.lower(Evenement.statut).notin_(OPEN_STATES),
            )
        )
        .order_by(Evenement.date_evenement.desc())
    )
    events_open = q.all()
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
        # Safe date formatting
        try:
            dstr = r.date_evenement.strftime("%Y-%m-%d %H:%M") if getattr(r, 'date_evenement', None) else None
        except Exception:
            dstr = str(getattr(r, 'date_evenement', None))[:16] if getattr(r, 'date_evenement', None) else None
        client_events_open.append({
            "id": getattr(r, 'id', None),
            "date_evenement": dstr,
            "statut": getattr(r, 'statut', None),
            "commentaire": getattr(r, 'commentaire', None),
            "type_id": getattr(r, 'type_id', None),
            "type_libelle": getattr(r, 'type_libelle', None),
            "type_categorie": getattr(r, 'type_categorie', None),
            "affaire_id": getattr(r, 'affaire_id', None),
            "affaire_ref": _aff_ref.get(getattr(r, 'affaire_id', None)),
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

    return templates.TemplateResponse(
        "dashboard_client_detail.html",
        {
            "request": request,
            "client": client,
            "historique": historique,
            "last_row": last_row,
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
            # Tâches: assistance création locale
            "types": types,
            "categories": cats,
            "statuts": statuts,
            "status_ui": status_ui,
            "en_cours_id": en_cours_id,
            "clients_suggest": clients_suggest,
            "affaires_suggest": affaires_suggest,
            "client_fullname_default": client_fullname_default,
            # Messages/alertes en-tête
            "msgs_reg_count": msgs_reg_count,
            "msgs_nonreg_count": msgs_nonreg_count,
            "client_events_open": client_events_open,
            # KYC Actifs
            "kyc_actifs": kyc_actifs,
            "kyc_types_actifs": rows_types_actifs,
        }
    )


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
