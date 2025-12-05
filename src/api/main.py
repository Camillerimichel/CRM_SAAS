# ---------------- Imports principaux ----------------
from fastapi import FastAPI, Depends, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates   # <-- ajoute ceci
from fastapi.staticfiles import StaticFiles
import threading
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from src.database import get_db
from src.security.rbac import load_access, require_permission, extract_user_context, pick_scope
from fastapi import Query
from datetime import date, datetime
from pathlib import Path
import os
import time
from collections import deque, defaultdict
from src.security.auth import (
    hash_password,
    verify_password,
    encode_token,
    decode_token,
    encode_reset_token,
    decode_reset_token,
)
from src.services.mailer import send_email


# ---------------- Définition app FastAPI ----------------
app = FastAPI()
templates = Jinja2Templates(directory="src/api/templates")
from src.api import dashboard
from src.api import events
from src.api import groupes
app.include_router(dashboard.router)
app.include_router(events.router)
app.include_router(groupes.router)

# Servir l'app React compilée (http://localhost:8000/app)


# ---------------- Route d'accueil ----------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Redirection par défaut vers la page de connexion."""
    return RedirectResponse(url="/login", status_code=303)

_THIS_FILE = Path(__file__).resolve()
BASE_DIR = _THIS_FILE.parents[2]
# Lorsque le code est exécuté depuis /var/www/CRM_SAAS/src/api/main.py,
# BASE_DIR pointe sur /var/www/CRM_SAAS/src. On corrige pour remonter
# à la racine si besoin afin de trouver /frontend.
if not (BASE_DIR / "frontend").exists() and len(_THIS_FILE.parents) >= 3:
    BASE_DIR = _THIS_FILE.parents[3]

FRONTEND_BUILD_PATH = BASE_DIR / "frontend" / "build"
FAVICON_PATH = BASE_DIR / "frontend" / "public" / "favicon.ico"

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH)
    raise HTTPException(status_code=404, detail="Favicon non trouvée")

if FRONTEND_BUILD_PATH.exists():
    frontend_app = FastAPI()

    frontend_app.mount(
        "/",
        StaticFiles(directory=FRONTEND_BUILD_PATH, html=True),
        name="dashboard",
    )

    static_dir = FRONTEND_BUILD_PATH / "static"
    if static_dir.exists():
        frontend_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def _serve_frontend_asset(filename: str):
        asset_path = FRONTEND_BUILD_PATH / filename
        if asset_path.exists():
            return FileResponse(asset_path)
        raise HTTPException(status_code=404, detail=f"Fichier {filename} introuvable")

    @frontend_app.get("/manifest.json", include_in_schema=False)
    @frontend_app.get("/asset-manifest.json", include_in_schema=False)
    @frontend_app.get("/robots.txt", include_in_schema=False)
    @frontend_app.get("/logo192.png", include_in_schema=False)
    @frontend_app.get("/logo512.png", include_in_schema=False)
    @frontend_app.get("/apple-touch-icon.png", include_in_schema=False)
    @frontend_app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
    def serve_frontend_assets(request: Request):
        filename = request.url.path.lstrip("/")
        return _serve_frontend_asset(filename)

    app.mount("/dashboard", frontend_app)

logger = logging.getLogger("uvicorn.error")

_forgot_rate_limit = defaultdict(deque)
_FORGOT_MAX = 5
_FORGOT_WINDOW_SEC = 600


def _check_forgot_rate(request: Request):
    """Limiter les requêtes /password/forgot par IP pour limiter le spam."""
    ip = request.client.host if request and request.client else "unknown"
    now = time.time()
    dq = _forgot_rate_limit[ip]
    # purge
    while dq and dq[0] < now - _FORGOT_WINDOW_SEC:
        dq.popleft()
    if len(dq) >= _FORGOT_MAX:
        return False
    dq.append(now)
    return True


def _require_feature(request: Request, db: Session, feature: str, action: str):
    """Charge l'utilisateur courant et vérifie la permission demandée."""
    user_type, user_id, req_scope = extract_user_context(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Non authentifié")
    access = load_access(db, user_type=user_type, user_id=user_id)
    scope = pick_scope(access, req_scope)
    require_permission(access, feature, action, societe_id=scope)
    return access, scope


def _log_client_login(db: Session, client_user_id: int, client_id: int | None, broker_id: int | None, request: Request):
    """Enregistre la connexion client + met à jour last_login (pas bloquant)."""
    ip = request.client.host if request and request.client else None
    ua = (request.headers.get("user-agent") or "")[:500]
    try:
        db.execute(
            text("UPDATE auth_client_users SET last_login = NOW() WHERE id = :uid"),
            {"uid": client_user_id},
        )
        db.execute(
            text(
                """
                INSERT INTO auth_client_login_logs (client_user_id, client_id, broker_id, ip_address, user_agent, success)
                VALUES (:uid, :cid, :bid, :ip, :ua, 1)
                """
            ),
            {"uid": client_user_id, "cid": client_id, "bid": broker_id, "ip": ip, "ua": ua},
        )
        db.commit()
    except Exception:
        db.rollback()


def _update_auth_user_last_login(db: Session, user_id: int):
    try:
        db.execute(text("UPDATE auth_users SET last_login = NOW() WHERE id = :uid"), {"uid": user_id})
        db.commit()
    except Exception:
        db.rollback()


# ---------------- Auth simple (login/logout) ----------------
@app.post("/login")
def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    broker_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    accept = request.headers.get("accept", "")
    broker_id_int: int | None = None
    try:
        broker_id_int = int(broker_id) if broker_id not in (None, "") else None
    except Exception:
        broker_id_int = None

    try:
        # 1) Comptes staff (auth_users.user_type = 'staff') prioritaire
        staff_row = db.execute(
            text(
                """
                SELECT id, user_type, password_hash, actif, client_id, rh_id
                FROM auth_users
                WHERE login = :login AND user_type = 'staff'
                LIMIT 1
                """
            ),
            {"login": email},
        ).fetchone()

        user_name = None
        societe_id = None
        role_codes: list[str] = []
        client_id: int | None = None
        broker_cookie: int | None = None

        if staff_row:
            m = staff_row._mapping if hasattr(staff_row, "_mapping") else None
            uid = m.get("id") if m else staff_row[0]
            utype = "staff"
            pwd_hash = m.get("password_hash") if m else staff_row[2]
            actif = m.get("actif") if m else staff_row[3]
            if not actif:
                raise HTTPException(status_code=403, detail="Compte désactivé")
            if not verify_password(password, pwd_hash):
                raise HTTPException(status_code=401, detail="Identifiants invalides")
            _update_auth_user_last_login(db, uid)
            try:
                rows_roles = db.execute(
                    text(
                        """
                        SELECT ar.code
                        FROM auth_user_roles aur
                        JOIN auth_roles ar ON ar.id = aur.role_id
                        WHERE aur.user_type = :ut AND aur.user_id = :uid
                        """
                    ),
                    {"ut": utype, "uid": uid},
                ).fetchall()
                role_codes = [str(r.code if hasattr(r, "_mapping") else r[0]) for r in rows_roles or []]
                row_scope = db.execute(
                    text(
                        """
                        SELECT societe_id
                        FROM auth_user_roles
                        WHERE user_type = :ut AND user_id = :uid
                        ORDER BY (societe_id IS NULL) DESC, societe_id ASC
                        LIMIT 1
                        """
                    ),
                    {"ut": utype, "uid": uid},
                ).fetchone()
                if row_scope:
                    societe_id = row_scope[0] if not hasattr(row_scope, "_mapping") else row_scope._mapping.get("societe_id")
            except Exception:
                societe_id = None
            try:
                if m and m.get("rh_id"):
                    rh_row = db.execute(
                        text("SELECT prenom, nom FROM administration_RH WHERE id = :rid LIMIT 1"),
                        {"rid": m.get("rh_id")},
                    ).fetchone()
                    if rh_row:
                        _m = rh_row._mapping if hasattr(rh_row, "_mapping") else None
                        prenom = _m.get("prenom") if _m else (rh_row[0] if len(rh_row) > 0 else "")
                        nom = _m.get("nom") if _m else (rh_row[1] if len(rh_row) > 1 else "")
                        user_name = f"{prenom or ''} {nom or ''}".strip()
            except Exception:
                user_name = None
        else:
            # 2) Comptes clients dans auth_client_users (courtier requis si doublon)
            params = {"login": email}
            query = "SELECT id, client_id, broker_id, password_hash, status FROM auth_client_users WHERE login = :login"
            if broker_id_int is not None:
                query += " AND broker_id = :bid"
                params["bid"] = broker_id_int
            client_rows = db.execute(text(query), params).fetchall()
            if not client_rows:
                # 3) Fallback legacy (anciens comptes clients dans auth_users)
                legacy_row = db.execute(
                    text(
                        """
                        SELECT id, user_type, password_hash, actif, client_id
                        FROM auth_users
                        WHERE login = :login AND user_type = 'client'
                        LIMIT 1
                        """
                    ),
                    {"login": email},
                ).fetchone()
                if not legacy_row:
                    raise HTTPException(status_code=401, detail="Identifiants invalides")
                m = legacy_row._mapping if hasattr(legacy_row, "_mapping") else None
                uid = m.get("id") if m else legacy_row[0]
                utype = "client"
                pwd_hash = m.get("password_hash") if m else legacy_row[2]
                actif = m.get("actif") if m else legacy_row[3]
                client_id = m.get("client_id") if m else (legacy_row[4] if len(legacy_row) > 4 else None)
                if not actif:
                    raise HTTPException(status_code=403, detail="Compte désactivé")
                if not verify_password(password, pwd_hash):
                    raise HTTPException(status_code=401, detail="Identifiants invalides")
                _update_auth_user_last_login(db, uid)
                # Rôles via auth_user_roles (legacy)
                rows_roles = db.execute(
                    text(
                        """
                        SELECT ar.code
                        FROM auth_user_roles aur
                        JOIN auth_roles ar ON ar.id = aur.role_id
                        WHERE aur.user_type = 'client' AND aur.user_id = :uid
                        """
                    ),
                    {"uid": uid},
                ).fetchall()
                role_codes = [str(r.code if hasattr(r, "_mapping") else r[0]) for r in rows_roles or []]
            else:
                if broker_id_int is None and len(client_rows) > 1:
                    raise HTTPException(status_code=400, detail="Plusieurs courtiers utilisent ce login : précisez votre courtier.")
                client_row = client_rows[0]
                m = client_row._mapping if hasattr(client_row, "_mapping") else None
                uid = m.get("id") if m else client_row[0]
                utype = "client"
                client_id = m.get("client_id") if m else (client_row[1] if len(client_row) > 1 else None)
                broker_cookie = m.get("broker_id") if m else (client_row[2] if len(client_row) > 2 else None)
                pwd_hash = m.get("password_hash") if m else (client_row[3] if len(client_row) > 3 else None)
                status = m.get("status") if m else (client_row[4] if len(client_row) > 4 else None)
                if status and str(status) != "active":
                    raise HTTPException(status_code=403, detail="Compte inactif ou en attente de réinitialisation")
                if not verify_password(password, pwd_hash):
                    raise HTTPException(status_code=401, detail="Identifiants invalides")
                _log_client_login(db, client_user_id=uid, client_id=client_id, broker_id=broker_cookie, request=request)
                # Rôles via auth_client_user_roles
                rows_roles = db.execute(
                    text(
                        """
                        SELECT ar.code, cur.societe_id
                        FROM auth_client_user_roles cur
                        JOIN auth_roles ar ON ar.id = cur.role_id
                        WHERE cur.client_user_id = :uid
                        """
                    ),
                    {"uid": uid},
                ).fetchall()
                role_codes = [str(r.code if hasattr(r, "_mapping") else r[0]) for r in rows_roles or []]
                # Priorité au scope explicite, sinon courtier
                try:
                    row_scope = next((r for r in rows_roles if (hasattr(r, "_mapping") and r._mapping.get("societe_id") is not None) or (not hasattr(r, "_mapping") and r[1] is not None)), None)
                    if row_scope:
                        societe_id = row_scope._mapping.get("societe_id") if hasattr(row_scope, "_mapping") else row_scope[1]
                    elif broker_cookie is not None:
                        societe_id = broker_cookie
                except Exception:
                    societe_id = broker_cookie
            # Nom/prénom pour clients
            try:
                if client_id:
                    c_row = db.execute(
                        text("SELECT prenom, nom FROM mariadb_clients WHERE id = :cid LIMIT 1"),
                        {"cid": client_id},
                    ).fetchone()
                    if c_row:
                        _m = c_row._mapping if hasattr(c_row, "_mapping") else None
                        prenom = _m.get("prenom") if _m else (c_row[0] if len(c_row) > 0 else "")
                        nom = _m.get("nom") if _m else (c_row[1] if len(c_row) > 1 else "")
                        user_name = f"{prenom or ''} {nom or ''}".strip()
            except Exception:
                user_name = None
    except HTTPException as exc:
        if "text/html" in accept:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": exc.detail, "prefill_email": email, "prefill_password": password},
                status_code=exc.status_code,
            )
        raise

    # Générer token et cookies
    token = encode_token(user_id=uid, user_type=utype, societe_id=societe_id, client_id=client_id, broker_id=broker_cookie)
    max_age = 60 * 60 * 8

    def _with_cookies(response):
        response.set_cookie("auth_token", token, httponly=True, max_age=max_age, path="/")
        # Cookies de compatibilité pour les templates/anciens contrôles
        response.set_cookie("user_id", str(uid), max_age=max_age, path="/")
        response.set_cookie("user_type", utype, max_age=max_age, path="/")
        if client_id is not None:
            response.set_cookie("client_id", str(client_id), max_age=max_age, path="/")
        if broker_cookie is not None:
            response.set_cookie("broker_id", str(broker_cookie), max_age=max_age, path="/")
        if user_name:
            response.set_cookie("user_name", user_name, max_age=max_age, path="/")
        else:
            response.delete_cookie("user_name", path="/")
        if societe_id is not None:
            response.set_cookie("societe_id", str(societe_id), max_age=max_age, path="/")
        if role_codes:
            response.set_cookie("user_role", role_codes[0], max_age=max_age, path="/")
        return response

    # Redirection implicite côté UI : si Accept: text/html, renvoyer une redirection
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        target = "/dashboard/clients"
        if "superadmin" in role_codes:
            target = "/dashboard/superadmin"
        elif "dirigeant" in role_codes:
            target = "/dashboard/"
        return _with_cookies(
            HTMLResponse(
                content=f'<meta http-equiv="refresh" content="0; url={target}" />',
                status_code=303,
                headers={"Location": target},
            )
        )
    return _with_cookies(JSONResponse({"detail": "ok"}))


@app.post("/logout")
def logout(request: Request):
    accept = request.headers.get("accept", "")
    target = "/login"
    if "text/html" in accept:
        resp = RedirectResponse(target, status_code=303)
    else:
        resp = JSONResponse({"detail": "logged out"})
    resp.delete_cookie("auth_token", path="/")
    resp.delete_cookie("user_id", path="/")
    resp.delete_cookie("user_type", path="/")
    resp.delete_cookie("user_role", path="/")
    resp.delete_cookie("societe_id", path="/")
    resp.delete_cookie("client_id", path="/")
    resp.delete_cookie("broker_id", path="/")
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, db: Session = Depends(get_db)):
    """Formulaire simple de connexion (staff/clients)."""
    return templates.TemplateResponse("login.html", {"request": request})


# ---------------- Mot de passe oublié / reset ----------------
@app.get("/password/forgot", response_class=HTMLResponse)
def password_forgot_form(request: Request, db: Session = Depends(get_db)):
    courtiers = db.execute(
        text("SELECT id, nom FROM mariadb_societe_gestion WHERE nature = 'courtier' AND actif = 1 ORDER BY nom")
    ).fetchall()
    courtiers_ctx = [{"id": (c.id if hasattr(c, 'id') else c[0]), "nom": (c.nom if hasattr(c, 'nom') else c[1])} for c in courtiers or []]
    return templates.TemplateResponse("password_forgot.html", {"request": request, "courtiers": courtiers_ctx})


@app.post("/password/forgot", response_class=HTMLResponse)
def password_forgot(
    request: Request,
    email: str = Form(...),
    broker_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    courtiers = db.execute(
        text("SELECT id, nom FROM mariadb_societe_gestion WHERE nature = 'courtier' AND actif = 1 ORDER BY nom")
    ).fetchall()
    courtiers_ctx = [{"id": (c.id if hasattr(c, 'id') else c[0]), "nom": (c.nom if hasattr(c, 'nom') else c[1])} for c in courtiers or []]
    if not _check_forgot_rate(request):
        msg = "Trop de demandes. Réessayez dans quelques minutes."
        return templates.TemplateResponse("password_forgot.html", {"request": request, "error": msg, "courtiers": courtiers_ctx})
    # Lookup user (staff d'abord)
    row = db.execute(
        text("SELECT id, user_type FROM auth_users WHERE login = :login AND actif = 1 AND user_type = 'staff' LIMIT 1"),
        {"login": email},
    ).fetchone()
    client_portal = False
    if not row:
        # Comptes clients dans la nouvelle table
        params = {"login": email}
        broker_id_int = None
        try:
            broker_id_int = int(broker_id) if broker_id not in (None, "") else None
        except Exception:
            broker_id_int = None
        query = "SELECT id FROM auth_client_users WHERE login = :login AND status = 'active'"
        if broker_id_int is not None:
            query += " AND broker_id = :bid"
            params["bid"] = broker_id_int
        client_rows = db.execute(text(query), params).fetchall()
        if broker_id_int is None and len(client_rows) > 1:
            msg = "Plusieurs courtiers utilisent ce login : sélectionnez votre courtier."
            return templates.TemplateResponse(
                "password_forgot.html",
                {"request": request, "error": msg, "courtiers": courtiers_ctx},
            )
        if client_rows:
            row = client_rows[0]
            client_portal = True
        else:
            # Fallback legacy client dans auth_users
            row = db.execute(
                text("SELECT id, user_type FROM auth_users WHERE login = :login AND actif = 1 AND user_type = 'client' LIMIT 1"),
                {"login": email},
            ).fetchone()

    if not row:
        msg = "Si ce compte existe, un lien de réinitialisation a été généré."
        return templates.TemplateResponse("password_forgot.html", {"request": request, "message": msg, "courtiers": courtiers_ctx})
    m = row._mapping if hasattr(row, "_mapping") else None
    uid = m.get("id") if m else row[0]
    utype = m.get("user_type") if m else (row[1] if len(row) > 1 else "client")
    token = encode_reset_token(uid, utype, ttl_seconds=3600, client_portal=client_portal)
    reset_url = f"{str(request.base_url).rstrip('/')}/password/reset?token={token}"

    # Envoi email si configuré
    sent = send_email(
        to_email=email,
        subject="Réinitialisation de mot de passe",
        body=f"Bonjour,\n\nCliquez sur le lien suivant pour réinitialiser votre mot de passe (valide 1h) :\n{reset_url}\n\nSi vous n'êtes pas à l'origine de cette demande, ignorez ce message.",
    )
    msg = "Si ce compte existe, un lien de réinitialisation a été généré."
    ctx = {
        "request": request,
        "message": msg,
        "courtiers": courtiers_ctx,
    }
    # Si l'envoi a échoué, afficher le lien pour tests
    if not sent:
        ctx["reset_url"] = reset_url
        ctx["token"] = token
        ctx["warning"] = "Envoi email indisponible (SMTP non configuré) — utilisez ce lien pour tester."
    return templates.TemplateResponse("password_forgot.html", ctx)


@app.get("/password/reset", response_class=HTMLResponse)
def password_reset_form(request: Request, token: str):
    data = decode_reset_token(token, max_age=3600)
    if not data:
        return templates.TemplateResponse(
            "password_reset.html",
            {"request": request, "error": "Lien invalide ou expiré.", "token": None},
        )
    return templates.TemplateResponse(
        "password_reset.html",
        {"request": request, "token": token},
    )


@app.post("/password/reset", response_class=HTMLResponse)
def password_reset(
    request: Request,
    token: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db),
):
    data = decode_reset_token(token, max_age=3600)
    if not data:
        return templates.TemplateResponse(
            "password_reset.html",
            {"request": request, "error": "Lien invalide ou expiré.", "token": None},
        )
    uid = data.get("uid")
    client_portal = bool(data.get("cp"))
    pwd_hash = hash_password(new_password)
    try:
        if client_portal:
            db.execute(
                text(
                    """
                    UPDATE auth_client_users
                    SET password_hash = :p,
                        status = 'active',
                        password_updated_at = NOW()
                    WHERE id = :uid
                    """
                ),
                {"p": pwd_hash, "uid": uid},
            )
        else:
            db.execute(text("UPDATE auth_users SET password_hash = :p WHERE id = :uid"), {"p": pwd_hash, "uid": uid})
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse(
            "password_reset.html",
            {"request": request, "error": "Erreur lors de la mise à jour.", "token": token},
        )
    return templates.TemplateResponse(
        "password_reset.html",
        {"request": request, "message": "Mot de passe réinitialisé. Vous pouvez vous connecter.", "token": None},
    )

# Préchauffage du cache finance au démarrage pour éviter le premier chargement lent
def _warm_finance_cache():
    try:
        from src.api.dashboard import _build_finance_analysis
        from src.database import SessionLocal
        db = SessionLocal()
        try:
            _build_finance_analysis(db, finance_rh_id=None, finance_date_param=None, finance_valo_param=None)
            logger.info("Finance cache prewarmed successfully")
        finally:
            db.close()
    except Exception:
        logger.exception("Finance cache prewarm failed", exc_info=True)

@app.on_event("startup")
def _on_startup():
    if os.getenv("FASTAPI_SKIP_WARM") or os.getenv("PYTEST_CURRENT_TEST"):
        return
    # Préchauffage synchrone pour éviter le premier hit très lent (peut prendre ~40s)
    _warm_finance_cache()


# ---------------- Middleware auth cookie -> headers/state ----------------
@app.middleware("http")
async def auth_cookie_middleware(request: Request, call_next):
    token = request.cookies.get("auth_token")
    if token:
        data = decode_token(token)
        if data:
            request.state.user_ctx = {
                "user_id": data.get("uid"),
                "user_type": data.get("utype"),
                "societe_id": data.get("sid"),
                "client_id": data.get("cid"),
                "broker_id": data.get("bid"),
            }
    # Fallback si pas de token mais cookies explicites (compat ancien front)
    if not getattr(request.state, "user_ctx", None):
        uid = request.cookies.get("user_id")
        utype = request.cookies.get("user_type")
        soc = request.cookies.get("societe_id")
        client_cookie = request.cookies.get("client_id")
        broker_cookie = request.cookies.get("broker_id")
        try:
            uid_int = int(uid) if uid not in (None, "") else None
        except Exception:
            uid_int = None
        try:
            soc_int = int(soc) if soc not in (None, "") else None
        except Exception:
            soc_int = None
        try:
            client_int = int(client_cookie) if client_cookie not in (None, "") else None
        except Exception:
            client_int = None
        try:
            broker_int = int(broker_cookie) if broker_cookie not in (None, "") else None
        except Exception:
            broker_int = None
        if uid_int is not None and utype:
            request.state.user_ctx = {
                "user_id": uid_int,
                "user_type": utype,
                "societe_id": soc_int,
                "client_id": client_int,
                "broker_id": broker_int,
            }
    response = await call_next(request)
    return response

# ---------------- Middleware CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://72.61.94.45",
        "https://72.61.94.45",
        "http://72.61.94.45:8100",
        "https://72.61.94.45:8100",
        "http://72.61.94.45:8101",
        "https://72.61.94.45:8101",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Imports Services ----------------
from src.services.allocations import get_allocations, get_allocation, create_allocation
from src.services.clients import get_clients, get_client, create_client, update_client
from src.services.affaires import get_affaires, get_affaire, create_affaire
from src.services.documents import get_documents, get_document, create_document
from src.services.document_client import (
    create_document_client,
    get_documents_by_client,
    get_document_client,
    delete_document_client,
)
from src.services.supports import get_supports, get_support, create_support
from src.services.historiques import (
    get_historiques_personne, get_historique_personne, create_historique_personne,
    get_historiques_affaire, get_historique_affaire, create_historique_affaire,
    get_historiques_support, get_historique_support, create_historique_support
)
from src.services.reporting import get_all_clients, get_top_clients, get_all_affaires, get_all_allocations, get_all_supports

# ---------------- Imports Schémas ----------------
from src.schemas.client import ClientSchema, ClientCreateSchema, ClientUpdateSchema
from src.schemas.affaire import AffaireSchema, AffaireCreateSchema
from src.schemas.document import DocumentSchema, DocumentCreateSchema
from src.schemas.document_client import DocumentClientSchema, DocumentClientCreateSchema
from src.schemas.support import SupportSchema, SupportCreateSchema
from src.schemas.allocation import AllocationSchema, AllocationCreateSchema
from src.schemas.historique_personne import HistoriquePersonneSchema, HistoriquePersonneCreateSchema
from src.schemas.historique_affaire import HistoriqueAffaireSchema, HistoriqueAffaireCreateSchema
from src.schemas.historique_support import HistoriqueSupportSchema, HistoriqueSupportCreateSchema


# ---------------- Imports Models ----------------
from src.models.historique_personne import HistoriquePersonne
from src.models.historique_affaire import HistoriqueAffaire
from src.models.historique_support import HistoriqueSupport
from src.models.client import Client
from src.models.affaire import Affaire
from src.models.support import Support

# ---------------- Allocations ----------------
@app.get("/allocations/", response_model=list[AllocationSchema])
def read_allocations(db: Session = Depends(get_db)):
    return get_allocations(db)

@app.get("/allocations/{allocation_id}", response_model=AllocationSchema)
def read_allocation(allocation_id: int, db: Session = Depends(get_db)):
    db_allocation = get_allocation(db, allocation_id)
    if not db_allocation:
        raise HTTPException(status_code=404, detail="Allocation non trouvée")
    return db_allocation


@app.post("/allocations/", response_model=AllocationSchema)
def create_new_allocation(payload: AllocationCreateSchema, db: Session = Depends(get_db)):
    return create_allocation(
        date=payload.date,
        valo=payload.valo,
        mouvement=payload.mouvement,
        sicav=payload.sicav,
        perf_sicav_hebdo=payload.perf_sicav_hebdo,
        perf_sicav_52=payload.perf_sicav_52,
        volat=payload.volat,
        annee=payload.annee,
        nom=payload.nom,
    )

# ---------------- Clients ----------------
@app.get("/clients/", response_model=list[ClientSchema])
def read_clients(db: Session = Depends(get_db)):
    return get_clients(db)

@app.get("/clients/{client_id}/", response_model=ClientSchema)
def read_client(client_id: int, db: Session = Depends(get_db)):
    db_client = get_client(db, client_id)
    if not db_client:
        raise HTTPException(status_code=404, detail="Client not found")
    return db_client

@app.post("/clients/", response_model=ClientSchema)
def create_new_client(payload: ClientCreateSchema, db: Session = Depends(get_db)):
    return create_client(db, payload)

@app.put("/clients/{client_id}/", response_model=ClientSchema)
def update_existing_client(client_id: int, payload: ClientUpdateSchema, db: Session = Depends(get_db)):
    db_client = update_client(db, client_id, payload)
    if not db_client:
        raise HTTPException(status_code=404, detail="Client not found")
    return db_client

@app.delete("/clients/{client_id}/", response_model=ClientSchema)
def delete_client(client_id: int, db: Session = Depends(get_db)):
    client = get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    db.delete(client)
    db.commit()
    return client

# ---------------- Affaires ----------------
@app.get("/affaires/", response_model=list[AffaireSchema])
def read_affaires(db: Session = Depends(get_db)):
    return get_affaires(db)

@app.get("/affaires/{affaire_id}/", response_model=AffaireSchema)
def read_affaire(affaire_id: int, db: Session = Depends(get_db)):
    db_affaire = get_affaire(db, affaire_id)
    if not db_affaire:
        raise HTTPException(status_code=404, detail="Affaire not found")
    return db_affaire

@app.post("/affaires/", response_model=AffaireSchema)
def create_new_affaire(payload: AffaireCreateSchema, db: Session = Depends(get_db)):
    return create_affaire(
        db,
        payload.id_personne,
        payload.ref,
        payload.srri,
        payload.date_debut,
        payload.date_cle,
        payload.frais_negocies,
    )

@app.delete("/affaires/{affaire_id}/", response_model=AffaireSchema)
def delete_affaire(affaire_id: int, db: Session = Depends(get_db)):
    affaire = get_affaire(db, affaire_id)
    if not affaire:
        raise HTTPException(status_code=404, detail="Affaire not found")
    db.delete(affaire)
    db.commit()
    return affaire

# ---------------- Documents ----------------
@app.get("/documents/", response_model=list[DocumentSchema])
def read_documents(db: Session = Depends(get_db)):
    return get_documents(db)

@app.get("/documents/{document_id}/", response_model=DocumentSchema)
def read_document(document_id: int, db: Session = Depends(get_db)):
    db_doc = get_document(db, document_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_doc

@app.post("/documents/", response_model=DocumentSchema)
def create_new_document(payload: DocumentCreateSchema, db: Session = Depends(get_db)):
    return create_document(
        db,
        payload.documents,
        payload.niveau,
        payload.obsolescence_annees,
        payload.risque,
    )

# ---------------- Documents par client ----------------
@app.post("/documents_clients/", response_model=DocumentClientSchema)
def create_new_document_client(payload: DocumentClientCreateSchema, db: Session = Depends(get_db)):
    doc, err = create_document_client(db, payload)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return doc

@app.get("/documents_clients/{client_id}", response_model=list[DocumentClientSchema])
def read_documents_by_client(client_id: int, db: Session = Depends(get_db)):
    docs = get_documents_by_client(db, client_id)
    return docs or []

from fastapi import HTTPException

@app.get("/document_client/{doc_client_id}", response_model=DocumentClientSchema)
def read_document_client(doc_client_id: int, db: Session = Depends(get_db)):
    doc = get_document_client(db, doc_client_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return doc


@app.delete("/document_client/{doc_client_id}")
def remove_document_client(doc_client_id: int, db: Session = Depends(get_db)):
    deleted = delete_document_client(db, doc_client_id)
    if not deleted:
        return {"message": "Document non trouvé"}
    return {"message": "Document supprimé"}

# ---------------- Supports ----------------
@app.get("/supports/", response_model=list[SupportSchema])
def read_supports(request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "supports", "access")
    return get_supports(db)

@app.get("/supports/{support_id}/", response_model=SupportSchema)
def read_support(support_id: int, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "supports", "access")
    db_support = get_support(db, support_id)
    if not db_support:
        raise HTTPException(status_code=404, detail="Support not found")
    return db_support

@app.post("/supports/", response_model=SupportSchema)
def create_new_support(payload: SupportCreateSchema, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "supports", "access")
    return create_support(
        db,
        payload.code_isin,
        payload.nom,
        payload.cat_gene,
        payload.cat_principale,
        payload.cat_det,
        payload.cat_geo,
        payload.promoteur,
        payload.taux_retro,
        payload.SRRI,
    )

@app.delete("/supports/{support_id}/", response_model=SupportSchema)
def delete_support(support_id: int, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "supports", "access")
    support = get_support(db, support_id)
    if not support:
        raise HTTPException(status_code=404, detail="Support not found")
    db.delete(support)
    db.commit()
    return support

# ---------------- Historiques Personne ----------------
@app.get("/historiques/personne/", response_model=list[HistoriquePersonneSchema])
def read_historiques_personne(request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "read")
    return get_historiques_personne(db)

@app.get("/historiques/personne/{hist_id}/", response_model=HistoriquePersonneSchema)
def read_historique_personne(hist_id: int, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "read")
    return get_historique_personne(db, hist_id)

@app.post("/historiques/personne/", response_model=HistoriquePersonneSchema)
def create_new_historique_personne(payload: HistoriquePersonneCreateSchema, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "write")
    return create_historique_personne(
        db,
        payload.date,
        payload.valo,
        payload.mouvement,
        payload.volat,
        payload.annee,
    )

# ---------------- Historiques Affaire ----------------
@app.get("/historiques/affaire/", response_model=list[HistoriqueAffaireSchema])
def read_historiques_affaire(request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "read")
    return get_historiques_affaire(db)

@app.get("/historiques/affaire/{hist_id}/", response_model=HistoriqueAffaireSchema)
def read_historique_affaire(hist_id: int, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "read")
    return get_historique_affaire(db, hist_id)

@app.post("/historiques/affaire/", response_model=HistoriqueAffaireSchema)
def create_new_historique_affaire(payload: HistoriqueAffaireCreateSchema, request: Request, db: Session = Depends(get_db)):
    _require_feature(request, db, "data", "write")
    return create_historique_affaire(
        db,
        payload.date,
        payload.valo,
        payload.mouvement,
        payload.sicav,
        payload.perf_sicav_hebdo,
        payload.perf_sicav_52,
        payload.volat,
        payload.annee,
    )

# ---------------- Historiques Support ----------------
@app.get("/historiques/support/", response_model=list[HistoriqueSupportSchema])
def read_historiques_support(db: Session = Depends(get_db)):
    return get_historiques_support(db)

@app.get("/historiques/support/{hist_id}/", response_model=HistoriqueSupportSchema)
def read_historique_support(hist_id: int, db: Session = Depends(get_db)):
    return get_historique_support(db, hist_id)

@app.post("/historiques/support/", response_model=HistoriqueSupportSchema)
def create_new_historique_support(payload: HistoriqueSupportCreateSchema, db: Session = Depends(get_db)):
    return create_historique_support(
        db,
        payload.modif_quand,
        payload.source,
        payload.id_source,
        payload.date,
        payload.id_support,
        payload.nbuc,
        payload.vl,
        payload.prmp,
        payload.valo,
    )

# ---------------- Reporting ----------------
@app.get("/reporting/clients/", response_model=list[ClientSchema])
def reporting_clients(db: Session = Depends(get_db)):
    return get_all_clients(db)

@app.get("/reporting/top-clients/", response_model=list[ClientSchema])
def reporting_top_clients(limit: int = 5, db: Session = Depends(get_db)):
    return get_top_clients(db, limit)

@app.get("/reporting/affaires/", response_model=list[AffaireSchema])
def reporting_affaires(db: Session = Depends(get_db)):
    return get_all_affaires(db)

@app.get("/reporting/allocations/", response_model=list[AllocationSchema])
def reporting_allocations(db: Session = Depends(get_db)):
    return get_all_allocations(db)

@app.get("/reporting/supports/", response_model=list[SupportSchema])
def reporting_supports(db: Session = Depends(get_db)):
    return get_all_supports(db)

# ---------------- Dashboards HTML ----------------
# Routes HTML gérées dans src/api/dashboard.py via router
