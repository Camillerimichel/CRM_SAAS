"""RBAC helpers backed by auth_* tables (roles, permissions, scopes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

from fastapi import HTTPException, Request

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session


Permission = Tuple[str, str]


@dataclass
class Access:
    user_type: str
    user_id: int
    role_ids: Set[int]
    permissions: Set[Permission]
    # None => périmètre global, sinon id de la société
    societes: Set[int | None]

    def has_permission(self, feature: str, action: str, societe_id: Optional[int] = None) -> bool:
        if (feature, action) not in self.permissions:
            return False
        # Portée globale => OK
        if None in self.societes:
            return True
        # Sinon, l'utilisateur doit être rattaché à la société courante
        return societe_id in self.societes


def load_access(db: Session, user_type: str, user_id: int) -> Access:
    """Charge rôles/permissions + périmètre société pour un utilisateur."""
    roles_rows = db.execute(
        text(
            """
            SELECT aur.role_id, aur.societe_id
            FROM auth_user_roles aur
            WHERE aur.user_type = :ut AND aur.user_id = :uid
            """
        ),
        {"ut": user_type, "uid": user_id},
    ).fetchall()
    role_ids = {int(r.role_id) for r in roles_rows} if roles_rows else set()
    scopes = {r.societe_id if r.societe_id is not None else None for r in roles_rows} if roles_rows else set()

    if not role_ids:
        return Access(user_type=user_type, user_id=user_id, role_ids=set(), permissions=set(), societes=set())

    perm_rows = db.execute(
        text(
            """
            SELECT DISTINCT ap.feature, ap.action
            FROM auth_role_permissions arp
            JOIN auth_permissions ap ON ap.id = arp.permission_id
            WHERE arp.role_id IN :rids AND arp.allow = 1
            """
        ).bindparams(bindparam("rids", expanding=True)),
        {"rids": tuple(role_ids)},
    ).fetchall()
    perms: Set[Permission] = {(str(p.feature), str(p.action)) for p in perm_rows} if perm_rows else set()

    return Access(user_type=user_type, user_id=user_id, role_ids=role_ids, permissions=perms, societes=scopes or {None})


def require_permission(access: Access, feature: str, action: str, societe_id: Optional[int] = None) -> None:
    """Lève une exception HTTP 403 si la permission n'est pas accordée."""

    if not access.has_permission(feature, action, societe_id=societe_id):
        raise HTTPException(status_code=403, detail="Accès refusé")


def ensure_client_ownership(access: Access, owner_id: int) -> None:
    """Pour les rôles client : restreindre aux données dont il est propriétaire."""
    if access.user_type == "client" and access.user_id != owner_id:
        raise HTTPException(status_code=403, detail="Accès refusé (périmètre client)")


def extract_user_context(request: Request) -> tuple[str, Optional[int], Optional[int]]:
    """Récupère user_type/user_id/societe_id depuis headers, cookies, state ou query (fallback)."""
    # Contexte injecté par middleware (token)
    state_ctx = getattr(request.state, "user_ctx", None)
    if state_ctx:
        return (
            (state_ctx.get("user_type") or "staff"),
            state_ctx.get("user_id"),
            state_ctx.get("societe_id"),
        )
    # user type
    user_type = (
        request.headers.get("X-User-Type")
        or (request.cookies.get("user_type") if request and getattr(request, "cookies", None) else None)
        or request.query_params.get("user_type")
        or "staff"
    )
    user_type = user_type.strip().lower()
    # user id
    user_id_raw = (
        request.headers.get("X-User-Id")
        or (request.cookies.get("user_id") if request and getattr(request, "cookies", None) else None)
        or request.query_params.get("user_id")
    )
    soc_raw = (
        request.headers.get("X-Societe-Id")
        or (request.cookies.get("societe_id") if request and getattr(request, "cookies", None) else None)
        or request.query_params.get("societe_id")
    )
    user_id = int(user_id_raw) if user_id_raw not in (None, "") else None
    try:
        societe_id = int(soc_raw) if soc_raw not in (None, "") else None
    except Exception:
        societe_id = None
    return user_type, user_id, societe_id


def pick_scope(access: Access, requested_societe_id: Optional[int]) -> Optional[int]:
    """Sélectionne une portée société cohérente avec les rôles de l'utilisateur."""
    if requested_societe_id is not None:
        return requested_societe_id
    if not access.societes:
        return None
    # Si une seule portée, on la prend par défaut
    if len(access.societes) == 1:
        return next(iter(access.societes))
    # Sinon pas de scope déterminé (exigera un scope explicite)
    return None
