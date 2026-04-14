from contextvars import ContextVar
from typing import Iterable, List, Type

from sqlalchemy import event
from sqlalchemy.orm import Session, with_loader_criteria

from src.models import models as models_registry

current_societe_context = ContextVar("current_societe_id", default=None)


def set_current_societe(societe_id: int | None, allowed_societe_ids: Iterable[int] | None = None):
    """Bascule la société active et son périmètre étendu dans le contexte courant."""
    allowed_ids = tuple(sorted({int(sid) for sid in (allowed_societe_ids or []) if sid is not None}))
    payload = {
        "root_societe_id": societe_id,
        "allowed_societe_ids": allowed_ids,
    }
    return current_societe_context.set(payload)


def get_current_societe():
    ctx = current_societe_context.get(None)
    if not ctx:
        return None
    if isinstance(ctx, dict):
        return ctx.get("root_societe_id")
    return ctx


def get_current_societe_scope_ids() -> tuple[int, ...]:
    ctx = current_societe_context.get(None)
    if not ctx:
        return tuple()
    if isinstance(ctx, dict):
        allowed_ids = ctx.get("allowed_societe_ids") or tuple()
        return tuple(int(sid) for sid in allowed_ids if sid is not None)
    try:
        return (int(ctx),)
    except Exception:
        return tuple()


def reset_current_societe(token):
    if token is not None:
        current_societe_context.reset(token)


def _collect_societe_models() -> List[Type]:
    models: List[Type] = []
    for attr_name in dir(models_registry):
        cls = getattr(models_registry, attr_name)
        if isinstance(cls, type) and hasattr(cls, "__tablename__") and hasattr(cls, "id_societe_gestion"):
            models.append(cls)
    return models


SOCIETE_FILTERED_MODELS = _collect_societe_models()


@event.listens_for(Session, "do_orm_execute")
def _with_societe_scope(execute_state):
    if not execute_state.is_select or execute_state.is_column_load:
        return
    societe_ids = get_current_societe_scope_ids()
    if not societe_ids or not SOCIETE_FILTERED_MODELS:
        return
    filters = [
        with_loader_criteria(
            model,
            lambda cls, scope_ids=societe_ids: cls.id_societe_gestion.in_(scope_ids),
            include_aliases=True,
        )
        for model in SOCIETE_FILTERED_MODELS
    ]
    execute_state.statement = execute_state.statement.options(*filters)
