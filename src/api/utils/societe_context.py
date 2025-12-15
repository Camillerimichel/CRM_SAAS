from contextvars import ContextVar
from typing import Iterable, List, Type

from sqlalchemy import event
from sqlalchemy.orm import Session, with_loader_criteria

from src.models import models as models_registry

current_societe_context = ContextVar("current_societe_id", default=None)


def set_current_societe(societe_id: int):
    """Bascule la société active dans le contexte courant."""
    return current_societe_context.set(societe_id)


def get_current_societe():
    return current_societe_context.get(None)


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
    societe_id = get_current_societe()
    if societe_id is None or not SOCIETE_FILTERED_MODELS:
        return
    filters = [
        with_loader_criteria(
            model,
            lambda cls, sid=societe_id: cls.id_societe_gestion == sid,
            include_aliases=True,
        )
        for model in SOCIETE_FILTERED_MODELS
    ]
    execute_state.statement = execute_state.statement.options(*filters)
