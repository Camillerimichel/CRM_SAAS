from fastapi import HTTPException, Request

from sqlalchemy.orm import Query


def get_societe_id_from_request(request: Request) -> int:
    societe_id = getattr(request.state, "id_societe_gestion", None)
    if societe_id is None:
        raise HTTPException(status_code=403, detail="Contexte société de gestion manquant")
    return societe_id


def apply_societe_filter(query: Query, model, request: Request) -> Query:
    if not hasattr(model, "id_societe_gestion"):
        raise HTTPException(
            status_code=400,
            detail=f"Le modèle {getattr(model, '__name__', str(model))} n’a pas d’id_societe_gestion",
        )
    return query.filter(model.id_societe_gestion == get_societe_id_from_request(request))
