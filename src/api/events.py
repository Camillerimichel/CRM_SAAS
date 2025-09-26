from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Any

from src.database import get_db

from src.schemas.evenement import EvenementSchema, EvenementCreateSchema, EvenementUpdateSchema, TacheCreateSchema
from src.schemas.statut_evenement import StatutEvenementSchema, StatutEvenementCreateSchema
from src.schemas.type_evenement import TypeEvenementSchema, TypeEvenementCreateSchema
from src.schemas.evenement_statut import EvenementStatutSchema, EvenementStatutCreateSchema
from src.schemas.evenement_intervenant import (
    EvenementIntervenantSchema,
    EvenementIntervenantCreateSchema,
)
from src.schemas.evenement_lien import EvenementLienSchema, EvenementLienCreateSchema
from src.schemas.evenement_envoi import EvenementEnvoiSchema, EvenementEnvoiCreateSchema
from src.schemas.modele_document import ModeleDocumentSchema, ModeleDocumentCreateSchema

from src.services.evenements import (
    list_evenements,
    get_evenement,
    create_evenement,
    update_evenement,
    delete_evenement,
    list_statuts,
    create_statut,
    list_types,
    create_type,
    add_statut_to_evenement,
    get_evenement_statuts,
    add_intervenant,
    list_intervenants,
    delete_intervenant,
    add_lien,
    list_liens,
    delete_lien,
    create_envoi,
    update_envoi_statut,
    list_envois,
    list_modeles,
    get_modele,
    create_modele,
    update_modele,
    delete_modele,
    vue_reclamations,
    vue_suivi_evenement,
    create_tache,
)
from src.services.modele_render import render_modele as render_modele_util


router = APIRouter(prefix="", tags=["Evenements"])


# -------- CRUD Evenements --------
@router.get("/evenements/", response_model=list[EvenementSchema])
def api_list_evenements(
    db: Session = Depends(get_db),
    type_id: int | None = Query(None),
    statut: str | None = Query(None),
    client_id: int | None = Query(None),
    affaire_id: int | None = Query(None),
    support_id: int | None = Query(None),
    intervenant: str | None = Query(None),
    categorie: str | None = Query(None),
):
    return list_evenements(
        db,
        type_id=type_id,
        statut=statut,
        client_id=client_id,
        affaire_id=affaire_id,
        support_id=support_id,
        intervenant=intervenant,
        categorie=categorie,
    )


@router.get("/evenements/{evenement_id}", response_model=EvenementSchema)
def api_get_evenement(evenement_id: int, db: Session = Depends(get_db)):
    ev = get_evenement(db, evenement_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evènement introuvable")
    return ev


@router.post("/evenements/", response_model=EvenementSchema, summary="Créer un évènement")
def api_create_evenement(payload: EvenementCreateSchema, db: Session = Depends(get_db)):
    return create_evenement(db, payload)


@router.put("/evenements/{evenement_id}", response_model=EvenementSchema, summary="Mettre à jour un évènement")
def api_update_evenement(evenement_id: int, payload: EvenementUpdateSchema, db: Session = Depends(get_db)):
    ev = update_evenement(db, evenement_id, payload)
    if not ev:
        raise HTTPException(status_code=404, detail="Evènement introuvable")
    return ev


@router.delete("/evenements/{evenement_id}", summary="Supprimer un évènement")
def api_delete_evenement(evenement_id: int, db: Session = Depends(get_db)):
    ok = delete_evenement(db, evenement_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Evènement introuvable")
    return {"message": "Supprimé"}


# -------- Rattachements pratiques --------
@router.get("/clients/{client_id}/evenements", response_model=list[EvenementSchema])
def api_evenements_par_client(client_id: int, db: Session = Depends(get_db)):
    return list_evenements(db, client_id=client_id)


@router.get("/affaires/{affaire_id}/evenements", response_model=list[EvenementSchema])
def api_evenements_par_affaire(affaire_id: int, db: Session = Depends(get_db)):
    return list_evenements(db, affaire_id=affaire_id)


# -------- Statuts --------
@router.get("/statuts_evenement/", response_model=list[StatutEvenementSchema])
def api_list_statuts(db: Session = Depends(get_db)):
    return list_statuts(db)


@router.post("/statuts_evenement/", response_model=StatutEvenementSchema)
def api_create_statut(payload: StatutEvenementCreateSchema, db: Session = Depends(get_db)):
    return create_statut(db, payload.libelle)


@router.post("/evenements/{evenement_id}/statut", response_model=EvenementStatutSchema, summary="Ajouter un statut et MAJ statut courant")
def api_add_statut(evenement_id: int, payload: EvenementStatutCreateSchema, db: Session = Depends(get_db)):
    res = add_statut_to_evenement(db, evenement_id, payload)
    if isinstance(res, str):
        if res == "evenement_not_found":
            raise HTTPException(status_code=404, detail="Evènement introuvable")
        if res == "statut_not_found":
            raise HTTPException(status_code=404, detail="Statut introuvable")
        if res == "status_downgrade_forbidden":
            raise HTTPException(status_code=400, detail="Transition de statut interdite (régression)")
        if res == "dependency_blocking":
            raise HTTPException(status_code=400, detail="Dépendance bloquante non terminée")
        raise HTTPException(status_code=400, detail=res)
    return res


@router.get("/evenements/{evenement_id}/statuts", response_model=list[EvenementStatutSchema])
def api_list_statuts_evenement(evenement_id: int, db: Session = Depends(get_db)):
    return get_evenement_statuts(db, evenement_id)


# -------- Types d'évènement --------
@router.get("/types_evenement/", response_model=list[TypeEvenementSchema])
def api_list_types(db: Session = Depends(get_db)):
    return list_types(db)


@router.post("/types_evenement/", response_model=TypeEvenementSchema)
def api_create_type(payload: TypeEvenementCreateSchema, db: Session = Depends(get_db)):
    return create_type(db, payload.libelle, payload.categorie)


# -------- Intervenants --------
@router.post("/evenements/{evenement_id}/intervenants", response_model=EvenementIntervenantSchema, summary="Ajouter un intervenant")
def api_add_intervenant(evenement_id: int, payload: EvenementIntervenantCreateSchema, db: Session = Depends(get_db)):
    res = add_intervenant(db, evenement_id, payload)
    if isinstance(res, str):
        raise HTTPException(status_code=404, detail="Evènement introuvable")
    return res


@router.get("/evenements/{evenement_id}/intervenants", response_model=list[EvenementIntervenantSchema])
def api_list_intervenants(evenement_id: int, db: Session = Depends(get_db)):
    return list_intervenants(db, evenement_id)


@router.delete("/intervenants/{intervenant_id}")
def api_delete_intervenant(intervenant_id: int, db: Session = Depends(get_db)):
    ok = delete_intervenant(db, intervenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Intervenant introuvable")
    return {"message": "Supprimé"}


# -------- Liens --------
@router.post("/evenements/{evenement_id}/liens", response_model=EvenementLienSchema, summary="Ajouter un lien d'évènement")
def api_add_lien(evenement_id: int, payload: EvenementLienCreateSchema, db: Session = Depends(get_db)):
    res = add_lien(db, evenement_id, payload)
    if isinstance(res, str):
        if res == "evenement_not_found":
            raise HTTPException(status_code=404, detail="Evènement introuvable")
        if res == "evenement_cible_not_found":
            raise HTTPException(status_code=404, detail="Evènement cible introuvable")
        raise HTTPException(status_code=400, detail=res)
    return res


@router.get("/evenements/{evenement_id}/liens", response_model=list[EvenementLienSchema])
def api_list_liens(evenement_id: int, db: Session = Depends(get_db)):
    return list_liens(db, evenement_id)


@router.delete("/liens/{lien_id}")
def api_delete_lien(lien_id: int, db: Session = Depends(get_db)):
    ok = delete_lien(db, lien_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Lien introuvable")
    return {"message": "Supprimé"}


# -------- Envois --------
@router.post("/evenements/{evenement_id}/envois", response_model=EvenementEnvoiSchema, summary="Préparer un envoi lié à l'évènement")
def api_create_envoi(evenement_id: int, payload: EvenementEnvoiCreateSchema, db: Session = Depends(get_db)):
    res = create_envoi(db, evenement_id, payload)
    if isinstance(res, str):
        if res == "evenement_not_found":
            raise HTTPException(status_code=404, detail="Evènement introuvable")
        if res == "modele_not_found":
            raise HTTPException(status_code=404, detail="Modèle introuvable")
        raise HTTPException(status_code=400, detail=res)
    return res


@router.put("/envois/{envoi_id}/statut", response_model=EvenementEnvoiSchema)
def api_update_envoi_statut(envoi_id: int, statut: str, db: Session = Depends(get_db)):
    ev = update_envoi_statut(db, envoi_id, statut)
    if not ev:
        raise HTTPException(status_code=404, detail="Envoi introuvable")
    return ev


@router.get("/evenements/{evenement_id}/envois", response_model=list[EvenementEnvoiSchema])
def api_list_envois(evenement_id: int, db: Session = Depends(get_db)):
    return list_envois(db, evenement_id)


# -------- Modèles de documents --------
@router.get("/modeles/", response_model=list[ModeleDocumentSchema])
def api_list_modeles(db: Session = Depends(get_db)):
    return list_modeles(db)


@router.get("/modeles/{modele_id}", response_model=ModeleDocumentSchema)
def api_get_modele(modele_id: int, db: Session = Depends(get_db)):
    m = get_modele(db, modele_id)
    if not m:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return m


@router.post("/modeles/", response_model=ModeleDocumentSchema)
def api_create_modele(payload: ModeleDocumentCreateSchema, db: Session = Depends(get_db)):
    return create_modele(db, payload.nom, payload.canal, payload.contenu, objet=payload.objet, actif=payload.actif)


@router.put("/modeles/{modele_id}", response_model=ModeleDocumentSchema)
def api_update_modele(modele_id: int, payload: ModeleDocumentCreateSchema, db: Session = Depends(get_db)):
    m = update_modele(db, modele_id, nom=payload.nom, canal=payload.canal, contenu=payload.contenu, objet=payload.objet, actif=payload.actif)
    if not m:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return m


@router.delete("/modeles/{modele_id}")
def api_delete_modele(modele_id: int, db: Session = Depends(get_db)):
    ok = delete_modele(db, modele_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return {"message": "Supprimé"}


@router.post("/modeles/{modele_id}/render")
def api_render_modele(modele_id: int, data: dict[str, Any] | None = None, db: Session = Depends(get_db)):
    try:
        return render_modele_util(db, modele_id, data or {})
    except ValueError:
        raise HTTPException(status_code=404, detail="Modèle introuvable")


# -------- Reporting & vues --------
@router.get("/reporting/vue_reclamations")
def api_vue_reclamations(
    db: Session = Depends(get_db),
    statut: str | None = Query(None),
    client_id: int | None = Query(None),
):
    rows = vue_reclamations(db, statut=statut, client_id=client_id)
    return [dict(r) for r in rows]


@router.get("/reporting/vue_suivi_evenement")
def api_vue_suivi_evenement(
    db: Session = Depends(get_db),
    statut: str | None = Query(None),
    type_id: int | None = Query(None),
):
    rows = vue_suivi_evenement(db, statut=statut, type_id=type_id)
    return [dict(r) for r in rows]
@router.post("/taches/", response_model=EvenementSchema, summary="Créer une tâche (type auto)")
def api_create_tache(payload: TacheCreateSchema, db: Session = Depends(get_db)):
    return create_tache(db, payload)
