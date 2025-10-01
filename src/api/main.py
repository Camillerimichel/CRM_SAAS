# ---------------- Imports principaux ----------------
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates   # <-- ajoute ceci
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database import get_db
from fastapi import Query
from datetime import date


# ---------------- Définition app FastAPI ----------------
app = FastAPI()
templates = Jinja2Templates(directory="src/api/templates")
from src.api import dashboard
from src.api import events
app.include_router(dashboard.router)
app.include_router(events.router)

# Debug : afficher toutes les routes connues
for route in app.routes:
    print(">>> ROUTE:", route.path, "→", route.name)


# ---------------- Route d'accueil ----------------
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l’API CRM_SAAS. Consultez /docs pour la documentation."}

# ---------------- Middleware CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
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
def read_supports(db: Session = Depends(get_db)):
    return get_supports(db)

@app.get("/supports/{support_id}/", response_model=SupportSchema)
def read_support(support_id: int, db: Session = Depends(get_db)):
    db_support = get_support(db, support_id)
    if not db_support:
        raise HTTPException(status_code=404, detail="Support not found")
    return db_support

@app.post("/supports/", response_model=SupportSchema)
def create_new_support(payload: SupportCreateSchema, db: Session = Depends(get_db)):
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
def delete_support(support_id: int, db: Session = Depends(get_db)):
    support = get_support(db, support_id)
    if not support:
        raise HTTPException(status_code=404, detail="Support not found")
    db.delete(support)
    db.commit()
    return support

# ---------------- Historiques Personne ----------------
@app.get("/historiques/personne/", response_model=list[HistoriquePersonneSchema])
def read_historiques_personne(db: Session = Depends(get_db)):
    return get_historiques_personne(db)

@app.get("/historiques/personne/{hist_id}/", response_model=HistoriquePersonneSchema)
def read_historique_personne(hist_id: int, db: Session = Depends(get_db)):
    return get_historique_personne(db, hist_id)

@app.post("/historiques/personne/", response_model=HistoriquePersonneSchema)
def create_new_historique_personne(payload: HistoriquePersonneCreateSchema, db: Session = Depends(get_db)):
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
def read_historiques_affaire(db: Session = Depends(get_db)):
    return get_historiques_affaire(db)

@app.get("/historiques/affaire/{hist_id}/", response_model=HistoriqueAffaireSchema)
def read_historique_affaire(hist_id: int, db: Session = Depends(get_db)):
    return get_historique_affaire(db, hist_id)

@app.post("/historiques/affaire/", response_model=HistoriqueAffaireSchema)
def create_new_historique_affaire(payload: HistoriqueAffaireCreateSchema, db: Session = Depends(get_db)):
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
