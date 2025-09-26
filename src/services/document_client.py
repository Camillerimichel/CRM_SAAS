from sqlalchemy.orm import Session
from src.models.document_client import DocumentClient
from src.models.client import Client
from src.models.document import Document
from src.schemas.document_client import DocumentClientCreateSchema

# ---------------- CREATE ----------------
def create_document_client(db: Session, payload: DocumentClientCreateSchema):
    # validations d'existence basiques
    if not db.query(Client.id).filter(Client.id == payload.id_client).first():
        return None, "Client introuvable"
    if not db.query(Document.id_document_base).filter(Document.id_document_base == payload.id_document_base).first():
        return None, "Document de base introuvable"

    # récupérer le nom complet du client pour nom_client
    client_row = db.query(Client).filter(Client.id == payload.id_client).first()
    full_name = None
    if client_row:
        nom = (client_row.nom or "").strip()
        prenom = (client_row.prenom or "").strip()
        full_name = (nom + (" " + prenom if prenom else "")).strip() or None

    doc = DocumentClient(
        id_client=payload.id_client,
        nom_client=full_name,
        id_document_base=payload.id_document_base,
        nom_document=payload.nom_document,
        date_creation=payload.date_creation,
        date_obsolescence=payload.date_obsolescence,
        obsolescence=payload.obsolescence,
    )
    db.add(doc)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        return None, f"Erreur en base: {e}"
    db.refresh(doc)
    return doc, None

# ---------------- READ ----------------
def get_documents_by_client(db: Session, client_id: int):
    return db.query(DocumentClient).filter(DocumentClient.id_client == client_id).all()

def get_document_client(db: Session, doc_client_id: int):
    return db.query(DocumentClient).filter(DocumentClient.id == doc_client_id).first()

# ---------------- DELETE ----------------
def delete_document_client(db: Session, doc_client_id: int):
    doc = db.query(DocumentClient).filter(DocumentClient.id == doc_client_id).first()
    if not doc:
        return None
    db.delete(doc)
    db.commit()
    return doc
