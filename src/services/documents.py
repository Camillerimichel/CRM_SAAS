from sqlalchemy import text
from sqlalchemy.orm import Session
from src.models.document import Document


# Récupérer tous les documents
def get_documents(db: Session):
    documents = db.query(Document).all()
    return [d for d in documents if d is not None]


# Récupérer un document par id
def get_document(db: Session, document_id: int):
    return db.query(Document).filter(Document.id_document_base == document_id).first()


# Ajouter un document
def create_document(
    db: Session,
    documents: str | None = "",
    niveau: str | None = "",
    obsolescence_annees: int = None,
    risque: str | None = None,
):
    documents = documents or ""
    niveau = niveau or ""
    next_id = db.execute(text("SELECT COALESCE(MAX(id_document_base), 0) + 1 FROM Documents")).scalar()
    if next_id is None:
        raise RuntimeError("Impossible de calculer l'identifiant du document de base")
    db.execute(
        text(
            """
            INSERT INTO Documents (id_document_base, documents, niveau, obsolescence__annes, risque)
            VALUES (:id_document_base, :documents, :niveau, :obsolescence_annes, :risque)
            """
        ),
        {
            "id_document_base": int(next_id),
            "documents": documents,
            "niveau": niveau,
            "obsolescence_annes": obsolescence_annees,
            "risque": risque,
        },
    )
    db.commit()
    doc_id = int(next_id)
    try:
        refreshed = db.query(Document).filter(Document.id_document_base == doc_id).first()
        if refreshed is not None:
            return refreshed
    except Exception:
        # Certains schémas réels ne permettent pas toujours un refresh après insert.
        pass
    return db.query(Document).filter(Document.id_document_base == doc_id).first() or Document(
        id_document_base=doc_id,
        documents=documents,
        niveau=niveau,
        obsolescence_annees=obsolescence_annees,
        risque=risque,
    )


def update_document(
    db: Session,
    document_id: int,
    *,
    documents: str | None = None,
    niveau: str | None = None,
    obsolescence_annees: int | None = None,
    risque: str | None = None,
):
    doc = db.query(Document).filter(Document.id_document_base == document_id).first()
    if not doc:
        return None
    if documents is not None:
        doc.documents = documents
    if niveau is not None:
        doc.niveau = niveau
    if obsolescence_annees is not None:
        doc.obsolescence_annees = obsolescence_annees
    else:
        doc.obsolescence_annees = None
    if risque is not None:
        doc.risque = risque
    db.commit()
    try:
        db.refresh(doc)
    except Exception:
        pass
    return doc


def delete_document(db: Session, document_id: int):
    doc = db.query(Document).filter(Document.id_document_base == document_id).first()
    if not doc:
        return None
    db.delete(doc)
    db.commit()
    return doc
