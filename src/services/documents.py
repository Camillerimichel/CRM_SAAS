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
def create_document(db: Session, documents: str, niveau: str, obsolescence_annees: int = None, risque: str = None):
    document = Document(
        documents=documents,
        niveau=niveau,
        obsolescence_annees=obsolescence_annees,
        risque=risque,
    )
    db.add(document)
    db.commit()
    db.refresh(document)  # recharge l'objet pour éviter l'erreur de sérialisation
    return document
