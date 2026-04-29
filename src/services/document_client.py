from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from src.models.document_client import DocumentClient
from src.models.client import Client
from src.models.document import Document
from src.schemas.document_client import DocumentClientCreateSchema


def canonical_document_label(label: str | None) -> str | None:
    raw = (label or "").strip()
    if not raw:
        return raw
    try:
        import unicodedata as _ud

        norm = _ud.normalize("NFD", raw).encode("ascii", "ignore").decode("ascii").lower()
        norm = " ".join(norm.split())
    except Exception:
        norm = raw.lower()
    if norm in {"synthese", "synthese client", "rapport de synthese", "rapport synthese"}:
        return "Synthèse"
    if norm in {
        "rapport kyc",
        "rapport de kyc",
        "compte rendu dentretien",
        "compte rendu d entretien",
        "compte rendu entretien",
        "kyc",
    } or "kyc" in norm:
        return "Compte rendu d'entretien"
    return raw


def _document_client_payload(payload: DocumentClientCreateSchema, doc_id: int, full_name: str | None):
    return {
        "id": doc_id,
        "id_client": payload.id_client,
        "nom_client": full_name,
        "id_document_base": payload.id_document_base,
        "nom_document": payload.nom_document,
        "date_creation": payload.date_creation,
        "date_obsolescence": payload.date_obsolescence,
        "obsolescence": payload.obsolescence,
        "stored_filename": payload.stored_filename,
        "stored_path": payload.stored_path,
        "mime_type": payload.mime_type,
        "file_size": payload.file_size,
    }
def ensure_document_client_storage_columns(db: Session) -> bool:
    table = "Documents_client"
    desired = {
        "stored_filename": "TEXT",
        "stored_path": "TEXT",
        "mime_type": "TEXT",
        "file_size": "INTEGER",
    }
    existing: set[str] = set()
    try:
        rows = db.execute(text(f"PRAGMA table_info({table})")).fetchall()
        existing = {row[1] for row in rows}
    except Exception:
        pass
    if not existing:
        try:
            rows = db.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE()
                      AND table_name = :t
                    """
                ),
                {"t": table},
            ).fetchall()
            existing = {row[0] for row in rows}
        except Exception:
            existing = set()
    missing = [name for name in desired if name not in existing]
    added_any = False
    for name in missing:
        try:
            db.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {desired[name]}"))
            existing.add(name)
            added_any = True
        except Exception:
            continue
    if added_any:
        db.commit()
    return all(name in existing for name in desired)


def ensure_document_base_for_label(db: Session, label: str, *, niveau: str = "généré", risque: str | None = None) -> int:
    canonical_label = canonical_document_label(label) or (label or "").strip()
    alias_labels = {canonical_label}
    if canonical_label == "Compte rendu d'entretien":
        alias_labels.update({"Rapport KYC", "Rapport de KYC", "Compte rendu entretien"})
    elif canonical_label == "Synthèse":
        alias_labels.update({"Synthese", "Synthèse"})

    existing_rows = (
        db.query(Document)
        .filter(Document.documents.in_(sorted(alias_labels)))
        .order_by(Document.id_document_base.asc())
        .all()
    )
    if existing_rows:
        primary = existing_rows[0]
        changed = False
        for row in existing_rows:
            if row.documents != canonical_label:
                row.documents = canonical_label
                changed = True
        if changed:
            db.commit()
        if getattr(primary, "id_document_base", None) is not None:
            return int(primary.id_document_base)
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
            "documents": canonical_label,
            "niveau": niveau,
            "obsolescence_annes": None,
            "risque": risque,
        },
    )
    db.commit()
    refreshed = db.query(Document).filter(Document.id_document_base == int(next_id)).first()
    if refreshed and getattr(refreshed, "id_document_base", None) is not None:
        return int(refreshed.id_document_base)
    return int(next_id)

# ---------------- CREATE ----------------
def create_document_client(db: Session, payload: DocumentClientCreateSchema):
    # validations d'existence basiques
    if not db.query(Client.id).filter(Client.id == payload.id_client).first():
        return None, "Client introuvable"
    if payload.id_document_base is None:
        return None, "Document de base introuvable"
    if not db.query(Document.id_document_base).filter(Document.id_document_base == payload.id_document_base).first():
        # Certains schémas expirent ou décalent la session après insertion; on garde l'ID fourni.
        try:
            db.execute(text("SELECT 1 FROM Documents WHERE id_document_base = :id LIMIT 1"), {"id": payload.id_document_base}).fetchone()
        except Exception:
            pass

    # récupérer le nom complet du client pour nom_client
    client_row = db.query(Client).filter(Client.id == payload.id_client).first()
    full_name = None
    if client_row:
        nom = (client_row.nom or "").strip()
        prenom = (client_row.prenom or "").strip()
        full_name = (nom + (" " + prenom if prenom else "")).strip() or None

    # Pour les documents générés, on réécrit l'entrée existante du même type
    # au lieu d'empiler des doublons dans Documents_client.
    try:
        if payload.obsolescence == "généré" and (payload.stored_path or "").startswith("generated_clients/"):
            existing_doc = (
                db.query(DocumentClient)
                .filter(
                    DocumentClient.id_client == payload.id_client,
                    DocumentClient.id_document_base == payload.id_document_base,
                    DocumentClient.obsolescence == "généré",
                )
                .order_by(DocumentClient.id.desc())
                .first()
            )
            if existing_doc:
                existing_doc.nom_client = full_name
                existing_doc.nom_document = payload.nom_document
                existing_doc.date_creation = payload.date_creation
                existing_doc.date_obsolescence = payload.date_obsolescence
                existing_doc.obsolescence = payload.obsolescence
                existing_doc.stored_filename = payload.stored_filename
                existing_doc.stored_path = payload.stored_path
                existing_doc.mime_type = payload.mime_type
                existing_doc.file_size = payload.file_size
                db.commit()
                return _document_client_payload(payload, int(existing_doc.id), full_name), None
    except Exception:
        pass

    # Empêche les doubles inserts accidentels lors d'une double soumission quasi simultanée
    # ou d'un rafraîchissement rapide après génération.
    try:
        if payload.obsolescence == "généré" and (payload.stored_path or "").startswith("generated_clients/"):
            recent_threshold = datetime.utcnow() - timedelta(seconds=15)
            recent_filters = [
                DocumentClient.id_client == payload.id_client,
                DocumentClient.id_document_base == payload.id_document_base,
                DocumentClient.obsolescence == "généré",
                DocumentClient.date_creation.isnot(None),
            ]
            if payload.nom_document is not None:
                recent_filters.append(DocumentClient.nom_document == payload.nom_document)
            recent_doc = (
                db.query(DocumentClient)
                .filter(*recent_filters)
                .order_by(DocumentClient.id.desc())
                .first()
            )
            if recent_doc and getattr(recent_doc, "date_creation", None) and recent_doc.date_creation >= recent_threshold:
                return _document_client_payload(payload, int(recent_doc.id), full_name), None
    except Exception:
        pass

    doc_id = db.execute(text("SELECT COALESCE(MAX(id), 0) + 1 FROM Documents_client")).scalar()
    if doc_id is None:
        return None, "Impossible de calculer l'identifiant du document client"
    try:
        db.execute(
            text(
                """
                INSERT INTO Documents_client
                (id, id_client, nom_client, id_document_base, nom_document, date_creation, date_obsolescence, obsolescence, stored_filename, stored_path, mime_type, file_size)
                VALUES
                (:id, :id_client, :nom_client, :id_document_base, :nom_document, :date_creation, :date_obsolescence, :obsolescence, :stored_filename, :stored_path, :mime_type, :file_size)
                """
            ),
            {
                "id": int(doc_id),
                "id_client": payload.id_client,
                "nom_client": full_name,
                "id_document_base": payload.id_document_base,
                "nom_document": payload.nom_document,
                "date_creation": payload.date_creation,
                "date_obsolescence": payload.date_obsolescence,
                "obsolescence": payload.obsolescence,
                "stored_filename": payload.stored_filename,
                "stored_path": payload.stored_path,
                "mime_type": payload.mime_type,
                "file_size": payload.file_size,
            },
        )
        db.commit()
    except Exception as e:
        db.rollback()
        return None, f"Erreur en base: {e}"
    try:
        refreshed = db.query(DocumentClient).filter(DocumentClient.id == int(doc_id)).first()
        if refreshed is not None:
            return _document_client_payload(payload, int(doc_id), full_name), None
    except Exception:
        pass
    return _document_client_payload(payload, int(doc_id), full_name), None

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
