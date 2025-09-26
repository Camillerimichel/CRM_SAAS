from src.services.documents import (
    ajouter_document,
    lister_documents,
    modifier_document,
    supprimer_document,
)
from src.services.document_client import (
    lier_document_client,
    lister_documents_client,
    supprimer_document_client,
)
from datetime import datetime

if __name__ == "__main__":
    print("\n=== TEST DOCUMENTS ===")

    # CREATE
    ajouter_document("Carte identité", "Obligatoire", obsolescence=5, risque="faible")

    # READ
    print("\nDocuments après ajout :")
    lister_documents(5)

    # UPDATE
    print("\nMise à jour du document id=1 (exemple) :")
    modifier_document(1, documents="Passeport", risque="moyen")

    # DELETE
    print("\nSuppression du document id=1 (exemple) :")
    supprimer_document(1)

    print("\n=== TEST DOCUMENTS CLIENTS ===")

    # Lier document à un client existant
    lier_document_client(
        id_client=314,              # client existant
        id_document_base=2,         # document existant
        nom_document="Justificatif domicile",
        date_creation=datetime(2020, 1, 1),
        date_obsolescence=datetime(2025, 1, 1),
        obsolescence="valide"
    )

    # Lister les documents du client
    lister_documents_client(314)

    # Supprimer le lien
    supprimer_document_client(314, 2)

    # Vérifier suppression
    lister_documents_client(314)
