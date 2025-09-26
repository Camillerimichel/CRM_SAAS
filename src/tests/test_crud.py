from src.services.clients import (
    create_client,
    get_clients,
    update_client,
    delete_client,
)
from src.services.affaires import create_affaire, get_affaires


if __name__ == "__main__":
    print("\n=== TEST CRUD CLIENTS ===")

    # CREATE
    # create_client(db, client)  # Adapter selon la signature réelle

    # READ
    print("\nClients après ajout :")
    get_clients(None)  # Adapter selon la signature réelle

    # UPDATE
    print("\nMise à jour du client id=1 (exemple) :")
    update_client(None, 1, None)  # Adapter selon la signature réelle

    # DELETE
    print("\nSuppression du client id=1 (exemple) :")
    delete_client(None, 1)  # Adapter selon la signature réelle

    # READ
    print("\nClients après suppression :")
    get_clients(None)  # Adapter selon la signature réelle

    print("\n=== TEST CRUD AFFAIRES ===")

    # CREATE
    create_affaire(None, 314, "AFF-TEST", srri=3)  # Adapter selon la signature réelle

    # READ
    print("\nAffaires après ajout :")
    get_affaires(None)  # Adapter selon la signature réelle

    # UPDATE
    print("\nMise à jour de l'affaire id=1 (exemple) :")
    # update_affaire(None, 1, ...)  # À ajouter si existe

    # DELETE
    print("\nSuppression de l'affaire id=1 (exemple) :")
    # delete_affaire(None, 1)  # À ajouter si existe

    # READ
    print("\nAffaires après suppression :")
    get_affaires(None)  # Adapter selon la signature réelle
