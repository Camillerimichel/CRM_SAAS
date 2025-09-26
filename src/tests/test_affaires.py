from src.services.affaires import ajouter_affaire, lister_affaires

if __name__ == "__main__":
    # Exemple : ajouter une affaire pour le client id=1
    ajouter_affaire(1, "AFF-001", srri=3, frais_courtier=50.0)

    # Lister les 10 premières affaires
    lister_affaires(10)
