from src.services.reporting import get_all_affaires, get_top_clients

if __name__ == "__main__":
    print("\n=== REPORTING AFFAIRE ===")

    # Lister toutes les affaires
    get_all_affaires(None)

    # Lister les top clients
    get_top_clients(None)
