from src.services.supports import (
    ajouter_support,
    lister_supports,
    modifier_support,
    supprimer_support,
)

if __name__ == "__main__":
    # CREATE
    ajouter_support("TEST123456", "SupportTest", srri=3, promoteur="TestPromoteur")

    # READ
    print("\nSupports après ajout :")
    lister_supports(5)

    # UPDATE (exemple avec un support existant)
    print("\nMise à jour du support id=1 (exemple) :")
    modifier_support(1, nom="SupportModif", promoteur="PromoteurModif")

    # DELETE (exemple avec un support existant)
    print("\nSuppression du support id=1 (exemple) :")
    supprimer_support(1)

    # READ
    print("\nSupports après suppression :")
    lister_supports(5)
