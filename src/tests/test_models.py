from sqlalchemy import create_engine, inspect, text
import pandas as pd
from src.models.client import Client
from src.models.affaire import Affaire
from src.models.support import Support
from src.models.document import Document
from src.models.document_client import DocumentClient
from src.models.historique_personne import HistoriquePersonne
from src.models.historique_affaire import HistoriqueAffaire
from src.models.historique_support import HistoriqueSupport
from src.models.allocation import Allocation

# Connexion à la base SQLite
engine = create_engine("sqlite:///data/Base.sqlite")

# Fonction pour afficher infos + aperçu
def show_table_info(model_class):
    inspector = inspect(model_class)
    table_name = model_class.__tablename__

    print(f"\n=== Table: {table_name} ===")
    print("Colonnes :", [c.name for c in inspector.columns])

    # Nombre de lignes
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()
        print("Nombre de lignes :", count)

        # Aperçu des 5 premières lignes si table non vide
        if count > 0:
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5;", conn)
            print("Aperçu des 5 premières lignes :")
            print(df)

if __name__ == "__main__":
    show_table_info(Client)
    show_table_info(Affaire)
    show_table_info(Support)
    show_table_info(Document)
    show_table_info(DocumentClient)
    show_table_info(HistoriquePersonne)
    show_table_info(HistoriqueAffaire)
    show_table_info(HistoriqueSupport)
    show_table_info(Allocation)
