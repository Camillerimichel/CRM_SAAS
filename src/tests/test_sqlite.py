import pandas as pd
from sqlalchemy import create_engine

# Connexion à la base SQLite
engine = create_engine("sqlite:///data/Base.sqlite")

def test_clients():
    # Charger 5 premiers clients
    df = pd.read_sql("SELECT * FROM mariadb_clients LIMIT 5;", engine)
    print(df)

if __name__ == "__main__":
    test_clients()
