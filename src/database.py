from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# URL de connexion SQLite vers le fichier Base.sqlite à la racine du projet
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/Base.sqlite"

# Pour SQLite uniquement : check_same_thread doit être désactivé
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Factory de sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles ORM
Base = declarative_base()

# Dépendance FastAPI : ouverture/fermeture automatique de la session
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
