import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Connexion MySQL (tunnel SSH actif sur 127.0.0.1:3306)
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # Mot de passe contient '@@' → encodé en %40%40
    "mysql+pymysql://crm_user:Veduta1789%40%40@127.0.0.1:3306/MariaDB_CRM_SAAS?charset=utf8mb4",
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
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
