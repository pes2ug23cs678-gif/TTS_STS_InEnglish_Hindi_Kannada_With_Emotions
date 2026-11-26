from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend_api.config import DATABASE_URL

# SQLite (the local-dev fallback) disallows sharing a connection across threads by default,
# which FastAPI/uvicorn does; this connect arg relaxes that. Ignored for PostgreSQL etc.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
