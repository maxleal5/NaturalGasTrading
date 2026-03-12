"""Database connection and session management."""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()


def get_db_url() -> str:
    """Build database URL from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "natgas")
    user = os.getenv("POSTGRES_USER", "natgas")
    password = os.getenv("POSTGRES_PASSWORD", "natgas_dev")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine(poolclass=None):
    url = get_db_url()
    kwargs = {"pool_pre_ping": True}
    if poolclass:
        kwargs["poolclass"] = poolclass
    return create_engine(url, **kwargs)


_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Session:
    """Context manager for database sessions."""
    sess = SessionLocal(bind=engine())
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def apply_schema(schema_path: str = None) -> None:
    """Apply SQL schema file to the database."""
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), "../../db/schema.sql")
    schema_path = os.path.abspath(schema_path)
    with open(schema_path) as f:
        sql = f.read()
    with engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()


def refresh_materialized_view() -> None:
    """Refresh the weekly_analysis_master materialized view."""
    with engine().connect() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY weekly_analysis_master"))
        conn.commit()
