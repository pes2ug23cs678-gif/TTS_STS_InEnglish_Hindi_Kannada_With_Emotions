import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load configuration from the project-local .env file (real values live in .env, which is
# not committed; .env.example documents the keys). Also fall back to a .env found by walking
# up from the current working directory, so it works regardless of where uvicorn is launched.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # default search from CWD upward; does nothing if none found

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")

# ---- Sensible local-dev defaults so the backend runs out-of-the-box in VS Code ----
# No DATABASE_URL? Fall back to a local SQLite file instead of crashing. Set DATABASE_URL
# in .env (e.g. a PostgreSQL URL) for anything beyond local development.
if not DATABASE_URL:
    sqlite_path = PROJECT_ROOT / "sts_dev.db"
    DATABASE_URL = f"sqlite:///{sqlite_path}"
    logger.warning(
        "DATABASE_URL not set; using a local SQLite database at %s. "
        "Set DATABASE_URL in .env (e.g. a PostgreSQL URL) for production.", sqlite_path)

# No SECRET_KEY? Generate an ephemeral one so auth still works this run. It changes on every
# restart (so logins won't persist across restarts) — set a fixed SECRET_KEY in .env to keep
# tokens valid. Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning(
        "SECRET_KEY not set; generated a temporary one for this run only. "
        "Set a fixed SECRET_KEY in .env so login tokens survive restarts.")
