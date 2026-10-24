"""Central configuration: model ids, paths, env knobs.

Single source of truth so every story (and the Dockerfile model bake)
references the same identifiers (NFR-6).
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path | None = None) -> None:
    """Seed ``os.environ`` from a ``.env`` file (operator ergonomics).

    The API key belongs in ``<repo>/.env``, not in a crontab line or a
    wrapper script — cron does not read `.env`, and keys pasted into
    crontabs leak into logs and editor histories. The real environment
    always wins (``setdefault``, never override): an operator export or a
    CI secret beats the file. Malformed lines are skipped, values are never
    logged (NFR-5).
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export ") :]
        key, sep, value = text.partition("=")
        if not sep or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)

# --- Models (NFR-6: baked into the image; fingerprinted into the collection) ---
DENSE_MODEL_ID = "intfloat/multilingual-e5-small"
DENSE_DIM = 384
SPARSE_MODEL_ID = "Qdrant/bm25"
SPARSE_LANGUAGE = "portuguese"

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# `.env` seeds the environment before any knob below reads it — after this
# point every os.environ.get sees the file's values (real env still wins).
_load_dotenv()
DATA_DIR = Path(os.environ.get("MC_DATA_DIR", PROJECT_ROOT / "data"))
STATE_DIR = Path(os.environ.get("MC_STATE_DIR", PROJECT_ROOT / "state"))
QDRANT_DIR = DATA_DIR / "qdrant"
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
META_CACHE_DIR = DATA_DIR / "meta"
QUEUE_DIR = DATA_DIR / "queue"
TRANSCRIPTS_DIR = Path(
    os.environ.get("MC_TRANSCRIPTS_DIR", PROJECT_ROOT / "transcripts")
)

INGEST_DB_PATH = STATE_DIR / "ingest.db"
COLLECTION_NAME = "mc-transcripts-v1"

# --- Qdrant ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Optional API-key auth (defense in depth; OFF by default). Pair with the
# server-side QDRANT__SERVICE__API_KEY — see docker-compose.yml.
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

# --- API ---
API_PORT = int(os.environ.get("API_PORT", "8080"))
FRESHNESS_MAX_HOURS = int(os.environ.get("FRESHNESS_MAX_HOURS", "72"))
SEARCH_QUOTE_BOOST = float(os.environ.get("SEARCH_QUOTE_BOOST", "0.5"))
SEARCH_CACHE_SIZE = int(os.environ.get("SEARCH_CACHE_SIZE", "128"))

# --- Ingestion ---
CHANNELS = tuple(
    line.strip()
    for line in (
        os.environ.get(
            "MC_CHANNELS",
            # No hardcoded default — the channels to ingest live in config.yaml
            # (read first by discover.load_channels) or the MC_CHANNELS env.
            "",
        ).split(",")
    )
    if line.strip()
)
COOKIES_PATH = os.environ.get("MC_COOKIES_PATH", str(PROJECT_ROOT / "cookies.txt"))

# --- YouTube Data API v3 (stats-backfill --from-api, Story 9.3) ---
# A GCP key RESTRICTED to the YouTube Data API v3 (+ server IP) — not a
# cookie: the API path never touches the browser credential (NFR-5) and the
# ban surface is the key's project, never the Google account. Unset → the
# API pass fails loud with an operator-facing message (D18).
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY") or None
