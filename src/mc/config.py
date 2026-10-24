"""Central configuration: model ids, paths, env knobs.

Single source of truth so every story (and the Dockerfile model bake)
references the same identifiers (NFR-6).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Models (NFR-6: baked into the image; fingerprinted into the collection) ---
DENSE_MODEL_ID = "intfloat/multilingual-e5-small"
DENSE_DIM = 384
SPARSE_MODEL_ID = "Qdrant/bm25"
SPARSE_LANGUAGE = "portuguese"

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
