"""Ingestion state store: durable bookkeeping in state/ingest.db (Story 1.5).

Short-lived connections keep contention low: WAL mode + busy_timeout so a
cron run never blocks the API, which opens one per request for the
freshness line. Locking is transient and never destructive (review fix).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    video_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    ts TEXT NOT NULL,
    PRIMARY KEY (video_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Story 6.2: per-video archive facts feeding the landing-page "status do
-- acervo". Added to _SCHEMA so every existing state DB auto-migrates on the
-- next open. episode is INTEGER so MAX() is numeric, never lexical.
CREATE TABLE IF NOT EXISTS video_stats (
    video_id TEXT PRIMARY KEY,
    duration INTEGER,   -- total video length in seconds; NULL when unknown
    episode INTEGER,    -- numeric episode; NULL when unparseable
    date TEXT           -- ISO yyyy-mm-dd upload date; NULL when unknown
);
"""

_BUSY_TIMEOUT_MS = 5000


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._open()

    def _open(self) -> None:
        try:
            self._conn = self._connect()
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.OperationalError:
            # A locked/busy db is transient contention, not corruption —
            # recreating here would wipe all bookkeeping (review fix).
            raise
        except sqlite3.DatabaseError as exc:
            # Only a genuinely unreadable file ("not a database") warrants a
            # wipe; any other DatabaseError is raised so a transient fault
            # can't silently destroy the bookkeeping. The corrupt file is
            # preserved (renamed) rather than deleted, so it can be inspected.
            if "not a database" not in str(exc):
                raise
            logger.warning(
                "state db %s is not a valid database — preserving it and recreating from scratch",
                self.path,
            )
            self.path.replace(self.path.with_suffix(".corrupt"))
            self._conn = self._connect()
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    # --- processed videos ---------------------------------------------------
    def mark_processed(self, video_id: str, run_type: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO processed(video_id, run_type, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET run_type=excluded.run_type, "
                "ts=excluded.ts",
                (video_id, run_type, _now_iso()),
            )

    def is_processed(self, video_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row is not None

    def count_processed(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM processed").fetchone()[0]

    def processed_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT video_id FROM processed").fetchall()
        return [row[0] for row in rows]

    # --- archive stats (Story 6.2) -------------------------------------------
    def record_video_stats(
        self,
        video_id: str,
        *,
        duration: int | None = None,
        episode: int | None = None,
        date: str | None = None,
    ) -> None:
        """Idempotent per-video upsert; **non-null wins** on every column.

        Re-ingests upgrade rows and partial data never wipes a known value
        (``COALESCE(excluded.x, x)``), so every writer is safe to re-run. An
        all-None call still inserts the row — it marks the video "seen, stats
        unknown" so COUNT(*) stays honest.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO video_stats(video_id, duration, episode, date) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET "
                "duration=COALESCE(excluded.duration, video_stats.duration), "
                "episode=COALESCE(excluded.episode, video_stats.episode), "
                "date=COALESCE(excluded.date, video_stats.date)",
                (video_id, duration, episode, date),
            )

    def set_video_episode(self, video_id: str, episode: int | None) -> None:
        """Authoritative episode write — ``None`` CLEARS the stored value.

        ``record_video_stats`` deliberately never erases (COALESCE), which is
        right for ingest but wrong for repair: an episode known to be junk
        (parsed from a placeholder title or a bare digit run) must be replaced
        by the title-derived truth, including None. A missing row is a no-op —
        the record path creates rows; repair only touches existing ones.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE video_stats SET episode = ? WHERE video_id = ?",
                (episode, video_id),
            )

    def archive_stats(self) -> dict:
        """One cheap sqlite read of the archive numbers for the landing page.

        ``COUNT(*)`` videos, ``SUM(duration)`` seconds (NULL-ignoring, so
        ``None`` when no duration is known), and the (episode, date) pair of
        the highest known episode (numeric ORDER BY).
        """
        total = self._conn.execute("SELECT COUNT(*), SUM(duration) FROM video_stats").fetchone()
        last = self._conn.execute(
            "SELECT episode, date FROM video_stats "
            "WHERE episode IS NOT NULL ORDER BY episode DESC LIMIT 1"
        ).fetchone()
        return {
            "videos": total[0] or 0,
            "seconds": total[1],
            "last_episode": last[0] if last else None,
            "last_episode_date": last[1] if last else None,
        }

    def video_facts(self, video_id: str) -> dict:
        """Every stored archive fact for one video (Story 6.4 backfill): the
        row's ``duration``/``episode``/``date``, each ``None`` when still
        unknown — and all three ``None`` when the video has no row at all.
        Lets the backfill tell which fields are still missing, so a row that
        already has a duration still gains its episode/date."""
        row = self._conn.execute(
            "SELECT duration, episode, date FROM video_stats WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if row is None:
            return {"duration": None, "episode": None, "date": None}
        return {"duration": row[0], "episode": row[1], "date": row[2]}

    def video_duration(self, video_id: str) -> int | None:
        """The stored duration for one video (Story 6.4 backfill resumability):
        None when the video is absent or its duration is still unknown."""
        return self.video_facts(video_id)["duration"]

    # --- run success bookkeeping (NFR-4) -------------------------------------
    def record_success(self, run_type: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (f"last_success:{run_type}", _now_iso()),
            )

    def last_success(self, run_type: str = "update") -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (f"last_success:{run_type}",)
        ).fetchone()
        return row[0] if row else None

    # --- generic meta ---------------------------------------------------------
    def record_key(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_key(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # --- diagnostics ------------------------------------------------------------
    def journal_mode(self) -> str:
        return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    def busy_timeout_ms(self) -> int:
        return self._conn.execute("PRAGMA busy_timeout").fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
