"""`mc` command line: every operator task scriptable by cron (Story 0.3).

Subcommands are wired to their story implementations as the epic proceeds;
the skeleton exits non-zero for unknown commands and prints usage.

Story 10.4 (NFR-3): logging is configured here, not left to the library
default. Without a handler every INFO line — including `mc update`'s
"discovery summary" and "dry run: N new videos" — was dropped, so a cron run
that ingested nothing looked exactly like a healthy one.

Story 10.5: `mc health` is the cron freshness gate. It reads the state DB's
last successful ingest (`last_success("update")`), so it fires when the
archive stopped growing — unlike a log-mtime check, which `>>` refreshes on
every run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _configure_logging(level: str = "INFO") -> None:
    """Send library logs to stderr at ``level`` (cron captures stderr).

    ``force=True`` so an operator's pre-existing handler (or a re-run in the
    same process) cannot silently keep the level at WARNING."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _cmd_backfill(args: argparse.Namespace) -> int:
    from mc.ingest.backfill import run_backfill

    return run_backfill(
        from_db=args.from_db,
        batch_size=args.batch_size,
        limit=args.limit,
    )


def _cmd_update(args: argparse.Namespace) -> int:
    from mc.ingest.update import run_update

    return run_update(dry_run=args.dry_run)


def _cmd_push(args: argparse.Namespace) -> int:
    from mc.ingest.push import run_push

    return run_push()


def _cmd_apply(args: argparse.Namespace) -> int:
    from mc.ingest.apply import run_apply

    return run_apply()


def _cmd_stats_backfill(
    args: argparse.Namespace, lock_path: Path | None = None
) -> int:
    from qdrant_client import QdrantClient

    from mc import config
    from mc.config import DATA_DIR, QDRANT_URL
    from mc.ingest.lock import AlreadyRunning, RunLock
    from mc.ingest.stats_backfill import run_stats_backfill
    from mc.qdrant import client_kwargs

    # D18: an unconfigured key must not "succeed" having done nothing — and
    # must not even contend for the lock. Read via the config module at call
    # time so tests (and operator env) can patch it.
    if args.from_api and not config.YOUTUBE_API_KEY:
        print(
            "stats-backfill: --from-api needs YOUTUBE_API_KEY "
            "(a GCP key restricted to the YouTube Data API v3) — "
            "see docs/tasks/deploy-runbook.md"
        )
        return 1

    # The tab hydration pass (--from-tabs) is the longest job in the repo
    # (hours), and even the offline pass rewrites the same state DB — one
    # holder at a time, cron guard convention, like `mc update`. lock_path is
    # injectable like update/apply/backfill's — tests must not contend with a
    # live operator run holding the real lock.
    lock = RunLock(lock_path or DATA_DIR / "stats-backfill.lock")
    try:
        lock.acquire()
    except AlreadyRunning:
        print("stats-backfill: another run holds the lock — refusing to overlap")
        return 1
    try:
        # Index repair is ON by default — that is the point of the command: it also
        # fills the episode/date old points were indexed without, so the result
        # cards show them. --no-index keeps the run from touching Qdrant at all.
        client = (
            None if args.no_index else QdrantClient(url=QDRANT_URL, **client_kwargs(timeout=60))
        )
        # Story 8.1: --from-tabs hydrates exact upload dates straight from the
        # channel tabs (one yt-dlp pass per tab, hours) BEFORE the classic
        # pass, which then mops rows/payloads up from the now-richer cache.
        if args.from_tabs:
            from mc.ingest.stats_backfill import run_tab_backfill

            print(run_tab_backfill(client=client)["summary"])
        # Story 9.4: --from-api is the same pre-pass against the YouTube Data
        # API (minutes, ~150 quota units, no cookies). Quota exhaustion fails
        # LOUD (D18) — silence would hide a misconfigured deployment, and the
        # classic mop-up must not run on top of a dead run (writes are
        # idempotent, so the partial work already committed stays).
        if args.from_api:
            from mc.ingest.stats_backfill import run_api_backfill
            from mc.ingest.youtube_api import QuotaExceededError

            try:
                print(run_api_backfill(client=client)["summary"])
            except QuotaExceededError as exc:
                print(
                    f"stats-backfill: YouTube API quota exhausted — {exc}. "
                    "The quota resets daily and re-runs are idempotent."
                )
                return 1
        report = run_stats_backfill(
            fetch_missing=args.fetch_missing,
            limit=args.limit,
            from_db=args.from_db,
            client=client,
        )
        # summary() has no trailing newline; a plain print terminates the line so
        # the next shell prompt doesn't run into the numbers.
        print(report["summary"])
        # Story 7.3: the temporal sort modes need trustworthy payload dates, so the
        # same command also audits + backfills null-date payloads (cache → row →
        # set_payload). Needs the index client; --no-index keeps the run off Qdrant.
        if client is not None:
            from mc.ingest.date_backfill import run_date_backfill

            print(run_date_backfill(client=client)["summary"])
    finally:
        lock.release()
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    from mc.ingest.reindex import run_reindex

    return run_reindex(model=args.model)


def _cmd_eval(args: argparse.Namespace) -> int:
    from mc.eval import format_report, load_suite, run_eval

    report = run_eval(load_suite(args.suite), limit=args.limit)
    text = format_report(report)
    print(text, end="")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    # Story 5.4 DoD: hybrid must not regress verbatim queries vs sparse-only
    return 0 if report.mrr_hybrid >= report.mrr_sparse else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from mc.config import API_PORT

    uvicorn.run("mc.api.app:app", host="0.0.0.0", port=API_PORT)
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    """Report ingest freshness from the state DB; exit 1 when stale (Story 10.5)."""
    import json

    from mc.api.health import freshness
    from mc.config import INGEST_DB_PATH
    from mc.state import StateStore

    try:
        status, last_update, age_hours = freshness(
            StateStore(INGEST_DB_PATH), args.max_age_hours
        )
    except Exception as exc:  # noqa: BLE001 — an unreadable state DB is stale
        # Read-path failure isolation: never crash the cron job with a
        # traceback; a health probe that cannot read state must fail closed.
        print(f"health: cannot read the ingest state db: {exc}")
        return 1

    print(
        json.dumps(
            {
                "status": status,
                "last_update": last_update,
                "hours_since_update": None if age_hours is None else round(age_hours, 2),
                "max_age_hours": args.max_age_hours,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    from mc.config import FRESHNESS_MAX_HOURS

    parser = argparse.ArgumentParser(
        prog="mc",
        description="Morning Crypto semantic search operations CLI",
    )
    # Story 10.4: the level is a global option so every subcommand logs at
    # INFO by default (the cron log needs the discovery summary). It is read
    # from the environment at parser-build time — a crontab can set
    # MC_LOG_LEVEL=DEBUG without changing the job line.
    parser.add_argument(
        "--log-level",
        default=os.environ.get("MC_LOG_LEVEL", "INFO").upper(),
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log verbosity for every subcommand (env MC_LOG_LEVEL; default INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backfill", help="embed + index all chunks from the legacy DB")
    p.add_argument("--from-db", required=True, help="path to mc_transcripts.db")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_backfill)

    p = sub.add_parser("update", help="daily ingestion: discover→download→chunk→embed→upsert")
    p.add_argument("--dry-run", action="store_true", help="verify cookies, don't ingest")
    p.set_defaults(func=_cmd_update)

    p = sub.add_parser("push", help="ship the daily increment (Mode B)")
    p.set_defaults(func=_cmd_push)

    p = sub.add_parser("apply", help="embed + upsert queued increments (Mode B server)")
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser(
        "stats-backfill",
        help="backfill archive facts for already-indexed videos and repair "
        "their Qdrant payloads (no fetch by default)",
    )
    p.add_argument(
        "--fetch-missing",
        action="store_true",
        help="yt-dlp --dump-single-json for videos still lacking any fact — "
        "a duration or the episode/date of old pre-enrichment videos "
        "(upgrades the cache once; rate-limited, resumable)",
    )
    p.add_argument(
        "--no-index",
        action="store_true",
        help="only write the stats table; do not repair the Qdrant payloads "
        "(the episode/date old points were indexed without)",
    )
    p.add_argument(
        "--from-db",
        type=Path,
        default=None,
        metavar="MC_TRANSCRIPTS_DB",
        help="re-derive every episode from the chunked titles in a legacy "
        "mc_transcripts.db (authoritative and offline; junk episodes are "
        "cleared, rows/caches/payloads corrected)",
    )
    # D19: one fact source per run — the two passes would interleave two
    # writers on the same cache/rows for zero benefit. --from-tabs is
    # retained, superseded, as the offline/cookie fallback.
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--from-tabs",
        action="store_true",
        help="hydrate exact upload dates straight from the channel tabs: one "
        "uncapped yt-dlp full-extraction pass per /videos /streams /shorts "
        "tab (hours; randomized 3-8 s pacing; resumable/idempotent), then "
        "the classic pass aligns rows/payloads from the richer cache "
        "(offline/cookie fallback — prefer --from-api)",
    )
    source.add_argument(
        "--from-api",
        action="store_true",
        help="hydrate exact upload dates/durations/titles from the YouTube "
        "Data API v3 (needs YOUTUBE_API_KEY; minutes, ~150 quota units of "
        "the free 10,000/day, NO cookies and no bot gate; idempotent — a "
        "re-run over complete rows costs 0 units), then the classic pass "
        "aligns rows/payloads from the richer cache",
    )
    p.add_argument("--limit", type=int, default=None, help="cap videos considered this run")
    p.set_defaults(func=_cmd_stats_backfill)

    p = sub.add_parser("reindex", help="rebuild into a new collection and swap alias")
    p.add_argument("--model", required=True, help="dense model id to reindex with")
    p.set_defaults(func=_cmd_reindex)

    p = sub.add_parser("eval", help="sparse-only vs hybrid MRR on known quotes")
    p.add_argument("--suite", required=True, help="path to quotes.yaml")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default=None, help="write the markdown report here")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="run the search API")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser(
        "health",
        help="exit 1 when the last successful ingest is older than the window",
    )
    p.add_argument(
        "--max-age-hours",
        type=int,
        default=FRESHNESS_MAX_HOURS,
        help=f"staleness window in hours (env FRESHNESS_MAX_HOURS; default {FRESHNESS_MAX_HOURS})",
    )
    p.set_defaults(func=_cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
