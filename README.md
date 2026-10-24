# mc-semantic-search

Semantic search over the *Morning Crypto* transcript archive. Replaces the
SearXNG + SQLite FTS5 keyword pipeline with a Qdrant-backed hybrid
(dense + sparse) search behind a FastAPI + htmx UI.
Portuguese-first: users reach the **exact moment** an idea was said, even
when their wording paraphrases the transcript.

## Key features

- Hybrid search — dense `multilingual-e5-small` (384-dim) + sparse BM25, fused
  natively by Qdrant (RRF)
- Quote-aware ranking boost and highlighted snippets in result fragments
- Agent-friendly surface — `?render=json` envelope and a `/llms.txt` manual,
  advertised on the landing page itself (moltbook pattern)
- Archive status — a "status do acervo" block (last episode, video count,
  total duration) maintained by ingest and read by both the page and agents
- Two ingestion modes: **Mode A** (server self-update via cron) and **Mode B**
  (local scrape + push, fallback when the server IP gets bot-blocked)
- Ingestion bookkeeping in SQLite (`state/ingest.db`) — never on the search path
- Nightly Qdrant snapshots with rotation; MRR eval harness as a cutover gate
- Hardened defaults: loopback-only Qdrant, security headers, input caps,
  non-root container, credential guard

## Table of contents

- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting started (dev)](#getting-started-dev)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [API reference](#api-reference)
- [Operations](#operations)
- [Security](#security)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Further docs](#further-docs)

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (dev on 3.12), all deps pinned in `pyproject.toml` |
| Vector DB / search | Qdrant v1.12.4 (native hybrid RRF) |
| Dense embeddings | `intfloat/multilingual-e5-small` (384-dim, CPU) via `sentence-transformers` |
| Sparse vectors | `fastembed` BM25-style (`Qdrant/bm25`, Portuguese) |
| API | FastAPI + uvicorn |
| Frontend | Jinja2 + htmx + Alpine + Pico CSS (no Node/npm build) |
| Ingestion | yt-dlp (channel discovery + subtitle download) |
| Bookkeeping | SQLite `state/ingest.db` |
| Tooling | uv (venv), pytest + pytest-bdd, ruff, Docker Compose |

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip)
- Docker + Docker Compose v2 (runs Qdrant and the API container)
- `rsync` on the scraping box (Mode B only)
- A YouTube cookies file (needed for ingestion; see [Configuration](#configuration))

## Getting started (dev)

```bash
# 1. Clone and enter the repo
git clone <repo-url> mc-semantic-search && cd mc-semantic-search

# 2. Create the venv and install (dev tools — pytest, ruff — are regular deps)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# 3. Wire the git hooks (credential guard — rejects staged cookies files)
make hooks

# 4. Start qdrant + api (local ≈ production; api is mapped to host port 18080)
make up

# 5. Check it is alive
curl -s localhost:18080/health | jq .status   # "ok"

# 6. Open the UI (or use the API directly — see API reference)
xdg-open http://localhost:18080
```

By default the index is empty. Bootstrap it from the legacy SQLite DB
(~108k chunks, hours on CPU — run detached and tail the log):

```bash
.venv/bin/python -m mc.cli backfill --from-db extras/mc_transcripts.db \
  >> /var/log/mc-backfill.log 2>&1 &
```

Alternative: run the API from the local venv instead of the container
(it talks to the compose qdrant on `localhost:6333`):

```bash
make dev    # uvicorn mc.api.app:app on :8080
```

## Architecture

### Directory structure

```
src/mc/
├── api/            # FastAPI app, Jinja2 templates, static assets (htmx/alpine/pico)
├── ingest/
│   ├── discover.py     # channel listing via yt-dlp --dump-single-json
│   ├── download.py     # subtitle download (yt-dlp Python API), cookies, ignore list
│   ├── chunker.py      # transcript → N-segment chunks
│   ├── metadata.py     # episode/date/channel enrichment (yt-dlp, cached in data/meta)
│   ├── update.py       # daily orchestrator: discover→download→chunk→embed→upsert→stage
│   ├── push.py         # Mode B: rsync the staged increment to the server
│   ├── apply.py        # Mode B server: validate + embed + upsert queued increments
│   ├── backfill.py     # bootstrap from the legacy SQLite DB
│   ├── reindex.py      # rebuild into a new collection, swap alias atomically
│   ├── legacy.py       # legacy DB reader (read-only, quarantines bad rows)
│   ├── crosscheck.py   # transcript/DB consistency check
│   ├── lock.py         # flock-based RunLock (cron overlap guard)
│   └── models.py       # Chunk / report dataclasses
├── search/
│   ├── engine.py       # hybrid search: dense+sparse prefetch → RRF → results
│   ├── query.py        # query parsing/normalization (Portuguese-first, quoted phrases)
│   ├── ranking.py      # quote-aware boost (demotes non-literal matches)
│   ├── snippets.py     # <mark> snippets (HTML-escaped before marking)
│   └── cache.py        # query-embedding LRU cache (embedder starvation guard)
├── state.py        # SQLite bookkeeping: processed chunks + meta sentinel
├── qdrant.py       # client helpers, collection fingerprinting, upsert
├── embeddings.py   # dense + sparse embedders (global one-at-a-time lock)
├── eval.py         # MRR harness: sparse-only vs hybrid on known quotes
├── backup.py       # Qdrant snapshot create/restore + rotation
├── bake_models.py  # downloads model weights at image build time
├── config.py       # single source of truth: model ids, paths, env knobs
└── cli.py          # `mc` subcommands
```

### Request lifecycle (search)

1. Browser loads `GET /` (Jinja2 + htmx); typing fires `GET /results?query=…`
   on a debounce — the HTML fragment replaces the results list.
2. The endpoint caps input (`query` ≤ 512 chars, `limit` ≤ 50) and consults
   the embedding cache (LRU keyed on the normalized query).
3. On a cache miss the engine runs **both** embedders (dense + sparse) inside
   a one-at-a-time executor, then issues a Qdrant hybrid query (dense +
   sparse prefetch → RRF fusion) against the aliased collection.
4. Results pass the quote-aware boost and the snippet highlighter; each
   result is sanitized (https-only URLs, strict video-id validation) and
   rendered.

### Data flow (ingestion)

```
Mode A (server):   cron → mc update → yt-dlp discover/download → chunk → embed
                        → upsert into qdrant (collection mc-transcripts-v1) → stage increment
                        → data/queue/pending.jsonl

Mode B (fallback): local (residential IP) → mc update → chunk → stage
                        → data/queue/pending.jsonl → mc push (rsync)
                        → server data/queue/*.jsonl → mc apply → validate → embed → upsert
```

`mc update` upserts directly **and** stages the same increment, so Mode B
ships exactly what Mode A indexes. `mc apply` dedupes by content hash;
failed increments are quarantined under `data/queue/failed/` with a reason.

### State

SQLite (`state/ingest.db`, configurable via `MC_STATE_DIR`) holds three
tables: `processed` (which chunks/videos are already indexed — checkpoints
per video), `meta` (a sentinel recording run bookkeeping) and `video_stats`
(per-video duration/episode/date feeding the landing-page "status do acervo").
It is never consulted on the search path; only `/health`, the ingestion flow
and the landing page read it.

**Archive status.** The landing page's "status do acervo" block (last episode
loaded, videos catalogued, total duration in s/min/h/days) reads the
`video_stats` aggregates at render time — the same numbers are exposed to
agents as `archive_status` in the `?render=json` envelope and in `/llms.txt`.
`mc update` (Mode A) and `mc apply` (Mode B) maintain the rows as a side
effect of runs that already exist; the one-off `mc stats-backfill` bootstraps
the existing archive from the local transcripts cache (no yt-dlp fetch by
default — see the deploy runbook). That command also **repairs the Qdrant
payloads**: points indexed before the episode/date enrichment existed show
`ep —` / `sem data` in results, and the backfill writes the missing
episode/date/duration back into the payload — payload-only, no re-embedding,
only where the field is empty (`--no-index` skips it). With `--fetch-missing`,
a video missing episode/date is fetched once via yt-dlp even when its duration
is already known — that is what old pre-enrichment points need. An empty or
unreadable table simply hides the block; the page never breaks on it.

**Exact dates at scale (`--from-api`).** The temporal sort
(`?sort=recent|oldest`) needs an exact upload date per video, and yt-dlp
scraping died to the bot gate in production (a ~12 h run wrote zero rows).
`mc stats-backfill --from-api` gets the three facts the archive lacks
(exact `upload_date`, `duration`, `title` ⇒ episode) from the YouTube Data
API v3 instead: each configured channel's uploads playlist is walked (the
superset of /videos /streams /shorts) and unioned with the indexed
universe, the whole backlog costs ~150 of the 10,000 free daily quota
units, **no cookies are involved**, and the ban surface is the key's GCP
project — never the Google account. Writes reuse the same idempotent
doctrine (cache merge never regresses, rows are COALESCE, payloads
fill-empty), so a re-run over complete rows costs 0 quota units. Missing
key and exhausted quota fail loud with non-zero exit; key setup is in the
deploy runbook.

## Configuration

All knobs live in `src/mc/config.py` and are read from the environment at
import time. Model ids and the collection name are constants (single source
of truth; fingerprinted into the collection).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_API_KEY` | unset (auth off) | API key sent to Qdrant; set only together with the server-side `QDRANT__SERVICE__API_KEY` (see docker-compose.yml) |
| `API_PORT` | `8080` | Port for `mc serve` |
| `FRESHNESS_MAX_HOURS` | `72` | `/health` reports `stale` when the last successful ingest is older than this |
| `SEARCH_QUOTE_BOOST` | `0.5` | Score demotion factor for non-literal matches |
| `SEARCH_CACHE_SIZE` | `128` | Query-embedding LRU size |
| `MC_CHANNELS` | *(unset; see `config.yaml`)* | Comma-separated YouTube channel URLs to ingest; `config.yaml` lists them, this env overrides |
| `MC_COOKIES_PATH` | `<repo>/cookies.txt` | YouTube cookies file — **live credentials**: keep mode 0600, never commit |
| `MC_DATA_DIR` | `<repo>/data` | Root for metadata cache, queue, qdrant storage |
| `MC_STATE_DIR` | `<repo>/state` | SQLite ingest bookkeeping |
| `MC_TRANSCRIPTS_DIR` | `<repo>/transcripts` | Downloaded transcripts |
| `MC_PUSH_DEST` | unset | rsync destination for `mc push` (Mode B), e.g. `server:/opt/mc/data/queue/` — must be the directory server-side `mc apply` watches |
| `NOTIFY_WEBHOOK` | unset | Telegram-compatible webhook URL for cron failure alerts |
| `YOUTUBE_API_KEY` | unset | GCP key **restricted to the YouTube Data API v3** for `mc stats-backfill --from-api` — no cookies involved; setup, restriction and quota math in the deploy runbook |
| `MC_COMPOSE_TESTS` | unset | Set to `1` to run the compose smoke tests |

Constants (not env-tunable): dense model `intfloat/multilingual-e5-small`
(384-dim), sparse model `Qdrant/bm25` (Portuguese), collection
`mc-transcripts-v1` (searched by name; `mc reindex` swaps the alias
`mc-transcripts-v1-alias` to a freshly built collection).

### Channels (`config.yaml`)

The channels `mc update` ingests live in `config.yaml` (committed; public
URLs, not credentials). `discover.load_channels()` reads it first, then falls
back to the `MC_CHANNELS` env var. Each channel is listed across its
`/videos`, `/streams` and `/shorts` tabs and the video ids are unioned —
Morning Crypto episodes are live streams that only appear under `/streams`:

```yaml
channels:
  - "https://www.youtube.com/channel/UC…"
```

The channel list is configuration, never hardcoded in code (enforced by
`tests/test_discover.py::test_channels_come_from_config_not_hardcoded`).

### Cookies (ingestion credentials)

Subtitle downloads need YouTube cookies (bot detection). Export them to
`cookies.txt` (path above), `chmod 600` it, and treat it as a live
credential: it is never copied into the Docker image and never logged — the
container path, if used, is a **read-only** bind mount. Rotation and
stale-cookie detection are documented in
[docs/tasks/cookies.md](docs/tasks/cookies.md). Verify cookies without
ingesting:

```bash
mc update --dry-run
```

## CLI reference

| Command | What it does |
|---|---|
| `mc backfill --from-db PATH [--batch-size 256] [--limit N]` | Embed + index all unprocessed chunks from the legacy SQLite DB (bootstrap; hours on CPU) |
| `mc update [--dry-run]` | One daily pass: discover → download → chunk → embed → upsert → stage increment. `--dry-run` verifies cookies only |
| `mc push` | rsync the staged increment to `$MC_PUSH_DEST` (Mode B) |
| `mc apply` | Embed + upsert queued increments (Mode B server side); validates URLs/text/size, quarantines failures; no-op when the queue is empty |
| `mc stats-backfill [--fetch-missing] [--no-index] [--from-db PATH] [--from-tabs \| --from-api] [--limit N]` | Repair `video_stats` + old Qdrant payloads; `--from-api` hydrates exact upload dates/durations/titles from the YouTube Data API v3 (needs `YOUTUBE_API_KEY`; minutes, ~150 quota units, no cookies, idempotent — 0 units on a re-run over complete rows); `--from-tabs` is the retained offline/cookie fallback (hours, residential IP — run detached). The two are mutually exclusive |
| `mc reindex --model ID` | Rebuild every chunk into a new collection (re-embeds) and swap the alias in one atomic update |
| `mc eval --suite PATH [--limit 50] [--out FILE]` | Sparse-only vs hybrid MRR on known quotes; exits non-zero if hybrid regresses (cutover gate) |
| `mc serve` | Run the search API on `$API_PORT` |
| `python -m mc.backup create [--collection NAME] [--keep 7]` | Create a Qdrant snapshot; keep the last 7 (default collection is the live one) |
| `python -m mc.backup restore SNAPSHOT_PATH --to COLLECTION` | Restore a snapshot — `--to` is required and the target must NOT exist (refuses to overwrite); restore into a new name, then re-point the alias |

Each command is cron-friendly: exit codes are meaningful (0 = ok), runs are
serialized by flock locks, and point-level work is idempotent (content-hash
dedupe), so a retried run never duplicates data.

## API reference

| Endpoint | What |
|---|---|
| `GET /` | Search UI (HTML); `?query=` pre-fills the box; `?render=json` returns the agent envelope instead |
| `GET /search?query=…&limit=…` | JSON results |
| `GET /results?query=…` | HTML result fragment (htmx) |
| `GET /llms.txt` | Agent manual (llmstxt.org-style plain text) |
| `GET /health` | JSON probe: qdrant reachability + ingest freshness |
| `GET /static/*` | Static assets |

### Agents (moltbook pattern)

Muriel is agent-friendly: humans teach their agents to use the site — a
"Fale com a Muriel" box on the landing page hands visitors one paste-ready
phrase that points any agent at the machine-readable manual:

```text
quando o eddie falou sobre ilha das américas? https://muriel.eddieoz.com/llms.txt
```

The agent reads `/llms.txt`, learns the endpoints, and calls the JSON API.
Every `render=json` response is also self-describing.

- `GET /?query=TERMO&render=json` — the landing page answers in JSON: same
  envelope as `/search` plus an `agent` block (endpoints, params, result
  fields). Without a query it returns Muriel's self-description card.
- `GET /llms.txt` — the same manual as plain text for crawlers/agents.
- `GET /search?query=TERMO&limit=10` — the plain JSON API (unchanged).

`render` is case-insensitive and only the literal value `json` switches the
format; anything else (or its absence) renders HTML as before.

`sort` orders the returned set — `relevance` (default, unchanged), `recent`
(date descending) or `oldest` (date ascending); undated results sink to the
bottom in both date modes. It is post-retrieval only: the same ≤ 50 relevance
candidates are reordered, no extra query is run. The web UI exposes the same
three-way control next to the results; the chosen mode rides in the URL
(`?query=…&sort=…`) so a sort survives a new query and is shareable. Under
`recent`/`oldest`, per-result `score` stays a per-result relevance and is
**not** monotonic with position — don't re-rank by it.

Constraints: `query` 1–512 chars, `limit` default 10 / max 50. The engine
clamps both again internally, so direct callers cannot bypass the caps.

`GET /search` response:

```json
{
  "query": "bitcoin halving",
  "took_ms": 42.1,
  "sort": "relevance",
  "freshness": {"last_update": "2026-10-10T06:20:00+00:00"},
  "results": [
    {
      "title": "EP 123 — O halving explicado",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "start_seconds": 734,
      "date": "2024-04-19",
      "episode": "EP 123",
      "channel": "Morning Crypto",
      "snippet": "…e é por isso que o <mark>halving</mark> muda tudo…",
      "score": 0.83
    }
  ]
}
```

`GET /health` responses:

```json
200 {"status": "ok", "last_update": "2026-10-10T06:20:00+00:00", "fingerprint": "…", "qdrant": "reachable"}
200 {"status": "stale", "last_update": "2026-10-07T06:20:00+00:00", "fingerprint": "…", "qdrant": "reachable"}
503 {"status": "unreachable"}
```

## Operations

### Cron (Mode A — server self-update)

Install the example crontab (as root on the server) and adjust the repo path:

```bash
crontab /opt/mc/crontab.example
```

It schedules three jobs, each serialized by a lock in `/run/lock` and each
firing the `NOTIFY_WEBHOOK` on failure (non-zero exit **or** a stale update
log — set `NOTIFY_WEBHOOK` directly in the crontab; cron does not read your
shell environment):

| Job | Schedule | Command |
|---|---|---|
| Daily update | 06:15 | `docker compose run --rm api python -m mc.cli update` |
| Nightly snapshot | 03:00 | `.venv/bin/python -m mc.backup create --keep 7` (from host venv) |
| Weekly apply sanity | Sun 05:00 | `mc apply` — must be a no-op on Mode A; any other outcome is drift and an alert |

### Snapshots

Nightly snapshots land in `./snapshots` (mounted into qdrant at
`/qdrant/snapshots`) with 7-deep rotation. Create/restore drills, including
the "restore into a NEW collection, then swap the alias" procedure, are in
[docs/tasks/backup-restore.md](docs/tasks/backup-restore.md).

### Mode B (local scrape + push)

Mode A is the default; Mode B is the documented fallback for the top
operational risk: datacenter-IP cookie/bot-detection. The local machine
(residential IP) runs `mc update` + `mc push` on cron; the server only runs
`mc apply`. The failover table (how to detect Mode A failing and switch) is
in [docs/tasks/mode-b-runbook.md](docs/tasks/mode-b-runbook.md).

### Reindex (model swap)

When changing the embedding model, build into a fresh collection instead of
touching the live one:

```bash
mc reindex --model intfloat/multilingual-e5-small
```

This re-embeds every chunk into `<current>-v2`, then swaps the
`mc-transcripts-v1-alias` alias in one atomic update — the old collection
is never deleted, so rollback is re-pointing the alias by hand.

## Security

Posture as deployed (hardening applied after a red-team assessment):

- **Qdrant is loopback-only** (`127.0.0.1` binds in docker-compose.yml); the
  API container reaches it over the Docker network. Never expose 6333/6334.
- **Optional API-key auth**: set `QDRANT__SERVICE__API_KEY` on the qdrant
  service and `QDRANT_API_KEY` on the api service (same secret) — all
  clients attach the key automatically.
- **GET-only API**: no mutating endpoints; `/health` never discloses
  exception details.
- **Input caps**: query ≤ 512 chars, limit ≤ 50, enforced at the edge and
  again in the engine; the snippet highlighter caps its word set.
- **Response hardening**: CSP, `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy` on every response; result URLs
  are forced to `https://` and video ids are strictly validated before
  rendering.
- **Mode B apply validates queue input**: http(s) URLs only, non-empty text
  ≤ 50k chars, files ≤ 100 MB; violations are quarantined, never indexed.
- **Containers**: non-root user (uid 1000), `cap_drop: [ALL]`,
  `no-new-privileges`. Host bind mounts must be owned by uid 1000.
- **Credential guard**: `make hooks` wires a pre-commit hook that rejects
  staged cookies files (any name) and pasted cookie content.

## Testing

```bash
make test                          # full suite: pytest + BDD (unit)
MC_COMPOSE_TESTS=1 make test       # + compose smoke tests (needs docker)
make up                            # integration tests need the local qdrant
.venv/bin/python -m pytest tests/test_engine.py -q   # single file
.venv/bin/python -m pytest -m integration -q         # only integration tests
make lint                          # ruff
```

Suite layout: unit tests per module (`tests/test_*.py`), BDD-style flows
under `tests/bdd/`, integration tests marked `integration` (require the
local qdrant container from `make up`). Keep the container's `nofile`
ulimit at the compose-configured 65536 or RocksDB returns
`Too many open files` under load.

## Deployment

Full step-by-step: [docs/tasks/deploy-runbook.md](docs/tasks/deploy-runbook.md).
Summary:

1. **Pre-flight**: ≥ 6 GB free disk (≈ 2.5 GB image with models baked +
   volume), 4 GB RAM box, an API port, `NOTIFY_WEBHOOK` available.
2. **Clone + build**: `git clone <repo> /opt/mc && cd /opt/mc && docker
   compose build && docker compose up -d` — the build bakes the e5 + bm25
   weights into the image so container start does no HuggingFace network
   access. Verify with `curl -s localhost:18080/health | jq .status`.
3. **Backfill BEFORE routing traffic** (see [Getting started](#getting-started-dev)).
4. **Eval + cutover**: run `mc eval --suite tests/eval/quotes.yaml --out
   docs/tasks/eval-report.md`; on sign-off re-point nginx. Rollback is
   re-pointing nginx back (old files kept 2 weeks).
5. **Cron**: install `crontab.example` (see [Operations](#operations)).

Ownership note: bind mounts (`./data`, `./state`, `./snapshots`) must be
owned by uid 1000 to match the container's non-root user.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` → `503 {"status": "unreachable"}` | Qdrant down or `QDRANT_URL` wrong: `docker compose ps`, `docker compose logs qdrant` |
| `/health` → `"stale"` | Last successful ingest older than `FRESHNESS_MAX_HOURS` (72 h). Check the update log; run `mc update --dry-run` to verify cookies |
| Cron runs exit with `AlreadyRunning` | A previous run still holds the flock (`/run/lock/mc-*.lock`). Wait for it to finish or investigate before clearing |
| Ingestion silently skips videos | Check `ignored_videos.txt` next to the transcripts. Auth failures (expired cookies) are classified and alerted, not ignore-listed — rotate cookies per docs/tasks/cookies.md and retry |
| qdrant: `Too many open files` | RocksDB exhausted the fd ulimit; compose sets `nofile: 65536` — keep it |
| Port 8080 already in use | Compose maps the api to host port 18080 by default; adjust `ports:` in docker-compose.yml |
| Container: `Permission denied` on data dirs | Bind mounts must be owned by uid 1000 (the container's non-root user): `chown -R 1000:1000 data state snapshots` |
| `mc push` errors about `MC_PUSH_DEST` | Set the rsync destination, e.g. `MC_PUSH_DEST=server:/opt/mc/data/queue/` (must be the directory `mc apply` watches) |
| `restore` refuses: collection exists | Intentional — restore into a new collection name, then re-point the alias (docs/tasks/backup-restore.md) |
| Eval exits non-zero | Hybrid regressed vs sparse-only on the suite — investigate before cutover (docs/tasks/eval-report.md) |

## Further docs

| Doc | What |
|---|---|
| `docs/tasks/mc-semantic-search-epic.md` | Full story of the epic |
| `crontab.example` | Mode A crontab (update, snapshot, apply sanity, notify hook) |
| `docs/tasks/mode-b-runbook.md` | Mode B fallback: local scrape + push, failover table |
| `docs/tasks/deploy-runbook.md` | deployment, staging route, eval + cutover + rollback |
| `docs/tasks/cookies.md` | Cookie export/rotation + stale-cookie detection |
| `docs/tasks/backup-restore.md` | Qdrant snapshots: create/restore drill |
| `docs/tasks/eval-report.md` | Sparse-only vs hybrid MRR on known quotes (cutover evidence) |
| `docs/tasks/results-sorting-epic.md` | Results sorting: relevance / most recent / oldest (web UI control + `/search` & `render=json` knob) |
