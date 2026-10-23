# AGENTS.md — mc-semantic-search

Guidance for AI coding agents working in this repository. This file is the
agent-facing浓缩 of the conventions and invariants that the human-facing
[README.md](README.md) and [docs/tasks/](docs/tasks/) describe in full. Read it
before editing; the "Hard invariants" section lists rules that tests enforce and
that regressions quietly break.

## What this project is

Semantic search over the *Morning Crypto* YouTube transcript
archive. It replaces an old SearXNG + SQLite FTS5 keyword pipeline with a
Qdrant-backed **hybrid** (dense + sparse) search behind a FastAPI + htmx UI. Portuguese-first: a user reaches the **exact moment**
(`&t=` deep link, embedded `youtube-nocookie` player) an idea was said, even
when their wording paraphrases the transcript.

- Package: `mc` (src-layout, installed editable). Console script: `mc`.
- Language: Python 3.11+ (dev on 3.12). All deps pinned in `pyproject.toml`.
- License: MIT. Repo: `github.com/eddieoz/mc-semantic-search`.
- Current branch model: story-based TDD (see "Conventions").

## Build / run / test commands

Use the project venv (`.venv/bin/...`) or `uv`. `make` targets are the stable entry points.

```bash
make up                            # docker compose up -d --build (qdrant + api; local ≈ prod)
make down                          # docker compose down
make test                          # .venv pytest tests/  (unit + BDD; no docker needed)
MC_COMPOSE_TESTS=1 make test       # + compose smoke tests (needs docker)
make lint                          # ruff check .
make dev                           # uvicorn mc.api.app:app on :8080 (talks to compose qdrant on localhost:6333)
make hooks                         # git config core.hooksPath .githooks   (wire the credential guard)

.venv/bin/python -m pytest tests/test_query.py -q        # single file
.venv/bin/python -m pytest -m integration -q             # only integration tests (need local qdrant)
```

- `make up` maps the API to **host port 18080** (8080 is taken on the dev box); the
  container listens on 8080. `make dev` uses 8080 directly.
- Integration tests (marker `integration`) require the qdrant container from `make up`.
- The container's `nofile` ulimit is pinned to **65536** in compose; lowering it makes
  RocksDB die with `Too many open files`. Do not remove it.
- Verify after a change: `make lint` must be clean and `make test` green. `tests/test_repo_layout.py`
  runs `ruff check .` and asserts gitignore/hook/Dockerfile invariants as part of the suite.

## Repository layout

```
src/mc/
├── api/
│   ├── app.py            # FastAPI factory (create_app); routes, security headers, render=json, DI
│   ├── agent.py          # agent manual: /llms.txt + render=json "agent" block + landing box
│   └── health.py         # health_snapshot(): liveness + freshness + fingerprint
│   ├── templates/        # search.html, results.html (htmx partial), result.html (player)
│   └── static/           # vendored htmx/alpine/pico + css + favicon (NO external CDN)
├── ingest/
│   ├── discover.py       # channel listing via yt-dlp --dump-single-json
│   ├── download.py       # subtitle download (yt-dlp Python API), cookies, ignore list, 429 backoff
│   ├── chunker.py        # transcript → N-segment chunks (size=30), first-segment timestamp
│   ├── metadata.py       # episode/date/channel enrichment (yt-dlp, cached in data/meta)
│   ├── update.py         # daily orchestrator: discover→download→chunk→embed→upsert→stage
│   ├── push.py           # Mode B: rsync the staged increment to $MC_PUSH_DEST
│   ├── apply.py          # Mode B server: validate + embed + upsert queued increments
│   ├── backfill.py       # bootstrap from the legacy SQLite DB (resumable)
│   ├── reindex.py        # rebuild into a new collection, swap alias atomically
│   ├── stats_backfill.py # repair video_stats + old Qdrant payloads (episode/date/duration)
│   ├── legacy.py         # legacy DB reader (read-only; quarantines bad rows)
│   ├── crosscheck.py     # transcript/DB consistency check
│   ├── lock.py           # flock-based RunLock (cron overlap guard)
│   └── models.py         # Chunk / report dataclasses
├── search/
│   ├── engine.py         # hybrid search: dense+sparse prefetch → RRF → typed results
│   ├── query.py          # query parsing/normalization (Portuguese-first, quoted phrases)
│   ├── ranking.py        # quote-aware boost (demotes non-literal matches)
│   ├── snippets.py       # <mark> snippets (HTML-escaped before marking)
│   └── cache.py          # query-embedding LRU (embedder starvation guard)
├── state.py              # SQLite bookkeeping: processed + meta + video_stats
├── qdrant.py             # client helpers, collection fingerprint, atomic per-video upsert
├── embeddings.py         # dense (e5) + sparse (fastembed BM25) embedders; fingerprints
├── eval.py               # MRR harness: sparse-only vs hybrid on known quotes
├── backup.py             # Qdrant snapshot create/restore + rotation
├── bake_models.py        # downloads model weights at image build time
├── config.py             # single source of truth: model ids, paths, env knobs
└── cli.py                # `mc` subcommands
```

Other top-level items: `Dockerfile` (multi-stage, models baked), `docker-compose.yml`
(qdrant + api), `Makefile`, `pyproject.toml`, `crontab.example`, `README.md`,
`docs/tasks/` (runbooks + epic), `docs/superpowers/specs/` (design specs).

Runtime dirs, **gitignored**: `data/` (meta cache, queue, qdrant storage), `state/`
(SQLite `ingest.db`), `snapshots/` (Qdrant snapshots + BDD working dirs `mc-bdd-*`),
`transcripts/`. `.venv/`, `.serena/`, `.zcode/` are also ignored.

## Conventions (follow these)

- **Story-based TDD.** Development proceeds as 1-point stories grouped in themes
  (see `docs/tasks/mc-semantic-search-epic.md`). Convention: write the failing
  test first; every story's DoD is tests-first, all green, lint clean, docs touched.
  User-visible behavior gets a Gherkin feature under `tests/bdd/<story>.feature`
  with step defs in `tests/bdd/test_bdd_*.py`.
- **Module docstrings cite the story and NFR.** Each module's docstring names the
  story number(s) and the non-functional requirement it serves (e.g. "Story 4.2",
  "NFR-1"). Preserve this when editing — it is the project's traceability mechanism.
- **`from __future__ import annotations`** at the top of every module; type hints
  are used throughout (`X | None`, `list[str]`).
- **Conventional commits** with a scope, matching history: `feat(scope):`,
  `fix(scope):`, `docs(scope):`, `chore(scope):`, `style(scope):`. Recent examples:
  `fix(apply): mark applied videos processed (Mode B checkpoint)`.
- **Comments explain *why*/the constraint, not *what*.** Many comments record a
  review fix or a deliberate trade-off (e.g. "non-null wins", "None CLEARS"). Do
  not strip these when refactoring — they encode decisions.
- **ruff** config: line-length 100, rules `E,F,W,I,B,UP`. `src/mc/ingest/models.py`
  is the only per-file ignore (`RUF`, for wide dataclasses). Keep `make lint` green.
- **Dependency injection for testability.** `create_app(...)` takes injectable
  `client`/`state`/`cache` so tests build the app against a fake qdrant or tmp
  state store. New endpoints should stay DI-friendly.

## Hard invariants (tests enforce these — do not break)

- **Credential guard (NFR-5).** The YouTube `cookies.txt` is a live credential.
  It is gitignored (`cookies*`, `*cookies*`, `www.youtube.com_cookies*`) and a
  pre-commit hook (`.githooks/pre-commit`, wired via `make hooks`) rejects any
  staged cookie filename **or** pasted Netscape cookie line. Never commit it, even
  renamed or pasted. `docs/` is excluded from the filename check so
  `docs/tasks/cookies.md` stays editable.
- **Embedder runs one call at a time (NFR-1).** `embeddings.py` holds a module
  `torch_lock` and caps `TORCH_NUM_THREADS = 2`; `engine.py` runs both embedder
  calls in a `ThreadPoolExecutor(max_workers=1)`. A crawler must not be able to
  starve the embedder — the API-layer `ResultCache` (LRU, default 128) exists for
  this. Do not parallelize embedding.
- **Input caps, enforced at the edge *and* re-clamped in the engine.** Query ≤ 512
  chars, limit default 10 / max 50 (`MAX_QUERY_LENGTH`, `MAX_SEARCH_LIMIT`). The
  engine clamps again (`limit = min(max(limit,1),100)`, `normalized[:512]`) because
  `eval.py` bypasses the API `Query` constraints. Direct callers cannot bypass them.
- **Model fingerprint (NFR-6).** Model ids/dims are constants in `config.py`
  (`intfloat/multilingual-e5-small` 384-dim dense; `Qdrant/bm25` pt sparse). A
  fingerprint (sha256[:16] of `id:dim`) is stored in sentinel meta point id `0`.
  `ensure_collection` rejects a fingerprint mismatch with `ModelMismatchError` —
  to change models, run `mc reindex` (build a new collection, swap the alias),
  never write into a different model space.
- **Atomic per-video upsert (NFR-3).** `qdrant.upsert_video` deletes the video's
  old points and inserts the new ones in **one** `batch_update_points` call, so a
  crash can never half-commit a video. Point id is `uuid5(NAMESPACE_URL,
  f"{video_id}:{start_seconds}")` → stable → re-ingest is idempotent (content-hash
  dedupe). `processed` is marked only after a successful commit.
- **Failure isolation on the read path.** `/search`, `/health`, the freshness line,
  and the archive-status block each wrap state/qdrant access in try/except: an
  unreadable state DB hides the section or returns nulls, and a qdrant transport
  failure returns **503, never 500**. Preserve this pattern for any new read path.
- **State store is never on the search path.** `state/ingest.db` (WAL +
  `busy_timeout=5000`) holds `processed`, `meta` (run timestamps), and
  `video_stats`. It is read only by `/health`, ingestion, and the landing page.
  `record_video_stats` is COALESCE-based (**non-null wins**, never wipes a known
  value on re-ingest); `set_video_episode` is the authoritative repair write where
  `None` **clears** a junk episode. A corrupt db file is renamed `.corrupt` and
  recreated, never silently wiped.
- **Output sanitization (stored-XSS / phishing defense).** Result URLs are forced
  to `https://` YouTube only (`_safe_url`); video ids match `^[A-Za-z0-9_-]{6,20}$`.
  Snippet text is HTML-escaped before `<mark>` highlighting. Poisoned payloads
  (Mode B queue or a poisoned index) must never link out or inject markup.
- **`render=json` is the literal string `"json"`** (case-insensitive, capped at 16
  chars). Anything else renders HTML. Every `render=json` response is
  self-describing (an `agent` block); `/llms.txt` is the same manual as plain text.
- **Hybrid fusion is Qdrant-native RRF**, with prefetch oversampling `max(limit*3,
  30)` because RRF discards absolute scores (a bare `limit` pool ties out by point
  order before fusion). Quote-aware boost (`SEARCH_QUOTE_BOOST`, default 0.5)
  demotes non-literal matches but never changes result count.
- **Cron runs are serialized and alerting.** `ingest/lock.py` provides flock-based
  `RunLock`; locks live in `/run/lock` (root-only), not world-writable `/tmp`.
  Every CLI command has a meaningful exit code (0 = ok); `mc eval` exits non-zero
  if hybrid MRR regresses vs sparse-only (the cutover gate).

## Configuration

All knobs live in `src/mc/config.py`, read from the environment **at import time**.
Model ids, dims, and the collection name are constants (not env-tunable).

| Env var | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_API_KEY` | unset (auth off) | Sent to Qdrant only if set, paired with server `QDRANT__SERVICE__API_KEY` |
| `API_PORT` | `8080` | Port for `mc serve` / the container |
| `FRESHNESS_MAX_HOURS` | `72` | `/health` reports `stale` past this since last successful ingest |
| `SEARCH_QUOTE_BOOST` | `0.5` | Score demotion for non-literal matches |
| `SEARCH_CACHE_SIZE` | `128` | Query-embedding LRU size |
| `MC_CHANNELS` | *(unset; see `config.yaml`)* | Comma-separated YouTube channel URLs; `config.yaml` lists them, this env overrides |
| `MC_COOKIES_PATH` | `<repo>/cookies.txt` | YouTube cookies (live credential; 0600, never committed) |
| `MC_DATA_DIR` | `<repo>/data` | meta cache, queue, qdrant storage |
| `MC_STATE_DIR` | `<repo>/state` | SQLite ingest bookkeeping |
| `MC_TRANSCRIPTS_DIR` | `<repo>/transcripts` | Downloaded transcripts |
| `MC_PUSH_DEST` | unset | rsync destination for `mc push` (Mode B) |
| `NOTIFY_WEBHOOK` | unset | Telegram-compatible webhook for cron failure alerts |

**Channels** live in `config.yaml` (committed; public URLs). `discover.load_channels()`
reads it first, then the `MC_CHANNELS` env. Each channel is listed across its
`/videos`, `/streams`, `/shorts` tabs and the ids are unioned — so adding a
channel automatically covers all three tabs (no code change).

## CLI reference (`mc`)

| Command | What it does |
|---|---|
| `mc backfill --from-db PATH [--batch-size 256] [--limit N]` | Embed + index unprocessed chunks from the legacy SQLite DB (bootstrap; hours on CPU; resumable) |
| `mc update [--dry-run]` | One daily pass: discover→download→chunk→embed→upsert→stage. `--dry-run` verifies cookies only |
| `mc push` | rsync the staged increment to `$MC_PUSH_DEST` (Mode B) |
| `mc apply` | Embed + upsert queued increments (Mode B server); validates + quarantines failures; no-op when empty |
| `mc stats-backfill [--fetch-missing] [--no-index] [--from-db PATH] [--limit N]` | Repair `video_stats` + old Qdrant payloads; `--no-index` skips Qdrant, `--from-db` re-derives episodes offline |
| `mc reindex --model ID` | Rebuild every chunk into a new collection and swap the alias atomically (old kept for rollback) |
| `mc eval --suite PATH [--limit 50] [--out FILE]` | Sparse-only vs hybrid MRR; exits non-zero if hybrid regresses (cutover gate) |
| `mc serve` | Run the search API on `$API_PORT` |
| `python -m mc.backup create [--collection NAME] [--keep 7]` | Create a Qdrant snapshot; keep last 7 |
| `python -m mc.backup restore SNAPSHOT --to COLLECTION` | Restore — `--to` required and must NOT exist; restore into a new name, then swap the alias |

Each command is cron-friendly: meaningful exit codes, flock-serialized runs,
content-hash idempotency so a retried run never duplicates data.

## API reference

| Endpoint | What |
|---|---|
| `GET /` | Search UI (HTML); `?query=` pre-fills; `?render=json` returns the agent envelope |
| `GET /search?query=…&limit=…` | JSON results |
| `GET /results?query=…` | HTML result fragment (htmx) |
| `GET /llms.txt` | Agent manual (llmstxt.org-style plain text) |
| `GET /health` | JSON: qdrant reachability + ingest freshness + fingerprint |
| `GET /static/*` | Vendored assets (htmx/alpine/pico) — no external requests |

`/search` result keys (shared by `/search` and the `render=json` envelope):
`title, url, start_seconds, date, episode, channel, snippet, score`.
Security headers (CSP, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`) are
set on every response by an app middleware.

## Data flow (ingestion)

```
Mode A (server, default):  cron → mc update → discover/download → chunk → embed
        → upsert into qdrant (collection mc-transcripts-v1) → stage increment → data/queue/pending.jsonl

Mode B (fallback):  local residential IP → mc update → chunk → stage
        → mc push (rsync) → server queue → mc apply → validate → embed → upsert
```

Mode A is default; Mode B is the documented fallback for the #1 operational risk —
datacenter-IP cookie/bot-detection. `mc update` upserts directly **and** stages the
same increment, so Mode B ships exactly what Mode A indexes. `mc apply` dedupes by
content hash and quarantines failed increments under `data/queue/failed/`.

## Testing conventions

- Unit tests per module: `tests/test_*.py` (plain pytest).
- BDD flows: `tests/bdd/*.feature` (Gherkin) + `tests/bdd/test_bdd_*.py` step defs.
- Fixtures: `tests/fixtures/` (golden chunk JSON, crosscheck SRTs, download json3,
  legacy mini-DB, playlist JSON); eval suite `tests/eval/quotes.yaml`.
- Doc-consistency tests assert the human docs stay truthful: `test_repo_layout.py`
  (gitignore/hook/Dockerfile), `test_crontab_doc.py`, `test_runbook_doc.py`. When
  you change a command, path, or invariant, update the doc **and** its asserting
  test together.
- Golden tests compare against saved real outputs (e.g. `chunker_golden.json`,
  `download/sample.json3`) — keep them in sync when chunking/parsing rules change.
- Integration tests (marker `integration`) need the qdrant container (`make up`).

## Gotchas

- **Dockerfile copies the whole package**: `COPY src/mc/ /app/src/mc/`. A partial
  COPY (e.g. only `api/`) makes uvicorn die with `ModuleNotFoundError` at container
  start. `templates/`/`static/` live under `src/mc/api/`, not the repo root.
- **Bind-mount ownership**: host dirs `./data`, `./state`, `./snapshots` must be
  owned by uid 1000 (the container's non-root user) or the app gets `Permission
  denied`.
- **Qdrant is loopback-only** in compose (`127.0.0.1:6333/6334`); never expose it.
  The api container reaches it over the Docker network (`http://qdrant:6333`).
- **Restore never overwrites**: `mc backup restore --to` refuses if the target
  exists. Restore into a new collection, then re-point the `mc-transcripts-v1-alias`.
- **Old points show `ep —` / `sem data`** if indexed before episode/date
  enrichment existed; `mc stats-backfill` repairs the payloads (payload-only, no
  re-embedding) or, with `--fetch-missing`, fetches once via yt-dlp.
- **`mc update --dry-run`** verifies cookies without ingesting — use it to debug
  auth/bot failures. Auth failures are classified and alerted, not ignore-listed.

## Further reading

- `README.md` — full human-facing documentation (tech stack, deploy, troubleshooting).
- `docs/tasks/mc-semantic-search-epic.md` — the epic: stories, NFRs, decision log.
- `docs/tasks/deploy-runbook.md`, `mode-b-runbook.md`, `backup-restore.md`,
  `cookies.md`, `archive-status-epic.md` — operational runbooks.
- `docs/superpowers/specs/` — design specs for recent features.
