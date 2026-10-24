"""Story 9.4 — `config` module: `.env` seeding of the operator environment.

The YouTube API key (and any other operator knob) lives in `<repo>/.env`
rather than a crontab line or a wrapper script — cron does not read `.env`,
and keys pasted into crontabs end up in logs and editor histories. The
loader seeds ``os.environ`` at import time; a value already present in the
real environment always wins (``setdefault``), so an operator export or a
CI secret overrides the file.
"""

from __future__ import annotations

from pathlib import Path

from mc.config import _load_dotenv


def _write_env(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_missing_env_var_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_DOTENV_PROBE", raising=False)
    _write_env(tmp_path, "MC_DOTENV_PROBE=abc123\n")

    _load_dotenv(tmp_path / ".env")

    import os

    assert os.environ["MC_DOTENV_PROBE"] == "abc123"


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_DOTENV_PROBE", "from-env")
    _write_env(tmp_path, "MC_DOTENV_PROBE=from-file\n")

    _load_dotenv(tmp_path / ".env")

    import os

    assert os.environ["MC_DOTENV_PROBE"] == "from-env"


def test_comments_blanks_and_export_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_DOTENV_A", raising=False)
    monkeypatch.delenv("MC_DOTENV_B", raising=False)
    _write_env(
        tmp_path,
        "# a comment line\n"
        "\n"
        "MC_DOTENV_A=plain\n"
        "export MC_DOTENV_B=exported\n",
    )

    _load_dotenv(tmp_path / ".env")

    import os

    assert os.environ["MC_DOTENV_A"] == "plain"
    assert os.environ["MC_DOTENV_B"] == "exported"


def test_quotes_are_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_DOTENV_Q", raising=False)
    _write_env(tmp_path, 'MC_DOTENV_Q="quoted value"\n')

    _load_dotenv(tmp_path / ".env")

    import os

    assert os.environ["MC_DOTENV_Q"] == "quoted value"


def test_malformed_lines_are_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("MC_DOTENV_OK", raising=False)
    _write_env(tmp_path, "no equals sign here\n=orphan value\nMC_DOTENV_OK=1\n")

    _load_dotenv(tmp_path / ".env")

    import os

    assert os.environ["MC_DOTENV_OK"] == "1"


def test_missing_file_is_not_fatal(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")  # no exception


def test_gitignore_blocks_the_env_file():
    """.env holds live operator secrets — the repo layout test pins it."""
    gitignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert ".env" in gitignore
