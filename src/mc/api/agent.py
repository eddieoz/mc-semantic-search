"""Agent-facing self-description of Muriel — single source of truth.

Agents arrive at the landing page (the moltbook.com pattern) and need a
machine-readable card: what this archive is, the endpoints, the parameters
and the shape of a result. The same content renders three ways:

- ``GET /?render=json``  → JSON envelope carrying the ``agent`` block
- ``GET /llms.txt``      → llmstxt.org-style plain text
- ``GET /`` (HTML)       → the "Fale com a Muriel" box: the paste-ready
  phrase a human sends their agent (the moltbook box)

Keep the wording stable: agents (and their caches) parse it literally.
"""

from __future__ import annotations

AGENT_NAME = "Muriel"

# The archive's public identity. The paste-ready phrase must point at the
# canonical domain no matter where the page was rendered (localhost, mirror,
# proxy) — that URL is what carries the agent to the manual.
PUBLIC_URL = "https://muriel.eddieoz.com"

EXAMPLE_QUESTION = "quando o eddie falou sobre ilha das américas?"

# Relative endpoint recipes (SEU_TERMO = the agent's search terms), shown
# literally in every surface; the HTML/browser resolves them against the
# page origin, the JSON/llms.txt surfaces prepend the request origin.
ENDPOINTS = (
    "/?query=SEU_TERMO&render=json",
    "/search?query=SEU_TERMO&limit=10",
    "/llms.txt",
)


def endpoints(base_url: str | None = None) -> list[str]:
    if not base_url:
        return list(ENDPOINTS)
    return [f"{base_url}{path}" for path in ENDPOINTS]


def landing_box() -> dict:
    """The moltbook-style box on the landing page — written for the human
    who has an agent (Claude, GPT, …): one paste-ready phrase, then what
    happens next. No endpoint jargon here; the agent gets the full technical
    card when it follows the link to manual()/llms_txt()."""
    return {
        "title": "Fale com a Muriel 🤖",
        "lead": "para consultar o acervo, cole esta frase na conversa com o seu agente:",
        "question": EXAMPLE_QUESTION,
        "manual_path": "/llms.txt",
        "manual_url": f"{PUBLIC_URL}/llms.txt",
        "steps": (
            "o agente lê o manual e aprende a consultar o acervo",
            "o agente busca o momento exato em que o tema foi dito",
            "você recebe o episódio, o link do YouTube e o segundo exato",
        ),
    }


def manual(base_url: str | None = None) -> dict:
    """The agent card: how to search the archive and read a result."""
    search_url, api_url, manual_url = endpoints(base_url)
    return {
        "name": AGENT_NAME,
        "role": "bibliotecária do acervo Morning Crypto",
        "language": "pt-BR",
        "what": (
            "busca semântica que encontra o momento exato em que algo "
            "foi dito no Morning Crypto (YouTube)."
        ),
        "how_to_search": [
            f"GET {search_url} — busca; devolve JSON em vez da página HTML",
            f"GET {api_url} — API JSON pura (limit 1–50, padrão 10)",
            f"GET {manual_url} — este manual em texto puro",
        ],
        "params": {
            "query": "texto livre; entre aspas duplas exige o trecho literal",
            "limit": "1–50 (padrão 10)",
            "render": "valor json devolve JSON; sem ele, HTML",
        },
        "result_fields": {
            "title": "episódio (fallback: vídeo)",
            "url": "link do YouTube",
            "start_seconds": "segundo exato onde o trecho começa",
            "date": "data de publicação (AAAA-MM-DD)",
            "episode": "número do episódio",
            "channel": "canal",
            "snippet": "trecho da transcrição com <mark> no termo",
            "score": "relevância (maior = melhor)",
        },
        "note": "para assistir o momento: abra url e avance para start_seconds",
    }


def llms_txt(base_url: str | None = None) -> str:
    """llmstxt.org canonical shape: every H2 section is a link list, each
    line exactly ``- [name](url): optional notes``. Strict validators reject
    plain text inside sections, so params and field docs live in the notes
    (``query``/``render`` on the search link, ``limit`` on the API link) and
    each result field is its own list item pointing at the JSON envelope."""
    m = manual(base_url)
    search_url, api_url, manual_url = endpoints(base_url)
    envelope = search_url  # the URL that returns the documented fields
    return "\n".join(
        [
            f"# {AGENT_NAME} — arquivo Morning Crypto",
            "",
            f"> {m['role']}. {m['what']}",
            "",
            "## Como consultar",
            "",
            f"- [Busca render=json]({search_url}): substitua SEU_TERMO pelo "
            "termo; query é texto livre (aspas duplas exigem o trecho "
            "literal); devolve JSON em vez da página HTML",
            f"- [API /search]({api_url}): API JSON pura; limit 1–50 "
            "(padrão 10)",
            f"- [Manual /llms.txt]({manual_url}): este manual em texto puro",
            "- [Busca no Blog](https://eddieoz.com/llms.txt): Consulte também no "
            "Blog: artigos, documentos, entrevistas, tutoriais, apresentações e "
            "outros conteúdos",
            "",
            "## Campos de cada resultado",
            "",
            f"- [title]({envelope}): episódio (fallback: vídeo)",
            f"- [url]({envelope}): link do YouTube",
            f"- [start_seconds]({envelope}): segundo exato onde o trecho começa",
            f"- [date]({envelope}): data de publicação (AAAA-MM-DD)",
            f"- [episode]({envelope}): número do episódio",
            f"- [channel]({envelope}): canal",
            f"- [snippet]({envelope}): trecho da transcrição com <mark> no termo",
            f"- [score]({envelope}): relevância (maior = melhor)",
            f"- [como assistir]({envelope}): abra url e avance para start_seconds",
            "",
            "## Status do acervo",
            "",
            f"- [archive_status]({envelope}): números do acervo — vídeos, "
            "segundos, minutos, horas, dias e o último episódio (null quando "
            "desconhecido)",
            "",
        ]
    )
