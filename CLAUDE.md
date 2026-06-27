# SciAgent — multi-domain scientific AI platform

Cross-domain scientific research assistant. Users compose panels of 2–4 domain expert agents that debate a question, each grounded in peer-reviewed literature. The novel value is synthesis *across* domains, not within one.

## Project structure

```
sci-agents/
├── CLAUDE.md                  ← you are here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── cli.py                     ← CLI entry point
├── corpus/
│   ├── CLAUDE.md              ← layer spec (future: real retrieval pipeline)
│   ├── __init__.py
│   └── retriever.py           ← PaperChunk + Retriever interface
├── agents/
│   ├── CLAUDE.md              ← layer spec
│   ├── __init__.py
│   ├── base_agent.py          ← DomainAgent (uses web search for demo)
│   └── personas/
│       ├── _base.txt
│       ├── mathematics.txt
│       ├── ops_research.txt
│       ├── finance.txt
│       ├── engineering.txt
│       ├── strategy_game_theory.txt
│       ├── lean_process.txt
│       ├── business_economics.txt
│       ├── ai_ml_systems.txt
│       └── solution_architecture.txt
├── panel/
│   ├── CLAUDE.md              ← layer spec
│   ├── __init__.py
│   ├── models.py
│   ├── composer.py            ← PanelComposer orchestration loop
│   ├── moderator.py
│   └── synthesis.py
├── api/
│   ├── CLAUDE.md              ← layer spec
│   ├── __init__.py
│   └── main.py                ← FastAPI app (serves API + demo UI)
└── demo/
    ├── CLAUDE.md              ← layer spec
    └── index.html             ← single-file web UI
```

## Running

```bash
# Docker (recommended)
cp .env.example .env           # add your ANTHROPIC_API_KEY
docker compose up --build
# → http://localhost:8080

# Local
pip install -r requirements.txt
uvicorn api.main:app --port 8080

# CLI
python cli.py --question "..." --agents mathematics ops_research
```

## Core concepts

| Term | Meaning |
|---|---|
| Domain agent | LLM with a persona system prompt; retrieves literature via web search (demo) or corpus (future) |
| Panel | 2–4 domain agents + 1 moderator agent debating a user question |
| Moderator | Agent that surfaces contradictions and asks cross-domain questions between turns |
| Synthesis | Post-panel pass that maps agreements, conflicts, and novelty gaps |
| Corpus | Future: vector-indexed + BM25 over open-access papers. Demo uses Anthropic web search. |

## Tech stack

- **LLM:** Anthropic Claude API (claude-sonnet-4-6 for agents, claude-haiku-4-5-20251001 for moderator)
- **Retrieval (demo):** Anthropic web search tool scoped to academic sites (arxiv, ssrn, pubmed)
- **Retrieval (future):** Qdrant/pgvector + BM25 over ingested papers
- **API:** FastAPI, serves both REST API and static demo UI
- **Deployment:** Docker → GCP Cloud Run

## Environment variables

```
ANTHROPIC_API_KEY=             # required
LOG_LEVEL=INFO                 # optional
```

## Key constraints

- Every agent response must include citation objects (title + year + URL). No uncited claims.
- Panel runs cost ~8–15 LLM calls. Token usage is tracked and returned in API response.
- Demo is stateless. Do not persist user queries or panel outputs.
- A 2-agent × 2-round panel costs roughly $0.15–0.25 in API calls.
