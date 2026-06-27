# Layer 4: API

Parent context: see `../CLAUDE.md`
Depends on: `../panel/` (PanelComposer, PanelConfig, PanelResult)

## Responsibility

Expose the panel composer over HTTP (REST) and as an MCP tool server.
This layer owns request validation, serialisation, error handling, and auth stubs.

Do NOT build real auth or billing here — stubs only.

---

## Files to build here

```
api/
├── CLAUDE.md
├── main.py              ← FastAPI app, all routes
├── mcp_server.py        ← MCP tool server (thin wrapper over REST)
├── schemas.py           ← Pydantic request/response models
├── deps.py              ← dependency injection (composer, retriever, client)
└── run.py               ← entrypoint: uvicorn main:app
```

---

## REST API — endpoints

### POST /panels/run

Run a complete panel synchronously. Returns when synthesis is done.
Fine for demo; add streaming/async later.

**Request**
```json
{
  "question": "What scheduling approaches minimise makespan under stochastic processing times, and what are the practical barriers to implementing the theoretically optimal solution?",
  "domains": ["mathematics", "industrial_management"],
  "mode": "sequential",
  "turns_per_agent": 2
}
```

**Response** — `PanelResultResponse`
```json
{
  "panel_id": "uuid",
  "question": "...",
  "domains": ["mathematics", "industrial_management"],
  "turns": [
    {
      "turn_number": 1,
      "domain": "mathematics",
      "agent_name": "Dr. Virtanen (Mathematics)",
      "response": "...",
      "cited_papers": [
        {
          "paper_id": "arXiv:2301.00001",
          "title": "...",
          "authors": ["Smith, J."],
          "year": 2022,
          "source_url": "https://arxiv.org/abs/2301.00001"
        }
      ],
      "moderator_intervention": {
        "text": "...",
        "tension_detected": true,
        "tension_description": "..."
      }
    }
  ],
  "synthesis": {
    "agreements": ["..."],
    "conflicts": ["..."],
    "hypotheses": ["..."],
    "open_questions": ["..."],
    "bibliography": [...]
  },
  "meta": {
    "total_input_tokens": 12430,
    "total_output_tokens": 3210,
    "estimated_cost_usd": 0.18,
    "duration_seconds": 34.2
  }
}
```

### GET /agents

List available domain agents.

**Response**
```json
{
  "agents": [
    {
      "domain": "mathematics",
      "display_name": "Dr. Virtanen (Mathematics)",
      "description": "Stochastic optimisation, combinatorial problems, operations research theory",
      "source_corpus": "arXiv:math, SIAM journals"
    }
  ]
}
```

### GET /health

```json
{ "status": "ok", "demo_mode": true }
```

---

## Pydantic schemas

```python
# schemas.py
from pydantic import BaseModel, Field
from enum import Enum

class DebateMode(str, Enum):
    sequential = "sequential"
    adversarial = "adversarial"
    socratic = "socratic"

class PanelRunRequest(BaseModel):
    question: str = Field(..., min_length=20, max_length=500)
    domains: list[str] = Field(..., min_length=2, max_length=4)
    mode: DebateMode = DebateMode.sequential
    turns_per_agent: int = Field(default=2, ge=1, le=4)

class CitedPaper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int
    source_url: str

class TurnResponse(BaseModel):
    turn_number: int
    domain: str
    agent_name: str
    response: str
    cited_papers: list[CitedPaper]
    moderator_intervention: dict | None

class SynthesisResponse(BaseModel):
    agreements: list[str]
    conflicts: list[str]
    hypotheses: list[str]
    open_questions: list[str]
    bibliography: list[CitedPaper]

class PanelMetaResponse(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    duration_seconds: float

class PanelResultResponse(BaseModel):
    panel_id: str
    question: str
    domains: list[str]
    turns: list[TurnResponse]
    synthesis: SynthesisResponse
    meta: PanelMetaResponse
```

---

## FastAPI app structure

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SciAgent API",
    description="Cross-domain scientific panel AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo only — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/panels/run", response_model=PanelResultResponse)
async def run_panel(request: PanelRunRequest): ...

@app.get("/agents")
async def list_agents(): ...

@app.get("/health")
async def health(): ...
```

---

## MCP server

MCP wraps the REST API as tool definitions so Claude Code / Claude Desktop
can call your agents as native tools.

```python
# mcp_server.py
# Uses the `mcp` package: pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("sciagent")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_panel",
            description="Run a cross-domain scientific panel. Assemble 2-4 domain expert agents to debate a research question and produce a grounded synthesis with citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The research question for the panel"},
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 domain names e.g. ['mathematics', 'industrial_management']"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sequential", "adversarial"],
                        "default": "sequential"
                    }
                },
                "required": ["question", "domains"]
            }
        ),
        types.Tool(
            name="list_agents",
            description="List available scientific domain agents",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # Delegate to REST API (or call composer directly)
    ...

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
```

To use in Claude Desktop, add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "sciagent": {
      "command": "python",
      "args": ["/path/to/sciagent/api/mcp_server.py"]
    }
  }
}
```

---

## Error handling

```python
from fastapi import HTTPException

# Domain not found
if domain not in DOMAINS:
    raise HTTPException(400, f"Unknown domain '{domain}'. Available: {DOMAINS}")

# Too many domains
if len(request.domains) > 4:
    raise HTTPException(400, "Maximum 4 domains per panel")

# Anthropic API error — surface cleanly
except anthropic.APIError as e:
    raise HTTPException(502, f"LLM API error: {e.message}")
```

---

## Testing this layer

```bash
# Start server
uvicorn api.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/agents

# Run a panel (takes 30-60s)
curl -X POST http://localhost:8000/panels/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What scheduling approaches minimise makespan under stochastic processing times?",
    "domains": ["mathematics", "industrial_management"],
    "turns_per_agent": 2
  }' | python -m json.tool
```

Auto-docs available at `http://localhost:8000/docs` once running.
