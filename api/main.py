import os
import time
import logging

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.base_agent import AVAILABLE_DOMAINS, AGENT_DISPLAY_NAMES
from api.documents import store_document, get_documents
from panel.composer import PanelComposer
from panel.models import PanelConfig

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

app = FastAPI(title="SciAgent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


# --- Request / Response models ---

class PanelRunRequest(BaseModel):
    question: str = Field(..., min_length=20, max_length=500)
    domains: list[str] = Field(..., min_length=2, max_length=4)
    turns_per_agent: int = Field(default=2, ge=1, le=3)
    document_ids: list[str] = Field(default_factory=list)


class CitedPaper(BaseModel):
    title: str = ""
    year: int = 0
    url: str = ""
    doi: str = ""


class TurnResponse(BaseModel):
    turn_number: int
    domain: str
    agent_name: str
    response: str
    cited_papers: list[CitedPaper]
    confidence: float
    dissent: str | None
    moderator: dict | None


class SynthesisResponse(BaseModel):
    agreements: list[dict]
    conflicts: list[dict]
    novelty_gaps: list[dict]
    narrative: str
    bibliography: list[CitedPaper]


class PanelResultResponse(BaseModel):
    question: str
    domains: list[str]
    turns: list[TurnResponse]
    synthesis: SynthesisResponse
    meta: dict


class AgentInfo(BaseModel):
    domain: str
    display_name: str


# --- Routes ---

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    content = await file.read()
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(content) > max_size:
        raise HTTPException(400, "File too large (max 10 MB)")
    try:
        doc = store_document(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": doc.id, "filename": doc.filename, "char_count": doc.char_count, "preview": doc.text[:200]}


@app.get("/api/agents")
def list_agents():
    return {
        "agents": [
            AgentInfo(domain=d, display_name=AGENT_DISPLAY_NAMES[d])
            for d in AVAILABLE_DOMAINS
        ]
    }


@app.post("/api/panels/run", response_model=PanelResultResponse)
def run_panel(req: PanelRunRequest):
    for d in req.domains:
        if d not in AVAILABLE_DOMAINS:
            raise HTTPException(400, f"Unknown domain '{d}'. Available: {AVAILABLE_DOMAINS}")

    ref_docs = []
    if req.document_ids:
        docs = get_documents(req.document_ids)
        ref_docs = [{"filename": d.filename, "text": d.text} for d in docs]

    client = get_client()
    composer = PanelComposer(client=client)
    config = PanelConfig(
        question=req.question,
        domains=req.domains,
        turns_per_agent=req.turns_per_agent,
        reference_docs=ref_docs,
    )

    start = time.time()
    result = composer.run(config)
    duration = time.time() - start

    cost = (result.token_usage["input_tokens"] * 3 + result.token_usage["output_tokens"] * 15) / 1_000_000

    turns = []
    for t in result.turns:
        at = t.agent_turn
        turns.append(TurnResponse(
            turn_number=t.turn_number,
            domain=at.domain,
            agent_name=at.agent_name,
            response=at.content,
            cited_papers=[CitedPaper(**c) for c in at.citations],
            confidence=at.confidence,
            dissent=at.dissent,
            moderator={"text": t.moderator.text, "directed_at": t.moderator.directed_at, "tension_detected": t.moderator.tension_detected} if t.moderator else None,
        ))

    synthesis = SynthesisResponse(
        agreements=result.synthesis.agreements,
        conflicts=result.synthesis.conflicts,
        novelty_gaps=result.synthesis.novelty_gaps,
        narrative=result.synthesis.narrative,
        bibliography=[CitedPaper(**c) for c in result.synthesis.all_citations],
    )

    return PanelResultResponse(
        question=config.question,
        domains=config.domains,
        turns=turns,
        synthesis=synthesis,
        meta={
            "total_input_tokens": result.token_usage["input_tokens"],
            "total_output_tokens": result.token_usage["output_tokens"],
            "estimated_cost_usd": round(cost, 4),
            "duration_seconds": round(duration, 1),
        },
    )


# Serve demo UI — mount AFTER api routes
demo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo")
if os.path.isdir(demo_dir):
    app.mount("/static", StaticFiles(directory=demo_dir), name="static")

    @app.get("/")
    def serve_demo():
        return FileResponse(os.path.join(demo_dir, "index.html"))
