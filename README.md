# SciAgent — Cross-Domain Academic Expert Panel

Assemble a panel of 2–4 domain expert AI agents and make them debate your question. Each agent retrieves peer-reviewed literature via web search, cites it inline, and challenges the other agents from its own disciplinary lens. A moderator surfaces tensions between turns. A synthesis pass maps agreements, conflicts, novelty gaps, and concrete action points.

The value isn't the individual answers — it's what happens at the boundaries between domains.

![Demo UI showing a three-agent panel debate with synthesis tabs](https://raw.githubusercontent.com/mattipuh/sci-agents/main/demo/screenshot.png)

---

## How it works

```
User question
     │
     ▼
Agent 1 (e.g. Venture Finance)     ← web search → cites papers → responds
     │
  Moderator                        ← surfaces tension, asks cross-domain question
     │
Agent 2 (e.g. GTM & Growth)        ← reads prior turns → challenges + extends
     │
  Moderator
     │
Agent 3 (e.g. Entrepreneurship)    ← synthesises and dissents where warranted
     │
     ▼
Synthesis
  ├── Action points (prioritised, grounded in the debate)
  ├── Conflicts (explicit disagreements between agents)
  ├── Agreements (claims all agents aligned on)
  ├── Novelty gaps (unresolved questions worth investigating)
  └── Narrative (300–500 word prose synthesis)
```

Each agent can also be queried solo — useful when you want one expert's take without a full debate.

---

## Quickstart

**Requirements:** Docker, an [Anthropic API key](https://console.anthropic.com/)

```bash
git clone https://github.com/mattipuh/sci-agents.git
cd sci-agents
cp .env.example .env          # add your ANTHROPIC_API_KEY
docker compose up --build
```

Open http://localhost:8080

**Without Docker:**
```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8080
```

**CLI:**
```bash
python cli.py \
  --question "Should we pursue Series A now or grow to €1M ARR first?" \
  --agents venture_finance gtm_growth entrepreneurship_validation
```

---

## Cost

Each agent turn costs roughly $0.013 (Sonnet input + output + web search). A 3-agent × 2-round panel runs ~$0.15–0.25 total. The synthesis pass adds ~$0.05.

---

## The 15 expert personas

### Science & Engineering
| Domain | Agent | Expertise |
|---|---|---|
| `mathematics` | Dr. Virtanen | Stochastic optimisation, dynamical systems, formal models |
| `ops_research` | Prof. Koskinen | Scheduling, supply chain, production planning |
| `engineering` | Prof. Järvi | Systems engineering, manufacturing, physical constraints |
| `ai_ml_systems` | Dr. Korhonen | ML pipelines, MLOps, production AI, evaluation rigour |
| `solution_architecture` | Prof. Nieminen | Integration patterns, NFRs, TCO, architectural runway |

### Strategy & Business
| Domain | Agent | Expertise |
|---|---|---|
| `strategy_game_theory` | Prof. Laaksonen | Competitive dynamics, mechanism design, Nash equilibria |
| `lean_process` | Dr. Rantanen | Toyota Production System, Theory of Constraints, flow |
| `business_economics` | Prof. Salminen | Transaction costs, value creation/capture, org design |
| `finance` | Dr. Lehtinen | Financial system design, risk, regulatory economics |

### Startup & Scaleup
| Domain | Agent | Expertise |
|---|---|---|
| `entrepreneurship_validation` | Prof. Mäkinen | Customer discovery, lean validation, technology adoption |
| `consumer_market_research` | Dr. Heikkinen | Diffusion of innovations, jobs-to-be-done, WTP measurement |
| `gtm_growth` | Prof. Leinonen | Channel theory, growth loops, CAC/LTV, sales motion |
| `org_scaling` | Dr. Hämäläinen | Founder-to-CEO transition, high-growth HR, culture at scale |
| `venture_finance` | Prof. Karjalainen | Unit economics, funding dynamics, cap table, VC power law |
| `platform_business_model` | Dr. Väisänen | Network effects, platform economics, pricing strategy |

---

## Adding a new expert persona

Each persona is a single `.txt` file in `agents/personas/`. The structure is:

```
You are a professor of [field] with expertise in [areas].

Your reasoning style:
- How this expert frames problems
- What they're skeptical of
- What they prioritise

Citation discipline:
- What claims must be cited
- How to handle uncertainty

Cross-domain behaviour:
- How they challenge other agents
- What they push back on
- The one thing they're most likely to say

Tone: [one line]
```

Then register it in `agents/base_agent.py`:

```python
AVAILABLE_DOMAINS = [
    ...,
    "your_domain",
]

AGENT_DISPLAY_NAMES = {
    ...,
    "your_domain": "Prof. Surname (Field)",
}

DOMAIN_SEARCH_SCOPES = {
    ...,
    "your_domain": "site:arxiv.org OR site:ssrn.com",  # scoped to relevant sources
}
```

That's it. The agent is immediately available in the API and demo UI.

---

## API

The REST API is documented at http://localhost:8080/docs once running.

**Run a panel:**
```bash
curl -X POST http://localhost:8080/api/panels/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the research say about why most lean transformations fail?",
    "domains": ["lean_process", "org_scaling", "business_economics"],
    "turns_per_agent": 2
  }'
```

**Single expert:**
```bash
curl -X POST http://localhost:8080/api/panels/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does empirical research say about optimal equity splits between co-founders?",
    "domains": ["venture_finance"],
    "turns_per_agent": 2
  }'
```

**With company context** (frames responses to your situation):
```bash
curl -X POST http://localhost:8080/api/panels/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Should we build or buy our data pipeline?",
    "domains": ["ai_ml_systems", "solution_architecture", "business_economics"],
    "turns_per_agent": 2,
    "company_context": {
      "company_name": "Acme Analytics",
      "stage": "Series A",
      "team_size": "18",
      "market": "B2B SaaS, data infrastructure",
      "description": "Real-time analytics platform for e-commerce",
      "key_challenge": "Scaling ingestion pipeline to 10M events/day"
    }
  }'
```

**With reference documents** (upload PDFs/DOCX/TXT as debate context):
```bash
# Upload a document first
curl -X POST http://localhost:8080/api/documents/upload \
  -F "file=@your_report.pdf"
# Returns: {"id": "abc123", "filename": "your_report.pdf", "char_count": 12400}

# Use it in a panel run
curl -X POST http://localhost:8080/api/panels/run \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "domains": [...], "document_ids": ["abc123"]}'
```

---

## What domains are missing?

This is the question I'm most interested in. The current set covers science/engineering, strategy, and startup. Obvious gaps:

- **Healthcare / clinical research** — evidence-based medicine, clinical trial design, health economics
- **Climate & sustainability** — carbon economics, systems ecology, energy transition
- **Policy & regulation** — regulatory economics, public administration, legal theory
- **Cognitive science** — decision-making under uncertainty, behavioural economics, HCI
- **Supply chain & logistics** — global supply networks, resilience, last-mile
- **Education research** — learning science, pedagogy, curriculum design

If you build a persona for a domain that's missing, a PR is welcome. The bar is: the persona should have a distinct reasoning style that creates productive friction with the existing agents, and there should be enough peer-reviewed literature in its domain for web search to return useful results.

---

## Architecture

```
sci-agents/
├── agents/
│   ├── base_agent.py          ← DomainAgent class + registry
│   └── personas/              ← one .txt file per expert
├── panel/
│   ├── composer.py            ← orchestration loop
│   ├── moderator.py           ← cross-domain tension surfacer
│   ├── synthesis.py           ← structured output pass
│   └── models.py              ← dataclasses
├── api/
│   ├── main.py                ← FastAPI app
│   └── documents.py           ← file upload + text extraction
├── demo/
│   └── index.html             ← single-file demo UI
└── cli.py                     ← CLI entry point
```

**Tech stack:** Python 3.11, FastAPI, Anthropic Claude API (`claude-sonnet-4-6` for agents, `claude-haiku-4-5-20251001` for moderator), Docker.

Retrieval is currently Anthropic web search scoped to academic sites (arxiv, ssrn, pubmed, nber). A vector corpus pipeline (Qdrant/pgvector + BM25) is the planned upgrade for domain-specific literature collections.

---

## License

Academic and research use only. See [LICENSE](LICENSE).

For commercial licensing, contact the author.
