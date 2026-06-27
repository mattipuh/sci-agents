# Layer 3: Panel composer

Parent context: see `../CLAUDE.md`
Depends on: `../agents/` (DomainAgent), `../corpus/` (retriever)

## Responsibility

This layer owns the panel execution loop:
- Accepting a panel configuration (which agents, what question, what mode)
- Running the debate (turn-taking, routing, history management)
- Running the moderator between turns
- Running the synthesis pass at the end
- Returning a complete `PanelResult`

Nothing above this layer manages turn order or calls agents directly.

---

## Files to build here

```
panel/
├── CLAUDE.md
├── composer.py          ← PanelComposer: main orchestrator
├── moderator.py         ← ModeratorAgent: cross-domain tension surfacer
├── synthesis.py         ← SynthesisEngine: post-debate structured output
├── models.py            ← PanelConfig, PanelResult, PanelTurn dataclasses
└── prompts/
    ├── moderator.txt
    └── synthesis.txt
```

---

## Data models

```python
# models.py
from dataclasses import dataclass, field
from agents.base_agent import AgentTurn

@dataclass
class PanelConfig:
    question: str
    domains: list[str]              # 2–4 domain names
    mode: str = "sequential"        # "sequential" | "adversarial" | "socratic"
    turns_per_agent: int = 2        # how many times each agent speaks
    panel_id: str = ""              # auto-generated UUID if empty

@dataclass
class ModeratorIntervention:
    text: str                       # moderator's bridging question or observation
    tension_detected: bool          # did the moderator flag a cross-domain conflict?
    tension_description: str = ""

@dataclass
class PanelTurn:
    turn_number: int
    domain: str
    agent_turn: AgentTurn
    moderator: ModeratorIntervention | None   # None on first turn

@dataclass
class Synthesis:
    agreements: list[str]           # claims all agents aligned on
    conflicts: list[str]            # explicit tensions between domains
    hypotheses: list[str]           # novel cross-domain research directions
    open_questions: list[str]       # unresolved questions surfaced by agents
    all_cited_papers: list[dict]    # deduplicated bibliography

@dataclass
class PanelResult:
    config: PanelConfig
    turns: list[PanelTurn]
    synthesis: Synthesis
    total_tokens: dict              # input_tokens, output_tokens, total_cost_usd
```

---

## PanelComposer orchestration loop

```python
# composer.py
class PanelComposer:
    def run(self, config: PanelConfig) -> PanelResult:
        """
        Sequential mode execution:

        for round in range(config.turns_per_agent):
            for domain in config.domains:
                1. agent = get_agent(domain)
                2. if not first turn: run moderator on last 2 turns
                3. agent.respond(question, history)
                4. append AgentTurn to history
                5. append PanelTurn to result

        Then: run synthesis on full history
        """
```

Keep the loop synchronous for the demo. Async parallelism is a later optimisation.

### Conversation history format

Pass history to agents as Anthropic messages format:
```python
[
    {
        "role": "user",
        "content": f"[{turn.domain.upper()} AGENT - Turn {turn.turn_number}]\n{turn.agent_turn.response}"
    },
    ...
]
```

Each agent sees all prior turns as "user" messages, with domain labels.
This is unconventional but works well — agents read each other as peers,
not as assistant outputs.

---

## Moderator agent

The moderator runs BETWEEN turns (not as a participant).
It reads the last 2 agent turns and produces a `ModeratorIntervention`.

```
# moderator.txt system prompt

You are the moderator of a cross-disciplinary scientific panel.
Your job is to surface productive tensions and keep the debate generative.

After each agent speaks, you receive the last two turns.
Your output is a short bridging statement (1–3 sentences) that:
1. Acknowledges what was just said
2. Points at the most interesting cross-domain tension or gap
3. Poses ONE sharp question to the next agent

You do NOT state conclusions. You create productive pressure.

If the agents are agreeing too smoothly, introduce a challenge.
If they are talking past each other, name the real disagreement.

Output format — JSON only:
{
  "text": "...",
  "tension_detected": true,
  "tension_description": "..."
}
```

---

## Synthesis engine

Runs once after all agent turns complete. Reads the full panel transcript.

```
# synthesis.txt system prompt

You are producing the structured synthesis of a cross-disciplinary scientific panel.

You have received the full transcript of a debate between domain experts.
Each agent's turns are grounded in retrieved scientific literature.

Produce a JSON synthesis with these fields:

{
  "agreements": ["..."],           // 2–5 claims all agents aligned on, with citations
  "conflicts": ["..."],            // 2–5 explicit tensions between domains
  "hypotheses": ["..."],           // 2–5 novel cross-domain research directions that emerged
  "open_questions": ["..."],       // 2–5 unresolved questions worth investigating
}

Be specific. Reference paper titles and authors.
Conflicts and hypotheses are the highest-value outputs — spend your tokens there.
Do not pad with obvious observations.
```

---

## Debate modes

### Sequential (default — build this first)
Agents speak in round-robin order. Each hears all prior turns.
`Math → Ops → Math → Ops → Synthesis`

### Adversarial (add after sequential works)
Agents are prompted to find flaws in the previous agent's evidence.
Add to each agent's system prompt:
```
This is an adversarial panel. Your role is to identify the weakest empirical
claim in the previous agent's turn and challenge it with evidence from your domain.
```

### Socratic (add last)
The moderator never states conclusions — only asks questions.
Agents must surface assumptions rather than defend conclusions.
Hardest to implement but produces the most novel hypotheses.

---

## Testing this layer

```bash
python -c "
from corpus.stub_retriever import StubRetriever
from agents.agent_registry import get_agent
from panel.composer import PanelComposer
from panel.models import PanelConfig
from anthropic import Anthropic

client = Anthropic()
retriever = StubRetriever()

config = PanelConfig(
    question='What scheduling approaches minimise makespan in a job shop with stochastic processing times, and what are the practical barriers to implementing the theoretically optimal solution?',
    domains=['mathematics', 'industrial_management'],
    mode='sequential',
    turns_per_agent=2,
)

composer = PanelComposer(client=client, retriever=retriever)
result = composer.run(config)

print(f'Turns: {len(result.turns)}')
print(f'Agreements: {len(result.synthesis.agreements)}')
print(f'Conflicts: {len(result.synthesis.conflicts)}')
print(f'Hypotheses: {len(result.synthesis.hypotheses)}')
print(f'Total cost: \${result.total_tokens[\"total_cost_usd\"]:.4f}')
"
```

Expected: 4 turns, 2–3 agreements, 2–3 conflicts, 2–3 hypotheses, cost < $0.30.
