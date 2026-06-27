# Layer 2: Domain agents

Parent context: see `../CLAUDE.md`
Depends on: `../corpus/` (retriever interface only)

## Responsibility

Each domain agent is a Claude API call that:
1. Retrieves relevant paper chunks from its domain store
2. Composes a grounded response citing those chunks
3. Responds to what previous agents said (if in a panel turn)

This layer owns agent persona definitions and the retrieve-then-generate loop.
It does NOT own turn-taking, moderation, or synthesis — those live in `panel/`.

---

## Files to build here

```
agents/
├── CLAUDE.md
├── base_agent.py          ← DomainAgent class (the core abstraction)
├── personas.py            ← system prompt templates per domain
├── agent_registry.py      ← maps domain name → DomainAgent instance
└── prompts/
    ├── mathematics.txt
    ├── industrial_management.txt
    ├── biology.txt
    ├── finance.txt
    └── _base.txt          ← shared instructions all agents inherit
```

---

## DomainAgent class

```python
# base_agent.py
from anthropic import Anthropic
from corpus.retriever import Retriever, PaperChunk

class DomainAgent:
    def __init__(
        self,
        domain: str,
        persona: str,          # full system prompt
        retriever: Retriever,
        client: Anthropic,
        model: str = "claude-sonnet-4-6",
    ): ...

    def respond(
        self,
        question: str,
        conversation_history: list[dict],   # prior panel turns as messages
        top_k: int = 6,
    ) -> AgentTurn:
        """
        1. Build retrieval query from question + last 2 history turns
        2. Retrieve top_k chunks from own domain
        3. Build prompt: persona + retrieved chunks + history + question
        4. Call Claude API
        5. Return AgentTurn with response text + cited chunks
        """
        ...

@dataclass
class AgentTurn:
    domain: str
    agent_name: str            # display name e.g. "Dr. Chen (Mathematics)"
    response: str              # Claude's response text
    cited_chunks: list[PaperChunk]   # chunks that were passed as context
    token_usage: dict          # input_tokens, output_tokens (for cost tracking)
```

---

## System prompt structure (_base.txt + domain overlay)

Every agent inherits `_base.txt` which establishes:

```
You are a domain expert participating in a cross-disciplinary scientific panel.

Your role:
- Ground every claim in the papers retrieved for you (provided below as [CONTEXT])
- Cite papers inline using [Author, Year] notation
- Be direct about the limits of your domain's perspective
- Actively engage with what other panel members have said
- When you see a claim from another domain that your literature contradicts, say so explicitly

Format your response as flowing paragraphs. Do not use bullet lists.
End with one open question directed at the panel.

Length: 150–250 words per turn.
```

Then each `domain.txt` adds the persona overlay:

```
# mathematics.txt
You are a mathematical modeller specialising in stochastic optimisation,
combinatorial problems, and operations research theory. You think in terms of
formal models, convergence guarantees, and optimality bounds.

You are rigorous about the gap between theoretical results and empirical claims.
When an engineering or management colleague cites a heuristic, you probe whether
its performance guarantees hold under their stated assumptions.
```

---

## Agent registry

```python
# agent_registry.py
def get_agent(domain: str, retriever: Retriever, client: Anthropic) -> DomainAgent:
    """Returns a configured DomainAgent for the given domain."""
    ...

AGENT_DISPLAY_NAMES = {
    "mathematics": "Dr. Virtanen (Mathematics)",
    "industrial_management": "Prof. Koskinen (Industrial Management)",
    "biology": "Dr. Mäkinen (Biology)",
    "finance": "Dr. Lehtinen (Finance & Regulation)",
    "engineering": "Prof. Järvi (Engineering)",
    "computer_science": "Dr. Aaltonen (Computer Science)",
}
```

Use Finnish names as defaults — fits the demo context, easy to change.

---

## Retrieval query construction

Do not just pass the raw question to retrieve(). Build a richer query:

```python
def build_retrieval_query(question: str, history: list[dict]) -> str:
    """
    Combines the original question with key terms from the last 2 turns
    to bias retrieval toward the current thread of discussion.
    """
    recent_text = " ".join(
        turn["content"][-200:]   # last 200 chars of each recent turn
        for turn in history[-2:]
    )
    return f"{question} {recent_text}".strip()
```

This prevents agents from retrieving the same papers every turn as the
conversation evolves.

---

## Context window budget

A panel turn prompt has these components:
- System prompt (persona): ~300 tokens
- Retrieved chunks (6 × ~400 tokens): ~2400 tokens
- Conversation history: grows each turn, cap at last 6 turns × ~250 tokens = ~1500 tokens
- Question: ~50 tokens
- Total input: ~4250 tokens per agent turn

With `claude-sonnet-4-6` at $3/MTok input:
~4250 tokens × $3/1M = **~$0.013 per agent turn**

A 3-agent panel × 3 turns each = 9 agent turns + 1 synthesis = ~10 turns total
Estimated cost per panel run: **~$0.15–0.25** including synthesis

This is the cost floor. Price above it, not below it.

---

## Testing this layer

```bash
# Single agent smoke test
python -c "
import asyncio
from anthropic import Anthropic
from corpus.stub_retriever import StubRetriever
from agents.agent_registry import get_agent

client = Anthropic()
retriever = StubRetriever()
agent = get_agent('mathematics', retriever, client)

turn = agent.respond(
    question='What scheduling approaches minimise makespan under stochastic processing times?',
    conversation_history=[]
)
print(turn.agent_name)
print(turn.response[:300])
print(f'Cited {len(turn.cited_chunks)} chunks')
print(f'Tokens: {turn.token_usage}')
"
```

Expected: a 150–250 word response with at least one inline citation and
one open question at the end.
