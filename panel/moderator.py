import json

import anthropic

from agents.base_agent import AgentTurn
from panel.models import ModeratorIntervention


MODERATOR_SYSTEM = """You are a senior interdisciplinary research moderator.
Your job is NOT to answer questions — it is to ask better ones.

After reading the agents' responses, identify:
1. The most productive tension between domains that has not yet been addressed
2. A hidden assumption in one agent's argument that another agent should challenge
3. A case where two agents use the same term to mean different things

Generate a single, sharp cross-domain question directed at a specific agent.
The question should advance the debate, not summarise it.

Respond with ONLY a JSON object:
{"text": "Your question here", "directed_at": "domain_name", "tension_detected": true}"""

FALLBACK = ModeratorIntervention(
    text="What assumption made by one of the agents would most change the conclusion if it were false?",
    directed_at="all",
)


def run_moderator(
    client: anthropic.Anthropic,
    question: str,
    recent_turns: list[AgentTurn],
) -> tuple[ModeratorIntervention, dict]:
    turns_text = "\n\n".join(
        f"[{t.agent_name}]: {t.content[:500]}" for t in recent_turns
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=MODERATOR_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {question}\n\nAgent responses:\n{turns_text}"}],
        )
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        text = next((b.text for b in response.content if hasattr(b, "text")), "{}")
        clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(clean)
        return ModeratorIntervention(
            text=parsed.get("text", parsed.get("prompt", "")),
            directed_at=parsed.get("directed_at", "all"),
            tension_detected=parsed.get("tension_detected", False),
        ), usage
    except Exception:
        return FALLBACK, {"input_tokens": 0, "output_tokens": 0}
