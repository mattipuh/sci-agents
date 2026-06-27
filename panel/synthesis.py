import logging

import anthropic

from agents.base_agent import _extract_json
from panel.models import Synthesis, PanelTurn

log = logging.getLogger(__name__)


SYNTHESIS_SYSTEM = """You are a senior research synthesiser. You have just observed a multi-agent
scientific debate. Produce a structured analysis.

Your output MUST be valid JSON matching this schema exactly:
{
  "agreements": [
    {"claim": "...", "supported_by": ["domain1", "domain2"]}
  ],
  "conflicts": [
    {"description": "...", "position_a": {"domain": "...", "claim": "..."}, "position_b": {"domain": "...", "claim": "..."}}
  ],
  "novelty_gaps": [
    {"description": "...", "why_unresolved": "..."}
  ],
  "narrative": "300-500 word prose integrating the debate findings"
}

Be specific. Reference specific claims from specific agents. Do not generalise."""


def run_synthesis(
    client: anthropic.Anthropic,
    question: str,
    turns: list[PanelTurn],
) -> tuple[Synthesis, dict]:
    debate_text = f"Question: {question}\n\n=== Agent turns ===\n"
    for t in turns:
        at = t.agent_turn
        cites = "; ".join(c.get("title", "")[:60] for c in at.citations[:3])
        debate_text += f"\n[{at.agent_name}] (confidence {at.confidence}):\n{at.content}\nCitations: {cites}\n"
        if at.dissent:
            debate_text += f"Dissent: {at.dissent}\n"
        if t.moderator:
            debate_text += f"\n  → Moderator: {t.moderator.text}\n"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": debate_text}],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

    if response.stop_reason == "max_tokens":
        log.warning("Synthesis response truncated at max_tokens")

    text = next((b.text for b in response.content if hasattr(b, "text")), "{}")
    parsed = _extract_json(text)
    if not parsed:
        log.warning("Synthesis JSON extraction failed, raw text: %s", text[:200])
        parsed = {"agreements": [], "conflicts": [], "novelty_gaps": [], "narrative": text}

    all_citations = []
    seen = set()
    for t in turns:
        for c in t.agent_turn.citations:
            key = c.get("doi") or c.get("url") or c.get("title", "")
            if key and key not in seen:
                seen.add(key)
                all_citations.append(c)

    return Synthesis(
        agreements=parsed.get("agreements", []),
        conflicts=parsed.get("conflicts", []),
        novelty_gaps=parsed.get("novelty_gaps", []),
        narrative=parsed.get("narrative", ""),
        all_citations=all_citations,
    ), usage
