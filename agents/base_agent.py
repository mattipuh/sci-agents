import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)


AVAILABLE_DOMAINS = [
    "mathematics", "ops_research", "finance", "engineering",
    "strategy_game_theory", "lean_process", "business_economics",
    "ai_ml_systems", "solution_architecture",
    "entrepreneurship_validation", "consumer_market_research", "gtm_growth",
    "org_scaling", "venture_finance", "platform_business_model",
]

AGENT_DISPLAY_NAMES = {
    "mathematics": "Dr. Virtanen (Mathematics)",
    "ops_research": "Prof. Koskinen (Operations Research)",
    "finance": "Dr. Lehtinen (Finance & Regulation)",
    "engineering": "Prof. Järvi (Engineering)",
    "strategy_game_theory": "Prof. Laaksonen (Strategy & Game Theory)",
    "lean_process": "Dr. Rantanen (Lean & Process Engineering)",
    "business_economics": "Prof. Salminen (Business & Organizational Economics)",
    "ai_ml_systems": "Dr. Korhonen (AI/ML Systems)",
    "solution_architecture": "Prof. Nieminen (Solution Architecture)",
    "entrepreneurship_validation": "Prof. Mäkinen (Entrepreneurship & Validation)",
    "consumer_market_research": "Dr. Heikkinen (Consumer & Market Research)",
    "gtm_growth": "Prof. Leinonen (GTM & Growth)",
    "org_scaling": "Dr. Hämäläinen (Organizational Scaling)",
    "venture_finance": "Prof. Karjalainen (Venture Finance)",
    "platform_business_model": "Dr. Väisänen (Platform & Business Model)",
}

DOMAIN_SEARCH_SCOPES = {
    "mathematics": "site:arxiv.org",
    "ops_research": "site:arxiv.org/abs/math.OC OR site:arxiv.org/abs/cs.GT",
    "finance": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/q-fin",
    "engineering": "site:arxiv.org/abs/eess OR site:arxiv.org/abs/cs.SY",
    "strategy_game_theory": "site:ssrn.com OR site:arxiv.org/abs/cs.GT OR site:arxiv.org/abs/econ",
    "lean_process": "site:arxiv.org OR site:scholar.google.com OR site:emerald.com",
    "business_economics": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/econ",
    "ai_ml_systems": "site:arxiv.org/abs/cs.LG OR site:arxiv.org/abs/cs.AI OR site:arxiv.org/abs/cs.SE",
    "solution_architecture": "site:arxiv.org/abs/cs.SE OR site:ieeexplore.ieee.org OR site:acm.org",
    "entrepreneurship_validation": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/econ",
    "consumer_market_research": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/econ",
    "gtm_growth": "site:ssrn.com OR site:nber.org OR site:hbswk.hbs.edu",
    "org_scaling": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/econ",
    "venture_finance": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/q-fin",
    "platform_business_model": "site:ssrn.com OR site:nber.org OR site:arxiv.org/abs/econ",
}

PERSONAS_DIR = Path(__file__).parent / "personas"


@dataclass
class AgentTurn:
    domain: str
    agent_name: str
    content: str
    citations: list[dict]
    token_usage: dict = field(default_factory=dict)
    confidence: float = 0.7
    dissent: str | None = None


class DomainAgent:
    def __init__(self, domain: str, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6"):
        if domain not in AVAILABLE_DOMAINS:
            raise ValueError(f"Unknown domain: {domain}. Choose from {AVAILABLE_DOMAINS}")
        self.domain = domain
        self.client = client
        self.model = model
        self.display_name = AGENT_DISPLAY_NAMES[domain]
        self._persona = self._load_persona()

    def _load_persona(self) -> str:
        base = (PERSONAS_DIR / "_base.txt").read_text()
        domain_file = PERSONAS_DIR / f"{self.domain}.txt"
        overlay = domain_file.read_text() if domain_file.exists() else ""
        return f"{base}\n\n{overlay}"

    def respond(self, question: str, prior_turns: list[AgentTurn], reference_docs: list[dict] | None = None) -> AgentTurn:
        scope = DOMAIN_SEARCH_SCOPES.get(self.domain, "site:arxiv.org")

        prior_context = ""
        if prior_turns:
            prior_context = "\n\nPrevious agent responses:\n"
            for t in prior_turns:
                cites = ", ".join(c.get("title", "")[:50] for c in t.citations[:2])
                prior_context += f"\n[{t.agent_name}]: {t.content[:400]}...\n  (cited: {cites})\n"

        docs_context = ""
        if reference_docs:
            docs_context = "\n\nReference documents provided by the user (use these as additional context alongside your literature search):\n"
            for doc in reference_docs:
                text = doc["text"][:15000]
                docs_context += f"\n--- {doc['filename']} ---\n{text}\n"

        user_message = f"""Research question: {question}
{prior_context}{docs_context}

Search for relevant literature using the web search tool (scoped to {self.domain} literature: {scope}).
Then provide your expert response.

Your response MUST follow this exact JSON structure:
{{
  "content": "Your substantive response (200-400 words, flowing paragraphs, inline [Author, Year] citations)",
  "citations": [
    {{"title": "Paper title", "year": 2023, "url": "https://...", "doi": "10.xxx/xxx"}},
    {{"title": "Paper title", "year": 2021, "url": "https://...", "doi": ""}}
  ],
  "confidence": 0.8,
  "dissent": "If you disagree with a prior agent, state it here. Otherwise null."
}}

Provide ONLY the JSON object, no other text."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self._persona,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": user_message}],
        )

        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        if response.stop_reason == "max_tokens":
            log.warning("%s response truncated at max_tokens", self.display_name)

        parsed = _extract_json(text_content)
        if parsed and "content" in parsed:
            return AgentTurn(
                domain=self.domain,
                agent_name=self.display_name,
                content=parsed["content"],
                citations=parsed.get("citations", []),
                token_usage=usage,
                confidence=float(parsed.get("confidence", 0.7)),
                dissent=parsed.get("dissent"),
            )
        return AgentTurn(
            domain=self.domain,
            agent_name=self.display_name,
            content=text_content,
            citations=[],
            token_usage=usage,
            confidence=0.5,
        )


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text that may contain preamble or markdown fences."""
    # Try each { as a potential start and use json.JSONDecoder to find valid JSON
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        pos = text.find("{", idx)
        if pos == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = pos + 1
    return None
