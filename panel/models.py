from dataclasses import dataclass, field

from agents.base_agent import AgentTurn


@dataclass
class ModeratorIntervention:
    text: str
    directed_at: str
    tension_detected: bool = False


@dataclass
class PanelTurn:
    turn_number: int
    agent_turn: AgentTurn
    moderator: ModeratorIntervention | None = None


@dataclass
class Synthesis:
    agreements: list[dict]
    conflicts: list[dict]
    novelty_gaps: list[dict]
    narrative: str
    all_citations: list[dict]


@dataclass
class PanelConfig:
    question: str
    domains: list[str]
    turns_per_agent: int = 2
    reference_docs: list[dict] = field(default_factory=list)


@dataclass
class PanelResult:
    config: PanelConfig
    turns: list[PanelTurn]
    synthesis: Synthesis
    token_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
