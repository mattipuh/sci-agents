import logging
from typing import Generator

import anthropic

from agents.base_agent import DomainAgent, AgentTurn
from panel.models import PanelConfig, PanelResult, PanelTurn
from panel.moderator import run_moderator
from panel.synthesis import run_synthesis

log = logging.getLogger(__name__)


class PanelComposer:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def run(self, config: PanelConfig) -> PanelResult:
        """Run a full panel synchronously. Returns the complete result."""
        panel_turns: list[PanelTurn] = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for event in self.run_stream(config):
            u = event.get("token_usage", {})
            total_usage["input_tokens"] += u.get("input_tokens", 0)
            total_usage["output_tokens"] += u.get("output_tokens", 0)
            if event["type"] == "agent_turn":
                panel_turns.append(event["panel_turn"])

        log.info("Running synthesis")
        synthesis, synth_usage = run_synthesis(self.client, config.question, panel_turns)
        total_usage["input_tokens"] += synth_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += synth_usage.get("output_tokens", 0)

        return PanelResult(
            config=config,
            turns=panel_turns,
            synthesis=synthesis,
            token_usage=total_usage,
        )

    def run_stream(self, config: PanelConfig) -> Generator[dict, None, None]:
        """Yield events as the panel debate progresses."""
        agents = [DomainAgent(domain=d, client=self.client) for d in config.domains]
        all_agent_turns: list[AgentTurn] = []
        turn_number = 0

        for round_num in range(1, config.turns_per_agent + 1):
            yield {"type": "round_start", "round": round_num, "token_usage": {}}

            for agent in agents:
                turn_number += 1
                log.info("Round %d: %s responding", round_num, agent.display_name)
                agent_turn = agent.respond(config.question, all_agent_turns, reference_docs=config.reference_docs)
                all_agent_turns.append(agent_turn)

                moderator = None
                mod_usage = {}
                if len(all_agent_turns) >= 2:
                    moderator, mod_usage = run_moderator(
                        self.client, config.question, all_agent_turns[-2:]
                    )

                panel_turn = PanelTurn(
                    turn_number=turn_number,
                    agent_turn=agent_turn,
                    moderator=moderator,
                )

                combined_usage = {
                    "input_tokens": agent_turn.token_usage.get("input_tokens", 0) + mod_usage.get("input_tokens", 0),
                    "output_tokens": agent_turn.token_usage.get("output_tokens", 0) + mod_usage.get("output_tokens", 0),
                }
                yield {"type": "agent_turn", "panel_turn": panel_turn, "token_usage": combined_usage}
