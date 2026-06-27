"""
SciAgent CLI — run a cross-domain scientific panel from the terminal.

Usage:
  python cli.py
  python cli.py --question "..." --agents mathematics ops_research
  python cli.py --output report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

from agents.base_agent import AVAILABLE_DOMAINS, AGENT_DISPLAY_NAMES
from panel.composer import PanelComposer
from panel.models import PanelConfig

EXAMPLE_QUESTIONS = [
    "What scheduling approaches minimise makespan in a job shop with stochastic processing times?",
    "What does the academic literature say about the gap between regulatory compliance frameworks and actual operational risk reduction in financial institutions?",
    "What are the fundamental tradeoffs between energy density, cycle life, and safety in lithium-ion battery chemistries, and what do operations researchers say about which tradeoffs matter most at grid scale?",
]


def pick_question() -> str:
    print("\nExample questions (or type your own):\n")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q[:90]}...")
    choice = input("\nNumber or Enter to type your own: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_QUESTIONS):
        return EXAMPLE_QUESTIONS[int(choice) - 1]
    return input("Your question: ").strip()


def pick_agents() -> list[str]:
    print(f"\nAvailable: {', '.join(AVAILABLE_DOMAINS)}")
    raw = input("Choose 2-4 agents (space-separated): ").strip()
    chosen = [a.lower().replace("-", "_") for a in raw.split()]
    valid = [a for a in chosen if a in AVAILABLE_DOMAINS]
    if len(valid) < 2:
        print(f"Need at least 2 valid agents. Got: {valid}")
        sys.exit(1)
    return valid[:4]


def main():
    parser = argparse.ArgumentParser(description="SciAgent cross-domain panel")
    parser.add_argument("--question", type=str)
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--rounds", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--output", type=str, help="Write JSON report to file")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    print("\n=== SciAgent — cross-domain scientific panel ===\n")

    question = args.question or pick_question()
    domains = args.agents or pick_agents()

    print(f"\nQuestion: {question}")
    print(f"Panel: {' · '.join(AGENT_DISPLAY_NAMES.get(d, d) for d in domains)}\n")

    client = anthropic.Anthropic()
    composer = PanelComposer(client=client)
    config = PanelConfig(question=question, domains=domains, turns_per_agent=args.rounds)

    print("Running panel...\n")
    result = composer.run(config)

    for t in result.turns:
        at = t.agent_turn
        print(f"\n[{at.agent_name}]")
        print(at.content[:600])
        for c in at.citations[:3]:
            print(f"  → {c.get('title', '')} ({c.get('year', '')})")
        if t.moderator:
            print(f"\n  [MODERATOR] {t.moderator.text}")

    print("\n=== SYNTHESIS ===")
    for a in result.synthesis.agreements:
        print(f"  AGREE: {a.get('claim', a)}")
    for c in result.synthesis.conflicts:
        print(f"  CONFLICT: {c.get('description', c)}")
    for g in result.synthesis.novelty_gaps:
        print(f"  GAP: {g.get('description', g)}")
    print(f"\n{result.synthesis.narrative}")

    cost = (result.token_usage["input_tokens"] * 3 + result.token_usage["output_tokens"] * 15) / 1_000_000
    print(f"\nTokens: {result.token_usage['input_tokens']:,} in / {result.token_usage['output_tokens']:,} out  |  Cost: ${cost:.3f}")

    if args.output:
        report = {
            "question": config.question,
            "domains": config.domains,
            "synthesis": {
                "agreements": result.synthesis.agreements,
                "conflicts": result.synthesis.conflicts,
                "novelty_gaps": result.synthesis.novelty_gaps,
                "narrative": result.synthesis.narrative,
            },
            "bibliography": result.synthesis.all_citations,
            "token_usage": result.token_usage,
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
