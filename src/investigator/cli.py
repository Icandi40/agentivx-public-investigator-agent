"""Command line: ask one question, or chat with conversation memory.

uv run investigator ask "Is BANCO NACIONAL DE CUBA on the SDN list?"
uv run investigator chat
uv run investigator stats
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from investigator.graph_store import SanctionsGraph


def render(result: dict[str, Any]) -> str:
    lines = [f"[{result['status']}] {result['answer']}"]
    for claim in result.get("claims", []):
        ids = ", ".join(str(i) for i in claim.get("record_ids", [])) or "-"
        conf = claim.get("match_confidence", "")
        label = f" ({conf} match)" if conf in {"high", "medium", "low"} else ""
        lines.append(f"  - {claim['text']}{label} [SDN record {ids}]")
    for claim in result.get("dropped_claims", []):
        lines.append(f"  ! dropped unverified claim: {claim.get('text')} ({claim.get('reason')})")
    for cite in result.get("citations", []):
        lines.append(f"  source: {cite['source']} ({cite['publication_date']}), {cite['url']}")
    lines.append(f"  {result.get('latency_s', 0):.1f}s, ${result.get('cost_usd', 0):.4f}, model {result.get('model')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="investigator")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="print the raw result")
    sub.add_parser("chat", help="interactive session with memory")
    sub.add_parser("stats", help="print knowledge graph stats")
    args = parser.parse_args(argv)

    graph = SanctionsGraph.load()
    if args.command == "stats":
        print(json.dumps(graph.stats() | {"publication_date": graph.snapshot.publication_date}, indent=2))
        return 0

    from investigator.agent import InvestigatorAgent

    agent = InvestigatorAgent(graph)
    if args.command == "ask":
        result = agent.ask(args.question, thread_id="cli")
        if args.json:
            result.pop("trace", None)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(render(result))
        return 0

    print("Ask about the OFAC SDN list. Empty line to quit.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        print(render(agent.ask(question, thread_id="chat")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
