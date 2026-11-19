"""Exercise the LangGraph loop with a scripted fake model (no network)."""

from __future__ import annotations

import json
from typing import Any

from anthropic.types import Message

from investigator.agent import InvestigatorAgent, parse_json_answer
from investigator.graph_store import SanctionsGraph
from investigator.llm import ModelConfig


def _message(content: list[dict[str, Any]], stop: str) -> Message:
    return Message.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "fake",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
    )


def _final(answer: dict[str, Any]) -> Message:
    return _message([{"type": "text", "text": json.dumps(answer)}], "end_turn")


def _tool(name: str, args: dict[str, Any], tool_id: str) -> Message:
    return _message([{"type": "tool_use", "id": tool_id, "name": name, "input": args}], "tool_use")


PLAN = _message(
    [{"type": "text", "text": json.dumps({"intent": "name_lookup", "entities": ["X"], "steps": ["search"]})}],
    "end_turn",
)


class FakeMessages:
    def __init__(self, script: list[Message]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script: list[Message]) -> None:
        self.messages = FakeMessages(script)


def _agent(graph: SanctionsGraph, script: list[Message]) -> tuple[InvestigatorAgent, FakeClient]:
    client = FakeClient(script)
    agent = InvestigatorAgent(graph, config=ModelConfig(), client=client)  # type: ignore[arg-type]
    return agent, client


def _claim(field: str, value: str, ids: list[int]) -> dict[str, Any]:
    return {"text": f"{field} {value}", "field": field, "value": value, "record_ids": ids, "match_confidence": "exact"}


def test_tool_loop_and_verified_answer(graph: SanctionsGraph) -> None:
    answer = {"status": "answered", "answer": "Listed under CUBA.", "claims": [_claim("program", "CUBA", [306])]}
    agent, client = _agent(graph, [PLAN, _tool("get_record", {"record_id": 306}, "t1"), _final(answer)])
    result = agent.ask("Is BANCO NACIONAL DE CUBA listed?", thread_id="a")
    assert result["status"] == "answered"
    assert [c["verified"] for c in result["claims"]] == [True]
    assert result["dropped_claims"] == []
    assert result["citations"][0]["record_id"] == 306
    assert result["usage"]["calls"] == 3
    agent_request = client.messages.requests[1]
    assert agent_request["output_config"]["format"]["type"] == "json_schema"
    assert agent_request["messages"][-1]["role"] == "user"


def test_verifier_retry_then_drop(graph: SanctionsGraph) -> None:
    wrong = {"status": "answered", "answer": "Listed under SDGT.", "claims": [_claim("program", "SDGT", [306])]}
    agent, client = _agent(graph, [PLAN, _tool("get_record", {"record_id": 306}, "t1"), _final(wrong), _final(wrong)])
    result = agent.ask("Which program is BANCO NACIONAL DE CUBA under?", thread_id="b")
    assert result["verify_retries"] == 1
    assert result["claims"] == []
    assert len(result["dropped_claims"]) == 1
    assert not result["first_draft_checks"][0]["supported"]
    feedback = client.messages.requests[3]["messages"][-1]["content"]
    assert "Citation check failed" in feedback


def test_uncited_record_is_rejected(graph: SanctionsGraph) -> None:
    sneaky = {"status": "answered", "answer": "x", "claims": [_claim("program", "CYBER4", [55046])]}
    fixed = {"status": "answered", "answer": "x", "claims": []}
    agent, _ = _agent(graph, [PLAN, _final(sneaky), _final(fixed)])
    result = agent.ask("Is OLD VECTOR LLC listed?", thread_id="c")
    assert "never retrieved" in result["first_draft_checks"][0]["reason"]


def test_memory_across_turns(graph: SanctionsGraph) -> None:
    first = {
        "status": "answered",
        "answer": "Found it.",
        "claims": [_claim("listed", "IRAN INSURANCE COMPANY", [11631])],
    }
    second = {"status": "answered", "answer": "Program IRAN.", "claims": [_claim("program", "IRAN", [11631])]}
    agent, client = _agent(
        graph,
        [
            PLAN,
            _tool("search_entities", {"name": "IRAN INSURANCE COMPANY", "limit": 3}, "t1"),
            _final(first),
            PLAN,
            _final(second),
        ],
    )
    agent.ask("Look up IRAN INSURANCE COMPANY.", thread_id="mem")
    result = agent.ask("Which program is it under?", thread_id="mem")
    # The second turn cites a record retrieved in the first turn, which memory keeps.
    assert result["claims"] and result["claims"][0]["verified"]
    second_turn_request = client.messages.requests[4]
    assert len(second_turn_request["messages"]) > 2
    planner_prompt = client.messages.requests[3]["messages"][0]["content"]
    assert "Earlier question: Look up IRAN INSURANCE COMPANY." in planner_prompt


def test_tool_budget_forces_answer(graph: SanctionsGraph) -> None:
    calls = [_tool("search_entities", {"name": "X", "limit": 1}, f"t{i}") for i in range(6)]
    done = {"status": "not_found", "answer": "No match.", "claims": []}
    agent, client = _agent(graph, [PLAN, *calls, _final(done)])
    result = agent.ask("Is X listed?", thread_id="loop")
    assert result["tool_rounds"] == 6
    assert client.messages.requests[-1]["tool_choice"] == {"type": "none"}


def test_empty_answer_gets_one_retry(graph: SanctionsGraph) -> None:
    empty = {"status": "answered", "answer": "", "claims": []}
    good = {"status": "answered", "answer": "Listed under CUBA.", "claims": [_claim("program", "CUBA", [306])]}
    agent, client = _agent(graph, [PLAN, _final(empty), _tool("get_record", {"record_id": 306}, "t1"), _final(good)])
    result = agent.ask("Which program is BANCO NACIONAL DE CUBA under?", thread_id="empty")
    assert result["verify_retries"] == 1
    assert result["answer_problems"] == []
    assert result["claims"][0]["verified"]
    assert "answer text is empty" in client.messages.requests[2]["messages"][-1]["content"]


def test_parse_json_answer() -> None:
    assert parse_json_answer('```json\n{"status": "refused", "answer": "no"}\n```') == {
        "status": "refused",
        "answer": "no",
        "claims": [],
    }
    assert parse_json_answer("not json") is None
