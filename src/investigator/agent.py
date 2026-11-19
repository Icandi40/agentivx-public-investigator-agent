"""LangGraph agent: plan, call tools, answer in structured JSON, verify citations.

Graph:  START -> start_turn -> plan -> agent <-> tools
                                        agent -> verify -> (agent once, if claims fail) -> finalize -> END

Conversation memory is LangGraph's checkpointer: state (messages, evidence
trace, prior turns) is stored per thread_id, so follow-up questions see the
earlier turns and the records already retrieved.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, TypedDict

import anthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from investigator.citations import ANSWER_SCHEMA, check_answer
from investigator.graph_store import SanctionsGraph
from investigator.llm import ModelConfig, Usage, make_client
from investigator.tools import TOOL_DEFINITIONS, Toolbox, ToolTrace

MAX_TOOL_ROUNDS = 6
MAX_VERIFY_RETRIES = 1

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "name_lookup",
                "alias_lookup",
                "relationship",
                "aggregate",
                "identifier_lookup",
                "vessel",
                "follow_up",
                "speculation_or_private_info",
                "out_of_scope",
            ],
        },
        "entities": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "entities", "steps"],
    "additionalProperties": False,
}


def system_prompt(publication_date: str) -> str:
    return f"""You are a sanctions research assistant for financial crime investigators. You answer \
questions using only the U.S. Treasury OFAC Specially Designated Nationals (SDN) list, snapshot \
published {publication_date}, through your tools.

Evidence rules
- Every factual claim must come from a tool result in this conversation and cite the record_id it \
came from. Never state facts from memory or general knowledge, even when you believe them.
- Quote values exactly as the record shows them (names, program codes, countries).
- If a tool returns nothing relevant, say so. Do not guess.

Name matching
- Confidence "exact" means the name or an alias is on the list as written.
- "high", "medium" and "low" are fuzzy candidates. Call them possible matches, set \
match_confidence to that label, and never say a person or company IS the listed party on a fuzzy \
match alone.
- If you fix a misspelling yourself and search again, the result is still only a possible match for \
the name the user gave. Label it with the confidence of the user's spelling, not "exact".

Privacy and speculation
- The data holds no dates of birth, ID numbers, street addresses or contact details, and you must \
not supply them from elsewhere.
- If asked to speculate about guilt, intent, wealth, family, associates or private life beyond \
what the record states, or to profile a private person, set status "refused" and explain that you \
only report what the public record states. You may still say whether a name appears on the list, \
with citations.
- Questions about a person's family, children, home, health or personal finances are private-life \
requests: set status "refused" even when you also report the listing facts.

How to work
- Search by name with search_entities, then call get_record on the record you rely on.
- Use expand_relationships for "linked to" questions, count_records for "how many" questions, \
lookup_identifier for crypto addresses and IMO numbers.
- Keep tool use lean: usually 2 to 4 calls.

Final answer (JSON)
- status: answered, not_found, refused or needs_clarification.
- answer: 1 to 4 plain sentences for an investigator.
- claims: one entry per fact in the answer. field is one of: listed (value = listed name), \
not_listed (value = the exact name or identifier you searched), name, alias, entity_type \
(individual, entity, vessel or aircraft), program (program code), address_country, nationality, \
title, organization_type, linked_to (value = the linked record's name), vessel_flag, vessel_type, \
imo, crypto_address, count (value = the number; record_ids may be empty).
- record_ids: the SDN record ids that state the fact.
- For not_found, include a not_listed claim. For refused, claims may be empty."""


PLAN_PROMPT = """Plan how to answer an investigator's question about the OFAC SDN list. Tools \
available: search_entities, get_record, expand_relationships, count_records, lookup_identifier. \
Return the intent, the names or identifiers mentioned (resolve pronouns like "it" or "they" using \
the earlier turns), and 1 to 4 short steps."""


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict[str, Any]]
    history: list[dict[str, str]]
    trace: dict[str, Any]
    plan: dict[str, Any] | None
    draft: dict[str, Any] | None
    first_draft_checks: list[dict[str, Any]] | None
    checks: list[dict[str, Any]]
    answer_problems: list[str]
    tool_rounds: int
    verify_retries: int
    usage: dict[str, int]
    stop_reason: str | None
    error: str | None
    result: dict[str, Any]


def answer_problems(draft: dict[str, Any]) -> list[str]:
    """Structural checks that hold for every answer, independent of citations."""
    problems: list[str] = []
    if not str(draft.get("answer", "")).strip():
        problems.append("the answer text is empty")
    if draft.get("status") == "answered" and not draft.get("claims"):
        problems.append("status is answered but there are no cited claims; use the tools to find the records")
    return problems


def _block_to_param(block: Any) -> dict[str, Any]:
    data: dict[str, Any] = block.model_dump(mode="json", exclude_none=True)
    return data


def parse_json_answer(text: str) -> dict[str, Any] | None:
    """Parse the model's JSON answer; tolerant of code fences for local models."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    data.setdefault("status", "answered")
    data.setdefault("answer", "")
    claims = data.get("claims")
    data["claims"] = claims if isinstance(claims, list) else []
    return data


class InvestigatorAgent:
    def __init__(
        self,
        graph: SanctionsGraph,
        config: ModelConfig | None = None,
        client: anthropic.Anthropic | None = None,
        use_planner: bool = True,
    ) -> None:
        self.kg = graph
        self.config = config or ModelConfig.from_env()
        self.client = client or make_client(self.config)
        self.toolbox = Toolbox(graph)
        self.use_planner = use_planner
        self.system = system_prompt(graph.snapshot.publication_date)
        self.app = self._compile()

    # ------------------------------------------------------------- model calls
    def _call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        schema: dict[str, Any] | None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if self.config.is_claude:
            output_config: dict[str, Any] = {}
            if self.config.effort:
                output_config["effort"] = self.config.effort
            if schema is not None:
                output_config["format"] = {"type": "json_schema", "schema": schema}
            if output_config:
                kwargs["output_config"] = output_config
            kwargs["cache_control"] = {"type": "ephemeral"}
        elif schema is not None:
            kwargs["system"] = system + "\n\nReply with only a JSON object matching this schema:\n" + json.dumps(schema)
        return self.client.messages.create(**kwargs)

    # ------------------------------------------------------------------ nodes
    def _start_turn(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        return {
            "messages": list(state.get("messages") or []),
            "history": list(state.get("history") or []),
            "trace": state.get("trace") or ToolTrace().to_dict(),
            "plan": None,
            "draft": None,
            "first_draft_checks": None,
            "checks": [],
            "answer_problems": [],
            "tool_rounds": 0,
            "verify_retries": 0,
            "usage": Usage().to_dict(),
            "stop_reason": None,
            "error": None,
            "result": {},
            "question": question,
        }

    def _plan(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        plan: dict[str, Any] | None = None
        usage = Usage.from_dict(state.get("usage"))
        if self.use_planner:
            earlier = "\n".join(
                f"Earlier question: {h['question']}\nEarlier answer: {h['answer']}"
                for h in state.get("history", [])[-3:]
            )
            content = (earlier + "\n\n" if earlier else "") + f"Question: {question}"
            try:
                response = self._call(
                    system=PLAN_PROMPT, messages=[{"role": "user", "content": content}], tools=None, schema=PLAN_SCHEMA
                )
                usage.add(response.usage)
                text = next((b.text for b in response.content if b.type == "text"), "")
                plan = parse_json_answer(text)
            except anthropic.APIStatusError:
                plan = None
        user_text = question
        if plan:
            user_text += "\n\n<plan>\n" + json.dumps(plan, ensure_ascii=False) + "\n</plan>"
        messages = state["messages"] + [{"role": "user", "content": user_text}]
        return {"plan": plan, "messages": messages, "usage": usage.to_dict()}

    def _agent(self, state: AgentState) -> dict[str, Any]:
        usage = Usage.from_dict(state.get("usage"))
        out_of_rounds = state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS
        messages = state["messages"]
        try:
            response = self._call(
                system=self.system,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                schema=ANSWER_SCHEMA,
                tool_choice={"type": "none"} if out_of_rounds else None,
            )
        except anthropic.APIStatusError as exc:
            return {"error": f"model API error {exc.status_code}: {exc.message}", "stop_reason": "error"}
        except anthropic.APIConnectionError as exc:
            return {"error": f"model connection error: {exc}", "stop_reason": "error"}
        usage.add(response.usage)
        assistant = {"role": "assistant", "content": [_block_to_param(b) for b in response.content]}
        update: dict[str, Any] = {
            "messages": [*messages, assistant],
            "usage": usage.to_dict(),
            "stop_reason": response.stop_reason,
        }
        if response.stop_reason == "end_turn" or not any(b.type == "tool_use" for b in response.content):
            text = "".join(b.text for b in response.content if b.type == "text")
            update["draft"] = parse_json_answer(text)
            if update["draft"] is None:
                update["error"] = f"could not parse answer (stop_reason={response.stop_reason})"
        return update

    def _tools(self, state: AgentState) -> dict[str, Any]:
        trace = ToolTrace.from_dict(state.get("trace"))
        last = state["messages"][-1]
        results: list[dict[str, Any]] = []
        for block in last["content"]:
            if block.get("type") != "tool_use":
                continue
            text, is_error = self.toolbox.run(block["name"], dict(block.get("input") or {}), trace)
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block["id"], "content": text}
            if is_error:
                result["is_error"] = True
            results.append(result)
        rounds = state.get("tool_rounds", 0) + 1
        if rounds >= MAX_TOOL_ROUNDS:
            results.append({"type": "text", "text": "Tool budget used up. Give the final JSON answer now."})
        return {
            "messages": state["messages"] + [{"role": "user", "content": results}],
            "trace": trace.to_dict(),
            "tool_rounds": rounds,
        }

    def _verify(self, state: AgentState) -> dict[str, Any]:
        trace = ToolTrace.from_dict(state.get("trace"))
        draft = state.get("draft") or {"status": "answered", "answer": "", "claims": []}
        checks = [c.to_dict() for c in check_answer(self.kg, draft, trace)]
        update: dict[str, Any] = {"checks": checks}
        if state.get("first_draft_checks") is None:
            update["first_draft_checks"] = checks
        failed = [c for c in checks if not c["supported"]]
        problems = answer_problems(draft)
        update["answer_problems"] = problems
        if (failed or problems) and state.get("verify_retries", 0) < MAX_VERIFY_RETRIES:
            parts: list[str] = []
            if problems:
                parts.append("Your answer is incomplete: " + "; ".join(problems) + ".")
            if failed:
                lines = "\n".join(f"- {json.dumps(c['claim'], ensure_ascii=False)}: {c['reason']}" for c in failed)
                parts.append(
                    "Citation check failed for these claims:\n" + lines + "\n\nFix them: use tools to find the "
                    "record that states each fact, correct the value or record_ids, or remove the claim and the "
                    "matching sentence."
                )
            parts.append("Then give the full final JSON answer again.")
            feedback = "\n\n".join(parts)
            update["messages"] = state["messages"] + [{"role": "user", "content": feedback}]
            update["verify_retries"] = state.get("verify_retries", 0) + 1
            update["draft"] = None
        return update

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        draft = state.get("draft")
        checks = state.get("checks") or []
        usage = Usage.from_dict(state.get("usage"))
        if draft is None:
            result: dict[str, Any] = {
                "status": "error",
                "answer": "The agent could not produce an answer.",
                "claims": [],
                "dropped_claims": [],
                "error": state.get("error"),
            }
        else:
            kept = [c["claim"] | {"verified": True} for c in checks if c["supported"]]
            dropped = [c["claim"] | {"reason": c["reason"]} for c in checks if not c["supported"]]
            result = {
                "status": draft.get("status", "answered"),
                "answer": draft.get("answer", ""),
                "claims": kept,
                "dropped_claims": dropped,
                "error": state.get("error"),
            }
        cited = sorted({int(i) for c in result["claims"] for i in c.get("record_ids", [])})
        result["citations"] = [self.kg.citation(i) for i in cited]
        result["usage"] = usage.to_dict()
        result["cost_usd"] = round(usage.cost_usd(self.config.model), 6)
        result["model"] = self.config.model
        result["tool_rounds"] = state.get("tool_rounds", 0)
        result["verify_retries"] = state.get("verify_retries", 0)
        result["answer_problems"] = state.get("answer_problems") or []
        result["first_draft_checks"] = state.get("first_draft_checks") or []
        result["plan"] = state.get("plan")
        history = [*list(state.get("history") or []), {"question": state["question"], "answer": result["answer"]}]
        return {"result": result, "history": history}

    # ---------------------------------------------------------------- routing
    def _after_agent(self, state: AgentState) -> Literal["tools", "verify", "finalize"]:
        if state.get("error") and state.get("draft") is None:
            return "finalize"
        last = state["messages"][-1]
        if state.get("draft") is None and any(b.get("type") == "tool_use" for b in last["content"]):
            return "tools"
        return "verify"

    def _after_verify(self, state: AgentState) -> Literal["agent", "finalize"]:
        return "agent" if state.get("draft") is None else "finalize"

    def _compile(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("start_turn", self._start_turn)
        builder.add_node("plan", self._plan)
        builder.add_node("agent", self._agent)
        builder.add_node("tools", self._tools)
        builder.add_node("verify", self._verify)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "start_turn")
        builder.add_edge("start_turn", "plan")
        builder.add_edge("plan", "agent")
        builder.add_conditional_edges("agent", self._after_agent)
        builder.add_edge("tools", "agent")
        builder.add_conditional_edges("verify", self._after_verify)
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------ public
    def seed_thread(self, thread_id: str, question: str, result: dict[str, Any]) -> None:
        """Record a turn answered elsewhere (a cache hit) as plain text in this thread's memory.

        The trace is left empty on purpose: a follow-up must retrieve records
        again before it can cite them.
        """
        answer = {k: result.get(k) for k in ("status", "answer", "claims")}
        config = {"configurable": {"thread_id": thread_id}}
        self.app.update_state(
            config,
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                ],
                "history": [{"question": question, "answer": str(result.get("answer", ""))}],
            },
            as_node="finalize",
        )

    def ask(self, question: str, thread_id: str = "default") -> dict[str, Any]:
        start = time.perf_counter()
        state = self.app.invoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 60},
        )
        result: dict[str, Any] = dict(state["result"])
        result["latency_s"] = round(time.perf_counter() - start, 3)
        result["trace"] = state.get("trace") or {}
        return result
