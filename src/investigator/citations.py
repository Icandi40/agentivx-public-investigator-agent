"""Deterministic claim checking.

The agent's final answer is a list of structured claims: (field, value, record_ids).
Each claim is checked in code against the cited records, so "hallucination" has
a precise meaning here: a claim whose value is not present in any record it cites,
or that cites a record the agent never retrieved in this turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from investigator.graph_store import SanctionsGraph, normalize
from investigator.tools import ToolTrace

CLAIM_FIELDS = [
    "listed",
    "not_listed",
    "name",
    "alias",
    "entity_type",
    "program",
    "address_country",
    "nationality",
    "title",
    "organization_type",
    "linked_to",
    "vessel_flag",
    "vessel_type",
    "imo",
    "crypto_address",
    "count",
]

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "not_found", "refused", "needs_clarification"],
        },
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "field": {"type": "string", "enum": CLAIM_FIELDS},
                    "value": {"type": "string"},
                    "record_ids": {"type": "array", "items": {"type": "integer"}},
                    "match_confidence": {
                        "type": "string",
                        "enum": ["exact", "high", "medium", "low", "not_applicable"],
                    },
                },
                "required": ["text", "field", "value", "record_ids", "match_confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "answer", "claims"],
    "additionalProperties": False,
}

_TYPE_SYNONYMS = {
    "individual": "individual",
    "person": "individual",
    "entity": "entity",
    "company": "entity",
    "organization": "entity",
    "organisation": "entity",
    "group": "entity",
    "vessel": "vessel",
    "ship": "vessel",
    "aircraft": "aircraft",
}


def _name_match(value: str, candidates: list[str]) -> bool:
    nv = normalize(value)
    if not nv:
        return False
    for cand in candidates:
        nc = normalize(cand)
        if nv == nc or fuzz.token_sort_ratio(nv, nc) >= 92:
            return True
    return False


def _equal_any(value: str, candidates: list[str]) -> bool:
    nv = normalize(value)
    return bool(nv) and any(nv == normalize(c) for c in candidates)


def _contains_any(value: str, candidates: list[str]) -> bool:
    nv = normalize(value)
    return bool(nv) and any(nv == normalize(c) or nv in normalize(c) for c in candidates if c)


def record_supports(graph: SanctionsGraph, record_id: int, field: str, value: str) -> bool:
    """True when the record's own data contains this field value."""
    if not graph.has(record_id):
        return False
    ent = graph.get(record_id)
    if field == "listed":
        return True
    if field == "name":
        return _name_match(value, [ent.name])
    if field == "alias":
        return _name_match(value, [a.name for a in ent.aliases] + [ent.name])
    if field == "entity_type":
        return _TYPE_SYNONYMS.get(value.strip().lower()) == ent.entity_type
    if field == "program":
        return _equal_any(value, list(ent.programs))
    if field == "address_country":
        return _equal_any(value, ent.address_countries)
    if field == "nationality":
        return _equal_any(value, ent.nationalities)
    if field == "title":
        return bool(ent.title) and fuzz.token_set_ratio(normalize(value), normalize(ent.title or "")) >= 90
    if field == "organization_type":
        return _contains_any(value, [ent.organization_type or ""])
    if field == "linked_to":
        for other, _direction in graph.linked(record_id):
            other_ent = graph.get(other)
            if value.strip() == str(other) or _name_match(
                value, [other_ent.name] + [a.name for a in other_ent.aliases]
            ):
                return True
        return False
    if ent.vessel is None and field in {"vessel_flag", "vessel_type", "imo"}:
        return False
    if field == "vessel_flag":
        return _equal_any(value, [ent.vessel.flag or ""]) if ent.vessel else False
    if field == "vessel_type":
        return _contains_any(value, [ent.vessel.vessel_type or ""]) if ent.vessel else False
    if field == "imo":
        digits = re.sub(r"\D", "", value)
        return bool(ent.vessel and ent.vessel.imo and digits == ent.vessel.imo)
    if field == "crypto_address":
        return value.strip().lower() in {a.lower() for _c, a in ent.crypto_addresses}
    return False


@dataclass
class ClaimCheck:
    claim: dict[str, Any]
    supported: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "supported": self.supported, "reason": self.reason}


def check_claim(graph: SanctionsGraph, claim: dict[str, Any], trace: ToolTrace) -> ClaimCheck:
    field = claim.get("field", "")
    value = str(claim.get("value", ""))
    ids = [int(i) for i in claim.get("record_ids", [])]

    if field == "count":
        digits = re.sub(r"[^\d]", "", value)
        if digits and int(digits) in trace.counts:
            return ClaimCheck(claim, True, "matches a count returned by count_records")
        return ClaimCheck(claim, False, "count not returned by any count_records call this turn")

    if field == "not_listed":
        nv = normalize(value)
        for search in trace.searches:
            if normalize(search["query"]) == nv or fuzz.token_sort_ratio(normalize(search["query"]), nv) >= 90:
                if not search["has_exact"]:
                    return ClaimCheck(claim, True, "a search this turn found no exact match")
                return ClaimCheck(claim, False, "a search this turn did find an exact match")
        return ClaimCheck(claim, False, "no search for this value was run this turn")

    if not ids:
        return ClaimCheck(claim, False, "claim cites no record id")
    missing = [i for i in ids if not graph.has(i)]
    if missing:
        return ClaimCheck(claim, False, f"cited ids do not exist: {missing}")
    unretrieved = [i for i in ids if i not in trace.retrieved_ids]
    if unretrieved:
        return ClaimCheck(claim, False, f"cited ids were never retrieved this turn: {unretrieved}")
    if field not in CLAIM_FIELDS:
        return ClaimCheck(claim, False, f"unknown field {field!r}")
    if any(record_supports(graph, i, field, value) for i in ids):
        return ClaimCheck(claim, True, "value found in cited record")
    return ClaimCheck(claim, False, f"{field}={value!r} not found in cited records {ids}")


def check_answer(graph: SanctionsGraph, answer: dict[str, Any], trace: ToolTrace) -> list[ClaimCheck]:
    return [check_claim(graph, c, trace) for c in answer.get("claims", [])]


# ---------------------------------------------------------------- prose check
_SKIP_WORDS = {
    "THE",
    "THIS",
    "THAT",
    "THESE",
    "YES",
    "NO",
    "IT",
    "ITS",
    "A",
    "AN",
    "AND",
    "OR",
    "OF",
    "IN",
    "ON",
    "AT",
    "TO",
    "FOR",
    "IS",
    "ARE",
    "WAS",
    "BY",
    "AS",
    "WITH",
    "FROM",
    "BOTH",
    "ALSO",
    "HOWEVER",
    "BASED",
    "ACCORDING",
    "I",
    "WE",
    "NOT",
}
_ALWAYS_OK = {"OFAC", "SDN", "SDN LIST", "U S", "US", "TREASURY", "OFAC SDN", "SPECIALLY DESIGNATED NATIONALS"}
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;:,])\s+|\s*[()]\s*")
_CAP_RUN = re.compile(r"(?:\b[A-Z][A-Za-z0-9'.&-]*(?:\s+|$)){2,}")
_NUMBER = re.compile(r"\b\d[\d,]*\b")
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9]{20,}\b")


def salient_spans(text: str) -> list[str]:
    """Names, numbers and long identifiers in prose, used for a grounding heuristic.

    Capitalized runs are found within one clause at a time, so "Venezuela. The"
    is never read as a name.
    """
    spans: list[str] = []
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.rstrip(".,;:!? ")
        for m in _CAP_RUN.finditer(clause):
            words = [w.rstrip(".,;:!?'") for w in m.group(0).split()]
            words = [w for w in words if w]
            while words and normalize(words[0]) in _SKIP_WORDS:
                words = words[1:]
            if len(words) >= 2:
                spans.append(" ".join(words))
    spans.extend(_NUMBER.findall(text))
    spans.extend(_LONG_TOKEN.findall(text))
    return spans


def ungrounded_spans(answer_text: str, question: str, trace: ToolTrace, context: str = "") -> list[str]:
    """Spans in the prose answer found in none of: tool outputs, the question, or `context`.

    `context` holds facts the system prompt gives the model, such as the snapshot date.
    """
    known = " ".join(trace.outputs) + " " + question + " " + context
    corpus = normalize(known)
    corpus_digits = re.sub(r"\D+", " ", known)
    bad: list[str] = []
    for span in salient_spans(answer_text):
        ns = normalize(span)
        if not ns or ns in _ALWAYS_OK:
            continue
        if re.fullmatch(r"[\d,]+", span):
            digits = span.replace(",", "")
            if len(digits) < 2 or re.search(rf"\b{digits}\b", corpus_digits):
                continue
            bad.append(span)
            continue
        if ns not in corpus:
            bad.append(span)
    return bad
