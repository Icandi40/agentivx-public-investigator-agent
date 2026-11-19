"""Agent tools over the knowledge graph, plus a trace of what each run retrieved.

The trace is what makes citations checkable: a claim may only cite a record id
that some tool actually returned during the same turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from investigator.graph_store import SanctionsGraph, normalize

ENTITY_TYPE_VALUES = ["individual", "entity", "vessel", "aircraft"]


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_entities",
        "description": (
            "Find SDN records by name. Checks primary names and aliases, exact match first, "
            "then fuzzy match. Every candidate has a confidence label: exact, high, medium or low. "
            "Treat medium and low as possible matches only, never as confirmed identities."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person, company, vessel or alias name."},
                "limit": {"type": "integer", "description": "Max candidates, 1 to 10."},
            },
            "required": ["name", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_record",
        "description": (
            "Get the public SDN record for one record id: name, type, programs, aliases, "
            "country-level locations, nationality, vessel details, digital currency addresses, "
            "linked records and the citation."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "integer"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "expand_relationships",
        "description": (
            "Walk the 'Linked To' relationships OFAC publishes, 1 or 2 hops out from a record. "
            "Direction 'outgoing' means this record names the other one in its remarks."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "integer"},
                "depth": {"type": "integer", "enum": [1, 2]},
            },
            "required": ["record_id", "depth"],
            "additionalProperties": False,
        },
    },
    {
        "name": "count_records",
        "description": (
            "Count SDN records that match all given filters (program code such as SDGT or "
            "RUSSIA-EO14024, country of address or nationality, entity type). Pass null for "
            "filters you do not use. Returns the count and up to 10 example records."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "program": _nullable({"type": "string"}),
                "country": _nullable({"type": "string"}),
                "entity_type": _nullable({"type": "string", "enum": ENTITY_TYPE_VALUES}),
            },
            "required": ["program", "country", "entity_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_identifier",
        "description": (
            "Exact lookup of a digital currency (crypto) address or a vessel IMO number "
            "against the identifiers OFAC lists."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "kind": {"type": "string", "enum": ["crypto_address", "imo"]},
            },
            "required": ["value", "kind"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolTrace:
    """Everything the tools returned during one turn."""

    retrieved_ids: set[int] = field(default_factory=set)
    counts: list[int] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_ids": sorted(self.retrieved_ids),
            "counts": self.counts,
            "searches": self.searches,
            "outputs": self.outputs,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ToolTrace:
        if not data:
            return cls()
        return cls(
            retrieved_ids=set(data.get("retrieved_ids", [])),
            counts=list(data.get("counts", [])),
            searches=list(data.get("searches", [])),
            outputs=list(data.get("outputs", [])),
            calls=list(data.get("calls", [])),
        )


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, graph: SanctionsGraph) -> None:
        self.graph = graph

    def run(self, name: str, args: dict[str, Any], trace: ToolTrace) -> tuple[str, bool]:
        """Run one tool. Returns (json_text, is_error) and records the result in the trace."""
        trace.calls.append({"name": name, "input": args})
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                raise ToolError(f"unknown tool {name!r}")
            result = handler(args, trace)
            text = json.dumps(result, ensure_ascii=False)
            trace.outputs.append(text)
            return text, False
        except ToolError as exc:
            return json.dumps({"error": str(exc)}), True
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": f"bad input: {exc}"}), True

    # Each tool returns plain JSON-able data and registers retrieved record ids.
    def _record_summary(self, record_id: int, trace: ToolTrace) -> dict[str, Any]:
        ent = self.graph.get(record_id)
        trace.retrieved_ids.add(record_id)
        return {
            "record_id": record_id,
            "name": ent.name,
            "entity_type": ent.entity_type,
            "programs": list(ent.programs),
        }

    def _tool_search_entities(self, args: dict[str, Any], trace: ToolTrace) -> dict[str, Any]:
        name = str(args["name"]).strip()
        if not name:
            raise ToolError("name is empty")
        limit = max(1, min(int(args.get("limit") or 5), 10))
        matches = self.graph.search_names(name, limit=limit)
        candidates = []
        for m in matches:
            summary = self._record_summary(m.record_id, trace)
            summary.update(
                {
                    "matched_name": m.matched_name,
                    "match_kind": m.match_kind,
                    "score": m.score,
                    "confidence": m.confidence,
                }
            )
            candidates.append(summary)
        has_exact = any(m.confidence == "exact" for m in matches)
        trace.searches.append({"query": name, "has_exact": has_exact, "n": len(matches)})
        note = None
        if not matches:
            note = "No SDN record matches this name above the fuzzy threshold."
        elif not has_exact:
            note = "No exact match. Candidates are fuzzy matches; label them as possible matches only."
        return {"query": name, "candidates": candidates, "note": note}

    def _tool_get_record(self, args: dict[str, Any], trace: ToolTrace) -> dict[str, Any]:
        record_id = int(args["record_id"])
        if not self.graph.has(record_id):
            raise ToolError(f"no SDN record with id {record_id}")
        trace.retrieved_ids.add(record_id)
        record = self.graph.public_record(record_id)
        for link in record.get("linked_to", []):
            trace.retrieved_ids.add(int(link["record_id"]))
        return record

    def _tool_expand_relationships(self, args: dict[str, Any], trace: ToolTrace) -> dict[str, Any]:
        record_id = int(args["record_id"])
        depth = 2 if int(args.get("depth") or 1) >= 2 else 1
        if not self.graph.has(record_id):
            raise ToolError(f"no SDN record with id {record_id}")
        trace.retrieved_ids.add(record_id)
        related = self.graph.expand(record_id, depth=depth)
        for item in related:
            trace.retrieved_ids.add(int(item["record_id"]))
            item["programs"] = list(self.graph.get(int(item["record_id"])).programs)
        return {
            "record_id": record_id,
            "name": self.graph.get(record_id).name,
            "depth": depth,
            "related": related,
            "note": None if related else "This record has no published 'Linked To' relationships.",
        }

    def _tool_count_records(self, args: dict[str, Any], trace: ToolTrace) -> dict[str, Any]:
        program = args.get("program") or None
        country = args.get("country") or None
        entity_type = args.get("entity_type") or None
        if not (program or country or entity_type):
            raise ToolError("give at least one filter")
        if program and self.graph.canonical_program(program) is None:
            close = [p for p in self.graph.known_programs() if normalize(program) in normalize(p)][:10]
            raise ToolError(f"unknown program {program!r}; similar codes: {close}")
        if country and self.graph.canonical_country(country) is None:
            close = [c for c in self.graph.known_countries() if normalize(country) in normalize(c)][:10]
            raise ToolError(f"unknown country {country!r}; similar: {close}")
        if entity_type and entity_type not in ENTITY_TYPE_VALUES:
            raise ToolError(f"entity_type must be one of {ENTITY_TYPE_VALUES}")
        ids = self.graph.filter_records(program=program, country=country, entity_type=entity_type)
        trace.counts.append(len(ids))
        return {
            "filters": {"program": program, "country": country, "entity_type": entity_type},
            "count": len(ids),
            "examples": [self._record_summary(i, trace) for i in ids[:10]],
            "note": "Country matches either a listed address country or a listed nationality.",
        }

    def _tool_lookup_identifier(self, args: dict[str, Any], trace: ToolTrace) -> dict[str, Any]:
        value = str(args["value"]).strip()
        kind = args["kind"]
        if kind == "crypto_address":
            ids = self.graph.lookup_crypto(value)
        elif kind == "imo":
            ids = self.graph.lookup_imo(value)
        else:
            raise ToolError("kind must be crypto_address or imo")
        trace.searches.append({"query": value, "has_exact": bool(ids), "n": len(ids)})
        return {
            "value": value,
            "kind": kind,
            "matches": [self._record_summary(i, trace) for i in ids],
            "note": None if ids else "This identifier does not appear on the SDN list snapshot.",
        }
