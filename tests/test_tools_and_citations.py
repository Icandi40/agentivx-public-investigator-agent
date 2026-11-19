from __future__ import annotations

import json

from investigator.citations import check_claim, salient_spans, ungrounded_spans
from investigator.graph_store import SanctionsGraph
from investigator.tools import TOOL_DEFINITIONS, Toolbox, ToolTrace


def test_tool_schemas_are_strict() -> None:
    for tool in TOOL_DEFINITIONS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_search_registers_retrieved_ids(toolbox: Toolbox) -> None:
    trace = ToolTrace()
    text, is_error = toolbox.run("search_entities", {"name": "BIMEH IRAN", "limit": 3}, trace)
    assert not is_error
    data = json.loads(text)
    assert data["candidates"][0]["record_id"] == 11631
    assert 11631 in trace.retrieved_ids
    assert trace.searches[0]["has_exact"] is True


def test_get_record_registers_linked_ids(toolbox: Toolbox) -> None:
    trace = ToolTrace()
    toolbox.run("get_record", {"record_id": 54478}, trace)
    assert {54478, 47664} <= trace.retrieved_ids


def test_count_and_bad_filters(toolbox: Toolbox) -> None:
    trace = ToolTrace()
    text, is_error = toolbox.run("count_records", {"program": "CYBER4", "country": None, "entity_type": None}, trace)
    assert not is_error and json.loads(text)["count"] == 97
    assert trace.counts == [97]
    text, is_error = toolbox.run(
        "count_records", {"program": "NOT-A-PROGRAM", "country": None, "entity_type": None}, trace
    )
    assert is_error and "unknown program" in text
    _, is_error = toolbox.run("count_records", {"program": None, "country": None, "entity_type": None}, trace)
    assert is_error


def test_unknown_tool_and_missing_record(toolbox: Toolbox) -> None:
    trace = ToolTrace()
    assert toolbox.run("delete_everything", {}, trace)[1] is True
    assert toolbox.run("get_record", {"record_id": 1}, trace)[1] is True


def _claim(field: str, value: str, ids: list[int]) -> dict[str, object]:
    return {"text": "t", "field": field, "value": value, "record_ids": ids, "match_confidence": "exact"}


def test_claim_checks(graph: SanctionsGraph) -> None:
    trace = ToolTrace(retrieved_ids={306, 54478, 47664})
    assert check_claim(graph, _claim("program", "CUBA", [306]), trace).supported
    assert check_claim(graph, _claim("alias", "NATIONAL BANK OF CUBA", [306]), trace).supported
    assert check_claim(graph, _claim("linked_to", "ANSARALLAH", [54478]), trace).supported
    assert check_claim(graph, _claim("address_country", "Yemen", [54478]), trace).supported
    # wrong value
    bad = check_claim(graph, _claim("program", "SDGT", [306]), trace)
    assert not bad.supported and "not found" in bad.reason
    # real record, but never retrieved this turn
    unretrieved = check_claim(graph, _claim("program", "CYBER4", [55046]), trace)
    assert not unretrieved.supported and "never retrieved" in unretrieved.reason
    # no citation at all
    assert not check_claim(graph, _claim("program", "CUBA", []), trace).supported


def test_count_and_not_listed_claims(graph: SanctionsGraph) -> None:
    trace = ToolTrace(counts=[97], searches=[{"query": "ACME ROBOTICS INC", "has_exact": False, "n": 0}])
    assert check_claim(graph, _claim("count", "97", []), trace).supported
    assert not check_claim(graph, _claim("count", "98", []), trace).supported
    assert check_claim(graph, _claim("not_listed", "ACME ROBOTICS INC", []), trace).supported
    assert not check_claim(graph, _claim("not_listed", "SOMETHING ELSE", []), trace).supported


def test_prose_grounding() -> None:
    assert "Banco Nacional" in " ".join(salient_spans("The Banco Nacional is listed under 2 programs since 1999."))
    trace = ToolTrace(outputs=['{"name": "BANCO NACIONAL DE CUBA", "programs": ["CUBA"]}'])
    assert ungrounded_spans("BANCO NACIONAL DE CUBA is listed.", "q", trace) == []
    flagged = ungrounded_spans("It was listed in 1999 by Juan Perez.", "q", trace)
    assert "1999" in flagged
    assert any("Juan Perez" in span for span in flagged)
