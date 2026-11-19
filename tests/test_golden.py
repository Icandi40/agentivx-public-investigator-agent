"""Check every golden expectation against the data, so the eval cannot rest on a typo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investigator.graph_store import SanctionsGraph, normalize

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"

COUNT_FILTERS: dict[str, dict[str, str]] = {
    "count-01": {"program": "IRAN-EO13902", "entity_type": "vessel"},
    "count-02": {"country": "Yemen"},
    "count-03": {"program": "CYBER4"},
    "count-04": {"program": "DPRK3", "entity_type": "vessel"},
    "count-05": {"program": "GLOMAG", "entity_type": "individual"},
    "count-06": {"program": "CUBA"},
}

UNLISTED_NAMES = {
    "notfound-01": "BLUE HARBOR LOGISTICS GMBH",
    "notfound-02": "NORTHWIND AGRITECH PTY LTD",
    "notfound-03": "SILVERLAKE DENTAL GROUP",
    "notfound-04": "ACME ROBOTICS INC",
    "notfound-06": "OCEAN PEARL",
}


def cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_set_size_and_ids() -> None:
    data = cases()
    assert len(data) >= 60
    assert len({c["id"] for c in data}) == len(data)


def test_required_citations_and_values_exist(graph: SanctionsGraph) -> None:
    for case in cases():
        if case["id"] in COUNT_FILTERS:
            expected = len(graph.filter_records(**COUNT_FILTERS[case["id"]]))
            assert case["must_include"] == [str(expected)], case["id"]
            continue
        corpus = ""
        for record_id in case["must_cite"]:
            assert graph.has(record_id), case["id"]
            corpus += " " + json.dumps(graph.public_record(record_id), ensure_ascii=False)
        for value in case["must_include"]:
            assert normalize(value) in normalize(corpus), (case["id"], value)


FUZZY_NAMES = {
    "fuzzy-01": "TAMKO ESTABLISHMENT FOR OIL DERIVATIVE",
    "fuzzy-02": "IRAN INSURENCE COMPANY",
    "fuzzy-03": "HANSA SHIPING INC",
    "fuzzy-04": "GLAD GLORRY LIMITED",
    "fuzzy-05": "ZUPA HOLDINGS LIMITED",
    "fuzzy-06": "BANCO NACIONAL DE KUBA",
    "fuzzy-07": "Ilya Pavlovich KOZLOV",
}


def test_fuzzy_cases_match_but_not_exactly(graph: SanctionsGraph) -> None:
    fuzzy_cases = {c["id"]: c for c in cases() if c["expect_fuzzy_label"]}
    assert set(fuzzy_cases) == set(FUZZY_NAMES)
    for case_id, name in FUZZY_NAMES.items():
        assert name in fuzzy_cases[case_id]["turns"][-1]
        matches = {m.record_id: m for m in graph.search_names(name, 10)}
        for record_id in fuzzy_cases[case_id]["must_cite"]:
            assert record_id in matches, case_id
            assert matches[record_id].confidence != "exact", case_id


def test_unlisted_names_are_really_unlisted(graph: SanctionsGraph) -> None:
    for case_id, name in UNLISTED_NAMES.items():
        assert not any(m.confidence == "exact" for m in graph.search_names(name)), case_id
    assert graph.lookup_imo("9999999") == []
    assert graph.lookup_crypto("0x1111111111111111111111111111111111111111") == []
