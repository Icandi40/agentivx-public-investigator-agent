from __future__ import annotations

from investigator.graph_store import SanctionsGraph, confidence_for, normalize


def test_normalize() -> None:
    assert normalize("Petróleos de Venezuela, S.A.") == "PETROLEOS DE VENEZUELA S A"
    assert normalize("  ") == ""


def test_confidence_bands() -> None:
    assert confidence_for(99) == "high"
    assert confidence_for(90) == "medium"
    assert confidence_for(82) == "low"


def test_exact_and_alias_search(graph: SanctionsGraph) -> None:
    top = graph.search_names("BANCO NACIONAL DE CUBA")[0]
    assert (top.record_id, top.confidence, top.match_kind) == (306, "exact", "primary")
    alias = graph.search_names("BIMEH IRAN")[0]
    assert (alias.record_id, alias.confidence, alias.match_kind) == (11631, "exact", "alias")


def test_fuzzy_search_is_labeled(graph: SanctionsGraph) -> None:
    top = graph.search_names("HANSA SHIPING INC")[0]
    assert top.record_id == 58116
    assert top.confidence != "exact"


def test_unlisted_name_returns_nothing(graph: SanctionsGraph) -> None:
    assert graph.search_names("BLUE HARBOR LOGISTICS GMBH") == []


def test_relationships(graph: SanctionsGraph) -> None:
    linked = dict(graph.linked(54478))
    assert linked == {47664: "outgoing"}
    two_hop = graph.expand(54478, depth=2)
    assert any(item["record_id"] == 47664 and item["hop"] == 1 for item in two_hop)
    assert all(item["record_id"] != 54478 for item in two_hop)


def test_identifier_lookups(graph: SanctionsGraph) -> None:
    assert graph.lookup_crypto("0x12DE548F79A50D2BD05481C8515C1EF5183666A9") == [55046]
    assert graph.lookup_imo("IMO 9463528") == [53897]
    assert graph.lookup_imo("9999999") == []


def test_filters(graph: SanctionsGraph) -> None:
    assert len(graph.filter_records(program="cyber4")) == 97
    yemen = graph.filter_records(country="yemen")
    assert 54478 in yemen


def test_public_record_has_citation_and_no_private_fields(graph: SanctionsGraph) -> None:
    record = graph.public_record(306)
    assert record["citation"]["url"].endswith("id=306")
    assert set(record) <= {
        "record_id",
        "name",
        "entity_type",
        "programs",
        "aliases",
        "address_countries",
        "title",
        "nationalities",
        "organization_type",
        "digital_currency_addresses",
        "vessel",
        "linked_to",
        "linked_to_truncated",
        "citation",
    }


def test_mentions(graph: SanctionsGraph) -> None:
    found = graph.mentions("Is BANCO NACIONAL DE CUBA on the list under CUBA?")
    assert "record:306" in found
    assert "program:CUBA" in found
    assert graph.mentions("Is ACME ROBOTICS on the list?") == set()
