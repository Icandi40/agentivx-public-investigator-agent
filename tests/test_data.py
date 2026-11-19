from __future__ import annotations

import csv
import gzip

import pytest

from investigator.data import parse_programs, parse_remarks
from investigator.graph_store import DEFAULT_PROCESSED, SanctionsGraph

from .conftest import RAW_DIR


def test_parse_programs() -> None:
    assert parse_programs("SDGT] [IFSR") == ("SDGT", "IFSR")
    assert parse_programs("CUBA") == ("CUBA",)
    assert parse_programs("-0- ") == ()


def test_parse_remarks_keeps_public_facts_and_drops_personal_ones() -> None:
    # Fabricated values in the SDN remarks format; not taken from any real record.
    remarks = (
        "DOB 01 Jan 1900; POB Exampleville; nationality Tunisia; alt. nationality Italy; "
        "Passport X 000123 issued 01 Jan 1950; Gender Male; "
        "Digital Currency Address - XBT 1AbcDefGhiJklMnoPqrStuVwxYz12345; "
        "Vessel Registration Identification IMO 9187667; Organization Type: Shipping; "
        "Linked To: NATIONAL IRANIAN TANKER COMPANY."
    )
    facts = parse_remarks(remarks)
    assert facts.nationalities == ["Italy", "Tunisia"]
    assert facts.crypto == [("XBT", "1AbcDefGhiJklMnoPqrStuVwxYz12345")]
    assert facts.imo == "9187667"
    assert facts.organization_type == "Shipping"
    assert facts.linked_to == ["NATIONAL IRANIAN TANKER COMPANY"]
    dumped = repr(facts)
    for personal in ("1900", "000123", "Exampleville", "Male", "POB"):
        assert personal not in dumped


def test_processed_snapshot_holds_no_personal_identifiers() -> None:
    with gzip.open(DEFAULT_PROCESSED, "rt", encoding="utf-8") as fh:
        text = fh.read()
    for marker in ("DOB ", "Passport", "POB ", "Gender", "National ID No", "Tax ID No"):
        assert marker not in text


@pytest.mark.skipif(not (RAW_DIR / "SDN.CSV").exists(), reason="raw OFAC files not downloaded")
def test_graph_matches_raw_csv_independently(graph: SanctionsGraph) -> None:
    """Recount from the raw CSV with plain csv code, not the loader under test."""
    with (RAW_DIR / "SDN.CSV").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0].strip().isdigit()]
    assert len(rows) == len(graph.entities)
    cuba = [r for r in rows if "CUBA" in [p.strip(" []") for p in r[3].split("] [")]]
    assert len(cuba) == len(graph.filter_records(program="CUBA"))
    vessels_iran = [
        r for r in rows if r[2].strip() == "vessel" and "IRAN-EO13902" in [p.strip(" []") for p in r[3].split("] [")]
    ]
    assert len(vessels_iran) == len(graph.filter_records(program="IRAN-EO13902", entity_type="vessel"))


def test_graph_stats(graph: SanctionsGraph) -> None:
    stats = graph.stats()
    assert stats["records"] > 15000
    assert stats["linked_to_edges"] > 5000
    assert stats["unresolved_links"] < 50
    assert graph.snapshot.publication_date == "2026-11-17"
    assert isinstance(SanctionsGraph, type)
