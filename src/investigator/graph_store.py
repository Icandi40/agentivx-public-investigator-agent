"""Knowledge graph over the SDN list.

Nodes: one per SDN record ("entity", id), one per sanctions program
("program", code) and one per country ("country", name).
Edges: IN_PROGRAM, ADDRESS_IN, NATIONAL_OF and LINKED_TO (the "Linked To:"
relationships OFAC publishes in the remarks, resolved from names to record ids).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from rapidfuzz import fuzz, process

from investigator.data import Entity, Snapshot, read_processed

Confidence = Literal["exact", "high", "medium", "low"]

HIGH_SCORE = 95.0
MEDIUM_SCORE = 88.0
MIN_SCORE = 80.0

DEFAULT_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "sdn_public.json.gz"
OFAC_DETAIL_URL = "https://sanctionssearch.ofac.treas.gov/Details.aspx?id={id}"


def normalize(text: str) -> str:
    """Uppercase, strip accents and punctuation, collapse spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", no_marks)
    return " ".join(cleaned.upper().split())


def confidence_for(score: float) -> Confidence:
    if score >= HIGH_SCORE:
        return "high"
    if score >= MEDIUM_SCORE:
        return "medium"
    return "low"


@dataclass(frozen=True)
class NameMatch:
    record_id: int
    matched_name: str
    primary_name: str
    match_kind: Literal["primary", "alias"]
    score: float
    confidence: Confidence


def _entity_node(record_id: int) -> tuple[str, int]:
    return ("entity", record_id)


class SanctionsGraph:
    def __init__(self, entities: dict[int, Entity], snapshot: Snapshot) -> None:
        self.entities = entities
        self.snapshot = snapshot
        self.g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
        self._names: list[str] = []  # normalized names, parallel to _name_meta
        self._name_meta: list[tuple[int, str, Literal["primary", "alias"]]] = []
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._crypto: dict[str, list[int]] = defaultdict(list)
        self._imo: dict[str, list[int]] = defaultdict(list)
        self._countries: dict[str, str] = {}
        self._programs: dict[str, str] = {}
        self.unresolved_links: list[tuple[int, str]] = []
        self.ambiguous_links: list[tuple[int, str]] = []
        self._build()

    @classmethod
    def load(cls, path: Path = DEFAULT_PROCESSED) -> SanctionsGraph:
        snapshot, entities = read_processed(path)
        return cls(entities, snapshot)

    # ------------------------------------------------------------------ build
    def _add_name(self, record_id: int, name: str, kind: Literal["primary", "alias"]) -> None:
        norm = normalize(name)
        if not norm:
            return
        self._names.append(norm)
        self._name_meta.append((record_id, name, kind))
        if record_id not in self._exact[norm]:
            self._exact[norm].append(record_id)

    def _build(self) -> None:
        primary_by_name: dict[str, list[int]] = defaultdict(list)
        for ent in self.entities.values():
            node = _entity_node(ent.id)
            self.g.add_node(node, kind="entity", name=ent.name, entity_type=ent.entity_type)
            primary_by_name[normalize(ent.name)].append(ent.id)
            self._add_name(ent.id, ent.name, "primary")
            for alias in ent.aliases:
                self._add_name(ent.id, alias.name, "alias")
            for program in ent.programs:
                self._programs[program.upper()] = program
                self.g.add_edge(node, ("program", program), key="IN_PROGRAM", rel="IN_PROGRAM")
            for country in ent.address_countries:
                self._countries[country.upper()] = country
                self.g.add_edge(node, ("country", country), key="ADDRESS_IN", rel="ADDRESS_IN")
            for country in ent.nationalities:
                self._countries[country.upper()] = country
                self.g.add_edge(node, ("country", country), key="NATIONAL_OF", rel="NATIONAL_OF")
            for _currency, address in ent.crypto_addresses:
                self._crypto[address.lower()].append(ent.id)
            if ent.vessel and ent.vessel.imo:
                self._imo[ent.vessel.imo].append(ent.id)

        for ent in self.entities.values():
            for target_name in ent.linked_to_names:
                targets = primary_by_name.get(normalize(target_name), [])
                if not targets:
                    self.unresolved_links.append((ent.id, target_name))
                    continue
                if len(targets) > 1:
                    self.ambiguous_links.append((ent.id, target_name))
                for target in targets:
                    if target != ent.id:
                        self.g.add_edge(_entity_node(ent.id), _entity_node(target), key="LINKED_TO", rel="LINKED_TO")

    # ------------------------------------------------------------------ queries
    def stats(self) -> dict[str, int]:
        linked = sum(1 for _, _, k in self.g.edges(keys=True) if k == "LINKED_TO")
        return {
            "records": len(self.entities),
            "aliases": sum(len(e.aliases) for e in self.entities.values()),
            "programs": len(self._programs),
            "countries": len(self._countries),
            "linked_to_edges": linked,
            "unresolved_links": len(self.unresolved_links),
            "crypto_addresses": len(self._crypto),
            "graph_nodes": self.g.number_of_nodes(),
            "graph_edges": self.g.number_of_edges(),
        }

    def has(self, record_id: int) -> bool:
        return record_id in self.entities

    def get(self, record_id: int) -> Entity:
        return self.entities[record_id]

    def search_names(self, query: str, limit: int = 5) -> list[NameMatch]:
        """Exact match on primary names and aliases first, then fuzzy match."""
        norm = normalize(query)
        if not norm:
            return []
        best: dict[int, NameMatch] = {}
        for record_id in self._exact.get(norm, []):
            ent = self.entities[record_id]
            kind: Literal["primary", "alias"] = "primary" if normalize(ent.name) == norm else "alias"
            matched = (
                ent.name
                if kind == "primary"
                else next((a.name for a in ent.aliases if normalize(a.name) == norm), query)
            )
            best[record_id] = NameMatch(record_id, matched, ent.name, kind, 100.0, "exact")
        if len(best) < limit:
            hits = process.extract(
                norm,
                self._names,
                scorer=fuzz.token_sort_ratio,
                limit=limit * 6,
                score_cutoff=MIN_SCORE,
            )
            for _choice, score, index in hits:
                record_id, matched, kind = self._name_meta[index]
                if record_id in best and best[record_id].score >= score:
                    continue
                ent = self.entities[record_id]
                best[record_id] = NameMatch(
                    record_id, matched, ent.name, kind, round(float(score), 1), confidence_for(score)
                )
        ranked = sorted(best.values(), key=lambda m: (-m.score, m.record_id))
        return ranked[:limit]

    def mentions(self, text: str, max_ngram: int = 12) -> set[str]:
        """Record ids, programs, countries, numbers and identifiers named verbatim in `text`.

        Used as the semantic cache's anchor: two questions may share a cached
        answer only when they mention exactly the same things.
        """
        tokens = normalize(text).split()
        found: set[str] = set()
        countries = {normalize(c): c for c in self._countries.values()}
        programs = {normalize(p): p for p in self._programs.values()}
        for n in range(1, min(max_ngram, len(tokens)) + 1):
            for start in range(len(tokens) - n + 1):
                gram = " ".join(tokens[start : start + n])
                for record_id in self._exact.get(gram, []):
                    found.add(f"record:{record_id}")
                if gram in countries:
                    found.add(f"country:{countries[gram]}")
                if gram in programs:
                    found.add(f"program:{programs[gram]}")
        for token in re.findall(r"\b\d{2,}\b", text):
            found.add(f"number:{token}")
        for token in re.findall(r"\b[A-Za-z0-9]{20,}\b", text):
            found.add(f"identifier:{token.lower()}")
        return found

    def exact_ids_for_name(self, text: str) -> list[int]:
        return list(self._exact.get(normalize(text), []))

    def lookup_crypto(self, address: str) -> list[int]:
        return list(self._crypto.get(address.strip().lower(), []))

    def lookup_imo(self, imo: str) -> list[int]:
        digits = re.sub(r"\D", "", imo)
        return list(self._imo.get(digits, []))

    def canonical_country(self, country: str) -> str | None:
        return self._countries.get(country.strip().upper())

    def canonical_program(self, program: str) -> str | None:
        return self._programs.get(program.strip().upper())

    def known_countries(self) -> list[str]:
        return sorted(self._countries.values())

    def known_programs(self) -> list[str]:
        return sorted(self._programs.values())

    def linked(self, record_id: int) -> list[tuple[int, str]]:
        """Direct LINKED_TO neighbors with direction ('outgoing' = this record names the other)."""
        node = _entity_node(record_id)
        out: dict[int, str] = {}
        for _, target, key in self.g.out_edges(node, keys=True):
            if key == "LINKED_TO":
                out[target[1]] = "outgoing"
        for source, _, key in self.g.in_edges(node, keys=True):
            if key == "LINKED_TO" and source[1] not in out:
                out[source[1]] = "incoming"
        return sorted(out.items())

    def expand(self, record_id: int, depth: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        """Breadth-first walk over LINKED_TO edges up to `depth` hops."""
        seen = {record_id}
        frontier = [record_id]
        results: list[dict[str, Any]] = []
        for hop in range(1, depth + 1):
            next_frontier: list[int] = []
            for current in frontier:
                for other, direction in self.linked(current):
                    if other in seen:
                        continue
                    seen.add(other)
                    next_frontier.append(other)
                    results.append(
                        {
                            "record_id": other,
                            "name": self.entities[other].name,
                            "hop": hop,
                            "via": current,
                            "direction": direction,
                        }
                    )
            frontier = next_frontier
        return results[:limit]

    def filter_records(
        self,
        program: str | None = None,
        country: str | None = None,
        entity_type: str | None = None,
    ) -> list[int]:
        candidates: set[int] | None = None
        if program:
            node = ("program", self.canonical_program(program) or program)
            ids = {src[1] for src, _ in self.g.in_edges(node)} if node in self.g else set()
            candidates = ids
        if country:
            canon = self.canonical_country(country) or country
            node = ("country", canon)
            ids = {src[1] for src, _ in self.g.in_edges(node)} if node in self.g else set()
            candidates = ids if candidates is None else candidates & ids
        if candidates is None:
            candidates = set(self.entities)
        if entity_type:
            candidates = {i for i in candidates if self.entities[i].entity_type == entity_type}
        return sorted(candidates)

    def citation(self, record_id: int) -> dict[str, str | int]:
        return {
            "record_id": record_id,
            "source": "OFAC Specially Designated Nationals (SDN) list",
            "publication_date": self.snapshot.publication_date,
            "url": OFAC_DETAIL_URL.format(id=record_id),
        }

    def public_record(self, record_id: int) -> dict[str, Any]:
        """The record as the agent sees it. Only privacy-filtered fields exist upstream."""
        ent = self.entities[record_id]
        record: dict[str, Any] = {
            "record_id": ent.id,
            "name": ent.name,
            "entity_type": ent.entity_type,
            "programs": list(ent.programs),
            "aliases": [{"type": a.alias_type, "name": a.name} for a in ent.aliases],
            "address_countries": ent.address_countries,
        }
        if ent.title:
            record["title"] = ent.title
        if ent.nationalities:
            record["nationalities"] = ent.nationalities
        if ent.organization_type:
            record["organization_type"] = ent.organization_type
        if ent.crypto_addresses:
            record["digital_currency_addresses"] = [{"currency": c, "address": a} for c, a in ent.crypto_addresses]
        if ent.vessel:
            record["vessel"] = {k: v for k, v in vars(ent.vessel).items() if v}
        links = self.linked(ent.id)
        if links:
            record["linked_to"] = [
                {"record_id": other, "name": self.entities[other].name, "direction": direction}
                for other, direction in links[:25]
            ]
            if len(links) > 25:
                record["linked_to_truncated"] = len(links) - 25
        record["citation"] = self.citation(ent.id)
        return record
