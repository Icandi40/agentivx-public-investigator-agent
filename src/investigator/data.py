"""Load the OFAC SDN list (legacy CSV format) into typed records.

Privacy rule applied here, at ingest: only fields that describe the listing
itself are kept (names, aliases, programs, entity type, public titles,
country-level location, nationality, vessel identifiers, linked entities and
digital currency addresses that OFAC publishes for screening). Dates and places
of birth, passport and national ID numbers, tax IDs, emails, phone numbers and
street addresses are dropped and never reach the agent.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NULL = "-0-"

ENTITY_TYPES = {"individual", "vessel", "aircraft"}  # blank type means a company or group


@dataclass(frozen=True)
class Alias:
    alias_type: str  # aka, fka (formerly known as), nka (now known as)
    name: str


@dataclass(frozen=True)
class VesselInfo:
    call_sign: str | None
    vessel_type: str | None
    flag: str | None
    owner: str | None
    imo: str | None


@dataclass
class Entity:
    id: int
    name: str
    entity_type: str
    programs: tuple[str, ...]
    title: str | None = None
    aliases: list[Alias] = field(default_factory=list)
    address_countries: list[str] = field(default_factory=list)
    nationalities: list[str] = field(default_factory=list)
    linked_to_names: list[str] = field(default_factory=list)
    crypto_addresses: list[tuple[str, str]] = field(default_factory=list)
    organization_type: str | None = None
    vessel: VesselInfo | None = None


@dataclass(frozen=True)
class Snapshot:
    publication_date: str
    retrieved_at: str
    source_url: str


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == NULL:
        return None
    return value


def parse_programs(raw: str) -> tuple[str, ...]:
    """'SDGT] [IFSR' -> ('SDGT', 'IFSR')."""
    cleaned = _clean(raw)
    if cleaned is None:
        return ()
    parts = [p.strip(" []") for p in cleaned.split("] [")]
    return tuple(p for p in parts if p)


_CRYPTO_RE = re.compile(r"^Digital Currency Address - ([A-Z0-9]+) (\S+)$")
_IMO_RE = re.compile(r"^Vessel Registration Identification IMO (\d+)$")


@dataclass
class RemarkFacts:
    linked_to: list[str] = field(default_factory=list)
    nationalities: list[str] = field(default_factory=list)
    crypto: list[tuple[str, str]] = field(default_factory=list)
    imo: str | None = None
    organization_type: str | None = None


def parse_remarks(raw: str | None) -> RemarkFacts:
    """Pull the public, non-personal facts out of the free-text remarks column."""
    facts = RemarkFacts()
    cleaned = _clean(raw)
    if cleaned is None:
        return facts
    for part in cleaned.split(";"):
        item = part.strip().rstrip(".").strip()
        if item.startswith("alt. "):
            item = item[5:].strip()
        if item.startswith("Linked To:"):
            target = item[len("Linked To:") :].strip()
            if target:
                facts.linked_to.append(target)
        elif item.startswith("nationality "):
            facts.nationalities.append(item[len("nationality ") :].strip())
        elif item.startswith("citizen "):
            facts.nationalities.append(item[len("citizen ") :].strip())
        elif item.startswith("Organization Type:"):
            facts.organization_type = item[len("Organization Type:") :].strip() or None
        elif m := _CRYPTO_RE.match(item):
            facts.crypto.append((m.group(1), m.group(2)))
        elif m := _IMO_RE.match(item):
            facts.imo = m.group(1)
    facts.nationalities = sorted(set(facts.nationalities))
    return facts


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    # The files end with a DOS end-of-file marker (0x1A) on its own line.
    return [r for r in rows if r and r[0].strip() and r[0].strip() != "\x1a"]


def load_entities(raw_dir: Path) -> dict[int, Entity]:
    entities: dict[int, Entity] = {}
    for row in _read_rows(raw_dir / "SDN.CSV"):
        row = row + [""] * (12 - len(row))
        ent_id = int(row[0])
        raw_type = _clean(row[2])
        entity_type = raw_type if raw_type in ENTITY_TYPES else "entity"
        facts = parse_remarks(row[11])
        vessel = None
        if entity_type == "vessel":
            vessel = VesselInfo(
                call_sign=_clean(row[5]),
                vessel_type=_clean(row[6]),
                flag=_clean(row[9]),
                owner=_clean(row[10]),
                imo=facts.imo,
            )
        entities[ent_id] = Entity(
            id=ent_id,
            name=row[1].strip(),
            entity_type=entity_type,
            programs=parse_programs(row[3]),
            title=_clean(row[4]),
            nationalities=facts.nationalities,
            linked_to_names=facts.linked_to,
            crypto_addresses=facts.crypto,
            organization_type=facts.organization_type,
            vessel=vessel,
        )

    countries: dict[int, set[str]] = defaultdict(set)
    for row in _read_rows(raw_dir / "ADD.CSV"):
        row = row + [""] * (6 - len(row))
        country = _clean(row[4])
        if country:
            countries[int(row[0])].add(country)
    for ent_id, found in countries.items():
        if ent_id in entities:
            entities[ent_id].address_countries = sorted(found)

    for row in _read_rows(raw_dir / "ALT.CSV"):
        row = row + [""] * (5 - len(row))
        ent_id = int(row[0])
        name = _clean(row[3])
        if ent_id in entities and name:
            entities[ent_id].aliases.append(Alias(alias_type=_clean(row[2]) or "aka", name=name))
    return entities


def load_snapshot(raw_dir: Path) -> Snapshot:
    manifest = raw_dir / "MANIFEST.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return Snapshot(
            publication_date=data["publication_date"],
            retrieved_at=data["retrieved_at"],
            source_url=data["source_url"],
        )
    return Snapshot(publication_date="unknown", retrieved_at="unknown", source_url="")


PROCESSED_NAME = "sdn_public.json.gz"


def entity_to_dict(entity: Entity) -> dict[str, object]:
    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "programs": list(entity.programs),
        "title": entity.title,
        "aliases": [[a.alias_type, a.name] for a in entity.aliases],
        "address_countries": entity.address_countries,
        "nationalities": entity.nationalities,
        "linked_to_names": entity.linked_to_names,
        "crypto_addresses": [list(c) for c in entity.crypto_addresses],
        "organization_type": entity.organization_type,
        "vessel": None
        if entity.vessel is None
        else {
            "call_sign": entity.vessel.call_sign,
            "vessel_type": entity.vessel.vessel_type,
            "flag": entity.vessel.flag,
            "owner": entity.vessel.owner,
            "imo": entity.vessel.imo,
        },
    }


def entity_from_dict(d: dict[str, Any]) -> Entity:
    vessel = d.get("vessel")
    return Entity(
        id=int(d["id"]),
        name=d["name"],
        entity_type=d["entity_type"],
        programs=tuple(d["programs"]),
        title=d.get("title"),
        aliases=[Alias(alias_type=t, name=n) for t, n in d["aliases"]],
        address_countries=list(d["address_countries"]),
        nationalities=list(d["nationalities"]),
        linked_to_names=list(d["linked_to_names"]),
        crypto_addresses=[(c, a) for c, a in d["crypto_addresses"]],
        organization_type=d.get("organization_type"),
        vessel=None if vessel is None else VesselInfo(**vessel),
    )


def write_processed(raw_dir: Path, out_dir: Path) -> Path:
    """Write the privacy-filtered snapshot that the graph is built from."""
    entities = load_entities(raw_dir)
    snapshot = load_snapshot(raw_dir)
    payload = {
        "snapshot": {
            "publication_date": snapshot.publication_date,
            "retrieved_at": snapshot.retrieved_at,
            "source_url": snapshot.source_url,
        },
        "entities": [entity_to_dict(e) for e in sorted(entities.values(), key=lambda e: e.id)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / PROCESSED_NAME
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return out


def read_processed(path: Path) -> tuple[Snapshot, dict[int, Entity]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    snap = payload["snapshot"]
    snapshot = Snapshot(
        publication_date=snap["publication_date"],
        retrieved_at=snap["retrieved_at"],
        source_url=snap["source_url"],
    )
    entities = {e.id: e for e in (entity_from_dict(d) for d in payload["entities"])}
    return snapshot, entities
