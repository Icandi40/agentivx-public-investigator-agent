"""Deterministic grading of agent results against golden cases.

Definitions used in EVALS.md:
- Task success: status is allowed, every required record is cited, every
  required value appears in the answer or claims, nothing forbidden appears,
  and fuzzy matches are labeled as possible matches.
- Citation recall: required record ids that the final answer cites.
- Claim hallucination: a structured claim whose value is not stated in any
  record it cites (or that cites a record never retrieved). Measured on the
  model's first draft and on its final draft (after one verifier retry).
- Ungrounded prose span: a name, number or identifier in the prose answer that
  appears in neither the tool outputs nor the question (heuristic).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from investigator.citations import ungrounded_spans
from investigator.graph_store import normalize
from investigator.tools import ToolTrace

FUZZY_LABELS = {"high", "medium", "low"}


@dataclass
class Grade:
    case_id: str
    category: str
    success: bool
    status_ok: bool
    cite_ok: bool
    include_ok: bool
    exclude_ok: bool
    fuzzy_ok: bool
    required_cites: int
    cited_required: int
    draft_claims: int
    draft_failed: int
    final_claims: int
    final_failed: int
    ungrounded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _corpus(result: dict[str, Any]) -> str:
    parts = [str(result.get("answer", ""))]
    for claim in result.get("claims", []):
        parts.append(str(claim.get("text", "")))
        parts.append(str(claim.get("value", "")))
    return " ".join(parts)


def grade(case: dict[str, Any], result: dict[str, Any], question: str, context: str = "") -> Grade:
    notes: list[str] = []
    status = result.get("status")
    status_ok = status in case["allowed_status"]
    if not status_ok:
        notes.append(f"status {status!r} not in {case['allowed_status']}")

    claims = result.get("claims", [])
    cited = {int(i) for c in claims for i in c.get("record_ids", [])}
    required = [int(i) for i in case["must_cite"]]
    cited_required = sum(1 for i in required if i in cited)
    cite_ok = cited_required == len(required)
    if not cite_ok:
        notes.append(f"missing citations {[i for i in required if i not in cited]}")

    corpus = _corpus(result)
    ncorpus = normalize(corpus)
    missing = [s for s in case["must_include"] if normalize(s) not in ncorpus]
    include_ok = not missing
    if missing:
        notes.append(f"missing values {missing}")

    exclude_hits = [s for s in case["must_not_include"] if normalize(s) in ncorpus]
    if case.get("forbid_regex") and re.search(case["forbid_regex"], corpus):
        exclude_hits.append(f"regex:{case['forbid_regex']}")
    exclude_ok = not exclude_hits
    if exclude_hits:
        notes.append(f"forbidden content {exclude_hits}")

    fuzzy_ok = True
    if case.get("expect_fuzzy_label"):
        fuzzy_ok = any(
            c.get("match_confidence") in FUZZY_LABELS and set(c.get("record_ids", [])) & set(required) for c in claims
        )
        if not fuzzy_ok:
            notes.append("fuzzy match not labeled as possible match")

    draft_checks = result.get("first_draft_checks", [])
    final_failed = len(result.get("dropped_claims", []))
    trace = ToolTrace.from_dict(result.get("trace"))
    ungrounded = ungrounded_spans(str(result.get("answer", "")), question, trace, context)

    success = status_ok and cite_ok and include_ok and exclude_ok and fuzzy_ok
    return Grade(
        case_id=case["id"],
        category=case["category"],
        success=success,
        status_ok=status_ok,
        cite_ok=cite_ok,
        include_ok=include_ok,
        exclude_ok=exclude_ok,
        fuzzy_ok=fuzzy_ok,
        required_cites=len(required),
        cited_required=cited_required,
        draft_claims=len(draft_checks),
        draft_failed=sum(1 for c in draft_checks if not c["supported"]),
        final_claims=len(claims) + final_failed,
        final_failed=final_failed,
        ungrounded=ungrounded,
        notes=notes,
    )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered) + 0.5)))
    return ordered[rank - 1]


def summarize(grades: list[Grade], latencies: list[float], costs: list[float]) -> dict[str, Any]:
    n = len(grades)
    draft_claims = sum(g.draft_claims for g in grades)
    final_claims = sum(g.final_claims for g in grades)
    required = sum(g.required_cites for g in grades)
    answers_with_issue = sum(1 for g in grades if g.final_failed or g.ungrounded)
    by_cat: dict[str, list[Grade]] = {}
    for g in grades:
        by_cat.setdefault(g.category, []).append(g)
    return {
        "cases": n,
        "task_success_rate": sum(g.success for g in grades) / n if n else 0.0,
        "citation_recall": sum(g.cited_required for g in grades) / required if required else 0.0,
        "required_citations": required,
        "draft_claims": draft_claims,
        "draft_hallucination_rate": sum(g.draft_failed for g in grades) / draft_claims if draft_claims else 0.0,
        "final_claims": final_claims,
        "final_hallucination_rate": sum(g.final_failed for g in grades) / final_claims if final_claims else 0.0,
        "answers_with_ungrounded_content_rate": answers_with_issue / n if n else 0.0,
        "answers_with_ungrounded_prose_rate": sum(1 for g in grades if g.ungrounded) / n if n else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_s": percentile(latencies, 95),
        "cost_total_usd": sum(costs),
        "cost_per_question_usd": sum(costs) / n if n else 0.0,
        "by_category": {
            cat: {"cases": len(gs), "success_rate": sum(g.success for g in gs) / len(gs)}
            for cat, gs in sorted(by_cat.items())
        },
    }
