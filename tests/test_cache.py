from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from investigator.cache import CachedInvestigator, SemanticCache
from investigator.graph_store import SanctionsGraph


class BagOfWordsEmbedder:
    """Deterministic stand-in for a sentence model: hashed bag of words."""

    model_name = "bag-of-words-test"

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalized: npt.NDArray[np.float32] = out / np.maximum(norms, 1e-12)
        return normalized


RESULT = {"status": "answered", "answer": "CUBA program.", "claims": [], "dropped_claims": []}


def test_guard_blocks_near_miss(graph: SanctionsGraph) -> None:
    cache = SemanticCache(graph, BagOfWordsEmbedder(), threshold=0.6)
    assert cache.store("Is BANCO NACIONAL DE CUBA on the SDN list?", RESULT)
    hit = cache.lookup("Is BANCO NACIONAL DE CUBA on the SDN list?")
    assert hit.hit
    near = cache.lookup("Is BANCO NACIONAL DE PANAMA on the SDN list?")
    assert not near.hit


def test_without_guard_near_miss_hits(graph: SanctionsGraph) -> None:
    cache = SemanticCache(graph, BagOfWordsEmbedder(), threshold=0.6, entity_guard=False)
    cache.store("Is BANCO NACIONAL DE CUBA on the SDN list?", RESULT)
    assert cache.lookup("Is BANCO NACIONAL DE PANAMA on the SDN list?").hit


def test_no_anchor_questions_are_not_cached(graph: SanctionsGraph) -> None:
    cache = SemanticCache(graph, BagOfWordsEmbedder())
    assert not cache.store("Is ACME ROBOTICS on the list?", RESULT)
    assert cache.lookup("Is ACME ROBOTICS on the list?").reason.startswith("no anchor")


def test_answers_with_dropped_claims_are_not_cached(graph: SanctionsGraph) -> None:
    cache = SemanticCache(graph, BagOfWordsEmbedder())
    assert not cache.store("Is BANCO NACIONAL DE CUBA listed?", RESULT | {"dropped_claims": [{"x": 1}]})


class FakeAgent:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.seeded: list[tuple[str, str]] = []

    def ask(self, question: str, thread_id: str = "default") -> dict[str, Any]:
        self.asked.append(question)
        return dict(RESULT, latency_s=1.0, cost_usd=0.001)

    def seed_thread(self, thread_id: str, question: str, result: dict[str, Any]) -> None:
        self.seeded.append((thread_id, question))


def test_cached_investigator_routes_first_turns_only(graph: SanctionsGraph) -> None:
    agent = FakeAgent()
    wrapper = CachedInvestigator(agent, SemanticCache(graph, BagOfWordsEmbedder(), threshold=0.6))
    q = "Is BANCO NACIONAL DE CUBA on the SDN list?"
    assert wrapper.ask(q, "t1")["cache"]["hit"] is False
    second = wrapper.ask(q, "t2")
    assert second["cache"]["hit"] is True and second["cost_usd"] == 0.0
    assert agent.seeded == [("t2", q)]
    # Follow-up in the same thread always goes to the agent.
    assert wrapper.ask(q, "t2")["cache"]["hit"] is False
    assert len(agent.asked) == 2
