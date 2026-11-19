"""Semantic cache for first-turn questions, with an entity guard.

A plain embedding cache is risky for investigations: "Is BANCO NACIONAL DE CUBA
listed?" and "Is BANCO NACIONAL DE PANAMA listed?" embed almost identically but
need different answers. So a cached answer is reused only when
1. the questions' embeddings are similar (cosine >= threshold), and
2. both questions mention exactly the same records, programs, countries,
   numbers and identifiers (SanctionsGraph.mentions).
Questions that mention nothing the graph knows (a misspelled or unlisted name)
are never cached, because there is no anchor to check reuse against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from investigator.graph_store import SanctionsGraph

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THRESHOLD = 0.90


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


class FastEmbedder:
    """Local ONNX sentence embeddings via fastembed (no API calls)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized: npt.NDArray[np.float32] = vectors / np.maximum(norms, 1e-12)
        return normalized


@dataclass
class CacheEntry:
    question: str
    anchors: frozenset[str]
    vector: npt.NDArray[np.float32]
    result: dict[str, Any]


@dataclass
class CacheDecision:
    hit: bool
    reason: str
    similarity: float
    anchors: frozenset[str]
    matched_question: str | None = None
    result: dict[str, Any] | None = None
    lookup_ms: float = 0.0


@dataclass
class SemanticCache:
    graph: SanctionsGraph
    embedder: Embedder
    threshold: float = DEFAULT_THRESHOLD
    entity_guard: bool = True
    entries: list[CacheEntry] = field(default_factory=list)

    def anchors(self, question: str) -> frozenset[str]:
        return frozenset(self.graph.mentions(question))

    def lookup(self, question: str) -> CacheDecision:
        start = time.perf_counter()
        anchors = self.anchors(question)
        if self.entity_guard and not anchors:
            return CacheDecision(
                False,
                "no anchor: question names nothing on the list",
                0.0,
                anchors,
                lookup_ms=(time.perf_counter() - start) * 1000,
            )
        if not self.entries:
            return CacheDecision(False, "empty cache", 0.0, anchors, lookup_ms=(time.perf_counter() - start) * 1000)
        query = self.embedder.embed([question])[0]
        matrix = np.stack([e.vector for e in self.entries])
        sims = matrix @ query
        order = np.argsort(-sims)
        best_sim = float(sims[order[0]])
        for idx in order:
            sim = float(sims[idx])
            if sim < self.threshold:
                break
            entry = self.entries[int(idx)]
            if self.entity_guard and entry.anchors != anchors:
                continue
            return CacheDecision(
                True, "hit", sim, anchors, entry.question, entry.result, lookup_ms=(time.perf_counter() - start) * 1000
            )
        reason = "below threshold" if best_sim < self.threshold else "similar question names different things"
        return CacheDecision(False, reason, best_sim, anchors, lookup_ms=(time.perf_counter() - start) * 1000)

    def store(self, question: str, result: dict[str, Any]) -> bool:
        """Cache a verified answer. Returns False when the answer is not cacheable."""
        anchors = self.anchors(question)
        if self.entity_guard and not anchors:
            return False
        if result.get("status") not in {"answered", "not_found", "refused"}:
            return False
        if result.get("dropped_claims"):
            return False
        vector = self.embedder.embed([question])[0]
        self.entries.append(CacheEntry(question, anchors, vector, result))
        return True


class CachedInvestigator:
    """Wraps an agent: first-turn questions go through the cache, follow-ups never do."""

    def __init__(self, agent: Any, cache: SemanticCache) -> None:
        self.agent = agent
        self.cache = cache
        self._threads_with_history: set[str] = set()

    def ask(self, question: str, thread_id: str = "default") -> dict[str, Any]:
        first_turn = thread_id not in self._threads_with_history
        if first_turn:
            start = time.perf_counter()
            decision = self.cache.lookup(question)
            if decision.hit and decision.result is not None:
                result = dict(decision.result)
                result.update(
                    {
                        "cache": {
                            "hit": True,
                            "similarity": round(decision.similarity, 4),
                            "matched_question": decision.matched_question,
                        },
                        "latency_s": round(time.perf_counter() - start, 4),
                        "cost_usd": 0.0,
                    }
                )
                # Seed the thread so a follow-up question still has this turn as context.
                self.agent.seed_thread(thread_id, question, result)
                self._threads_with_history.add(thread_id)
                return result
        result = dict(self.agent.ask(question, thread_id=thread_id))
        self._threads_with_history.add(thread_id)
        if first_turn:
            self.cache.store(question, result)
        result["cache"] = {"hit": False}
        return result
