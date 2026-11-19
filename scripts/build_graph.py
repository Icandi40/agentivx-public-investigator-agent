"""Turn data/raw (OFAC CSVs) into the privacy-filtered snapshot and print graph stats.

Usage: uv run python scripts/build_graph.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from investigator.data import write_processed
from investigator.graph_store import SanctionsGraph

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    out = write_processed(ROOT / "data" / "raw", ROOT / "data" / "processed")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)")
    start = time.perf_counter()
    graph = SanctionsGraph.load(out)
    print(f"graph built in {time.perf_counter() - start:.2f}s")
    print(json.dumps(graph.stats(), indent=2))


if __name__ == "__main__":
    main()
