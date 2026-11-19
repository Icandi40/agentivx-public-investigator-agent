from __future__ import annotations

from pathlib import Path

import pytest

from investigator.graph_store import SanctionsGraph
from investigator.tools import Toolbox

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def graph() -> SanctionsGraph:
    return SanctionsGraph.load()


@pytest.fixture()
def toolbox(graph: SanctionsGraph) -> Toolbox:
    return Toolbox(graph)
