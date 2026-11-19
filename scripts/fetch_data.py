"""Download the current OFAC SDN list (legacy CSV files) into data/raw.

Usage: uv run python scripts/fetch_data.py

The Treasury Sanctions List Service redirects each export to a signed S3 URL
whose path contains the publication date. Only that date is recorded; the
signed URL itself is never stored.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BASE = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/"
FILES = ("SDN.CSV", "ALT.CSV", "ADD.CSV")
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    publication_date = "unknown"
    for name in FILES:
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "public-investigator-agent"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            match = re.search(r"/Published/[^/]+/(\d{4}-\d{2}-\d{2})/", response.geturl())
            if match:
                publication_date = match.group(1)
        (RAW_DIR / name).write_bytes(body)
        print(f"{name}: {len(body):,} bytes")
    manifest = {
        "source": "U.S. Treasury OFAC Specially Designated Nationals (SDN) list, legacy CSV",
        "source_url": BASE,
        "publication_date": publication_date,
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": list(FILES),
    }
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"publication date {publication_date}")


if __name__ == "__main__":
    main()
