"""Export both OpenAPI specs to docs/, so they can be read without running anything.

    python -m api.spec

FastAPI serves the same documents live at /openapi.json, but a reviewer should
not have to start a container to read an API contract. These files are
committed; regenerate them whenever an endpoint or model changes.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("docs")


def main() -> int:
    from api.ingest import app as ingest_app
    from api.processing import app as processing_app

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, app in (("ingest", ingest_app), ("processing", processing_app)):
        spec = app.openapi()
        path = OUT_DIR / f"openapi-{name}.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n")
        n_ops = sum(len(ops) for ops in spec["paths"].values())
        print(f"wrote {path}  ({n_ops} operations, "
              f"{len(spec['components']['schemas'])} schemas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
