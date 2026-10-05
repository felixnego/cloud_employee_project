"""Fold landed biometric batches into the warehouse.

Run after ingest, when the DuckDB write lock is free:

    python -m api.load

Kept separate from the API process for the reason the ingest module explains:
the API must never hold DuckDB's single write lock. Ingested rows land in
`audience.ingested_biometric_seconds` -- a table of their own rather than mixed
into the synthetic panel, because whether real and generated rows should ever
sit together is a decision for whoever has real data.
"""
from __future__ import annotations

import sys

from pipeline import config
from pipeline.transform import connect

LANDING_DIR = config.WAREHOUSE_DIR / "landing" / "biometrics"
TABLE = "audience.ingested_biometric_seconds"


def main() -> int:
    files = sorted(LANDING_DIR.glob("*.parquet"))
    if not files:
        print(f"nothing to load in {LANDING_DIR}")
        return 0

    pattern = str(LANDING_DIR / "*.parquet")
    with connect() as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS audience")
        con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE} AS
            SELECT * FROM read_parquet('{pattern}', union_by_name = true)
        """)
        n, sessions = con.execute(
            f"SELECT count(*), count(DISTINCT session_id) FROM {TABLE}").fetchone()

    print(f"loaded {len(files)} batch file(s) -> {TABLE}: "
          f"{n:,} rows across {sessions} session(s)")
    print("landing files are kept; delete them once you are satisfied with the load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
