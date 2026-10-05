"""Runs transforms/*.sql against the DuckDB file, in filename order.

Deliberately not dbt. For eight models with no cross-project lineage needs,
a 30-line runner keeps the stack to one dependency and the SQL stays plain
DuckDB that anyone can paste into a notebook. dbt-duckdb is the upgrade path
once there are enough models to need tests and docs -- the SQL ports as-is.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from . import config


def _render(sql: str) -> str:
    """A few path substitutions, so the SQL stays readable and copy-pasteable."""
    return (sql
            .replace("${RAW}", config.RAW_DIR.as_posix())
            .replace("${MARTS}", config.MART_DIR.as_posix())
            .replace("${AUDIENCE}", (config.WAREHOUSE_DIR / "audience").as_posix()))


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)


def run(directory: Path | None = None, verbose: bool = True) -> list[str]:
    """Execute every .sql file in `directory`, in filename order."""
    directory = directory or config.TRANSFORM_DIR
    config.MART_DIR.mkdir(parents=True, exist_ok=True)
    scripts = sorted(directory.glob("*.sql"))
    if not scripts:
        raise FileNotFoundError(f"no .sql files in {directory}")

    applied: list[str] = []
    with connect() as con:
        for path in scripts:
            if verbose:
                print(f"  [sql] {path.name}")
            con.execute(_render(path.read_text()))
            applied.append(path.name)
    return applied


def table_summary() -> "list[tuple]":
    with connect(read_only=True) as con:
        return con.execute("""
            SELECT table_name, estimated_size
            FROM duckdb_tables()
            WHERE schema_name = 'main'
            ORDER BY table_name
        """).fetchall()
