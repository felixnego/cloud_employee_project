"""CLI for the synthetic audience layer.

    python -m audience.run generate [--seed 42] [--n-viewers 300]
    python -m audience.run load        # build the audience + analysis schemas
    python -m audience.run validate
    python -m audience.run all

Separate from `pipeline.run` on purpose: the creative layer must stand on its
own, and this package is meant to be deletable the day real panel data arrives.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pipeline import transform

from . import config, generate as gen, validate


def _write(tables: dict[str, pd.DataFrame]) -> None:
    config.AUDIENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(config.AUDIENCE_DIR / f"{name}.parquet",
                      index=False, compression="zstd")
        print(f"  {name:<22} {len(df):>8,} rows")


def cmd_generate(args) -> int:
    print(f"generating panel: seed={args.seed or config.SEED}, "
          f"viewers={args.n_viewers or config.N_VIEWERS}")
    tables = gen.generate(seed=args.seed, n_viewers=args.n_viewers)
    _write(tables)
    print(f"\nwrote {len(tables)} tables to {config.AUDIENCE_DIR}")
    return 0


def cmd_load(args) -> int:
    print(f"building audience + analysis schemas in {config.DUCKDB_PATH}")
    transform.run(config.TRANSFORM_DIR)
    with transform.connect(read_only=True) as con:
        rows = con.execute("""
            SELECT schema_name || '.' || table_name AS name, estimated_size AS rows
            FROM duckdb_tables()
            WHERE schema_name IN ('audience', 'analysis')
            ORDER BY name
        """).fetchall()
    print()
    for name, n in rows:
        print(f"  {name:<36} {n:>9,} rows")
    return 0


def cmd_validate(args) -> int:
    with transform.connect(read_only=True) as con:
        tables = {t: con.execute(f"SELECT * FROM audience.{t}").df() for t in
                  ("viewers", "sessions", "biometric_seconds", "iat_scores")}
        creative_ids = {r[0] for r in con.execute(
            "SELECT creative_id FROM creatives").fetchall()}
    _, cseconds = gen.load_creative_features()

    results = validate.run_all(tables, cseconds, creative_ids)
    width = results.check.str.len().max()
    for r in results.itertuples():
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.check:<{width}}  {r.detail}")
    failed = (~results.passed).sum()
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


def cmd_all(args) -> int:
    cmd_generate(args)
    print()
    cmd_load(args)
    print()
    return cmd_validate(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audience.run",
                                 description="Synthetic audience panel generator")
    sub = ap.add_subparsers(dest="command", required=True)

    def flags(p):
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--n-viewers", type=int, default=None)

    p = sub.add_parser("generate", help="generate the panel and write Parquet")
    flags(p); p.set_defaults(fn=cmd_generate)
    sub.add_parser("load", help="build the audience + analysis schemas").set_defaults(fn=cmd_load)
    sub.add_parser("validate", help="integrity and coefficient-recovery checks").set_defaults(fn=cmd_validate)
    p = sub.add_parser("all", help="generate, load, validate")
    flags(p); p.set_defaults(fn=cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
