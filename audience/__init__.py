"""Synthetic audience data.

This package is one *producer* of the audience data contract defined in
`audience/contract.py`. A real ingest -- from a biometric vendor, a survey tool
and an IAT platform -- would be another producer writing the same tables.
Nothing downstream imports this package: marts and notebooks read the
`audience` schema, never the generator.

Deleting this directory and pointing an ingest at the same tables is the
intended migration path to real data.
"""
