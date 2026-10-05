"""Nine extractors, one interface.

Every module in this package exposes the same handful of module-level
attributes and a single function:

    NAME        str   stable key used in the manifest and in raw/ paths
    VERSION     str   bump to invalidate cached output for that extractor only
    FEATURE     str   one-line human description (surfaced in docs/schema.md)
    TOOL        str   the library or model doing the work
    GRAIN       str   the row grain of its output
    REQUIRES    tuple other extractors whose output it reads (usually empty)

    extract(prepared: Prepared) -> dict[table_name, DataFrame]

That uniformity is the whole point: the list of features is a registry entry,
not an architectural change. Adding a tenth signal means adding one file.
"""
