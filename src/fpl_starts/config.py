"""Default paths for this package's CLI tools and library functions.

All relative, resolved against the current working directory a command is
invoked from -- this package has no opinion on where "the repo" is, or on
where a consuming project's shared archive location might be. Every CLI
tool (`fpl-starts-archive`/`-derive`/`-predict`/`-score`) also takes an
explicit override (`--base-dir`, `--db-path`, `--predictions-dir`) for
anyone wiring this package into a larger pipeline -- see README.md's
"Running the pipeline" section.

The database lives under `db/`; `derived.rebuild()` (the only writer)
creates that directory on first use.
"""

RAW_DIR = "raw"
DERIVED_DB_PATH = "db/derived.db"
PREDICTIONS_DIR = "predictions"
