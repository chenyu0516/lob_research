"""
src/pipeline.py
---------------
Single entry point that wires Phase 2 and Phase 3 together.

For each raw CSV file:
    1. Run Stage A + Stage B  (source-specific ingestion)
    2. Write event table to partitioned Parquet

Usage — from project root:
    python -m src.pipeline --source coinbase --file data/raw/coinbase/20240115_BTC-USD.csv
    python -m src.pipeline --source databento --file data/raw/databento/AAPL.csv --symbol AAPL
    python -m src.pipeline --source coinbase --dir data/raw/coinbase/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logging

setup_logging()
log = structlog.get_logger("pipeline")


def run_file(
    path: Path,
    source: str,
    symbol: str | None = None,
) -> None:
    """
    Process a single raw CSV file end-to-end and write to Parquet.

    Parameters
    ----------
    path   : path to the raw CSV file
    source : "coinbase" or "databento"
    symbol : required for Databento if the CSV has no symbol column
    """
    source = source.lower()
    log.info("pipeline start", file=str(path), source=source)

    # ── Phase 2: ingest and normalize ────────────────────────────────────────
    if source == "coinbase":
        from src.ingestion.coinbase import process
        events = process(path)

    elif source == "databento":
        from src.ingestion.databento import process
        events = process(path, symbol=symbol)

    else:
        raise ValueError(
            f"Unknown source '{source}'. Expected 'coinbase' or 'databento'."
        )

    if events.empty:
        log.warning("no events produced — skipping storage", file=str(path))
        return

    log.info(
        "ingestion complete",
        rows=len(events),
        event_breakdown=events["event_type"].value_counts().to_dict(),
    )

    # ── Phase 3a: validate ────────────────────────────────────────────────────
    from src.storage.validator import run as validate
    report = validate(events)
    if not report.passed:
        log.warning("validation issues found — writing anyway", summary=report.summary())

    # ── Phase 3b: write to Parquet ────────────────────────────────────────────
    from src.storage.writer import StorageWriter
    writer  = StorageWriter()
    written = writer.write_events(events)

    log.info(
        "pipeline complete",
        file=str(path),
        partitions_written=len(written),
        paths=[str(p) for p in written],
    )


def run_directory(
    directory: Path,
    source: str,
    symbol: str | None = None,
    glob: str = "*.csv",
) -> None:
    """
    Process all CSV files in a directory.

    Parameters
    ----------
    directory : folder containing raw CSV files
    source    : "coinbase" or "databento"
    symbol    : passed through to run_file for Databento files
    glob      : file pattern to match, default "*.csv"
    """
    files = sorted(directory.glob(glob))
    if not files:
        log.warning("no files found", directory=str(directory), pattern=glob)
        return

    log.info("processing directory", directory=str(directory), file_count=len(files))
    failed: list[tuple[Path, str]] = []

    for i, f in enumerate(files, 1):
        log.info(f"processing file {i}/{len(files)}", file=f.name)
        try:
            run_file(f, source=source, symbol=symbol)
        except Exception as exc:
            log.error("file failed — continuing", file=str(f), error=str(exc))
            failed.append((f, str(exc)))

    log.info(
        "directory processing complete",
        total=len(files),
        succeeded=len(files) - len(failed),
        failed=len(failed),
    )
    if failed:
        log.warning("failed files", files=[(str(f), e) for f, e in failed])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LOB research pipeline — ingest raw CSV and write to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single Coinbase file (date and symbol parsed from filename)
  python -m src.pipeline --source coinbase --file data/raw/coinbase/20240115_BTC-USD.csv

  # single Databento file (symbol provided explicitly)
  python -m src.pipeline --source databento --file data/raw/databento/AAPL.csv --symbol AAPL

  # all CSV files in a directory
  python -m src.pipeline --source coinbase --dir data/raw/coinbase/
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="path to a single raw CSV file")
    group.add_argument("--dir",  type=Path, help="path to a directory of raw CSV files")
    parser.add_argument(
        "--source", required=True,
        choices=["coinbase", "databento"],
        help="data source",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="instrument symbol (required for Databento if not in CSV)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.file:
        run_file(args.file, source=args.source, symbol=args.symbol)
    else:
        run_directory(args.dir, source=args.source, symbol=args.symbol)