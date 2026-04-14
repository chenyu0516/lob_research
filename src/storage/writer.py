"""
src/storage/writer.py
---------------------
Writes the flat event table to Parquet using Hive-style partitioning.

Partition scheme:
    data/parquet/events/source=COINBASE/symbol=BTC-USDT/date=2026-03-22/part.parquet

Partition values are derived from the data itself:
    source  — from the 'source' column
    symbol  — from the 'symbol' column
    date    — derived from the ts column (nanoseconds → UTC date string)

Conflict behaviour on existing partition:
    MERGE (default) — read existing rows, concatenate new rows, deduplicate
                      on (order_id, session_id, event_seq), rewrite.
                      Required for midnight-spillover correctness: today's file
                      may contain rows that belong to yesterday's partition
                      (midnight_correction in Stage A), which must be merged
                      into the already-written yesterday partition rather than
                      overwriting it.
    OVERWRITE       — replace the existing partition entirely. Use when
                      reprocessing a file after a bug fix and you want a
                      clean slate. Note: this will also erase any midnight-
                      spillover rows from subsequent days that were previously
                      merged in. Only use this if you plan to reprocess all
                      affected files.
    ERROR           — raise FileExistsError. Useful for detecting accidental
                      double-writes in automated pipelines.
"""

from __future__ import annotations

from pathlib import Path
from enum import Enum

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
import yaml

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"

# Deduplication key — uniquely identifies one event row across any merge
_DEDUP_KEY = ["order_id", "session_id", "event_seq"]


class ConflictMode(str, Enum):
    MERGE     = "merge"
    OVERWRITE = "overwrite"
    ERROR     = "error"


class StorageWriter:
    """
    Writes flat event DataFrames to partitioned Parquet files.

    Usage:
        writer = StorageWriter()
        writer.write_events(events_df)                          # merge by default
        writer.write_events(events_df, conflict=ConflictMode.OVERWRITE)
    """

    def __init__(self, settings_path: Path = _SETTINGS_PATH) -> None:
        with open(settings_path) as f:
            cfg = yaml.safe_load(f)
        self._parquet_root = _PROJECT_ROOT / cfg["data"]["parquet_dir"] / "events"

    def write_events(
        self,
        df: pd.DataFrame,
        conflict: ConflictMode = ConflictMode.MERGE,
    ) -> list[Path]:
        """
        Write a flat event DataFrame to Parquet, partitioned by source/symbol/date.

        A single DataFrame may span multiple date partitions — most commonly
        when midnight_correction assigns some rows from today's file to
        yesterday's date. Each partition is handled independently.

        Parameters
        ----------
        df       : flat event table from Phase 2
        conflict : how to handle an already-existing partition file.
                   MERGE (default) — merge new rows into existing partition,
                                     deduplicate on (order_id, session_id, event_seq).
                   OVERWRITE       — replace existing partition entirely.
                   ERROR           — raise FileExistsError.

        Returns
        -------
        List of Paths to the written partition files.
        """
        if df.empty:
            log.warning("empty DataFrame passed to write_events — nothing written")
            return []

        written: list[Path] = []

        df = df.copy()
        df["_date"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")

        for (source, symbol, date), group in df.groupby(["source", "symbol", "_date"]):
            partition_dir  = self._parquet_root / f"source={source}" / f"symbol={symbol}" / f"date={date}"
            partition_file = partition_dir / "part.parquet"
            out            = group.drop(columns=["_date"])

            if partition_file.exists():
                if conflict == ConflictMode.ERROR:
                    raise FileExistsError(
                        f"Partition already exists: {partition_file}\n"
                        f"Use ConflictMode.MERGE or ConflictMode.OVERWRITE."
                    )

                elif conflict == ConflictMode.OVERWRITE:
                    log.warning(
                        "overwriting existing partition — any previously merged "
                        "midnight-spillover rows will be lost",
                        path=str(partition_file),
                    )

                elif conflict == ConflictMode.MERGE:
                    existing   = pq.ParquetFile(partition_file).read().to_pandas()
                    rows_before = len(existing)
                    out        = (
                        pd.concat([existing, out], ignore_index=True)
                        .drop_duplicates(subset=_DEDUP_KEY, keep="last")
                        .sort_values(["ts", "order_id", "event_seq"])
                        .reset_index(drop=True)
                    )
                    rows_added = len(out) - rows_before
                    log.info(
                        "merged into existing partition",
                        path=str(partition_file),
                        rows_before=rows_before,
                        rows_added=rows_added,
                        rows_after=len(out),
                    )

            partition_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(out, preserve_index=False)
            pq.write_table(table, partition_file, compression="snappy")

            log.info(
                "partition written",
                path=str(partition_file),
                rows=len(out),
                source=source,
                symbol=symbol,
                date=date,
            )
            written.append(partition_file)

        return written