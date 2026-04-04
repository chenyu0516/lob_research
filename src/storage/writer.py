"""
src/storage/writer.py
---------------------
Writes the flat event table to Parquet using Hive-style partitioning.

Partition scheme:
    data/parquet/events/source=COINBASE/symbol=BTC-USD/date=2024-01-15/part.parquet

Partition values are derived from the data itself:
    source  — from the 'source' column
    symbol  — from the 'symbol' column
    date    — derived from the minimum ts value in each group

Default behaviour on existing partition: overwrite.
This is intentional for a research workflow — re-processing the same file
after a bug fix should replace the old output cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
import yaml

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


class StorageWriter:
    """
    Writes flat event DataFrames to partitioned Parquet files.

    Usage:
        writer = StorageWriter()
        writer.write_events(events_df)
    """

    def __init__(self, settings_path: Path = _SETTINGS_PATH) -> None:
        with open(settings_path) as f:
            cfg = yaml.safe_load(f)
        self._parquet_root = _PROJECT_ROOT / cfg["data"]["parquet_dir"] / "events"

    def write_events(
        self,
        df: pd.DataFrame,
        overwrite: bool = True,
    ) -> list[Path]:
        """
        Write a flat event DataFrame to Parquet, partitioned by source/symbol/date.

        A single DataFrame may span multiple source/symbol/date combinations —
        each unique combination is written to its own partition file.

        Parameters
        ----------
        df        : flat event table from Phase 2
        overwrite : if True (default), replace any existing partition file.
                    if False, raise FileExistsError if the partition already exists.

        Returns
        -------
        List of Paths to the written partition files.
        """
        if df.empty:
            log.warning("empty DataFrame passed to write_events — nothing written")
            return []

        written: list[Path] = []

        # Derive date from ts (nanoseconds → date string)
        # Use the minimum ts in the group as the representative date
        df = df.copy()
        df["_date"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")

        for (source, symbol, date), group in df.groupby(["source", "symbol", "_date"]):
            partition_dir  = self._parquet_root / f"source={source}" / f"symbol={symbol}" / f"date={date}"
            partition_file = partition_dir / "part.parquet"

            if partition_file.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Partition already exists: {partition_file}\n"
                        f"Pass overwrite=True to replace it."
                    )
                log.info("overwriting existing partition", path=str(partition_file))

            partition_dir.mkdir(parents=True, exist_ok=True)

            # Drop the helper column before writing
            out = group.drop(columns=["_date"])

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