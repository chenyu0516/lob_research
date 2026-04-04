"""
src/storage/loader.py
---------------------
Reads partitioned Parquet event files back into pandas DataFrames.

Mirrors the partition scheme written by StorageWriter:
    data/parquet/events/source=COINBASE/symbol=BTC-USD/date=2024-01-15/part.parquet

Two public functions:
    load_events(source, symbol, dates)  — load one or more date partitions
    list_available(source, symbol)      — scan what data exists on disk
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import structlog
import yaml

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _parquet_root(settings_path: Path = _SETTINGS_PATH) -> Path:
    with open(settings_path) as f:
        cfg = yaml.safe_load(f)
    return _PROJECT_ROOT / cfg["data"]["parquet_dir"] / "events"


def load_events(
    source: str,
    symbol: str,
    dates: str | list[str] | tuple[str, str],
    settings_path: Path = _SETTINGS_PATH,
) -> pd.DataFrame:
    """
    Load event Parquet partitions for a given source, symbol, and date(s).

    Parameters
    ----------
    source  : e.g. "COINBASE" or "DATABENTO" (case-insensitive)
    symbol  : e.g. "BTC-USD" or "AAPL"
    dates   : one of:
                - a single date string      "2024-01-15"
                - a list of date strings    ["2024-01-15", "2024-01-16"]
                - a (start, end) date tuple ("2024-01-15", "2024-01-19")
                  — inclusive on both ends

    Returns
    -------
    DataFrame sorted by ts, concatenation of all requested partitions.
    Raises FileNotFoundError if any requested partition does not exist.
    """
    root   = _parquet_root(settings_path)
    source = source.upper()
    dates  = _resolve_dates(dates)

    frames: list[pd.DataFrame] = []
    for date in dates:
        partition_file = root / f"source={source}" / f"symbol={symbol}" / f"date={date}" / "part.parquet"
        if not partition_file.exists():
            raise FileNotFoundError(
                f"No data found for source={source}, symbol={symbol}, date={date}.\n"
                f"Expected path: {partition_file}\n"
                f"Run list_available() to see what is on disk."
            )
        log.info("loading partition", path=str(partition_file))
        frames.append(pq.read_table(partition_file).to_pandas())

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    log.info(
        "load complete",
        source=source,
        symbol=symbol,
        dates=dates,
        total_rows=len(result),
    )
    return result


def list_available(
    source: str | None = None,
    symbol: str | None = None,
    settings_path: Path = _SETTINGS_PATH,
) -> pd.DataFrame:
    """
    Scan the Parquet directory and return a summary of available partitions.

    Parameters
    ----------
    source  : optional filter, e.g. "COINBASE"
    symbol  : optional filter, e.g. "BTC-USD"

    Returns
    -------
    DataFrame with columns: source, symbol, date, rows, path
    One row per existing partition file, sorted by source / symbol / date.
    Empty DataFrame if no partitions exist yet.
    """
    root = _parquet_root(settings_path)

    if not root.exists():
        log.info("parquet root does not exist yet", path=str(root))
        return pd.DataFrame(columns=["source", "symbol", "date", "rows", "path"])

    records: list[dict] = []

    # Walk source= / symbol= / date= directory structure
    for source_dir in sorted(root.glob("source=*")):
        src_val = source_dir.name.split("=", 1)[1]
        if source and src_val.upper() != source.upper():
            continue

        for symbol_dir in sorted(source_dir.glob("symbol=*")):
            sym_val = symbol_dir.name.split("=", 1)[1]
            if symbol and sym_val != symbol:
                continue

            for date_dir in sorted(symbol_dir.glob("date=*")):
                date_val      = date_dir.name.split("=", 1)[1]
                partition_file = date_dir / "part.parquet"
                if not partition_file.exists():
                    continue

                # Read row count from Parquet metadata — no data scan needed
                meta = pq.read_metadata(partition_file)
                rows = sum(
                    meta.row_group(i).num_rows for i in range(meta.num_row_groups)
                )
                records.append({
                    "source": src_val,
                    "symbol": sym_val,
                    "date":   date_val,
                    "rows":   rows,
                    "path":   str(partition_file),
                })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["source", "symbol", "date"]).reset_index(drop=True)

    log.info("available partitions found", count=len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_dates(dates: str | list[str] | tuple[str, str]) -> list[str]:
    """
    Normalise the dates argument into a sorted list of date strings.

    Single string  → ["2024-01-15"]
    List           → sorted list as-is
    Tuple (s, e)   → all dates from s to e inclusive
    """
    if isinstance(dates, str):
        return [dates]

    if isinstance(dates, (list, set)):
        return sorted(dates)

    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        return [
            d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, end, freq="D")
        ]

    raise TypeError(
        f"dates must be a string, list of strings, or (start, end) tuple. "
        f"Got: {type(dates)}"
    )