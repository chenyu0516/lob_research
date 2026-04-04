"""
src/ingestion/stage_a.py
------------------------
General Stage A normalizer. Takes any raw CSV DataFrame and a source name,
reads the corresponding section from schema_map.yaml, and returns a
standardized intermediate DataFrame.

This module contains zero source-specific logic. All translation rules
live exclusively in config/schema_map.yaml. To support a new data source,
add a new section to that config file — no code changes needed here.

Stage A responsibilities:
    - Column renaming
    - Timestamp parsing → int64 nanoseconds UTC
    - Side mapping → BID | ASK
    - Price scaling
    - Size resolution from priority column list
    - Preservation of any auxiliary columns declared in the config
      (e.g. new_size, new_price for Coinbase SET events)
    - Optional date injection from filename when the CSV only carries
      time-of-day (controlled via schema_map.yaml filename_parsing block)
    - Sort by timestamp

Stage A does NOT:
    - Enforce the restricted event catalog (ADD | MODIFY | FILL | CANCEL)
    - Compute remaining_size
    - Drop event types
    - Contain any if/else branching on source name

Filename convention for date injection:
    YYYYMMDD_SYMBOL.csv   e.g.  20240115_BTC-USD.csv
    Date format and separator are configurable in schema_map.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCHEMA_MAP_PATH = _PROJECT_ROOT / "config" / "schema_map.yaml"

# Columns always produced by Stage A regardless of source
STAGE_A_COLS = [
    "ts",          # int64 nanoseconds UTC
    "symbol",      # str
    "order_id",    # str
    "side",        # BID | ASK | NaN
    "price",       # float64, scaled
    "size",        # float64, resolved from priority list
    "raw_type",    # str, original source event type code — for Stage B
    "raw_reason",  # str, original reason field if present — for Stage B
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_from_file(
    path: Path | str,
    source: str,
    schema_map_path: Path = _SCHEMA_MAP_PATH,
) -> pd.DataFrame:
    """
    Read a CSV file, parse date and symbol from the filename, and run Stage A.

    Filename must follow the convention: YYYYMMDD_SYMBOL.csv
    The date and symbol separator and date format are read from
    schema_map.yaml under the source's `filename_parsing` block.

    If `filename_parsing.date_injection` is false in the config, the date
    parsing step is skipped and this function behaves identically to calling
    run() directly with the raw DataFrame.

    Parameters
    ----------
    path            : path to the raw CSV file
    source          : must match a top-level key in schema_map.yaml
    schema_map_path : path to schema_map.yaml

    Returns
    -------
    Same as run() — DataFrame with STAGE_A_COLS plus aux_cols.
    """
    path = Path(path)

    with open(schema_map_path) as f:
        cfg: dict = yaml.safe_load(f)

    source_key = source.lower()
    src_cfg    = cfg[source_key]
    fp_cfg     = src_cfg.get("filename_parsing", {})

    date_injection = fp_cfg.get("date_injection", False)
    sep            = fp_cfg.get("filename_sep", "_")
    date_fmt       = fp_cfg.get("date_format", "%Y%m%d")

    # ── Parse date and symbol from filename ───────────────────────────────────
    stem = path.stem                    # e.g. "20240115_BTC-USD"
    parts = stem.split(sep, maxsplit=1) # split on first separator only

    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse filename '{stem}'. "
            f"Expected format: DATE{sep}SYMBOL  (e.g. 20240115{sep}BTC-USD)\n"
            f"Check filename_parsing.filename_sep in schema_map.yaml."
        )

    date_str, symbol = parts[0], parts[1]
    log.info("parsed filename", date=date_str, symbol=symbol, source=source_key)

    # ── Read CSV ──────────────────────────────────────────────────────────────
    raw = pd.read_csv(path, low_memory=False)
    log.info("loaded raw csv", path=str(path), rows=len(raw))

    # ── Inject date into time-only timestamp column ───────────────────────────
    if date_injection:
        ts_col = src_cfg["column_map"]["ts"]
        if ts_col not in raw.columns:
            raise KeyError(
                f"[{source_key}] Timestamp column '{ts_col}' not found in CSV. "
                f"Available columns: {list(raw.columns)}"
            )

        # Validate the date string against the declared format
        try:
            pd.to_datetime(date_str, format=date_fmt)
        except ValueError:
            raise ValueError(
                f"Date string '{date_str}' does not match "
                f"date_format '{date_fmt}' in schema_map.yaml."
            )

        # Combine date prefix with time-of-day string
        # e.g. "20240115" + " " + "09:30:00.123456" → "2024-01-15 09:30:00.123456"
        date_prefix = pd.to_datetime(date_str, format=date_fmt).strftime("%Y-%m-%d")
        raw[ts_col]  = date_prefix + " " + raw[ts_col].astype(str)
        log.info("date injected into timestamp column",
                 ts_col=ts_col, date_prefix=date_prefix)

    return run(raw, source=source_key, schema_map_path=schema_map_path,
               symbol_override=symbol)


def run(
    raw: pd.DataFrame,
    source: str,
    schema_map_path: Path = _SCHEMA_MAP_PATH,
    symbol_override: str | None = None,
) -> pd.DataFrame:
    """
    Apply Stage A translation to a raw CSV DataFrame.

    Parameters
    ----------
    raw             : DataFrame from pd.read_csv()
    source          : must match a top-level key in schema_map.yaml
    schema_map_path : path to schema_map.yaml, defaults to config/schema_map.yaml
    symbol_override : if provided, overrides the symbol column for all rows.
                      run_from_file() passes the filename-parsed symbol here.

    Returns
    -------
    DataFrame with columns defined in STAGE_A_COLS plus any auxiliary columns
    declared under `aux_cols` in the source's schema_map section.
    """
    with open(schema_map_path) as f:
        cfg: dict = yaml.safe_load(f)

    source_key = source.lower()
    if source_key not in cfg:
        raise ValueError(
            f"Source '{source_key}' not found in schema_map.yaml. "
            f"Available sources: {list(cfg.keys())}"
        )

    src_cfg = cfg[source_key]
    out = pd.DataFrame(index=raw.index)

    # ── Timestamp ────────────────────────────────────────────────────────────
    ts_col = _require(raw, src_cfg["column_map"]["ts"], source_key, "timestamp")
    fmt    = src_cfg.get("timestamp_format", "iso8601")

    if fmt == "unix_ns":
        out["ts"] = pd.to_numeric(raw[ts_col], errors="coerce").astype("int64")
    else:
        # Covers both iso8601 and the combined date+time string after injection
        out["ts"] = pd.to_datetime(raw[ts_col], utc=True).astype("int64")

    # ── Symbol ────────────────────────────────────────────────────────────────
    if symbol_override:
        out["symbol"] = symbol_override
    else:
        sym_col = src_cfg["column_map"].get("symbol")
        if sym_col and sym_col in raw.columns:
            out["symbol"] = raw[sym_col].astype(str)
        else:
            raise KeyError(
                f"[{source_key}] Cannot resolve symbol: no 'symbol' column_map "
                f"entry and no symbol_override provided."
            )

    # ── Order ID ──────────────────────────────────────────────────────────────
    oid_col = src_cfg["column_map"].get("order_id", "order_id")
    out["order_id"] = (
        raw[oid_col].astype(str) if oid_col in raw.columns
        else pd.Series("", index=raw.index)
    )

    # ── Side ──────────────────────────────────────────────────────────────────
    side_col = src_cfg["column_map"].get("side", "side")
    side_map: dict = src_cfg.get("side_map", {})
    if side_col in raw.columns:
        out["side"] = raw[side_col].map(side_map)
    else:
        out["side"] = np.nan

    # ── Price ─────────────────────────────────────────────────────────────────
    price_col = src_cfg["column_map"].get("price", "price")
    scale     = float(src_cfg.get("price_scale", 1.0))
    out["price"] = pd.to_numeric(
        raw[price_col] if price_col in raw.columns else np.nan,
        errors="coerce",
    ) * scale

    # ── Size — resolve from priority list ─────────────────────────────────────
    size = pd.Series(np.nan, index=raw.index, dtype=float)
    for col in src_cfg.get("size_col_priority", ["size"]):
        if col in raw.columns:
            size = size.fillna(pd.to_numeric(raw[col], errors="coerce"))
    out["size"] = size

    # ── Raw event type — preserved as-is for Stage B ──────────────────────────
    et_col = src_cfg.get("event_type_col", "type")
    _require(raw, et_col, source_key, "event_type_col")
    out["raw_type"] = raw[et_col].astype(str)

    # ── Raw reason — preserved for Stage B ────────────────────────────────────
    reason_col = src_cfg.get("reason_col", "reason")
    out["raw_reason"] = (
        raw[reason_col].fillna("").astype(str)
        if reason_col in raw.columns
        else pd.Series("", index=raw.index)
    )

    # ── Auxiliary columns — source-specific fields Stage B may need ───────────
    for col in src_cfg.get("aux_cols", []):
        out[col] = (
            pd.to_numeric(raw[col], errors="coerce")
            if col in raw.columns
            else pd.Series(np.nan, index=raw.index, dtype=float)
        )

    # ── Sort and finalize ─────────────────────────────────────────────────────
    out = out.sort_values("ts").reset_index(drop=True)

    log.info(
        "stage A complete",
        source=source_key,
        rows=len(out),
        raw_types=out["raw_type"].value_counts().to_dict(),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require(df: pd.DataFrame, col: str, source: str, field: str) -> str:
    if col not in df.columns:
        raise KeyError(
            f"[{source}] Expected column '{col}' (mapped from '{field}') "
            f"not found in CSV.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update the '{source}' section of config/schema_map.yaml."
        )
    return col