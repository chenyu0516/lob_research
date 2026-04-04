# LOB Research

An empirical L3 limit order book research system for historical market data. Processes raw order-level CSV data from Coinbase (crypto) and Databento (equities) into a unified flat event table and partitioned Parquet storage, enabling order lifetime analysis, statistical research, and visualization.

---

## Table of contents

1. [Setup](#setup)
2. [Project structure](#project-structure)
3. [Running the pipeline](#running-the-pipeline)
4. [Data sources and file naming](#data-sources-and-file-naming)
5. [Pipeline architecture](#pipeline-architecture)
6. [Internal unified schema](#internal-unified-schema)
7. [Coinbase processing — technical details](#coinbase-processing--technical-details)
8. [Databento processing — technical details](#databento-processing--technical-details)
9. [Order lifetime table](#order-lifetime-table)
10. [Storage and loading](#storage-and-loading)
11. [Partition conflict modes and midnight spillover](#partition-conflict-modes-and-midnight-spillover)
12. [Data validation](#data-validation)
13. [Configuration reference](#configuration-reference)

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for QuestDB)
- uv (recommended)

### Install

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
cd lob_research
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Start QuestDB

```bash
docker run -d \
  --name questdb \
  -p 9000:9000 \
  -p 9009:9009 \
  -v $(pwd)/data/questdb:/var/lib/questdb \
  questdb/questdb
```

QuestDB web console: http://localhost:9000

### Verify setup

```bash
uv run python smoke_test.py
```

---

## Project structure

```
lob_research/
├── config/
│   ├── settings.yaml              # Paths, ports, runtime intervals
│   └── schema_map.yaml            # Column normalization rules for each source
├── data/
│   ├── raw/
│   │   ├── coinbase/              # Raw Coinbase L3 CSV files
│   │   └── databento/             # Raw Databento MBO CSV files
│   ├── processed/                 # Intermediate processed files
│   └── parquet/
│       └── events/                # Partitioned Parquet event tables
│           └── source=COINBASE/
│               └── symbol=BTC-USDT/
│                   └── date=2026-03-23/
│                       └── part.parquet
├── src/
│   ├── ingestion/
│   │   ├── stage_a.py             # General Stage A: mechanical CSV translation
│   │   ├── coinbase.py            # Coinbase Stage B: stateful event processing
│   │   └── databento.py           # Databento Stage B: stateful event processing
│   ├── bookbuilder/
│   │   └── lifetime.py            # Order lifetime summary builder
│   ├── storage/
│   │   ├── writer.py              # Parquet writer with Hive partitioning
│   │   ├── loader.py              # Parquet reader and partition lister
│   │   └── validator.py           # Data quality checks
│   ├── stats/plugins/             # User-defined stat plugins (Phase 6)
│   ├── viz/                       # Dash visualization dashboard (Phase 7)
│   ├── utils/
│   │   └── logger.py              # Structured logging setup (structlog)
│   └── pipeline.py                # End-to-end pipeline entry point
├── tests/
│   └── test_normalizer.py
├── smoke_test.py
└── pyproject.toml
```

---

## Running the pipeline

```bash
# Single Coinbase file (date and symbol parsed from filename)
uv run python src/pipeline.py --source coinbase --file data/raw/coinbase/20260323_BTC-USDT.csv

# Single Databento file (symbol provided explicitly)
uv run python src/pipeline.py --source databento --file data/raw/databento/AAPL.csv --symbol AAPL

# All CSV files in a directory
uv run python src/pipeline.py --source coinbase --dir data/raw/coinbase/
```

From a Jupyter notebook or script:

```python
import sys
sys.path.insert(0, "/path/to/lob_research")

from src.pipeline import run_file
run_file("data/raw/coinbase/20260323_BTC-USDT.csv", source="coinbase")
```

---

## Data sources and file naming

### Coinbase

Raw CSV files must follow the naming convention:

```
YYYYMMDD_SYMBOL.csv
e.g. 20260323_BTC-USDT.csv
```

The date is injected into the timestamp column during processing because Coinbase's CSV export contains time-of-day only (`HH:MM:SS.ffffff`), not full datetime values.

**Midnight correction:** Events near session boundaries can have `time_exchange` showing a late time (e.g. `23:59:xx`) while `time_coinapi` (the receive time) shows an early time of the filename date (e.g. `00:00:xx`). These rows are detected automatically and assigned `filename_date - 1 day` rather than the filename date. Detection thresholds are configurable in `schema_map.yaml` under `filename_parsing.midnight_correction`.

### Databento

Raw CSV files follow the MBO (Market By Order) schema. The `symbol` column is present in the export so no metadata file is required. Timestamps (`ts_event`) are already int64 nanoseconds UTC and require no date injection.

---

## Pipeline architecture

The pipeline is divided into two phases:

```
Raw CSV
   │
   ▼
Stage A  (src/ingestion/stage_a.py)
   │  Mechanical translation driven entirely by schema_map.yaml:
   │  column renaming, timestamp parsing, side mapping, price scaling,
   │  size resolution, date injection, midnight correction.
   │  Output: intermediate DataFrame with raw_type preserved as-is.
   │
   ▼
Stage B  (src/ingestion/coinbase.py or databento.py)
   │  Source-specific stateful processing.
   │  Enforces restricted event catalog: ADD | MODIFY | FILL | CANCEL.
   │  Tracks per-order state (remaining_size, price, session_id).
   │  Output: flat event table.
   │
   ▼
Validator  (src/storage/validator.py)
   │  Non-fatal quality checks. Logs violations, continues regardless.
   │
   ▼
Parquet writer  (src/storage/writer.py)
   │  Hive-style partitioned Parquet:
   │  data/parquet/events/source=X/symbol=Y/date=Z/part.parquet
   │
   ▼
Partitioned Parquet files
```

---

## Internal unified schema

All downstream modules (bookbuilder, storage, stats, viz) work exclusively with this schema. Source differences are fully resolved by Stage A and Stage B.

| Column | Type | Description |
|---|---|---|
| `order_id` | str / int | Exchange-assigned order identifier |
| `session_id` | int | Increments on each SNAPSHOT/CLEAR — composite key with `order_id` |
| `symbol` | str | Instrument symbol e.g. `BTC-USDT`, `AAPL` |
| `source` | str | `COINBASE` or `DATABENTO` |
| `side` | str | `BID` or `ASK` |
| `event_type` | str | `ADD`, `MODIFY`, `FILL`, or `CANCEL` |
| `event_seq` | int | Per-order monotonic sequence counter, starts at 0 |
| `ts` | int64 | Nanoseconds since Unix epoch, UTC |
| `price` | float64 | Price in native currency units |
| `size` | float64 | Event size (matched qty, set qty, deleted qty, etc.) |
| `remaining_size` | float64 | Remaining order size on book after this event |
| `reason` | str | Detailed reason code — see tables below |

### Reason codes

| Reason | Event type | Meaning |
|---|---|---|
| `PARTIAL_FILL` | FILL | Non-trade subtraction, order still live |
| `FULL_FILL` | FILL | Non-trade subtraction, order fully consumed |
| `PARTIAL_FILL_TRADE` | FILL | Trade-driven subtraction, order still live |
| `FULL_FILL_TRADE` | FILL | Trade-driven subtraction, order fully consumed |
| `SIZE_CHANGE` | MODIFY | Order size updated |
| `PRICE_CHANGE` | MODIFY | Order price updated |
| `PARTIAL_DELETE` | MODIFY | Size partially removed, order still live |
| `CANCELLED` | CANCEL | Order removed from book |
| `SNAPSHOT_RESET` | CANCEL | Synthetic cancel emitted before a snapshot clear |

### Why `(order_id, session_id)` as composite key

Coinbase reuses `order_id` integers across sessions. A session is the period between two consecutive SNAPSHOT events. Within a session, `order_id` is unique. The `session_id` counter (starting at 0, incrementing on each SNAPSHOT) disambiguates order lifetimes across sessions, making `(order_id, session_id)` a reliable composite primary key for any single file.

---

## Coinbase processing — technical details

### Raw event types

| Raw type | Book effect | Internal mapping |
|---|---|---|
| `ADD` | New order placed on book | `ADD` |
| `SUB` | Size subtracted (non-trade) | `FILL` |
| `MATCH` | Size subtracted (trade execution) | `FILL` |
| `SET` | New absolute price and/or size | `MODIFY` |
| `DELETE` | Order removed, partial or full | `MODIFY` (partial) or `CANCEL` (full) |
| `SNAPSHOT` | Full book reset | Synthetic `CANCEL` for all live orders, then `ADD` for each snapshot row |

### Size semantics

- **Subtraction events (SUB, MATCH, DELETE):** the `size` column contains the amount being removed from the order. `remaining_size` is computed as `current_remaining - size`. If the result reaches zero, the order is closed.
- **Set events (SET):** the `size` column contains the new absolute size. `remaining_size` is assigned directly (not subtracted).

### Implicit patterns — pre-processing

Before the main event loop, Stage B scans the entire file to detect two implicit event patterns that Coinbase encodes as multi-row sequences:

#### Pattern 1: zero-DELETE + paired ADD (no-op)

A DELETE with `size = 0.0` accompanied by an ADD for the same `order_id` at the exact same nanosecond timestamp means "nothing changed for this order." Both rows are identified in pre-processing and skipped entirely in the main loop.

```
Example:
  order_id=X, ts=T, DELETE, size=0.0   ← no-op marker
  order_id=X, ts=T, ADD,    price=P    ← cancels the delete
  → both rows skipped
```

Standalone zero-DELETEs with no paired ADD (e.g. data quality issues) are also silently ignored.

#### Pattern 2: SUB + paired ADD (implicit reprice)

Coinbase does not always use a `SET` event for price changes. Instead it sometimes encodes a reprice as a full SUB (completely draining the order at the old price) followed immediately by an ADD at the same timestamp with the new price and same size.

```
Example:
  order_id=X, ts=T, SUB, price=70489.20, size=0.1484   ← drains old price level
  order_id=X, ts=T, ADD, price=70490.75, size=0.1484   ← re-adds at new price
  → SUB skipped, ADD converted to MODIFY PRICE_CHANGE
```

Without this detection, the ADD would reinitialize `order_state` and reset `event_seq` to 0 for the same `order_id` in the same session, producing duplicate `(order_id, session_id, event_seq)` triplets in the event table.

Both patterns are detected using a vectorized merge on `(order_id, ts)` before the main loop begins, so the loop itself remains O(1) per row.

### SNAPSHOT handling

On the first row of a SNAPSHOT sequence:
1. Synthetic `CANCEL` events with `reason=SNAPSHOT_RESET` are emitted for every currently tracked order, using their last known price and remaining size.
2. `order_state` is cleared entirely.
3. `session_id` is incremented.

Each SNAPSHOT row is then processed as a fresh `ADD`, rebuilding order state from the snapshot. This ensures every order in the event table has a terminal event and no lifecycle is left dangling.

---

## Databento processing — technical details

### Raw action codes

| Action | Book effect | Internal mapping |
|---|---|---|
| `A` | New order on book | `ADD` |
| `M` | Price and/or size changed (set semantics) | `MODIFY` |
| `C` | Size subtracted (subtraction semantics) | `FILL` (in fill sequence) or `CANCEL` / `MODIFY PARTIAL_DELETE` |
| `F` | Resting order filled notification | Provides `order_id` context — no state change |
| `T` | Aggressing order traded | Skipped (`order_id = 0`, no lifetime) |
| `R` | Clear all resting orders | Synthetic `CANCEL` for all live orders, then clear |
| `N` | No book action | Skipped |

### Sequence-group processing

Unlike Coinbase where rows can be processed individually, Databento requires grouping rows by `sequence` number before processing. The context of the full group determines how individual rows are handled:

- If a group contains an `F` action → it is a **fill sequence**
- If a fill sequence also contains a `T` action → it is a **trade-driven fill**

Within a fill sequence, `C` does the actual size subtraction and drives `remaining_size`. `F` provides the `order_id` for the lifetime record but makes no state changes. `T` is always skipped since its `order_id` is 0.

### Modify fallback to ADD

If a `M` (Modify) arrives for an `order_id` not in `order_state`, it is treated as an `ADD`. This is consistent with Databento's own reference implementation and handles orders established before the data window starts.

### R (Clear) handling

Identical to Coinbase SNAPSHOT: synthetic `CANCEL` with `reason=SNAPSHOT_RESET` for all live orders, state cleared, `session_id` incremented.

---

## Order lifetime table

Built on demand from the event table using `src/bookbuilder/lifetime.py`. Not stored permanently — computed when needed for statistical analysis.

```python
from src.bookbuilder.lifetime import build
from src.storage.loader import load_events

events   = load_events("COINBASE", "BTC-USDT", "2026-03-23")
lifetime = build(events)
```

### Lifetime table schema

| Column | Description |
|---|---|
| `order_id` | Exchange order identifier |
| `session_id` | Session within file |
| `symbol` | Instrument |
| `source` | Data source |
| `born_ts` | Timestamp of ADD event (ns UTC) |
| `born_price` | Price at ADD |
| `born_size` | Size at ADD |
| `died_ts` | Timestamp of terminal event (NaN if `OPEN_AT_EOD`) |
| `died_price` | Price at terminal event (NaN if `OPEN_AT_EOD`) |
| `outcome` | `FILLED`, `CANCELLED`, or `OPEN_AT_EOD` |
| `duration_ns` | `died_ts - born_ts` (NaN if `OPEN_AT_EOD`) |
| `fill_count` | Number of FILL events |
| `partial_fill_count` | Number of PARTIAL_FILL* reason events |
| `modify_count` | Number of MODIFY events |
| `total_filled_size` | Sum of size across all FILL events |
| `cancel_size` | Size at terminal CANCEL event |
| `anomalies` | Pipe-separated anomaly codes, empty if clean |

### Anomaly codes

| Code | Meaning |
|---|---|
| `OVERFILL` | `total_filled_size > born_size` |
| `MULTI_ADD` | More than one ADD event for the same order |
| `NEG_REMAINING` | `remaining_size` went below zero at any point |
| `BAD_SEQ` | `event_seq` is not monotonically increasing |

---

## Storage and loading

### Writing

```python
from src.storage.writer import StorageWriter, ConflictMode

writer = StorageWriter()
writer.write_events(events_df)                                    # MERGE by default
writer.write_events(events_df, conflict=ConflictMode.OVERWRITE)   # replace partition
writer.write_events(events_df, conflict=ConflictMode.ERROR)       # raise if exists
```

### Loading

```python
from src.storage.loader import load_events, list_available

# Single date
events = load_events("COINBASE", "BTC-USDT", "2026-03-23")

# List of dates
events = load_events("COINBASE", "BTC-USDT", ["2026-03-23", "2026-03-24"])

# Date range (inclusive)
events = load_events("COINBASE", "BTC-USDT", ("2026-03-01", "2026-03-31"))

# See what's available
inventory = list_available()
inventory = list_available(source="COINBASE", symbol="BTC-USDT")
```

### Partition scheme

```
data/parquet/events/
└── source=COINBASE/
    └── symbol=BTC-USDT/
        └── date=2026-03-23/
            └── part.parquet
```

Hive-style partitioning allows PyArrow, DuckDB, pandas, and most ML frameworks to read specific partitions without scanning the full dataset.

---

## Partition conflict modes and midnight spillover

### The midnight spillover problem

Coinbase CSV exports contain time-of-day only in the timestamp column. Stage A injects the date from the filename, but events that occurred just before midnight (e.g. `time_exchange = 23:59:xx`) while the receive time shows an early time of the filename date (`time_coinapi = 00:00:xx`) are assigned `filename_date - 1 day` via the midnight correction. This means today's file will produce rows that belong to yesterday's Parquet partition.

If the writer simply overwrote yesterday's partition, all of yesterday's original data would be replaced with only the small handful of midnight-spillover rows from today. The MERGE mode exists to handle this correctly.

### Conflict modes

| Mode | Behaviour | When to use |
|---|---|---|
| `MERGE` (default) | Read existing partition, concatenate new rows, deduplicate on `(order_id, session_id, event_seq)`, rewrite | Normal daily processing — handles midnight spillover safely |
| `OVERWRITE` | Replace the existing partition entirely | Reprocessing a file after a bug fix |
| `ERROR` | Raise `FileExistsError` if partition exists | Automated pipelines where double-writes should be caught explicitly |

### Reprocessing safely with OVERWRITE

If you use `OVERWRITE` to reprocess a file, any midnight-spillover rows from a subsequent day's file that were previously merged into the partition will be lost. The writer logs a warning when this happens.

**Example scenario:** You reprocess `20260322_BTC-USDT.csv` with OVERWRITE after fixing a bug. The `date=2026-03-22` partition is replaced. But `20260323_BTC-USDT.csv` previously contributed midnight-spillover rows to that partition — those are now gone.

**The fix:** after reprocessing the original file, also reprocess the subsequent file with MERGE:

```bash
# Reprocess the buggy file
uv run python src/pipeline.py --source coinbase --file data/raw/coinbase/20260322_BTC-USDT.csv

# Re-merge spillover rows from the next day's file
uv run python src/pipeline.py --source coinbase --file data/raw/coinbase/20260323_BTC-USDT.csv
```

### Processing order matters

When processing a full directory, files are processed in alphabetical order. Since filenames start with `YYYYMMDD`, alphabetical and chronological order are the same. This means yesterday's full partition is always written before today's spillover rows arrive to be merged in — the correct order for MERGE to work properly.

---

## Data validation

Validation runs automatically inside the pipeline after Stage B, before writing to Parquet. It is non-fatal — violations are logged and the file is written regardless.

| Check | Description |
|---|---|
| `CHECK_UNKNOWN_EVENT` | `event_type` values outside `ADD\|MODIFY\|FILL\|CANCEL` |
| `CHECK_UNKNOWN_SIDE` | `side` values other than `BID\|ASK` |
| `CHECK_NEG_REMAINING` | Any row with `remaining_size < 0` |
| `CHECK_DUP_SEQ` | Duplicate `(order_id, session_id, event_seq)` triplets |
| `CHECK_MULTI_ADD` | More than one ADD per `(order_id, session_id)` |
| `CHECK_NO_ADD` | Order with no ADD event (expected for mid-session files) |
| `CHECK_OVERFILL` | Total filled size exceeds born size |
| `CHECK_TS_ORDER` | Timestamps not monotonically increasing within an order |
| `CHECK_MIXED_SYMBOL` | Same `order_id` appears under multiple symbols |

To inspect validation results manually:

```python
from src.storage.validator import run as validate

report = validate(events_df)
print(report.summary())
print(report.violations["CHECK_OVERFILL"])   # list of offending order_ids
```

---

## Configuration reference

### `config/settings.yaml`

Controls paths, QuestDB connection, and runtime intervals. Edit before first run to match your environment.

### `config/schema_map.yaml`

Maps raw CSV column names to the internal schema for each source. The convention is `internal_name: raw_column_name`. If your CSV export has different column names, update the `column_map` section for the relevant source. Always verify your actual column names first:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/raw/coinbase/YOUR_FILE.csv', nrows=2).columns.tolist())"
```

Key fields to verify in the Coinbase section: `entry_px` (price column), `entry_size` (size column), `update_type` (event type column), `is_buy` (side column).

Key field to verify in the Databento section: `price_scale` (default `1e-9` — confirm this matches your publisher).