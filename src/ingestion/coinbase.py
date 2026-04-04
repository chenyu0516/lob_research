"""
src/ingestion/coinbase.py
--------------------------
Coinbase-specific Stage B: stateful event catalog enforcement,
remaining_size tracking, output of unified flat event table.

Stage A (mechanical column translation) is handled by the general
src/ingestion/stage_a.py module — not here.

Coinbase L3 raw event types and their semantics:
    ADD       new order added to the book
    SUB       volume subtracted from an order (non-trade)
    MATCH     volume subtracted from an order (trade execution)
    SET       new absolute price and/or size assigned to an order
    DELETE    order removed from the book (full or partial)
    SNAPSHOT  full book snapshot — all prior state must be discarded first
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.ingestion import stage_a as _stage_a

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent

EVENT_TABLE_COLS = [
    "order_id",
    "symbol",
    "source",
    "event_type",     # ADD | MODIFY | FILL | CANCEL
    "event_seq",      # monotonic int, per-order sequence number starting at 0
    "ts",             # int64, nanoseconds since Unix epoch UTC
    "price",          # float64
    "size",           # float64 — event size: matched qty, set qty, etc.
    "remaining_size", # float64 — remaining on book after this event
    "reason",         # see constants below
]

# Reason constants
_PARTIAL_FILL    = "PARTIAL_FILL"
_FULL_FILL       = "FULL_FILL"
_PARTIAL_FILL_T  = "PARTIAL_FILL_TRADE"   # MATCH-driven partial fill
_FULL_FILL_T     = "FULL_FILL_TRADE"      # MATCH-driven full fill
_SIZE_CHANGE     = "SIZE_CHANGE"
_PRICE_CHANGE    = "PRICE_CHANGE"
_PARTIAL_DELETE  = "PARTIAL_DELETE"
_CANCELLED       = "CANCELLED"
_SNAPSHOT_RESET  = "SNAPSHOT_RESET"


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — stateful event catalog enforcement
# ─────────────────────────────────────────────────────────────────────────────

def stage_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Stage A output into the unified flat event table.

    Walks events chronologically, maintaining per-order state:
        remaining_size, current_price, event_seq

    Raw type → internal type mapping:
        ADD      → ADD
        SUB      → FILL  (subtraction semantics, non-trade)
        MATCH    → FILL  (subtraction semantics, trade execution)
        SET      → MODIFY (set semantics — may emit two records if both
                           price and size changed)
        DELETE   → MODIFY (PARTIAL_DELETE, size > 0 and order survives)
                   CANCEL (CANCELLED, order fully removed)
                   ignored (size == 0)
        SNAPSHOT → synthetic CANCEL (SNAPSHOT_RESET) for all live orders,
                   then ADD for each order in the snapshot
    """
    records: list[dict] = []

    # { order_id: { "remaining": float, "price": float, "seq": int } }
    order_state: dict[str, dict] = {}

    # Tracks whether the previous row was a SNAPSHOT row.
    # Used to detect the start of a new snapshot sequence and trigger
    # a state reset exactly once per snapshot, not once per snapshot row.
    _in_snapshot = False

    for row in df.itertuples(index=False):
        oid      = row.order_id
        raw_type = row.raw_type

        # ── ADD ───────────────────────────────────────────────────────────────
        # New order arrives on the book. Initialize state.
        if raw_type == "ADD":
            _in_snapshot = False
            initial_size = _float(row.size)
            order_state[oid] = {
                "remaining": initial_size,
                "price":     _float(row.price),
                "seq":       0,
            }
            records.append(_record(
                row=row, order_id=oid,
                event_type="ADD", event_seq=0,
                size=initial_size, remaining_size=initial_size,
                reason=None,
            ))

        # ── SUB ───────────────────────────────────────────────────────────────
        # Volume subtracted from an existing order (non-trade path).
        # size column = amount removed, subtraction semantics.
        elif raw_type == "SUB":
            _in_snapshot = False
            if oid not in order_state:
                log.warning("SUB for untracked order — skipping", order_id=oid)
                continue

            subtracted = _float(row.size)
            order_state[oid]["remaining"] = max(
                order_state[oid]["remaining"] - subtracted, 0.0
            )
            remaining = order_state[oid]["remaining"]
            reason = _FULL_FILL if remaining <= 0.0 else _PARTIAL_FILL

            records.append(_record(
                row=row, order_id=oid,
                event_type="FILL",
                event_seq=_advance_seq(order_state, oid),
                size=subtracted, remaining_size=remaining,
                reason=reason,
            ))

        # ── MATCH ─────────────────────────────────────────────────────────────
        # Volume subtracted due to a confirmed trade execution.
        # Mechanics identical to SUB; reason codes distinguish trade vs non-trade.
        elif raw_type == "MATCH":
            _in_snapshot = False
            if oid not in order_state:
                log.warning("MATCH for untracked order — skipping", order_id=oid)
                continue

            matched = _float(row.size)
            order_state[oid]["remaining"] = max(
                order_state[oid]["remaining"] - matched, 0.0
            )
            remaining = order_state[oid]["remaining"]
            reason = _FULL_FILL_T if remaining <= 0.0 else _PARTIAL_FILL_T

            records.append(_record(
                row=row, order_id=oid,
                event_type="FILL",
                event_seq=_advance_seq(order_state, oid),
                size=matched, remaining_size=remaining,
                reason=reason,
            ))

        # ── SET ───────────────────────────────────────────────────────────────
        # Absolute new price and/or size assigned to the order (set semantics).
        # May emit two MODIFY records if both changed.
        elif raw_type == "SET":
            _in_snapshot = False
            if oid not in order_state:
                log.warning("SET for untracked order — skipping", order_id=oid)
                continue

            new_size  = _float(row.new_size)
            new_price = _float(row.new_price)

            has_size_change  = not np.isnan(new_size)
            has_price_change = not np.isnan(new_price)

            if has_size_change:
                order_state[oid]["remaining"] = new_size
                records.append(_record(
                    row=row, order_id=oid,
                    event_type="MODIFY",
                    event_seq=_advance_seq(order_state, oid),
                    size=new_size, remaining_size=new_size,
                    reason=_SIZE_CHANGE,
                ))

            if has_price_change:
                order_state[oid]["price"] = new_price
                records.append(_record(
                    row=row, order_id=oid,
                    event_type="MODIFY",
                    event_seq=_advance_seq(order_state, oid),
                    size=order_state[oid]["remaining"],
                    remaining_size=order_state[oid]["remaining"],
                    reason=_PRICE_CHANGE,
                    price_override=new_price,
                ))

        # ── DELETE ────────────────────────────────────────────────────────────
        # Order removed from the book.
        # size == 0          → ignore entirely
        # remaining - size > 0 → partial delete, order survives: MODIFY PARTIAL_DELETE
        # remaining - size <= 0 → full delete, order gone: CANCEL CANCELLED
        elif raw_type == "DELETE":
            _in_snapshot = False
            if oid not in order_state:
                log.warning("DELETE for untracked order — skipping", order_id=oid)
                continue

            delete_size = _float(row.size)

            # Ignore zero-size deletes
            if delete_size == 0.0:
                log.debug("zero-size DELETE ignored", order_id=oid)
                continue

            new_remaining = max(order_state[oid]["remaining"] - delete_size, 0.0)

            if new_remaining > 0.0:
                # Partial delete — order still lives
                order_state[oid]["remaining"] = new_remaining
                records.append(_record(
                    row=row, order_id=oid,
                    event_type="MODIFY",
                    event_seq=_advance_seq(order_state, oid),
                    size=delete_size, remaining_size=new_remaining,
                    reason=_PARTIAL_DELETE,
                ))
            else:
                # Full delete — order is gone
                records.append(_record(
                    row=row, order_id=oid,
                    event_type="CANCEL",
                    event_seq=_advance_seq(order_state, oid),
                    size=delete_size, remaining_size=0.0,
                    reason=_CANCELLED,
                ))
                del order_state[oid]

        # ── SNAPSHOT ──────────────────────────────────────────────────────────
        # Full book snapshot. On the first SNAPSHOT row of a sequence:
        #   1. Emit synthetic CANCEL (SNAPSHOT_RESET) for every tracked order
        #   2. Clear order_state entirely
        # Then treat every SNAPSHOT row (including the first) as a fresh ADD.
        elif raw_type == "SNAPSHOT":
            if not _in_snapshot:
                # First row of a new snapshot sequence — reset all live state
                _emit_snapshot_resets(order_state, row, records)
                order_state.clear()
                _in_snapshot = True

            # Treat snapshot row as a fresh ADD
            initial_size = _float(row.size)
            order_state[oid] = {
                "remaining": initial_size,
                "price":     _float(row.price),
                "seq":       0,
            }
            records.append(_record(
                row=row, order_id=oid,
                event_type="ADD", event_seq=0,
                size=initial_size, remaining_size=initial_size,
                reason=None,
            ))

        else:
            log.debug("unrecognized raw event type — skipping",
                      raw_type=raw_type, order_id=oid)

    if order_state:
        log.info(
            "orders still open at end of file (no terminal event seen)",
            count=len(order_state),
        )

    result = pd.DataFrame(records, columns=EVENT_TABLE_COLS)
    log.info(
        "stage B complete",
        rows=len(result),
        event_breakdown=result["event_type"].value_counts().to_dict(),
        reason_breakdown=result["reason"].value_counts().to_dict(),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def process(path: Path | str) -> pd.DataFrame:
    """
    Load a Coinbase L3 CSV and return the processed flat event table.

    Filename must follow the convention: YYYYMMDD_SYMBOL.csv
    Date and symbol are parsed from the filename automatically.
    Date injection into the timestamp column is controlled via
    schema_map.yaml under coinbase.filename_parsing.date_injection.

    Usage:
        from src.ingestion.coinbase import process
        events = process("data/raw/coinbase/20240115_BTC-USD.csv")
    """
    intermediate = _stage_a.run_from_file(path, source="coinbase")
    return stage_b(intermediate)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _advance_seq(order_state: dict, oid: str) -> int:
    order_state[oid]["seq"] += 1
    return order_state[oid]["seq"]


def _float(val) -> float:
    """Safe float conversion — returns NaN for missing values."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _emit_snapshot_resets(
    order_state: dict,
    current_row,
    records: list[dict],
) -> None:
    """
    Emit a synthetic CANCEL (SNAPSHOT_RESET) for every currently tracked order.
    Uses the last known price and remaining_size for each order.
    The timestamp is taken from the triggering SNAPSHOT row.
    """
    for oid, state in order_state.items():
        records.append({
            "order_id":       oid,
            "symbol":         current_row.symbol,
            "source":         "COINBASE",
            "event_type":     "CANCEL",
            "event_seq":      state["seq"] + 1,
            "ts":             current_row.ts,
            "price":          state["price"],
            "size":           state["remaining"],
            "remaining_size": 0.0,
            "reason":         _SNAPSHOT_RESET,
        })


def _record(
    row,
    order_id: str,
    event_type: str,
    event_seq: int,
    size: float,
    remaining_size: float,
    reason: str | None,
    price_override: float | None = None,
) -> dict:
    return {
        "order_id":       order_id,
        "symbol":         row.symbol,
        "source":         "COINBASE",
        "event_type":     event_type,
        "event_seq":      event_seq,
        "ts":             row.ts,
        "price":          price_override if price_override is not None else row.price,
        "size":           size,
        "remaining_size": remaining_size,
        "reason":         reason,
    }