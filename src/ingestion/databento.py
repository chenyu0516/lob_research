"""
src/ingestion/databento.py
--------------------------
Databento-specific Stage B: stateful event catalog enforcement,
remaining_size tracking, output of unified flat event table.

Stage A (mechanical column translation) is handled by the general
src/ingestion/stage_a.py module — not here.

Databento MBO raw action codes and their semantics:
    A   new order inserted into the book
    M   price and/or size changed (set semantics for both)
    C   size subtracted from order (subtraction semantics)
    F   resting order filled — no book state change, carries order_id
    T   aggressing order traded — no book state change, order_id = 0
    R   clear all resting orders (equivalent to Coinbase SNAPSHOT)
    N   no book action — skip

Fill sequence detection:
    Events sharing the same sequence number are processed as a group.
    If a group contains F → it is a fill sequence.
    If a fill sequence also contains T → trade-driven fill (FULL/PARTIAL_FILL_TRADE).
    If a fill sequence has no T → non-trade fill (FULL/PARTIAL_FILL).
    Within a fill sequence, C does the size subtraction and drives remaining_size.
    F provides the resting order_id for the lifetime record.
    T is always skipped (order_id = 0, no lifetime).
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
    "price",          # float64, scaled
    "size",           # float64
    "remaining_size", # float64
    "reason",         # PARTIAL_FILL | FULL_FILL | PARTIAL_FILL_TRADE | FULL_FILL_TRADE
                      # SIZE_CHANGE | PRICE_CHANGE | PARTIAL_DELETE | CANCELLED
                      # SNAPSHOT_RESET | None
]

# Reason constants
_PARTIAL_FILL       = "PARTIAL_FILL"
_FULL_FILL          = "FULL_FILL"
_PARTIAL_FILL_T     = "PARTIAL_FILL_TRADE"
_FULL_FILL_T        = "FULL_FILL_TRADE"
_SIZE_CHANGE        = "SIZE_CHANGE"
_PRICE_CHANGE       = "PRICE_CHANGE"
_PARTIAL_DELETE     = "PARTIAL_DELETE"
_CANCELLED          = "CANCELLED"
_SNAPSHOT_RESET     = "SNAPSHOT_RESET"


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — stateful event catalog enforcement
# ─────────────────────────────────────────────────────────────────────────────

def stage_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Stage A output into the unified flat event table.

    Events are grouped by sequence number before processing. This is necessary
    because distinguishing a trade-driven fill from a non-trade fill requires
    knowing whether a T record exists in the same sequence group as F and C.

    Processing logic per sequence group:
        - Detect fill sequence: F present in group
        - Detect trade fill:    T present in fill sequence
        - A   → ADD, initialize order state
        - M   → MODIFY, set semantics, one record per changed field
        - C   → drives remaining_size update:
                  in fill sequence → FILL event
                  standalone       → CANCEL (full) or MODIFY PARTIAL_DELETE
        - F   → provides order_id for FILL event, no state change
        - T   → skip (order_id = 0)
        - R   → synthetic CANCEL (SNAPSHOT_RESET) for all live orders, clear state
        - N   → skip
    """
    records: list[dict] = []
    session_id: int = 0

    # { order_id: { "remaining": float, "price": float, "side": str, "session_id": int, "seq": int } }
    order_state: dict[int, dict] = {}

    # Group by sequence number, preserving chronological order of groups
    # sort=False keeps original row order within groups
    for _, group in df.groupby("sequence", sort=False):
        actions_in_group = set(group["raw_type"].tolist())
        is_fill_sequence = "F" in actions_in_group
        is_trade_fill    = is_fill_sequence and "T" in actions_in_group

        for row in group.sort_values("ts").itertuples(index=False):
            action = row.raw_type
            oid    = row.order_id

            # ── N / T — skip ──────────────────────────────────────────────────
            if action in ("N", "T"):
                continue

            # ── R — clear all resting orders ──────────────────────────────────
            elif action == "R":
                _emit_snapshot_resets(order_state, row, records)
                order_state.clear()
                session_id += 1

            # ── A — new order on the book ─────────────────────────────────────
            elif action == "A":
                initial_size = _float(row.size)
                order_state[oid] = {
                    "remaining":  initial_size,
                    "price":      _float(row.price),
                    "side":       row.side,
                    "session_id": session_id,
                    "seq":        0,
                }
                records.append(_record(
                    row=row, order_id=oid,
                    session_id=session_id,
                    event_type="ADD", event_seq=0,
                    size=initial_size, remaining_size=initial_size,
                    reason=None,
                ))

            # ── M — modify price and/or size (set semantics) ──────────────────
            # Databento always sends the full new price and size.
            # Compare against stored state to determine what actually changed.
            # Special case: if order not found, treat as ADD (per Databento spec).
            elif action == "M":
                if oid not in order_state:
                    log.warning(
                        "M for unknown order — treating as ADD",
                        order_id=oid,
                    )
                    initial_size = _float(row.size)
                    order_state[oid] = {
                        "remaining":  initial_size,
                        "price":      _float(row.price),
                        "side":       row.side,
                        "session_id": session_id,
                        "seq":        0,
                    }
                    records.append(_record(
                        row=row, order_id=oid,
                        session_id=session_id,
                        event_type="ADD", event_seq=0,
                        size=initial_size, remaining_size=initial_size,
                        reason=None,
                    ))
                    continue

                new_size  = _float(row.size)
                new_price = _float(row.price)
                old_size  = order_state[oid]["remaining"]
                old_price = order_state[oid]["price"]

                has_size_change  = not np.isnan(new_size)  and new_size  != old_size
                has_price_change = not np.isnan(new_price) and new_price != old_price

                if has_size_change:
                    order_state[oid]["remaining"] = new_size
                    records.append(_record(
                        row=row, order_id=oid,
                        session_id=order_state[oid]["session_id"],
                        event_type="MODIFY",
                        event_seq=_advance_seq(order_state, oid),
                        size=new_size, remaining_size=new_size,
                        reason=_SIZE_CHANGE,
                    ))

                if has_price_change:
                    order_state[oid]["price"] = new_price
                    records.append(_record(
                        row=row, order_id=oid,
                        session_id=order_state[oid]["session_id"],
                        event_type="MODIFY",
                        event_seq=_advance_seq(order_state, oid),
                        size=order_state[oid]["remaining"],
                        remaining_size=order_state[oid]["remaining"],
                        reason=_PRICE_CHANGE,
                        price_override=new_price,
                    ))

            # ── F — fill notification ─────────────────────────────────────────
            # F carries the resting order_id but does not change book state.
            # The paired C in this sequence will drive the actual size update.
            # We skip F here — it is handled alongside C below.
            elif action == "F":
                continue

            # ── C — cancel / fill cleanup ─────────────────────────────────────
            elif action == "C":
                if oid not in order_state:
                    log.warning("C for untracked order — skipping", order_id=oid)
                    continue

                subtracted = _float(row.size)
                order_state[oid]["remaining"] = max(
                    order_state[oid]["remaining"] - subtracted, 0.0
                )
                remaining = order_state[oid]["remaining"]

                if is_fill_sequence:
                    # C is removing the filled quantity — emit as FILL
                    if is_trade_fill:
                        reason = _FULL_FILL_T if remaining <= 0.0 else _PARTIAL_FILL_T
                    else:
                        reason = _FULL_FILL if remaining <= 0.0 else _PARTIAL_FILL

                    records.append(_record(
                        row=row, order_id=oid,
                        session_id=order_state[oid]["session_id"],
                        event_type="FILL",
                        event_seq=_advance_seq(order_state, oid),
                        size=subtracted, remaining_size=remaining,
                        reason=reason,
                    ))
                    if remaining <= 0.0:
                        del order_state[oid]

                else:
                    # Standalone cancel — partial or full
                    if remaining <= 0.0:
                        records.append(_record(
                            row=row, order_id=oid,
                            session_id=order_state[oid]["session_id"],
                            event_type="CANCEL",
                            event_seq=_advance_seq(order_state, oid),
                            size=subtracted, remaining_size=0.0,
                            reason=_CANCELLED,
                        ))
                        del order_state[oid]
                    else:
                        records.append(_record(
                            row=row, order_id=oid,
                            session_id=order_state[oid]["session_id"],
                            event_type="MODIFY",
                            event_seq=_advance_seq(order_state, oid),
                            size=subtracted, remaining_size=remaining,
                            reason=_PARTIAL_DELETE,
                        ))

            else:
                log.debug("unrecognized action — skipping",
                          action=action, order_id=oid)

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

def process(path: Path | str, symbol: str | None = None,
            meta_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load a Databento MBO CSV and return the processed flat event table.

    Usage:
        from src.ingestion.databento import process
        events = process("data/raw/databento/AAPL.csv", symbol="AAPL")
    """
    raw = pd.read_csv(path, low_memory=False)
    log.info("loaded raw csv", path=str(path), rows=len(raw))
    intermediate = _stage_a.run(raw, source="databento", symbol_override=symbol)
    return stage_b(intermediate)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _advance_seq(order_state: dict, oid: int) -> int:
    order_state[oid]["seq"] += 1
    return order_state[oid]["seq"]


def _float(val) -> float:
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
    Timestamp is taken from the triggering R row.
    """
    for oid, state in order_state.items():
        records.append({
            "order_id":       oid,
            "session_id":     state["session_id"],
            "symbol":         current_row.symbol,
            "source":         "DATABENTO",
            "side":           state["side"],
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
    order_id: int,
    session_id: int,
    event_type: str,
    event_seq: int,
    size: float,
    remaining_size: float,
    reason: str | None,
    price_override: float | None = None,
) -> dict:
    return {
        "order_id":       order_id,
        "session_id":     session_id,
        "symbol":         row.symbol,
        "source":         "DATABENTO",
        "side":           row.side,
        "event_type":     event_type,
        "event_seq":      event_seq,
        "ts":             row.ts,
        "price":          price_override if price_override is not None else row.price,
        "size":           size,
        "remaining_size": remaining_size,
        "reason":         reason,
    }