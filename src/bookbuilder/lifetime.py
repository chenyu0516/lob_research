"""
src/bookbuilder/lifetime.py
----------------------------
Builds the order lifetime summary table from the flat event table
produced by Phase 2 (src/ingestion/coinbase.py or databento.py).

Input:  flat event table  — one row per event, all orders all sources
Output: lifetime table    — one row per order, derived summary fields

Lifetime table schema:
    order_id            str / int
    symbol              str
    source              str
    born_ts             int64   nanoseconds UTC — timestamp of ADD event
    born_price          float64
    born_size           float64
    died_ts             int64   nanoseconds UTC — timestamp of terminal event
                                NaN if outcome is OPEN_AT_EOD
    died_price          float64 — price at terminal event
                                NaN if outcome is OPEN_AT_EOD
    outcome             str     FILLED | CANCELLED | OPEN_AT_EOD
    duration_ns         float64 died_ts - born_ts
                                NaN if outcome is OPEN_AT_EOD
    fill_count          int     number of FILL events
    partial_fill_count  int     number of PARTIAL_FILL* reason events
    modify_count        int     number of MODIFY events
    total_filled_size   float64 sum of size across all FILL events
    cancel_size         float64 size at the terminal CANCEL event (0 if filled)
    anomalies           str     pipe-separated anomaly codes, empty string if clean
                                OVERFILL       — total_filled_size > born_size
                                MULTI_ADD      — more than one ADD event seen
                                NEG_REMAINING  — remaining_size went negative
                                                 at any point in the event sequence
                                BAD_SEQ        — event_seq is not monotonically
                                                 increasing (gaps or resets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Terminal event types — an order's life ends at one of these
_TERMINAL_EVENT_TYPES = {"FILL", "CANCEL"}

# Reason codes that indicate a partial fill (any source)
_PARTIAL_FILL_REASONS = {"PARTIAL_FILL", "PARTIAL_FILL_TRADE"}

LIFETIME_COLS = [
    "order_id",
    "session_id",
    "symbol",
    "source",
    "born_ts",
    "born_price",
    "born_size",
    "died_ts",
    "died_price",
    "outcome",
    "duration_ns",
    "fill_count",
    "partial_fill_count",
    "modify_count",
    "total_filled_size",
    "cancel_size",
    "anomalies",
]


def build(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lifetime summary table from a flat event table.

    Parameters
    ----------
    events : DataFrame conforming to EVENT_TABLE_COLS from Phase 2.
             Must contain at minimum: order_id, symbol, source,
             event_type, event_seq, ts, price, size, remaining_size, reason.

    Returns
    -------
    DataFrame with columns defined in LIFETIME_COLS.
    One row per unique order_id.
    """
    if events.empty:
        log.warning("empty event table passed to lifetime builder")
        return pd.DataFrame(columns=LIFETIME_COLS)

    records: list[dict] = []
    skipped = 0

    for (order_id, session_id), group in events.groupby(["order_id", "session_id"], sort=False):
        group = group.sort_values("event_seq")
        anomaly_flags: list[str] = []

        # ── Locate the ADD event ───────────────────────────────────────────────
        add_rows = group[group["event_type"] == "ADD"]
        if add_rows.empty:
            # Orders with no ADD visible in this file window — skip.
            # This is expected for mid-session files where the order was
            # placed before the data window starts.
            log.debug(
                "order has no ADD event in this file — skipping",
                order_id=order_id,
            )
            skipped += 1
            continue

        # MULTI_ADD — more than one ADD seen for the same order_id
        # Indicates duplicate data or an exchange re-using order IDs
        if len(add_rows) > 1:
            anomaly_flags.append("MULTI_ADD")
            log.warning("multiple ADD events for same order_id", order_id=order_id,
                        count=len(add_rows))

        add        = add_rows.iloc[0]
        born_ts    = int(add["ts"])
        born_price = float(add["price"])
        born_size  = float(add["size"])

        # ── Locate the terminal event ─────────────────────────────────────────
        # Use iloc[-1] after sorting by event_seq to get the true closing event.
        # An order can have multiple FILL events (partial fills) before closure —
        # all are of type FILL, so terminal_rows may contain several.
        # iloc[-1] gives the last one, which is the actual order-closing event.
        # iloc[0] would give the first partial fill, which is incorrect.
        terminal_rows = group[group["event_type"].isin(_TERMINAL_EVENT_TYPES)]

        # ── Statistics across full lifetime ───────────────────────────────────
        fill_events        = group[group["event_type"] == "FILL"]
        fill_count         = len(fill_events)
        partial_fill_count = int(fill_events["reason"].isin(_PARTIAL_FILL_REASONS).sum())
        modify_count       = int((group["event_type"] == "MODIFY").sum())
        total_filled_size  = float(fill_events["size"].sum()) if fill_count > 0 else 0.0

        # ── Anomaly: OVERFILL ─────────────────────────────────────────────────
        # total filled volume exceeded the original order size.
        # Caused by data gaps, duplicate match events, or exchange edge cases.
        if total_filled_size > born_size + 1e-9:   # tolerance for float rounding
            anomaly_flags.append("OVERFILL")
            log.warning(
                "overfill detected",
                order_id=order_id,
                born_size=born_size,
                total_filled_size=total_filled_size,
            )

        # ── Anomaly: NEG_REMAINING ────────────────────────────────────────────
        # remaining_size went below zero at some point in the event sequence.
        # Should not happen if subtraction logic is correct — indicates either
        # a data quality issue or a bug in Stage B.
        if "remaining_size" in group.columns:
            if (group["remaining_size"] < -1e-9).any():
                anomaly_flags.append("NEG_REMAINING")
                log.warning(
                    "negative remaining_size detected",
                    order_id=order_id,
                    min_remaining=float(group["remaining_size"].min()),
                )

        # ── Anomaly: BAD_SEQ ──────────────────────────────────────────────────
        # event_seq is not monotonically increasing.
        # Could indicate out-of-order delivery or a Stage B sequencing bug.
        seq = group["event_seq"].tolist()
        if seq != sorted(seq):
            anomaly_flags.append("BAD_SEQ")
            log.warning("non-monotonic event_seq", order_id=order_id, seq=seq)

        # ── Derive outcome and terminal fields ────────────────────────────────
        if terminal_rows.empty:
            outcome     = "OPEN_AT_EOD"
            died_ts     = np.nan
            died_price  = np.nan
            duration_ns = np.nan
            cancel_size = 0.0

        else:
            terminal    = terminal_rows.iloc[-1]
            died_ts     = int(terminal["ts"])
            died_price  = float(terminal["price"])
            duration_ns = float(died_ts - born_ts)

            if terminal["event_type"] == "FILL":
                outcome     = "FILLED"
                cancel_size = 0.0
            else:
                outcome     = "CANCELLED"
                cancel_size = float(terminal["size"])

        records.append({
            "order_id":           order_id,
            "session_id":         session_id,
            "symbol":             group["symbol"].iloc[0],
            "source":             group["source"].iloc[0],
            "born_ts":            born_ts,
            "born_price":         born_price,
            "born_size":          born_size,
            "died_ts":            died_ts,
            "died_price":         died_price,
            "outcome":            outcome,
            "duration_ns":        duration_ns,
            "fill_count":         fill_count,
            "partial_fill_count": partial_fill_count,
            "modify_count":       modify_count,
            "total_filled_size":  total_filled_size,
            "cancel_size":        cancel_size,
            "anomalies":          "|".join(anomaly_flags),  # empty string = clean
        })

    if skipped:
        log.info(
            "orders skipped — no ADD event in file window",
            count=skipped,
            hint="expected for files that start mid-session",
        )

    result = pd.DataFrame(records, columns=LIFETIME_COLS)

    anomalous = result[result["anomalies"] != ""]
    log.info(
        "lifetime build complete",
        total_orders=len(result),
        anomalous_orders=len(anomalous),
        outcome_breakdown=result["outcome"].value_counts().to_dict(),
        anomaly_breakdown=(
            anomalous["anomalies"].value_counts().to_dict()
            if not anomalous.empty else {}
        ),
    )
    return result