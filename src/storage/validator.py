"""
src/storage/validator.py
------------------------
Data quality checks on the flat event table produced by Phase 2,
run before writing to Parquet.

Checks are non-fatal — violations are logged and recorded in the
returned ValidationReport. The pipeline continues regardless of
violations, but the report makes issues visible immediately.

All order-level checks are fully vectorized — no Python-level loops.
The composite key (order_id, session_id) is used throughout since
order_id integers are reused across sessions.

Checks performed:
    CHECK_UNKNOWN_EVENT   event_type values outside the restricted catalog
    CHECK_UNKNOWN_SIDE    side values other than BID / ASK
    CHECK_NEG_REMAINING   remaining_size < 0 on any event row
    CHECK_DUP_SEQ         duplicate (order_id, session_id, event_seq) triplets
    CHECK_MULTI_ADD       more than one ADD per (order_id, session_id)
    CHECK_NO_ADD          no ADD event for an (order_id, session_id) pair
    CHECK_OVERFILL        total filled size exceeds born size
    CHECK_TS_ORDER        ts not non-decreasing within an order lifecycle
    CHECK_MIXED_SYMBOL    same (order_id, session_id) spans multiple symbols
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

VALID_EVENT_TYPES = {"ADD", "MODIFY", "FILL", "CANCEL"}
VALID_SIDES       = {"BID", "ASK"}
_KEY              = ["order_id", "session_id"]


@dataclass
class ValidationReport:
    """
    Summary of all validation findings for one event table.

    Attributes
    ----------
    total_rows      : total number of rows checked
    total_orders    : total number of unique (order_id, session_id) pairs
    violations      : dict mapping check name → list of offending identifiers
    passed          : True if no violations found across all checks
    """
    total_rows:   int = 0
    total_orders: int = 0
    violations:   dict[str, list] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(len(v) == 0 for v in self.violations.values())

    def summary(self) -> str:
        lines = [
            f"Validation report — {self.total_rows} rows, "
            f"{self.total_orders} orders",
        ]
        if self.passed:
            lines.append("  ALL CHECKS PASSED")
        else:
            for check, items in self.violations.items():
                if items:
                    lines.append(f"  FAIL  {check}: {len(items)} violation(s)")
        return "\n".join(lines)


def run(events: pd.DataFrame) -> ValidationReport:
    """
    Run all quality checks on a flat event table.

    Parameters
    ----------
    events : flat event DataFrame from Phase 2

    Returns
    -------
    ValidationReport — inspect .passed and .violations for details.
    Call .summary() for a human-readable overview.
    """
    report = ValidationReport(
        total_rows=len(events),
        total_orders=0 if events.empty else events.groupby(_KEY).ngroups,
    )

    if events.empty:
        log.warning("empty event table passed to validator")
        return report

    # ── Row-level checks ──────────────────────────────────────────────────────
    # These require no groupby — single vectorized operations across all rows.

    # CHECK_UNKNOWN_EVENT
    bad_types = events.loc[
        ~events["event_type"].isin(VALID_EVENT_TYPES), "event_type"
    ].unique().tolist()
    report.violations["CHECK_UNKNOWN_EVENT"] = bad_types
    if bad_types:
        log.warning("unknown event types found", values=bad_types)

    # CHECK_UNKNOWN_SIDE — NaN allowed (Databento T/F/R rows have no side)
    bad_sides = events.loc[
        events["side"].notna() & ~events["side"].isin(VALID_SIDES), "side"
    ].unique().tolist()
    report.violations["CHECK_UNKNOWN_SIDE"] = bad_sides
    if bad_sides:
        log.warning("unknown side values found", values=bad_sides)

    # CHECK_NEG_REMAINING
    neg_mask   = events["remaining_size"] < -1e-9
    neg_orders = events.loc[neg_mask, _KEY].drop_duplicates().values.tolist()
    report.violations["CHECK_NEG_REMAINING"] = neg_orders
    if neg_orders:
        log.warning(
            "negative remaining_size detected",
            order_count=len(neg_orders),
            min_value=float(events.loc[neg_mask, "remaining_size"].min()),
        )

    # CHECK_DUP_SEQ — duplicate (order_id, session_id, event_seq) triplets
    dup_mask   = events.duplicated(
        subset=["order_id", "session_id", "event_seq"], keep=False
    )
    dup_orders = events.loc[dup_mask, _KEY].drop_duplicates().values.tolist()
    report.violations["CHECK_DUP_SEQ"] = dup_orders
    if dup_orders:
        log.warning(
            "duplicate (order_id, session_id, event_seq) triplets",
            order_count=len(dup_orders),
        )

    # ── Order-level checks — fully vectorized, no Python loop ─────────────────

    add_events  = events[events["event_type"] == "ADD"]
    fill_events = events[events["event_type"] == "FILL"]

    # CHECK_MULTI_ADD
    # Count ADD events per (order_id, session_id) — flag where count > 1
    add_counts = add_events.groupby(_KEY).size()
    multi_add  = add_counts[add_counts > 1].reset_index()[_KEY].values.tolist()
    report.violations["CHECK_MULTI_ADD"] = multi_add
    if multi_add:
        log.warning("orders with multiple ADD events", count=len(multi_add))

    # CHECK_NO_ADD
    # Find (order_id, session_id) pairs that have no ADD event
    all_orders   = events[_KEY].drop_duplicates()
    orders_w_add = add_events[_KEY].drop_duplicates()
    no_add = all_orders.merge(
        orders_w_add, on=_KEY, how="left", indicator=True
    )
    no_add = no_add.loc[no_add["_merge"] == "left_only", _KEY].values.tolist()
    report.violations["CHECK_NO_ADD"] = no_add
    if no_add:
        log.info(
            "orders with no ADD event (mid-session start)",
            count=len(no_add),
            hint="expected for files starting mid-session",
        )

    # CHECK_OVERFILL
    # Born size = size of first ADD per (order_id, session_id)
    # Total filled = sum of FILL sizes per (order_id, session_id)
    # Overfill: total_filled > born_size + tolerance
    born_sizes   = (
        add_events.sort_values("event_seq")
        .groupby(_KEY)["size"]
        .first()
        .rename("born_size")
    )
    fill_totals  = fill_events.groupby(_KEY)["size"].sum().rename("total_filled")
    comparison   = born_sizes.to_frame().join(fill_totals, how="inner")
    overfill_mask = comparison["total_filled"] > comparison["born_size"] + 1e-9
    overfill = comparison[overfill_mask].reset_index()[_KEY].values.tolist()
    report.violations["CHECK_OVERFILL"] = overfill
    if overfill:
        log.warning("overfilled orders detected", count=len(overfill))

    # CHECK_TS_ORDER
    # ts must be non-decreasing within each (order_id, session_id) group.
    # Sort by (order_id, session_id, event_seq), then compute per-group ts diff.
    # Any negative diff indicates an out-of-order timestamp.
    ordered = events.sort_values(_KEY + ["event_seq"])
    ts_diff  = ordered.groupby(_KEY, sort=False)["ts"].diff()
    bad_ts   = ordered.loc[
        ts_diff.notna() & (ts_diff < 0), _KEY
    ].drop_duplicates().values.tolist()
    report.violations["CHECK_TS_ORDER"] = bad_ts
    if bad_ts:
        log.warning("out-of-order timestamps within order", count=len(bad_ts))

    # CHECK_MIXED_SYMBOL
    # Count distinct symbols per (order_id, session_id) — flag where count > 1
    sym_counts   = events.groupby(_KEY)["symbol"].nunique()
    mixed_symbol = sym_counts[sym_counts > 1].reset_index()[_KEY].values.tolist()
    report.violations["CHECK_MIXED_SYMBOL"] = mixed_symbol
    if mixed_symbol:
        log.warning("order_id appears under multiple symbols", count=len(mixed_symbol))

    log.info(
        "validation complete",
        passed=report.passed,
        summary=report.summary(),
    )
    return report