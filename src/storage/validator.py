"""
src/storage/validator.py
------------------------
Data quality checks on the flat event table produced by Phase 2,
run before writing to Parquet.

Checks are non-fatal — violations are logged and recorded in the
returned ValidationReport. The pipeline continues regardless of
violations, but the report makes issues visible immediately.

Checks performed:
    CHECK_UNKNOWN_EVENT     event_type values outside the restricted catalog
    CHECK_UNKNOWN_SIDE      side values other than BID / ASK
    CHECK_NEG_REMAINING     remaining_size < 0 on any event row
    CHECK_MULTI_ADD         order_id with more than one ADD event
    CHECK_NO_ADD            order_id with no ADD event (mid-session start)
    CHECK_OVERFILL          sum of FILL sizes exceeds the ADD size for an order
    CHECK_DUP_SEQ           duplicate (order_id, event_seq) pairs
    CHECK_TS_ORDER          ts not monotonically increasing within an order
    CHECK_MIXED_SYMBOL      order_id appears under more than one symbol
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

VALID_EVENT_TYPES = {"ADD", "MODIFY", "FILL", "CANCEL"}
VALID_SIDES       = {"BID", "ASK"}


@dataclass
class ValidationReport:
    """
    Summary of all validation findings for one event table.

    Attributes
    ----------
    total_rows      : total number of rows checked
    total_orders    : total number of unique order_ids checked
    violations      : dict mapping check name → list of offending order_ids
                      (or row indices for row-level checks)
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
        total_orders=events["order_id"].nunique(),
    )

    if events.empty:
        log.warning("empty event table passed to validator")
        return report

    # ── Row-level checks (no groupby needed) ─────────────────────────────────

    # CHECK_UNKNOWN_EVENT — event_type outside restricted catalog
    bad_types = events.loc[
        ~events["event_type"].isin(VALID_EVENT_TYPES), "event_type"
    ].unique().tolist()
    report.violations["CHECK_UNKNOWN_EVENT"] = bad_types
    if bad_types:
        log.warning("unknown event types found", values=bad_types)

    # CHECK_UNKNOWN_SIDE — side values other than BID / ASK
    # NaN is allowed (T, F, R rows in Databento have no side)
    bad_sides = events.loc[
        events["side"].notna() & ~events["side"].isin(VALID_SIDES), "side"
    ].unique().tolist()
    report.violations["CHECK_UNKNOWN_SIDE"] = bad_sides
    if bad_sides:
        log.warning("unknown side values found", values=bad_sides)

    # CHECK_NEG_REMAINING — remaining_size < 0
    neg_mask   = events["remaining_size"] < -1e-9
    neg_orders = events.loc[neg_mask, "order_id"].unique().tolist()
    report.violations["CHECK_NEG_REMAINING"] = neg_orders
    if neg_orders:
        log.warning(
            "negative remaining_size detected",
            order_count=len(neg_orders),
            min_value=float(events.loc[neg_mask, "remaining_size"].min()),
        )

    # CHECK_DUP_SEQ — duplicate (order_id, session_id, event_seq) triplets
    # Using the composite key because order_id is reused across sessions
    dup_mask = events.duplicated(subset=["order_id", "session_id", "event_seq"], keep=False)
    dup_orders = events.loc[dup_mask, "order_id"].unique().tolist()
    report.violations["CHECK_DUP_SEQ"] = dup_orders
    if dup_orders:
        log.warning("duplicate (order_id, session_id, event_seq) triplets", order_count=len(dup_orders))

    # ── Order-level checks (groupby order_id) ────────────────────────────────
    multi_add    : list = []
    no_add       : list = []
    overfill     : list = []
    ts_out_order : list = []
    mixed_symbol : list = []

    for oid, group in events.groupby("order_id", sort=False):
        group = group.sort_values("event_seq")

        add_rows  = group[group["event_type"] == "ADD"]
        fill_rows = group[group["event_type"] == "FILL"]

        # CHECK_MULTI_ADD
        if len(add_rows) > 1:
            multi_add.append(oid)

        # CHECK_NO_ADD
        if add_rows.empty:
            no_add.append(oid)

        # CHECK_OVERFILL
        if not add_rows.empty and not fill_rows.empty:
            born_size        = float(add_rows.iloc[0]["size"])
            total_filled     = float(fill_rows["size"].sum())
            if total_filled > born_size + 1e-9:
                overfill.append(oid)

        # CHECK_TS_ORDER — ts must be non-decreasing within order
        ts_vals = group["ts"].tolist()
        if ts_vals != sorted(ts_vals):
            ts_out_order.append(oid)

        # CHECK_MIXED_SYMBOL — same order_id should not span multiple symbols
        if group["symbol"].nunique() > 1:
            mixed_symbol.append(oid)

    report.violations["CHECK_MULTI_ADD"]    = multi_add
    report.violations["CHECK_NO_ADD"]       = no_add
    report.violations["CHECK_OVERFILL"]     = overfill
    report.violations["CHECK_TS_ORDER"]     = ts_out_order
    report.violations["CHECK_MIXED_SYMBOL"] = mixed_symbol

    if multi_add:
        log.warning("orders with multiple ADD events", count=len(multi_add))
    if no_add:
        log.info(
            "orders with no ADD event (mid-session start)",
            count=len(no_add),
            hint="expected for files starting mid-session",
        )
    if overfill:
        log.warning("overfilled orders detected", count=len(overfill))
    if ts_out_order:
        log.warning("out-of-order timestamps within order", count=len(ts_out_order))
    if mixed_symbol:
        log.warning("order_id appears under multiple symbols", count=len(mixed_symbol))

    log.info(
        "validation complete",
        passed=report.passed,
        summary=report.summary(),
    )
    return report