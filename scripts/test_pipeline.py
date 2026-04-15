"""
scripts/test_pipeline.py
------------------------
Comprehensive pytest suite for the LOB research pipeline.

All tests are self-contained — no raw data files are required.
Synthetic DataFrames and temp files are constructed inline.

Coverage:
    Stage A   — column renaming, side mapping, timestamp parsing, price scale,
                size priority, date injection, midnight correction
    Coinbase  — ADD / SUB / MATCH / SET / DELETE / SNAPSHOT, reprice pairs,
                zero-DELETE no-ops, event_seq ordering
    Databento — A / M / C / F / T / R / N, fill sequences, trade fills,
                sequence-NaN handling, session_id + side in output
    Lifetime  — FILLED / CANCELLED / OPEN_AT_EOD outcomes, all anomaly codes
    Validator — all nine quality checks
    Writer    — MERGE / OVERWRITE / ERROR conflict modes, multi-date split
    Loader    — single date, list of dates, date range, missing partition error
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import pandas as pd
import pytest
import yaml

from src.ingestion import stage_a
from src.ingestion.coinbase import stage_b as cb_stage_b
from src.ingestion.databento import stage_b as db_stage_b
from src.bookbuilder.lifetime import build as build_lifetime
from src.storage.validator import run as validate
from src.storage.writer import StorageWriter, ConflictMode
from src.storage.loader import load_events, list_available

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
_SCHEMA_MAP   = _PROJECT_ROOT / "config" / "schema_map.yaml"

# Anchor timestamp: 2024-03-23 00:00:00 UTC expressed in nanoseconds
_TS_BASE = 1_711_152_000_000_000_000


def _ns(ms_offset: int = 0) -> int:
    """Return nanosecond timestamp anchored to _TS_BASE + ms_offset milliseconds."""
    return _TS_BASE + ms_offset * 1_000_000


# ── Synthetic DataFrame builders ─────────────────────────────────────────────

def _cb_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a minimal Stage-A output DataFrame suitable for Coinbase stage_b.
    Each dict in rows overrides the defaults for that row.
    """
    defaults = dict(
        ts=_ns(0), symbol="BTC-USDT", order_id="100",
        side="BID", price=100.0, size=1.0,
        raw_type="ADD", new_size=np.nan, new_price=np.nan,
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _db_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a minimal Stage-A output DataFrame suitable for Databento stage_b.
    sequence is int here; stage_b will fillna+cast regardless.
    """
    defaults = dict(
        ts=_ns(0), symbol="AAPL", order_id=1,
        side="BID", price=150.0, size=100.0,
        raw_type="A", sequence=1, flags=0,
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _event_df(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic unified event table for validator / lifetime tests."""
    defaults = dict(
        order_id="1", session_id=0, symbol="BTC-USDT", source="COINBASE",
        side="BID", event_type="ADD", event_seq=0,
        ts=_ns(0), price=100.0, size=1.0, remaining_size=1.0, reason=None,
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ── Writer / Loader fixture ───────────────────────────────────────────────────

@pytest.fixture
def tmp_settings(tmp_path):
    """
    Create a minimal settings.yaml that points the Parquet root at a temp dir.
    Because pathlib treats an absolute right-hand path as the result of /,
    setting parquet_dir to an absolute string bypasses _PROJECT_ROOT in writer.py.

    Returns (settings_path, parquet_events_root).
    """
    parquet_root = tmp_path / "parquet"
    parquet_root.mkdir()
    cfg = {"data": {"parquet_dir": str(parquet_root)}}
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.dump(cfg))
    return settings_path, parquet_root / "events"


# ─────────────────────────────────────────────────────────────────────────────
# Stage A tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStageA:

    def _coinbase_raw(self, rows: list[dict]) -> pd.DataFrame:
        """Minimal Coinbase raw CSV frame (column names from schema_map.yaml)."""
        defaults = dict(
            time_exchange="2024-03-23 09:30:00.000000",
            order_id="100", is_buy=1,
            entry_px=100.0, entry_sx=1.0,
            update_type="ADD",
            new_size=np.nan, new_price=np.nan,
        )
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def _databento_raw(self, rows: list[dict]) -> pd.DataFrame:
        """Minimal Databento raw CSV frame."""
        defaults = dict(
            ts_event=_ns(0), order_id=1,
            side="B", price=150_000_000_000, size=100,
            action="A", sequence=1, flags=0, symbol="AAPL",
        )
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def test_coinbase_side_integer_columns_map_correctly(self):
        """is_buy int 1/0 → BID/ASK via astype(str) + string-keyed map."""
        raw = self._coinbase_raw([
            {"is_buy": 1}, {"is_buy": 0},
        ])
        out = stage_a.run(raw, source="coinbase", schema_map_path=_SCHEMA_MAP,
                          symbol_override="BTC-USDT")
        assert out["side"].tolist() == ["BID", "ASK"]

    def test_databento_price_scale_applied(self):
        """Databento price_scale=1e-9 converts raw fixed-point to float."""
        raw = self._databento_raw([{"price": 150_000_000_000}])  # 150.0
        out = stage_a.run(raw, source="databento", schema_map_path=_SCHEMA_MAP,
                          symbol_override="AAPL")
        assert abs(out["price"].iloc[0] - 150.0) < 1e-6

    def test_databento_unix_ns_timestamp_passthrough(self):
        """unix_ns timestamps are preserved as-is (no parsing)."""
        raw = self._databento_raw([{"ts_event": _ns(500)}])
        out = stage_a.run(raw, source="databento", schema_map_path=_SCHEMA_MAP,
                          symbol_override="AAPL")
        assert out["ts"].iloc[0] == _ns(500)

    def test_output_sorted_by_ts(self):
        """Stage A output must be sorted by ts regardless of input order."""
        raw = self._databento_raw([
            {"ts_event": _ns(200), "sequence": 2},
            {"ts_event": _ns(100), "sequence": 1},
        ])
        out = stage_a.run(raw, source="databento", schema_map_path=_SCHEMA_MAP,
                          symbol_override="AAPL")
        assert out["ts"].tolist() == sorted(out["ts"].tolist())

    def test_size_col_priority_first_non_null_wins(self):
        """size_col_priority picks the first non-null column value."""
        raw = self._coinbase_raw([{"entry_sx": 3.5}])
        out = stage_a.run(raw, source="coinbase", schema_map_path=_SCHEMA_MAP,
                          symbol_override="BTC-USDT")
        assert out["size"].iloc[0] == 3.5

    def test_symbol_override_applied(self):
        """symbol_override is used instead of reading from a CSV column."""
        raw = self._coinbase_raw([{}])
        out = stage_a.run(raw, source="coinbase", schema_map_path=_SCHEMA_MAP,
                          symbol_override="ETH-USDT")
        assert out["symbol"].iloc[0] == "ETH-USDT"

    def test_databento_side_mapping(self):
        """Databento side codes B→BID, A→ASK, N→NaN."""
        raw = self._databento_raw([
            {"side": "B"}, {"side": "A"}, {"side": "N"},
        ])
        out = stage_a.run(raw, source="databento", schema_map_path=_SCHEMA_MAP,
                          symbol_override="AAPL")
        sides = out.sort_values("ts")["side"].tolist()
        assert sides[0] == "BID"
        assert sides[1] == "ASK"
        assert pd.isna(sides[2])

    def test_date_injection_and_midnight_correction(self, tmp_path):
        """
        run_from_file: rows with late exchange time + early receive time are
        assigned filename_date - 1; all other rows get filename_date.
        """
        # File named 20240116 → filename_date = 2024-01-16
        # Row 1: exchange 23:59, coinapi 00:00 → should land on 2024-01-15
        # Row 2: exchange 09:30, coinapi 09:30 → should land on 2024-01-16
        rows = [
            "time_exchange;time_coinapi;order_id;is_buy;entry_px;entry_sx;update_type;new_size;new_price",
            "23:59:00.000000;00:00:01.000000;100;1;100.0;1.0;ADD;;",
            "09:30:00.000000;09:30:01.000000;101;0;100.0;1.0;ADD;;",
        ]
        csv_path = tmp_path / "20240116_BTC-USDT.csv"
        csv_path.write_text("\n".join(rows))

        out = stage_a.run_from_file(csv_path, source="coinbase",
                                    schema_map_path=_SCHEMA_MAP)

        dates = pd.to_datetime(out["ts"], unit="ns", utc=True).dt.date.astype(str).tolist()
        assert "2024-01-15" in dates, "Midnight-correction row missing"
        assert "2024-01-16" in dates, "Normal row missing"


# ─────────────────────────────────────────────────────────────────────────────
# Coinbase Stage B tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCoinbaseStageB:

    def test_add_initialises_state_and_produces_add_event(self):
        df  = _cb_df([{"raw_type": "ADD", "order_id": "1", "size": 5.0, "ts": _ns(0)}])
        out = cb_stage_b(df)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["event_type"]     == "ADD"
        assert row["event_seq"]      == 0
        assert row["remaining_size"] == 5.0
        assert row["size"]           == 5.0
        assert row["reason"]         is None or pd.isna(row["reason"])

    def test_sub_partial_emits_fill_partial(self):
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "SUB", "order_id": "1", "size": 2.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert fill["reason"]         == "PARTIAL_FILL"
        assert fill["remaining_size"] == 3.0
        assert fill["size"]           == 2.0

    def test_sub_full_emits_fill_full(self):
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "size": 3.0, "ts": _ns(0)},
            {"raw_type": "SUB", "order_id": "1", "size": 3.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert fill["reason"]         == "FULL_FILL"
        assert fill["remaining_size"] == 0.0

    def test_match_partial_emits_fill_trade(self):
        df = _cb_df([
            {"raw_type": "ADD",   "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "MATCH", "order_id": "1", "size": 2.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert fill["reason"] == "PARTIAL_FILL_TRADE"

    def test_match_full_emits_full_fill_trade(self):
        df = _cb_df([
            {"raw_type": "ADD",   "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "MATCH", "order_id": "1", "size": 5.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert fill["reason"]         == "FULL_FILL_TRADE"
        assert fill["remaining_size"] == 0.0

    def test_set_size_change(self):
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "SET", "order_id": "1", "size": 0.0, "ts": _ns(1),
             "new_size": 3.0, "new_price": np.nan},
        ])
        out = cb_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"]         == "SIZE_CHANGE"
        assert mod["remaining_size"] == 3.0
        assert mod["size"]           == 3.0

    def test_set_price_change(self):
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "price": 100.0, "size": 5.0, "ts": _ns(0)},
            {"raw_type": "SET", "order_id": "1", "size": 0.0,   "ts": _ns(1),
             "new_size": np.nan, "new_price": 105.0},
        ])
        out = cb_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"] == "PRICE_CHANGE"
        assert mod["price"]  == 105.0

    def test_set_both_changes_emits_two_modify_records(self):
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "price": 100.0, "size": 5.0, "ts": _ns(0)},
            {"raw_type": "SET", "order_id": "1", "size": 0.0,    "ts": _ns(1),
             "new_size": 4.0, "new_price": 102.0},
        ])
        out = cb_stage_b(df)
        mods = out[out["event_type"] == "MODIFY"]
        assert len(mods) == 2
        reasons = set(mods["reason"].tolist())
        assert reasons == {"SIZE_CHANGE", "PRICE_CHANGE"}

    def test_delete_partial_emits_modify_partial_delete(self):
        df = _cb_df([
            {"raw_type": "ADD",    "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "DELETE", "order_id": "1", "size": 2.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"]         == "PARTIAL_DELETE"
        assert mod["remaining_size"] == 3.0

    def test_delete_full_emits_cancel_and_removes_from_state(self):
        df = _cb_df([
            {"raw_type": "ADD",    "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "DELETE", "order_id": "1", "size": 5.0, "ts": _ns(1)},
            # A subsequent SUB for the same order should be skipped (not in state)
            {"raw_type": "SUB",    "order_id": "1", "size": 1.0, "ts": _ns(2)},
        ])
        out = cb_stage_b(df)
        cancels = out[out["event_type"] == "CANCEL"]
        assert len(cancels) == 1
        assert cancels.iloc[0]["reason"]         == "CANCELLED"
        assert cancels.iloc[0]["remaining_size"] == 0.0
        # The orphan SUB produces no event
        assert len(out[out["event_type"] == "FILL"]) == 0

    def test_zero_delete_without_paired_add_is_ignored(self):
        df = _cb_df([
            {"raw_type": "ADD",    "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "DELETE", "order_id": "1", "size": 0.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        # Zero DELETE with no paired ADD → silently ignored
        assert len(out) == 1
        assert out.iloc[0]["event_type"] == "ADD"

    def test_zero_delete_add_noop_pair_both_skipped(self):
        """
        A zero-size DELETE + ADD at the exact same (order_id, ts) is a no-op.
        Both rows are skipped; the original order state is unchanged.
        """
        df = _cb_df([
            {"raw_type": "ADD",    "order_id": "1", "size": 5.0, "ts": _ns(0)},
            # no-op pair
            {"raw_type": "DELETE", "order_id": "1", "size": 0.0, "ts": _ns(1)},
            {"raw_type": "ADD",    "order_id": "1", "size": 5.0, "ts": _ns(1)},
            # real event after the no-op
            {"raw_type": "SUB",    "order_id": "1", "size": 1.0, "ts": _ns(2)},
        ])
        out = cb_stage_b(df)
        # Only: original ADD + FILL from the SUB
        assert len(out) == 2
        assert out.iloc[0]["event_type"] == "ADD"
        assert out.iloc[1]["event_type"] == "FILL"

    def test_reprice_pair_detected_as_modify_price_change(self):
        """
        SUB (full drain, size=5) + ADD (same size=5, new price) at the same
        nanosecond → SUB skipped, ADD converted to MODIFY PRICE_CHANGE.
        Fix 2: sizes must match.
        Fix 4: remaining_size taken from ADD's row.size, not stale state.
        """
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "price": 100.0, "size": 5.0, "ts": _ns(0)},
            # reprice pair at ts=1
            {"raw_type": "SUB", "order_id": "1", "price": 100.0, "size": 5.0, "ts": _ns(1)},
            {"raw_type": "ADD", "order_id": "1", "price": 105.0, "size": 5.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        # Should have: ADD (ts=0) + MODIFY PRICE_CHANGE (ts=1) — no FILL
        assert "FILL" not in out["event_type"].values
        mods = out[out["event_type"] == "MODIFY"]
        assert len(mods) == 1
        mod = mods.iloc[0]
        assert mod["reason"]         == "PRICE_CHANGE"
        assert mod["price"]          == 105.0
        assert mod["remaining_size"] == 5.0  # taken from ADD row.size (Fix 4)
        assert mod["size"]           == 5.0

    def test_partial_sub_with_different_add_size_is_not_reprice(self):
        """
        SUB size=3 + ADD size=5 at same (order_id, ts): sizes differ →
        NOT a reprice. The SUB is processed as a real partial fill.
        """
        df = _cb_df([
            {"raw_type": "ADD", "order_id": "1", "price": 100.0, "size": 5.0, "ts": _ns(0)},
            # SUB and ADD with different sizes at the same ts → not a reprice
            {"raw_type": "SUB", "order_id": "1", "price": 100.0, "size": 3.0, "ts": _ns(1)},
            {"raw_type": "ADD", "order_id": "1", "price": 105.0, "size": 5.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        # The SUB must produce a FILL (remaining 5-3=2)
        fills = out[out["event_type"] == "FILL"]
        assert len(fills) == 1
        assert fills.iloc[0]["reason"]         == "PARTIAL_FILL"
        assert fills.iloc[0]["remaining_size"] == 2.0

    def test_snapshot_emits_cancel_for_live_orders_and_increments_session(self):
        """
        First SNAPSHOT row: CANCEL (SNAPSHOT_RESET) emitted for every live order,
        order_state cleared, session_id incremented.
        """
        df = _cb_df([
            {"raw_type": "ADD",      "order_id": "1", "size": 5.0,  "ts": _ns(0)},
            {"raw_type": "ADD",      "order_id": "2", "size": 10.0, "ts": _ns(1)},
            {"raw_type": "SNAPSHOT", "order_id": "3", "size": 7.0,  "ts": _ns(2)},
        ])
        out = cb_stage_b(df)

        resets = out[out["reason"] == "SNAPSHOT_RESET"]
        assert len(resets) == 2, "Expected one SNAPSHOT_RESET per live order"
        reset_oids = set(resets["order_id"].tolist())
        assert reset_oids == {"1", "2"}
        # Resets must have remaining_size=0
        assert (resets["remaining_size"] == 0.0).all()

        # The snapshot row itself is added as a fresh ADD in the new session
        snap_add = out[(out["order_id"] == "3") & (out["event_type"] == "ADD")]
        assert len(snap_add) == 1
        assert snap_add.iloc[0]["session_id"] == 1

    def test_snapshot_multiple_rows_only_reset_once(self):
        """Multiple consecutive SNAPSHOT rows should not trigger repeated resets."""
        df = _cb_df([
            {"raw_type": "ADD",      "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "SNAPSHOT", "order_id": "2", "size": 3.0, "ts": _ns(1)},
            {"raw_type": "SNAPSHOT", "order_id": "3", "size": 4.0, "ts": _ns(1)},
        ])
        out = cb_stage_b(df)
        resets = out[out["reason"] == "SNAPSHOT_RESET"]
        # Only order "1" should get a reset; orders 2 and 3 are fresh ADDs
        assert len(resets) == 1
        assert resets.iloc[0]["order_id"] == "1"

    def test_event_seq_starts_at_zero_and_increments(self):
        """event_seq for an order: ADD=0, FILL=1, FILL=2, …"""
        df = _cb_df([
            {"raw_type": "ADD",   "order_id": "1", "size": 5.0, "ts": _ns(0)},
            {"raw_type": "MATCH", "order_id": "1", "size": 2.0, "ts": _ns(1)},
            {"raw_type": "MATCH", "order_id": "1", "size": 2.0, "ts": _ns(2)},
        ])
        out = cb_stage_b(df)
        seqs = out["event_seq"].tolist()
        assert seqs == [0, 1, 2]

    def test_session_id_in_all_output_rows(self):
        """session_id column must be present and populated."""
        df  = _cb_df([{"raw_type": "ADD", "order_id": "1", "size": 1.0}])
        out = cb_stage_b(df)
        assert "session_id" in out.columns
        assert out["session_id"].notna().all()

    def test_side_in_all_output_rows(self):
        """side column must be present and contain BID or ASK."""
        df  = _cb_df([{"raw_type": "ADD", "order_id": "1", "side": "ASK"}])
        out = cb_stage_b(df)
        assert "side" in out.columns
        assert out["side"].iloc[0] == "ASK"


# ─────────────────────────────────────────────────────────────────────────────
# Databento Stage B tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabentoStageB:

    def test_a_action_produces_add(self):
        df  = _db_df([{"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0)}])
        out = db_stage_b(df)
        assert len(out) == 1
        assert out.iloc[0]["event_type"]     == "ADD"
        assert out.iloc[0]["remaining_size"] == 100.0
        assert out.iloc[0]["event_seq"]      == 0

    def test_m_size_change(self):
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "price": 150.0, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "M", "order_id": 1, "price": 150.0, "size":  80, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"]         == "SIZE_CHANGE"
        assert mod["remaining_size"] == 80.0

    def test_m_price_change(self):
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "price": 150.0, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "M", "order_id": 1, "price": 155.0, "size": 100, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"] == "PRICE_CHANGE"
        assert mod["price"]  == 155.0

    def test_m_unknown_order_treated_as_add(self):
        """M for an order not in state → ADD (Databento reference behaviour)."""
        df  = _db_df([{"raw_type": "M", "order_id": 99, "price": 150.0, "size": 50}])
        out = db_stage_b(df)
        assert out.iloc[0]["event_type"] == "ADD"

    def test_standalone_c_full_produces_cancel(self):
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "C", "order_id": 1, "size": 100, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        cancel = out[out["event_type"] == "CANCEL"].iloc[0]
        assert cancel["reason"]         == "CANCELLED"
        assert cancel["remaining_size"] == 0.0

    def test_standalone_c_partial_produces_modify_partial_delete(self):
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "C", "order_id": 1, "size":  40, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        mod = out[out["event_type"] == "MODIFY"].iloc[0]
        assert mod["reason"]         == "PARTIAL_DELETE"
        assert mod["remaining_size"] == 60.0

    def test_fill_sequence_non_trade(self):
        """F + C in the same sequence → FILL with non-trade reason."""
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "F", "order_id": 1, "size": 100, "ts": _ns(1), "sequence": 2},
            {"raw_type": "C", "order_id": 1, "size": 100, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert fill["reason"] in ("FULL_FILL", "PARTIAL_FILL")
        assert "TRADE" not in fill["reason"]

    def test_fill_sequence_trade(self):
        """T + F + C in the same sequence → FILL with trade reason."""
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "T", "order_id": 0, "size": 100, "ts": _ns(1), "sequence": 2},
            {"raw_type": "F", "order_id": 1, "size": 100, "ts": _ns(1), "sequence": 2},
            {"raw_type": "C", "order_id": 1, "size": 100, "ts": _ns(1), "sequence": 2},
        ])
        out = db_stage_b(df)
        fill = out[out["event_type"] == "FILL"].iloc[0]
        assert "TRADE" in fill["reason"]

    def test_t_and_n_actions_produce_no_events(self):
        df = _db_df([
            {"raw_type": "T", "order_id": 0, "sequence": 1},
            {"raw_type": "N", "order_id": 0, "sequence": 2},
        ])
        out = db_stage_b(df)
        assert len(out) == 0

    def test_r_clears_state_emits_snapshot_resets_increments_session(self):
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "A", "order_id": 2, "size":  50, "ts": _ns(1), "sequence": 2},
            {"raw_type": "R", "order_id": 0,              "ts": _ns(2), "sequence": 3},
            {"raw_type": "A", "order_id": 3, "size":  75, "ts": _ns(3), "sequence": 4},
        ])
        out = db_stage_b(df)
        resets = out[out["reason"] == "SNAPSHOT_RESET"]
        assert len(resets) == 2
        assert set(resets["order_id"].tolist()) == {1, 2}

        # Post-reset ADD should be in session 1
        fresh_add = out[(out["order_id"] == 3) & (out["event_type"] == "ADD")]
        assert fresh_add.iloc[0]["session_id"] == 1

    def test_session_id_column_present(self):
        """Fix 1: session_id must be in Databento output (was missing before fix)."""
        df  = _db_df([{"raw_type": "A"}])
        out = db_stage_b(df)
        assert "session_id" in out.columns
        assert out["session_id"].notna().all()

    def test_side_column_present(self):
        """Fix 1: side must be in Databento output (was missing before fix)."""
        df  = _db_df([{"raw_type": "A", "side": "ASK"}])
        out = db_stage_b(df)
        assert "side" in out.columns
        assert out["side"].iloc[0] == "ASK"

    def test_sequence_nan_rows_are_not_dropped(self):
        """
        Fix 3: NaN sequence values must be filled (→ 0) before groupby so
        those rows are not silently lost.
        """
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": np.nan},
        ])
        out = db_stage_b(df)
        # The ADD must not have been dropped by the groupby
        assert len(out) == 1
        assert out.iloc[0]["event_type"] == "ADD"

    def test_r_action_with_nan_sequence_still_fires(self):
        """
        Fix 3: an R event with NaN sequence must still clear order_state.
        Without the fillna fix it would be dropped by groupby and never fire.
        """
        df = _db_df([
            {"raw_type": "A", "order_id": 1, "size": 100, "ts": _ns(0), "sequence": 1},
            {"raw_type": "R", "order_id": 0,              "ts": _ns(1), "sequence": np.nan},
        ])
        out = db_stage_b(df)
        resets = out[out["reason"] == "SNAPSHOT_RESET"]
        assert len(resets) == 1, "R with NaN sequence must still emit SNAPSHOT_RESET"


# ─────────────────────────────────────────────────────────────────────────────
# Lifetime builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLifetimeBuilder:

    def test_filled_outcome(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "size": 5.0,
             "remaining_size": 0.0, "ts": _ns(100), "reason": "FULL_FILL_TRADE"},
        ])
        lt = build_lifetime(events)
        assert len(lt) == 1
        assert lt.iloc[0]["outcome"]    == "FILLED"
        assert lt.iloc[0]["fill_count"] == 1
        assert lt.iloc[0]["born_size"]  == 5.0

    def test_cancelled_outcome(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",    "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "CANCEL", "event_seq": 1, "size": 5.0,
             "remaining_size": 0.0, "ts": _ns(200), "reason": "CANCELLED"},
        ])
        lt = build_lifetime(events)
        assert lt.iloc[0]["outcome"]      == "CANCELLED"
        assert lt.iloc[0]["cancel_size"]  == 5.0
        assert lt.iloc[0]["fill_count"]   == 0

    def test_open_at_eod_outcome(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD", "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
        ])
        lt = build_lifetime(events)
        assert lt.iloc[0]["outcome"]     == "OPEN_AT_EOD"
        assert pd.isna(lt.iloc[0]["died_ts"])
        assert pd.isna(lt.iloc[0]["duration_ns"])

    def test_partial_fills_counted(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 10.0,
             "remaining_size": 10.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "size": 3.0,
             "remaining_size": 7.0,  "ts": _ns(10), "reason": "PARTIAL_FILL_TRADE"},
            {"order_id": "1", "event_type": "FILL", "event_seq": 2, "size": 7.0,
             "remaining_size": 0.0,  "ts": _ns(20), "reason": "FULL_FILL_TRADE"},
        ])
        lt = build_lifetime(events)
        row = lt.iloc[0]
        assert row["fill_count"]            == 2
        assert row["partial_fill_count"]    == 1
        assert row["total_filled_size"]     == 10.0
        assert row["outcome"]               == "FILLED"

    def test_modify_count(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",    "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "MODIFY", "event_seq": 1, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(1),  "reason": "PRICE_CHANGE"},
            {"order_id": "1", "event_type": "MODIFY", "event_seq": 2, "size": 4.0,
             "remaining_size": 4.0, "ts": _ns(2),  "reason": "SIZE_CHANGE"},
            {"order_id": "1", "event_type": "CANCEL", "event_seq": 3, "size": 4.0,
             "remaining_size": 0.0, "ts": _ns(3),  "reason": "CANCELLED"},
        ])
        lt = build_lifetime(events)
        assert lt.iloc[0]["modify_count"] == 2

    def test_anomaly_overfill(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "size": 6.0,
             "remaining_size": 0.0, "ts": _ns(1),  "reason": "FULL_FILL_TRADE"},
        ])
        lt = build_lifetime(events)
        assert "OVERFILL" in lt.iloc[0]["anomalies"]

    def test_anomaly_multi_add(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD", "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "ADD", "event_seq": 1, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(1)},
        ])
        lt = build_lifetime(events)
        assert "MULTI_ADD" in lt.iloc[0]["anomalies"]

    def test_anomaly_neg_remaining(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 5.0,
             "remaining_size":  5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "size": 5.0,
             "remaining_size": -1.0, "ts": _ns(1),  "reason": "FULL_FILL"},
        ])
        lt = build_lifetime(events)
        assert "NEG_REMAINING" in lt.iloc[0]["anomalies"]

    def test_anomaly_bad_seq(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 3, "size": 5.0,
             "remaining_size": 0.0, "ts": _ns(1),  "reason": "FULL_FILL"},
        ])
        lt = build_lifetime(events)
        assert "BAD_SEQ" in lt.iloc[0]["anomalies"]

    def test_order_with_no_add_is_skipped(self):
        """Orders without an ADD event in the window are excluded from output."""
        events = _event_df([
            {"order_id": "1", "event_type": "FILL", "event_seq": 0, "size": 5.0,
             "remaining_size": 0.0, "reason": "FULL_FILL"},
        ])
        lt = build_lifetime(events)
        assert len(lt) == 0

    def test_empty_input_returns_empty_lifetime(self):
        lt = build_lifetime(_event_df([]))
        assert lt.empty

    def test_clean_order_has_empty_anomalies_string(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "size": 5.0,
             "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "size": 5.0,
             "remaining_size": 0.0, "ts": _ns(1), "reason": "FULL_FILL_TRADE"},
        ])
        lt = build_lifetime(events)
        assert lt.iloc[0]["anomalies"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# Validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidator:

    def _clean(self) -> pd.DataFrame:
        return _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0,
             "size": 5.0, "remaining_size": 5.0, "ts": _ns(0)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1,
             "size": 5.0, "remaining_size": 0.0, "ts": _ns(1),
             "reason": "FULL_FILL_TRADE"},
        ])

    def test_clean_table_passes_all_checks(self):
        report = validate(self._clean())
        assert report.passed

    def test_unknown_event_type_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0},
            {"order_id": "1", "event_type": "BOGUS", "event_seq": 1},
        ])
        report = validate(events)
        assert report.violations["CHECK_UNKNOWN_EVENT"] != []

    def test_unknown_side_flagged(self):
        events = _event_df([{"side": "UNKNOWN"}])
        report = validate(events)
        assert report.violations["CHECK_UNKNOWN_SIDE"] != []

    def test_nan_side_not_flagged(self):
        """NaN side is valid (Databento T/F/R rows have no side)."""
        events = _event_df([{"side": np.nan}])
        report = validate(events)
        assert report.violations["CHECK_UNKNOWN_SIDE"] == []

    def test_neg_remaining_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0,
             "remaining_size":  5.0},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1,
             "remaining_size": -1.0},
        ])
        report = validate(events)
        assert report.violations["CHECK_NEG_REMAINING"] != []

    def test_duplicate_seq_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_seq": 0},
            {"order_id": "1", "event_seq": 0},  # duplicate
        ])
        report = validate(events)
        assert report.violations["CHECK_DUP_SEQ"] != []

    def test_multi_add_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD", "event_seq": 0},
            {"order_id": "1", "event_type": "ADD", "event_seq": 1},
        ])
        report = validate(events)
        assert report.violations["CHECK_MULTI_ADD"] != []

    def test_no_add_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "FILL", "event_seq": 0,
             "remaining_size": 0.0},
        ])
        report = validate(events)
        assert report.violations["CHECK_NO_ADD"] != []

    def test_overfill_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0,
             "size": 5.0, "remaining_size": 5.0},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1,
             "size": 8.0, "remaining_size": 0.0, "reason": "FULL_FILL_TRADE"},
        ])
        report = validate(events)
        assert report.violations["CHECK_OVERFILL"] != []

    def test_ts_out_of_order_flagged(self):
        events = _event_df([
            {"order_id": "1", "event_type": "ADD",  "event_seq": 0, "ts": _ns(100)},
            {"order_id": "1", "event_type": "FILL", "event_seq": 1, "ts": _ns(50)},
        ])
        report = validate(events)
        assert report.violations["CHECK_TS_ORDER"] != []

    def test_mixed_symbol_flagged(self):
        events = pd.DataFrame([
            {"order_id": "1", "session_id": 0, "symbol": "BTC-USDT", "source": "COINBASE",
             "side": "BID", "event_type": "ADD", "event_seq": 0,
             "ts": _ns(0), "price": 100.0, "size": 1.0, "remaining_size": 1.0, "reason": None},
            {"order_id": "1", "session_id": 0, "symbol": "ETH-USDT", "source": "COINBASE",
             "side": "BID", "event_type": "FILL", "event_seq": 1,
             "ts": _ns(1), "price": 100.0, "size": 1.0, "remaining_size": 0.0, "reason": "FULL_FILL"},
        ])
        report = validate(events)
        assert report.violations["CHECK_MIXED_SYMBOL"] != []

    def test_empty_table_returns_report_without_error(self):
        report = validate(_event_df([]))
        assert report.total_rows == 0


# ─────────────────────────────────────────────────────────────────────────────
# Writer / Loader tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_events(
    source: str = "COINBASE",
    symbol: str = "BTC-USDT",
    date_ts: int = _ns(0),          # determines the partition date
    order_id: str = "1",
    session_id: int = 0,
    event_seq: int = 0,
    n: int = 3,
) -> pd.DataFrame:
    """Build a small synthetic event table ready for writing."""
    rows = []
    for i in range(n):
        rows.append({
            "order_id":       order_id,
            "session_id":     session_id,
            "symbol":         symbol,
            "source":         source,
            "side":           "BID",
            "event_type":     "ADD" if i == 0 else "FILL",
            "event_seq":      event_seq + i,
            "ts":             date_ts + i * 1_000_000,
            "price":          100.0,
            "size":           1.0,
            "remaining_size": float(n - 1 - i),
            "reason":         None if i == 0 else "PARTIAL_FILL_TRADE",
        })
    return pd.DataFrame(rows)


class TestWriter:

    def test_write_creates_correct_hive_path(self, tmp_settings):
        settings_path, events_root = tmp_settings
        df     = _make_events()
        writer = StorageWriter(settings_path=settings_path)
        paths  = writer.write_events(df)

        assert len(paths) == 1
        p = paths[0]
        assert "source=COINBASE" in str(p)
        assert "symbol=BTC-USDT" in str(p)
        assert "part.parquet"    in str(p)
        assert p.exists()

    def test_merge_adds_new_rows_to_existing_partition(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)

        first  = _make_events(order_id="1", event_seq=0, n=2)
        second = _make_events(order_id="2", event_seq=0, n=2)

        writer.write_events(first)
        writer.write_events(second, conflict=ConflictMode.MERGE)

        loaded = load_events("COINBASE", "BTC-USDT", "2024-03-23",
                             settings_path=settings_path)
        order_ids = set(loaded["order_id"].tolist())
        assert "1" in order_ids
        assert "2" in order_ids

    def test_merge_deduplicates_on_composite_key(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)

        df = _make_events(n=3)
        writer.write_events(df)
        # Write the exact same rows again — count must not double
        writer.write_events(df, conflict=ConflictMode.MERGE)

        loaded = load_events("COINBASE", "BTC-USDT", "2024-03-23",
                             settings_path=settings_path)
        assert len(loaded) == 3

    def test_overwrite_replaces_partition(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)

        original    = _make_events(order_id="original", n=5)
        replacement = _make_events(order_id="replacement", n=2)

        writer.write_events(original)
        writer.write_events(replacement, conflict=ConflictMode.OVERWRITE)

        loaded = load_events("COINBASE", "BTC-USDT", "2024-03-23",
                             settings_path=settings_path)
        assert len(loaded) == 2
        assert "replacement" in loaded["order_id"].tolist()
        assert "original"    not in loaded["order_id"].tolist()

    def test_error_mode_raises_on_existing_partition(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)

        df = _make_events()
        writer.write_events(df)
        with pytest.raises(FileExistsError):
            writer.write_events(df, conflict=ConflictMode.ERROR)

    def test_multi_date_dataframe_written_to_separate_partitions(self, tmp_settings):
        """
        A DataFrame containing rows from two different UTC dates must be split
        into two separate partition files.
        """
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)

        # 2024-03-23 vs 2024-03-24 (add 86400 seconds = 86_400_000 ms)
        ts_day1 = _ns(0)
        ts_day2 = _ns(0) + 86_400 * 1_000 * 1_000_000  # +1 day in ns

        df = pd.concat([
            _make_events(order_id="day1", date_ts=ts_day1, n=1),
            _make_events(order_id="day2", date_ts=ts_day2, n=1),
        ], ignore_index=True)

        paths = writer.write_events(df)
        assert len(paths) == 2
        date_strs = {str(p) for p in paths}
        assert any("2024-03-23" in s for s in date_strs)
        assert any("2024-03-24" in s for s in date_strs)

    def test_empty_dataframe_writes_nothing(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        paths  = writer.write_events(_make_events(n=0))
        assert paths == []


class TestLoader:

    def _write_date(self, writer, date_offset_days: int, order_id: str) -> None:
        ts = _ns(0) + date_offset_days * 86_400 * 1_000 * 1_000_000
        df = _make_events(order_id=order_id, date_ts=ts, n=1)
        writer.write_events(df)

    def test_load_single_date(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        self._write_date(writer, 0, "day0")

        loaded = load_events("COINBASE", "BTC-USDT", "2024-03-23",
                             settings_path=settings_path)
        assert len(loaded) > 0
        assert "day0" in loaded["order_id"].tolist()

    def test_load_list_of_dates(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        self._write_date(writer, 0, "day0")
        self._write_date(writer, 1, "day1")

        loaded = load_events("COINBASE", "BTC-USDT",
                             ["2024-03-23", "2024-03-24"],
                             settings_path=settings_path)
        oids = set(loaded["order_id"].tolist())
        assert "day0" in oids
        assert "day1" in oids

    def test_load_date_range(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        for i in range(3):
            self._write_date(writer, i, f"day{i}")

        loaded = load_events("COINBASE", "BTC-USDT",
                             ("2024-03-23", "2024-03-25"),
                             settings_path=settings_path)
        oids = set(loaded["order_id"].tolist())
        assert {"day0", "day1", "day2"}.issubset(oids)

    def test_load_result_sorted_by_ts(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        self._write_date(writer, 0, "day0")
        self._write_date(writer, 1, "day1")

        loaded = load_events("COINBASE", "BTC-USDT",
                             ("2024-03-23", "2024-03-24"),
                             settings_path=settings_path)
        assert loaded["ts"].is_monotonic_increasing

    def test_missing_partition_raises_file_not_found(self, tmp_settings):
        settings_path, _ = tmp_settings
        with pytest.raises(FileNotFoundError):
            load_events("COINBASE", "BTC-USDT", "2099-01-01",
                        settings_path=settings_path)

    def test_list_available_returns_inventory(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        self._write_date(writer, 0, "day0")
        self._write_date(writer, 1, "day1")

        inv = list_available(settings_path=settings_path)
        assert len(inv) == 2
        assert set(inv.columns) >= {"source", "symbol", "date", "rows", "path"}

    def test_list_available_filter_by_source(self, tmp_settings):
        settings_path, _ = tmp_settings
        writer = StorageWriter(settings_path=settings_path)
        self._write_date(writer, 0, "day0")

        inv = list_available(source="COINBASE", settings_path=settings_path)
        assert (inv["source"] == "COINBASE").all()

    def test_list_available_empty_when_no_data(self, tmp_settings):
        settings_path, _ = tmp_settings
        inv = list_available(settings_path=settings_path)
        assert inv.empty
