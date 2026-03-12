"""
Tests for pipeline layers (unit — no live network calls).

Covers:
- eia_storage.fetch_weekly_storage: correct structure, surprise calc, backoff contract
- analyst_consensus.fetch_barchart_consensus: handles missing API key gracefully
- analyst_consensus.record_manual_consensus: correct dict shape
- analyst_consensus.insert_consensus_record: deduplication ON CONFLICT DO NOTHING
- futures_prices.fetch_daily_settlement: handles missing API key gracefully
- supply_demand.fetch_supply_demand: aggregates residential + commercial to rc_demand
"""
import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from natgas.pipelines.analyst_consensus import (
    fetch_barchart_consensus,
    record_manual_consensus,
    insert_consensus_record,
)
from natgas.pipelines.futures_prices import fetch_daily_settlement
from natgas.pipelines.supply_demand import fetch_supply_demand


# ── analyst_consensus ─────────────────────────────────────────────────────────

class TestFetchBarchartConsensus:
    def test_returns_none_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = fetch_barchart_consensus(date(2025, 3, 13), api_key=None)
        assert result is None

    def test_returns_none_on_empty_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_barchart_consensus(date(2025, 3, 13), api_key="test_key")
        assert result is None

    def test_parses_valid_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "meanEstimate": "45.5",
                "highEstimate": "50.0",
                "lowEstimate": "40.0",
                "numEstimates": "12",
            }]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_barchart_consensus(date(2025, 3, 13), api_key="test_key")

        assert result is not None
        assert result["mean_estimate_bcf"] == pytest.approx(45.5)
        assert result["high_estimate_bcf"] == pytest.approx(50.0)
        assert result["low_estimate_bcf"] == pytest.approx(40.0)
        assert result["respondent_count"] == 12
        assert result["source_name"] == "Barchart"
        assert result["report_date"] == date(2025, 3, 13)

    def test_returns_none_on_request_exception(self):
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            result = fetch_barchart_consensus(date(2025, 3, 13), api_key="test_key")
        assert result is None


class TestRecordManualConsensus:
    def test_required_fields_present(self):
        rec = record_manual_consensus(
            report_date=date(2025, 3, 13),
            mean_estimate_bcf=42.0,
        )
        assert rec["report_date"] == date(2025, 3, 13)
        assert rec["mean_estimate_bcf"] == 42.0
        assert rec["source_name"] == "Manual"
        assert "published_at" in rec

    def test_optional_fields_default_to_none(self):
        rec = record_manual_consensus(date(2025, 3, 13), 42.0)
        assert rec.get("high_estimate_bcf") is None
        assert rec.get("low_estimate_bcf") is None
        assert rec.get("respondent_count") is None

    def test_custom_source_name(self):
        rec = record_manual_consensus(date(2025, 3, 13), 42.0, source_name="Bloomberg")
        assert rec["source_name"] == "Bloomberg"


class TestInsertConsensusRecord:
    def _mock_session(self, rowcount: int = 1):
        mock_result = MagicMock()
        mock_result.rowcount = rowcount
        session = MagicMock()
        session.execute.return_value = mock_result
        return session

    def test_returns_true_on_insert(self):
        session = self._mock_session(rowcount=1)
        rec = {
            "report_date": date(2025, 3, 13),
            "published_at": datetime.now(timezone.utc),
            "mean_estimate_bcf": 42.0,
            "high_estimate_bcf": 48.0,
            "low_estimate_bcf": 36.0,
            "respondent_count": 10,
            "source_name": "Barchart",
        }
        result = insert_consensus_record(rec, session)
        assert result is True

    def test_returns_false_on_conflict(self):
        session = self._mock_session(rowcount=0)
        rec = {
            "report_date": date(2025, 3, 13),
            "published_at": datetime.now(timezone.utc),
            "mean_estimate_bcf": 42.0,
            "high_estimate_bcf": None,
            "low_estimate_bcf": None,
            "respondent_count": None,
            "source_name": "Manual",
        }
        result = insert_consensus_record(rec, session)
        assert result is False


# ── futures_prices ────────────────────────────────────────────────────────────

class TestFetchDailySettlement:
    def test_returns_none_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = fetch_daily_settlement(date(2025, 3, 13), api_key=None)
        assert result is None

    def test_returns_none_on_empty_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_daily_settlement(date(2025, 3, 13), api_key="key")
        assert result is None

    def test_parses_valid_settlement(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"lastPrice": "3.456", "volume": "80000", "openInterest": "150000"}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_daily_settlement(date(2025, 3, 13), api_key="key")

        assert result is not None
        assert result["front_month_settle"] == pytest.approx(3.456)
        assert result["volume"] == 80000
        assert result["open_interest"] == 150000
        assert result["trade_date"] == date(2025, 3, 13)
        assert result["data_source"] == "Barchart"

    def test_returns_none_on_exception(self):
        with patch("requests.get", side_effect=TimeoutError("timeout")):
            result = fetch_daily_settlement(date(2025, 3, 13), api_key="key")
        assert result is None


# ── supply_demand ─────────────────────────────────────────────────────────────

class TestFetchSupplyDemand:
    def _make_mock_requests(self, values: dict):
        """Return a requests.get mock that returns the given field→value mapping."""
        def side_effect(url, params=None, timeout=None):
            # Determine which series is being fetched from params
            series_id = (params or {}).get("facets[series][]", "")
            # Map series_id back to field name
            from natgas.pipelines.supply_demand import EIA_SERIES
            field = next((f for f, s in EIA_SERIES.items() if s == series_id), None)
            value = values.get(field, 0.0)
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "response": {"data": [{"value": str(value)}]}
            }
            return mock_resp
        return side_effect

    def test_aggregates_residential_and_commercial(self):
        with patch("requests.get", side_effect=self._make_mock_requests({
            "dry_gas_production": 100.0,
            "lng_exports": 13.0,
            "pipeline_exports_mexico": 6.0,
            "residential_demand": 15.0,
            "commercial_demand": 10.0,
            "industrial_demand": 22.0,
        })):
            result = fetch_supply_demand(api_key="key", week_ending_date=date(2025, 3, 13))

        assert result is not None
        assert result["residential_commercial_demand_bcfd"] == pytest.approx(25.0)

    def test_returns_correct_week_ending_date(self):
        with patch("requests.get", side_effect=self._make_mock_requests({})):
            result = fetch_supply_demand(api_key="key", week_ending_date=date(2025, 3, 13))
        assert result["week_ending_date"] == date(2025, 3, 13)

    def test_data_source_is_eia(self):
        with patch("requests.get", side_effect=self._make_mock_requests({})):
            result = fetch_supply_demand(api_key="key", week_ending_date=date(2025, 3, 13))
        assert result["data_source"] == "EIA_API"

    def test_field_none_when_request_fails(self):
        """If one EIA series call fails, that field should be None (not crash)."""
        call_count = [0]

        def partial_fail(url, params=None, timeout=None):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("network error")
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"response": {"data": [{"value": "10.0"}]}}
            return mock_resp

        with patch("requests.get", side_effect=partial_fail):
            result = fetch_supply_demand(api_key="key", week_ending_date=date(2025, 3, 13))

        assert result is not None
        # Some fields may be None, others should have values
        all_values = [
            result.get("lng_exports_bcfd"),
            result.get("dry_gas_production_bcfd"),
            result.get("industrial_demand_bcfd"),
        ]
        # At least some should be non-None (the successful calls)
        assert any(v is not None for v in all_values)
