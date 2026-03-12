"""
Tests for natgas.analysis.bias_correction

Covers:
- BiasCorrector.correct_hdd_cdd with no DB (returns raw unchanged minus zero bias)
- Bias application: corrected >= 0 always (no negative HDD/CDD)
- Cache: second call for same (model, region, lead) uses cache
- correct_hdd_cdd with a mock DB that returns known bias values
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import date

from natgas.analysis.bias_correction import BiasCorrector, NORMAL_MOS_WINDOW_DAYS


# ── No-DB (offline) mode ──────────────────────────────────────────────────────

class TestBiasCorrectorNoDB:
    def setup_method(self):
        self.corrector = BiasCorrector(db_session=None)

    def test_returns_raw_when_no_db(self):
        hdd_c, cdd_c, bias_h, bias_c = self.corrector.correct_hdd_cdd(
            raw_hdd=10.0, raw_cdd=0.0,
            model_name="GFS", region="national", lead_days=5,
        )
        assert hdd_c == pytest.approx(10.0)
        assert cdd_c == pytest.approx(0.0)
        assert bias_h == pytest.approx(0.0)
        assert bias_c == pytest.approx(0.0)

    def test_corrected_hdd_never_negative(self):
        # If somehow bias > raw, corrected must be clamped to 0
        corrector = BiasCorrector(db_session=None)
        # Inject a positive bias manually
        corrector._bias_cache[("GFS", "national", 5)] = (20.0, 0.0)
        hdd_c, _, _, _ = corrector.correct_hdd_cdd(
            raw_hdd=5.0, raw_cdd=0.0,
            model_name="GFS", region="national", lead_days=5,
        )
        assert hdd_c == 0.0

    def test_corrected_cdd_never_negative(self):
        corrector = BiasCorrector(db_session=None)
        corrector._bias_cache[("GFS", "national", 5)] = (0.0, 15.0)
        _, cdd_c, _, _ = corrector.correct_hdd_cdd(
            raw_hdd=0.0, raw_cdd=3.0,
            model_name="GFS", region="national", lead_days=5,
        )
        assert cdd_c == 0.0

    def test_cache_populated_on_first_call(self):
        self.corrector.correct_hdd_cdd(
            raw_hdd=8.0, raw_cdd=0.0,
            model_name="EURO_HRES", region="midwest", lead_days=3,
        )
        assert ("EURO_HRES", "midwest", 3) in self.corrector._bias_cache

    def test_cache_reused_on_second_call(self):
        """compute_rolling_bias should only be called once per (model, region, lead)."""
        with patch.object(self.corrector, "compute_rolling_bias", return_value=(2.0, 1.0)) as mock_bias:
            self.corrector.correct_hdd_cdd(12.0, 5.0, "GFS", "northeast", 7)
            self.corrector.correct_hdd_cdd(15.0, 8.0, "GFS", "northeast", 7)
        mock_bias.assert_called_once()


# ── With mock DB ──────────────────────────────────────────────────────────────

class TestBiasCorrectorWithDB:
    def _make_mock_session(self, bias_hdd: float, bias_cdd: float):
        """Return a mock SQLAlchemy session that returns known bias values."""
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, idx: [bias_hdd, bias_cdd][idx]
        mock_result.__iter__ = lambda self: iter([bias_hdd, bias_cdd])
        mock_result.__bool__ = lambda self: True

        # Mimic (result[0], result[1])
        fetchone_return = (bias_hdd, bias_cdd)

        mock_execute = MagicMock()
        mock_execute.fetchone.return_value = fetchone_return

        mock_session = MagicMock()
        mock_session.execute.return_value = mock_execute
        return mock_session

    def test_applies_known_bias(self):
        session = self._make_mock_session(bias_hdd=3.0, bias_cdd=1.5)
        corrector = BiasCorrector(db_session=session)
        hdd_c, cdd_c, bh, bc = corrector.correct_hdd_cdd(
            raw_hdd=10.0, raw_cdd=5.0,
            model_name="GFS", region="national", lead_days=5,
        )
        assert hdd_c == pytest.approx(7.0)
        assert cdd_c == pytest.approx(3.5)
        assert bh == pytest.approx(3.0)
        assert bc == pytest.approx(1.5)

    def test_rolling_bias_uses_correct_window(self):
        """compute_rolling_bias passes mos_window_days to the SQL query."""
        session = self._make_mock_session(0.0, 0.0)
        corrector = BiasCorrector(db_session=session, mos_window_days=7)
        assert corrector.mos_window_days == 7
        corrector.correct_hdd_cdd(5.0, 0.0, "AIFS", "texas", 10)
        # Session.execute should have been called (with a window-based query)
        assert session.execute.called

    def test_zero_bias_when_db_returns_none(self):
        session = MagicMock()
        session.execute.return_value.fetchone.return_value = (None, None)
        corrector = BiasCorrector(db_session=session)
        hdd_c, cdd_c, bh, bc = corrector.correct_hdd_cdd(
            raw_hdd=8.0, raw_cdd=2.0, model_name="GEFS", region="southeast", lead_days=14,
        )
        assert hdd_c == pytest.approx(8.0)
        assert cdd_c == pytest.approx(2.0)
        assert bh == pytest.approx(0.0)
        assert bc == pytest.approx(0.0)

    def test_default_mos_window(self):
        session = self._make_mock_session(0.0, 0.0)
        corrector = BiasCorrector(db_session=session)
        assert corrector.mos_window_days == NORMAL_MOS_WINDOW_DAYS
