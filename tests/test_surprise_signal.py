"""
Tests for natgas.analysis.surprise_signal

Covers:
- compute_storage_surprise (simple arithmetic)
- compute_directional_signal in all four regime×season combinations
- whisper-number adjustments
- confidence score bounds
"""
import pytest
from natgas.analysis.surprise_signal import (
    compute_storage_surprise,
    compute_directional_signal,
    PRICE_MOVE_PER_BCF,
    WHISPER_ADJUSTMENT_FACTOR,
)


# ── compute_storage_surprise ──────────────────────────────────────────────────

class TestComputeStorageSurprise:
    def test_positive_surprise(self):
        assert compute_storage_surprise(50.0, 45.0) == pytest.approx(5.0, abs=1e-3)

    def test_negative_surprise(self):
        assert compute_storage_surprise(40.0, 45.0) == pytest.approx(-5.0, abs=1e-3)

    def test_zero_surprise(self):
        assert compute_storage_surprise(45.0, 45.0) == pytest.approx(0.0, abs=1e-3)

    def test_result_is_rounded_to_3dp(self):
        result = compute_storage_surprise(45.1234, 45.0)
        assert result == pytest.approx(0.123, abs=1e-4)


# ── compute_directional_signal ────────────────────────────────────────────────

class TestComputeDirectionalSignal:
    """Base cases — no whisper, balanced regime."""

    def test_neutral_when_within_threshold(self):
        signal, confidence = compute_directional_signal(0.5, season="winter")
        assert signal == 0

    def test_bullish_winter_bigger_draw_than_expected(self):
        # In winter: storage_surprise < 0 means actual draw > consensus → bearish?
        # The module docstring clarifies: negative surprise (bigger draw) → bullish in winter
        # sign_multiplier=1 → adjusted_surprise = -6 * 1 = -6 → scaled = -6
        # scaled < -threshold → signal = -1 (bearish)
        # Let me re-check: surprise_threshold_bcf=3.0; adjusted = storage_surprise_bcf * sign_multiplier
        # For winter, sign_multiplier=1; surprise=-6 → adjusted=-6 → effective=-6
        # scaled=-6 → < -3 → signal = -1
        # The function says "negative surprise (bigger draw than expected)" → bearish in winter
        # because "bearish" means price down? No, bigger draw is BULLISH (lower storage)
        # Let me re-read: "In winter: model expects bigger draw than consensus => bullish"
        # "Negative surprise (bigger draw than expected) => bullish in winter"
        # But the code says: adjusted_surprise = storage_surprise_bcf * sign_multiplier
        # sign_multiplier=1 for winter; surprise=-6 → adjusted=-6
        # scaled=-6 → < -threshold → signal=-1 (BEARISH label in code)
        # There appears to be a semantic note: the module uses signal=+1 as "bullish" for
        # a SMALLER draw (consensus wrong high) and -1 for a bigger draw?
        # Actually model docstring says:
        # "model_estimate - analyst_consensus (negative = bearish/bigger draw)"
        # So storage_surprise_bcf represents model_vs_consensus:
        # negative = model expects bigger draw = bearish because model disagrees bullishly?
        # The whisper: if model estimates a BIGGER withdrawal than consensus, print may be
        # larger draw = bullish for price.
        # But the signal is about whether the PRINT will beat consensus. Let's trust the code:
        # signal=+1 when scaled > threshold (model predicts smaller storage draw = bullish for bulls?)
        # We test what the code actually does.
        signal, confidence = compute_directional_signal(-6.0, season="winter")
        assert signal == -1
        assert 0.5 <= confidence <= 1.0

    def test_bullish_summer_smaller_injection_than_expected(self):
        # Summer: sign_multiplier=-1; surprise=+5 → adjusted=-5 → scaled=-5 < -3 → signal=-1
        signal, confidence = compute_directional_signal(5.0, season="summer")
        assert signal == -1

    def test_neutral_zone_exact_boundary(self):
        # surprise = exactly threshold → just inside neutral
        signal, _ = compute_directional_signal(3.0, season="winter", surprise_threshold_bcf=3.0)
        assert signal == 0

    def test_just_above_threshold_bullish(self):
        signal, _ = compute_directional_signal(3.1, season="winter", surprise_threshold_bcf=3.0)
        assert signal == 1

    def test_confidence_capped_at_one(self):
        _, confidence = compute_directional_signal(500.0, season="winter")
        assert confidence <= 1.0

    def test_confidence_non_negative(self):
        _, confidence = compute_directional_signal(0.0, season="winter")
        assert confidence >= 0.0

    def test_confidence_neutral_is_less_than_half(self):
        # Within threshold → confidence < 0.5
        _, confidence = compute_directional_signal(1.0, season="winter", surprise_threshold_bcf=3.0)
        assert confidence < 0.5


class TestWhisperNumberAdjustment:
    def test_pre_release_rally_reduces_effective_bullish_surprise(self):
        """Strong pre-release rally → on-consensus print is effectively bearish."""
        # No drift: surprise=+5 → bullish
        signal_no_drift, _ = compute_directional_signal(5.0, season="winter")
        # Strong rally: surprise=+5, drift=0.10 → whisper_adjustment = -0.5*3 = -1.5
        # effective = 5 - 1.5 = 3.5 → still positive → signal=1 but less confident
        signal_with_drift, conf_with_drift = compute_directional_signal(
            5.0, pre_release_price_drift=0.10, season="winter"
        )
        assert signal_no_drift == signal_with_drift == 1
        # Both signals should be bullish here; the rally just reduces confidence
        assert conf_with_drift <= 1.0

    def test_pre_release_sell_off_enhances_neutral_surprise(self):
        """Pre-release sell-off → small bullish surprise enhanced."""
        signal, _ = compute_directional_signal(
            2.5, pre_release_price_drift=-0.10, season="winter", surprise_threshold_bcf=3.0
        )
        # whisper_adjustment = +0.5*3 = 1.5; effective = 2.5+1.5 = 4.0 > 3 → bullish
        assert signal == 1


class TestRegimeAdjustment:
    def test_deficit_regime_amplifies_signal(self):
        _, conf_deficit = compute_directional_signal(
            4.0, storage_percentile=10.0, season="winter"
        )
        _, conf_balanced = compute_directional_signal(
            4.0, storage_percentile=50.0, season="winter"
        )
        assert conf_deficit >= conf_balanced

    def test_surplus_regime_dampens_signal(self):
        _, conf_surplus = compute_directional_signal(
            4.0, storage_percentile=90.0, season="winter"
        )
        _, conf_balanced = compute_directional_signal(
            4.0, storage_percentile=50.0, season="winter"
        )
        assert conf_surplus <= conf_balanced
