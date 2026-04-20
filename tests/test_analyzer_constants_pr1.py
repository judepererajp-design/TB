"""
Tests for the PR-1 analyzer-infrastructure constants and the
``validate_analyzer_constants`` cross-class invariant validator.
"""

from __future__ import annotations

import importlib

from config.constants import (
    BTCDominance,
    Correlation,
    EnergyModel,
    Equilibrium,
    InstitutionalFlow,
    LeverageMapper,
    Liquidation,
    MarketBrain,
    NoTradeZones,
    OrderFlow,
    ParabolicDetector,
    Sentiment,
    VolatilityStructure,
    Volume,
)
from config.schema import validate_analyzer_constants, validate_config


class TestDefaults:
    def test_energy_weights_sum_to_one(self):
        total = (
            EnergyModel.WEIGHT_MOMENTUM
            + EnergyModel.WEIGHT_PARTICIPATION
            + EnergyModel.WEIGHT_POSITIONING
            + EnergyModel.WEIGHT_FLOW
        )
        assert abs(total - 1.0) < 1e-6

    def test_energy_thresholds_ordered(self):
        assert EnergyModel.EXHAUSTED < EnergyModel.LOW_ENERGY < EnergyModel.HIGH_ENERGY

    def test_market_brain_tiers_ordered(self):
        assert MarketBrain.CONFLUENCE_MODERATE_MIN < MarketBrain.CONFLUENCE_STRONG_MIN
        assert MarketBrain.CONFLUENCE_MODERATE_PTS <= MarketBrain.CONFLUENCE_STRONG_PTS

    def test_no_trade_zones_atr_monotonic(self):
        assert (
            NoTradeZones.ATR_BLOCK_RATIO_CHOPPY
            <= NoTradeZones.ATR_BLOCK_RATIO_TREND
            <= NoTradeZones.ATR_BLOCK_RATIO_VOLATILE
        )
        assert NoTradeZones.CONFIRMATION_BARS >= 1

    def test_correlation_clamp_valid(self):
        assert Correlation.BETA_CLAMP_MIN < 0 < Correlation.BETA_CLAMP_MAX
        assert 0 <= Correlation.MODERATE_THRESHOLD < Correlation.STRONG_THRESHOLD <= 1

    def test_parabolic_sane(self):
        assert ParabolicDetector.EXHAUSTION_MIN_SIGNALS >= 1
        assert ParabolicDetector.MIN_PRICE_USD > 0

    def test_sentiment_bounds(self):
        assert Sentiment.API_MIN < Sentiment.API_MAX
        assert 0 < Sentiment.EWMA_DECAY < 1

    def test_btc_dominance_ordering(self):
        assert (
            BTCDominance.SHARP_FALL_PCT
            < BTCDominance.MODERATE_FALL_PCT
            < 0
            < BTCDominance.MODERATE_RISE_PCT
            < BTCDominance.SHARP_RISE_PCT
        )
        assert BTCDominance.HISTORY_MAX_POINTS > 0

    def test_equilibrium(self):
        assert Equilibrium.DEAD_ZONE_PCT_CHOPPY < Equilibrium.DEAD_ZONE_PCT_DEFAULT
        assert 0 < Equilibrium.MIN_CHOP_FOR_RULES <= 1

    def test_orderflow(self):
        assert 0 <= OrderFlow.CVD_RESET_UTC_HOUR <= 23
        assert 0 < OrderFlow.ABSORPTION_DELTA_RATIO <= 1
        assert OrderFlow.BLOCK_TRADE_USD_MIN > 0

    def test_leverage_mapper_ordering(self):
        assert LeverageMapper.EFFECTIVE_LEVERAGE_HIGH < LeverageMapper.EFFECTIVE_LEVERAGE_EXTREME
        assert LeverageMapper.CASCADE_MIN_USD > 0

    def test_liquidation_timings_positive(self):
        for attr in ("OI_REFRESH_SECS", "LIQ_REFRESH_SECS",
                     "DEDUP_BUFFER_SIZE", "DEDUP_TTL_SECS", "REPLAY_WINDOW_SECS"):
            assert getattr(Liquidation, attr) > 0

    def test_institutional_flow_ordered(self):
        assert (
            InstitutionalFlow.FAST_INTERVAL_SECS
            < InstitutionalFlow.SLOW_INTERVAL_SECS
            < InstitutionalFlow.COT_INTERVAL_SECS
        )
        assert InstitutionalFlow.PCTILE_30D_WINDOW < InstitutionalFlow.PCTILE_90D_WINDOW

    def test_volume(self):
        assert Volume.LOG_FLOOR > 0
        assert Volume.BASELINE_BARS >= Volume.MIN_SAMPLES

    def test_volatility_percentiles_ordered(self):
        assert (
            0
            < VolatilityStructure.VOL_EXTREME_LOW_PCTILE
            < VolatilityStructure.VOL_LOW_PCTILE
            < VolatilityStructure.VOL_HIGH_PCTILE
            < VolatilityStructure.VOL_EXTREME_HIGH_PCTILE
            < 100
        )

    def test_garch_stationary(self):
        # GARCH(1,1) stationarity: α + β < 1
        s = VolatilityStructure.GARCH_ALPHA + VolatilityStructure.GARCH_BETA
        assert s < 1.0


class TestInvariantValidator:
    def test_clean_defaults_pass(self):
        errors = validate_analyzer_constants()
        assert errors == [], f"Default constants failed invariants: {errors}"

    def test_full_validate_config_pipes_through(self):
        # An empty config gets section-missing errors, but the analyzer-constants
        # invariants should still report 0 contribution.
        ok, errors = validate_config({})
        # Any invariant-level regression would show up as one of our specific
        # messages — assert none are present.
        marker_strings = [
            "EnergyModel:",
            "MarketBrain:",
            "NoTradeZones:",
            "Correlation:",
            "Sentiment:",
            "BTCDominance:",
            "Equilibrium:",
            "OrderFlow.",
            "LeverageMapper",
            "Liquidation.",
            "InstitutionalFlow:",
            "Volume.",
            "VolatilityStructure:",
        ]
        for m in marker_strings:
            assert not any(m in e for e in errors), f"Unexpected invariant failure: {[e for e in errors if m in e]}"

    def test_invariant_fires_on_bad_energy_weights(self, monkeypatch):
        monkeypatch.setattr(EnergyModel, "WEIGHT_MOMENTUM", 0.9)
        errors = validate_analyzer_constants()
        assert any("EnergyModel:" in e and "sum to 1.0" in e for e in errors)

    def test_invariant_fires_on_reversed_api_bounds(self, monkeypatch):
        monkeypatch.setattr(Sentiment, "API_MIN", 100)
        monkeypatch.setattr(Sentiment, "API_MAX", 0)
        errors = validate_analyzer_constants()
        assert any("Sentiment" in e and "API_MIN" in e for e in errors)

    def test_invariant_fires_on_non_stationary_garch(self, monkeypatch):
        monkeypatch.setattr(VolatilityStructure, "GARCH_ALPHA", 0.6)
        monkeypatch.setattr(VolatilityStructure, "GARCH_BETA", 0.5)
        errors = validate_analyzer_constants()
        assert any("GARCH" in e for e in errors)
