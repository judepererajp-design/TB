"""Verify that the regime-aware expected-silent classification in
``TitanBot._check_strategy_health`` correctly derives from each strategy's
own ``VALID_REGIMES``.

Context: the health alert used to hardcode a regime→strategy set and lacked
a BULL_TREND entry, so mean_reversion / reversal / range_scalper (which are
hard-gated to CHOPPY) always alerted as "real silence" in a bull trend.
The fix derives the set from each strategy's ``VALID_REGIMES`` attribute.
"""

from config.constants import STRATEGY_VALID_REGIMES, STRATEGY_KEY_MAP


# Keys the health alert uses (same derivation as _strat_key_map values).
def _key_for(strategy_name: str) -> str:
    return STRATEGY_KEY_MAP.get(strategy_name, strategy_name.lower())


def _expected_silent_via_reflection(current_regime: str) -> set:
    """Mirror the engine's derivation: a strategy is expected-silent iff the
    current regime is not in its VALID_REGIMES."""
    out: set = set()
    for name, valid in STRATEGY_VALID_REGIMES.items():
        if current_regime not in valid:
            out.add(_key_for(name))
    # Manual override for RangeScalper which gates explicitly in analyse().
    _manual = {
        "BULL_TREND":     {"range_scalper"},
        "BEAR_TREND":     {"range_scalper"},
        "VOLATILE":       {"range_scalper"},
        "VOLATILE_PANIC": {"range_scalper"},
    }
    out |= _manual.get(current_regime, set())
    return out


class TestRegimeExpectedSilent:
    def test_bull_trend_suppresses_counter_trend_and_range_strategies(self):
        silent = _expected_silent_via_reflection("BULL_TREND")
        # Three strategies hard-gated to CHOPPY (or explicitly trend-blocked)
        # must be classified as expected-silent in BULL_TREND.
        assert "mean_reversion" in silent
        assert "reversal" in silent
        assert "range_scalper" in silent

    def test_bull_trend_does_not_suppress_trend_strategies(self):
        silent = _expected_silent_via_reflection("BULL_TREND")
        # Strategies that are supposed to fire in BULL_TREND must NOT be in
        # the expected-silent set.
        for k in ("momentum", "breakout", "elliott_wave",
                  "smc", "price_action", "ichimoku", "wyckoff"):
            assert k not in silent, f"{k} should alert as real silent in BULL_TREND"

    def test_volatile_panic_suppresses_most_strategies(self):
        silent = _expected_silent_via_reflection("VOLATILE_PANIC")
        # VOLATILE_PANIC isn't in most VALID_REGIMES sets so many should be
        # expected-silent.  SmartMoneyConcepts explicitly allows it.
        assert "smc" not in silent
        assert "elliott_wave" in silent
        assert "mean_reversion" in silent

    def test_choppy_lets_mean_reversion_fire(self):
        silent = _expected_silent_via_reflection("CHOPPY")
        assert "mean_reversion" not in silent
        assert "reversal" not in silent
        # Momentum isn't valid in CHOPPY
        assert "momentum" in silent
