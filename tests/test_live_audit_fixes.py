def test_effective_rr_matches_aggregator_long():
    from strategies.base import BaseStrategy

    # Worst-case fill for LONG = entry_high (101).
    # reward = 108 - 101 = 7, risk = 101 - 96 = 5  →  RR = 1.4
    rr = BaseStrategy.calculate_effective_rr(
        direction="LONG",
        entry_low=99.0,
        entry_high=101.0,
        stop_loss=96.0,
        tp2=108.0,
    )

    assert rr == 1.4


def test_effective_rr_matches_aggregator_short():
    from strategies.base import BaseStrategy

    # Worst-case fill for SHORT = entry_low (99).
    # reward = 99 - 92 = 7, risk = 104 - 99 = 5  →  RR = 1.4
    rr = BaseStrategy.calculate_effective_rr(
        direction="SHORT",
        entry_low=99.0,
        entry_high=101.0,
        stop_loss=104.0,
        tp2=92.0,
    )

    assert rr == 1.4


def test_smc_blocks_non_reversal_counter_trend_setup():
    from strategies.smc import SmartMoneyConcepts

    allowed = SmartMoneyConcepts._allows_counter_trend_setup(
        setup_type="BreakOfStructure",
        has_choch=False,
        choch_direction="",
        direction="SHORT",
    )

    assert allowed is False


def test_smc_allows_counter_trend_liquidity_sweep():
    from strategies.smc import SmartMoneyConcepts

    allowed = SmartMoneyConcepts._allows_counter_trend_setup(
        setup_type="LiquiditySweep",
        has_choch=False,
        choch_direction="",
        direction="SHORT",
    )

    assert allowed is True


def test_smc_allows_counter_trend_choch_reversal():
    from strategies.smc import SmartMoneyConcepts

    allowed = SmartMoneyConcepts._allows_counter_trend_setup(
        setup_type="OrderBlock",
        has_choch=True,
        choch_direction="BEARISH",
        direction="SHORT",
    )

    assert allowed is True
