from pathlib import Path


def test_effective_rr_matches_aggregator_long():
    from strategies.base import BaseStrategy

    rr = BaseStrategy.calculate_effective_rr(
        direction="LONG",
        entry_low=99.0,
        entry_high=101.0,
        stop_loss=96.0,
        tp2=108.0,
    )

    assert rr == 2.0


def test_effective_rr_matches_aggregator_short():
    from strategies.base import BaseStrategy

    rr = BaseStrategy.calculate_effective_rr(
        direction="SHORT",
        entry_low=99.0,
        entry_high=101.0,
        stop_loss=104.0,
        tp2=92.0,
    )

    assert rr == 2.0


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


def test_smc_and_elliott_use_effective_rr_helper():
    repo_root = Path(__file__).resolve().parents[1]
    smc_source = repo_root.joinpath("strategies", "smc.py").read_text()
    elliott_source = repo_root.joinpath("strategies", "elliott_wave.py").read_text()

    assert "calculate_effective_rr(" in smc_source
    assert "calculate_effective_rr(" in elliott_source
