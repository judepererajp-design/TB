import sys
import types
from unittest.mock import MagicMock

if 'config.loader' not in sys.modules:
    _loader = types.ModuleType('config.loader')
    _loader.cfg = MagicMock()
    sys.modules['config.loader'] = _loader

from signals.context_contracts import build_execution_context, build_setup_context, enrich_setup_context_with_eq
from strategies.base import SignalDirection, SignalResult


def test_build_setup_context_from_smc_signal():
    sig = SignalResult(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        strategy="SmartMoneyConcepts",
        confidence=84.0,
        entry_low=100.0,
        entry_high=102.0,
        stop_loss=97.0,
        tp1=106.0,
        tp2=110.0,
        rr_ratio=2.0,
        raw_data={
            "setup_type": "OrderBlock",
            "bos_level": 99.0,
            "ob_low": 98.0,
            "ob_high": 101.0,
            "has_choch": True,
            "choch_direction": "BULLISH",
            "structure_trend": "BEARISH",
            "atr": 2.5,
        },
        timeframe="4h",
        setup_class="swing",
    )

    setup_context = build_setup_context(sig)
    assert setup_context["structure"]["bos"] is True
    assert setup_context["structure"]["choch"] is True
    assert setup_context["pattern"]["order_block"]["low"] == 98.0
    assert setup_context["location"]["entry_low"] == 100.0


def test_build_execution_context_uses_structured_raw_data():
    sig = SignalResult(
        symbol="ETHUSDT",
        direction=SignalDirection.SHORT,
        strategy="WyckoffAccDist",
        confidence=79.0,
        entry_low=100.0,
        entry_high=101.0,
        stop_loss=103.0,
        tp1=96.0,
        tp2=93.0,
        rr_ratio=2.5,
        raw_data={
            "trigger_quality_score": 0.61,
            "trigger_quality_label": "MEDIUM",
            "trigger_quality_volume_context": "NORMAL",
            "spread_bps": 18.0,
            "whale_intent": "DISTRIBUTION",
            "whale_intent_bias": "BEARISH",
            "whale_intent_confidence": 0.72,
            "volume_quality_score": 66.0,
            "volume_quality_label": "MEDIUM",
            "funding_rate": 0.03,
            "oi_change_24h": -12.0,
        },
    )

    execution_context = build_execution_context(
        sig,
        session_name="New York Open",
        is_killzone=True,
        is_dead_zone=False,
        is_weekend=False,
        eq_zone="premium",
        eq_zone_depth=0.71,
        volume_context="NORMAL",
        volume_score=64.0,
        derivatives_score=58.0,
        sentiment_score=61.0,
        volatility_regime="HIGH",
        transition_type="breakout",
        transition_risk=0.18,
        trade_type="LOCAL_CONTINUATION_SHORT",
        trade_type_strength=2,
    )

    assert execution_context["whales"]["aligned"] is True
    assert execution_context["liquidity"]["spread_bps"] == 18.0
    assert execution_context["positioning"]["derivatives_score"] == 58.0
    assert execution_context["market"]["transition_type"] == "breakout"

    enriched = enrich_setup_context_with_eq({"location": {}}, eq_zone="premium", eq_distance=0.71)
    assert enriched["location"]["entry_zone"] == "premium"
    assert enriched["location"]["eq_distance"] == 0.71


def test_build_execution_context_preserves_zero_values():
    sig = SignalResult(
        symbol="XRPUSDT",
        direction=SignalDirection.LONG,
        strategy="SmartMoneyConcepts",
        confidence=75.0,
        entry_low=1.0,
        entry_high=1.02,
        stop_loss=0.98,
        tp1=1.05,
        tp2=1.08,
        raw_data={
            "trigger_quality_score": 0.0,
            "spread_bps": 0.0,
            "whale_intent_bias": "BEARISH",
            "whale_buy_ratio": 0.0,
        },
    )

    execution_context = build_execution_context(
        sig,
        session_name="London Open",
        is_killzone=True,
        is_dead_zone=False,
        is_weekend=False,
        eq_zone="discount",
        eq_zone_depth=0.0,
        volume_context="NORMAL",
        volume_score=0.0,
    )

    assert execution_context["trigger"]["score"] == 0.0
    assert execution_context["liquidity"]["spread_bps"] == 0.0
    assert execution_context["whales"]["buy_ratio"] == 0.0
