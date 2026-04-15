"""
Tests for 5-area enhancement — tests requiring real numpy/pandas.

These tests use subprocess to run in a clean Python process where
conftest.py's numpy/pandas mocks don't exist.  This ensures the
analyzer modules get real numpy arrays.

Each test function writes a mini test script to /tmp, runs it via
subprocess, and checks the exit code + stdout.
"""

import json
import subprocess
import sys
import os

import pytest

PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_test(script: str) -> tuple:
    """Run a Python script in a clean subprocess. Returns (ok, output)."""
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        timeout=30,
    )
    ok = result.returncode == 0
    output = result.stdout + result.stderr
    return ok, output


# ════════════════════════════════════════════════════════════════
# POWER ALIGNMENT v2 — COMPUTE LOGIC
# ════════════════════════════════════════════════════════════════

class TestPowerAlignmentV2Compute:

    def test_all_strong_returns_strong_tier(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
sig = MagicMock(); sig.direction = MagicMock(value='LONG')
s = ScoredSignal(base_signal=sig)
s.technical_score = 85; s.volume_score = 80; s.orderflow_score = 75
s.derivatives_score = 80; s.sentiment_score = 75; s.correlation_score = 50
s.regime = 'UNKNOWN'
aligned, reason, tier, score, boost = SignalAggregator._compute_power_alignment_v2(s)
assert aligned is True, f'Expected aligned, got {aligned}'
assert tier == 'STRONG', f'Expected STRONG, got {tier} (score={score:.1f})'
assert boost > 0, f'Expected boost > 0, got {boost}'
assert score >= 72, f'Expected score >= 72, got {score:.1f}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_moderate_scores_return_moderate_tier(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
sig = MagicMock(); sig.direction = MagicMock(value='LONG')
s = ScoredSignal(base_signal=sig)
s.technical_score = 68; s.volume_score = 65; s.orderflow_score = 58
s.derivatives_score = 68; s.sentiment_score = 62; s.correlation_score = 50
s.regime = 'UNKNOWN'
aligned, reason, tier, score, boost = SignalAggregator._compute_power_alignment_v2(s)
assert aligned is True, f'Expected aligned, got {aligned}'
assert tier in ('MODERATE', 'STRONG'), f'Expected MODERATE/STRONG, got {tier} (score={score:.1f})'
assert boost >= 0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_not_aligned_returns_empty(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
sig = MagicMock(); sig.direction = MagicMock(value='LONG')
s = ScoredSignal(base_signal=sig)
s.technical_score = 50; s.volume_score = 50; s.orderflow_score = 50
s.derivatives_score = 50; s.sentiment_score = 50; s.correlation_score = 50
aligned, reason, tier, score, boost = SignalAggregator._compute_power_alignment_v2(s)
assert aligned is False
assert tier == ''
assert boost == 0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_htf_concordance_bonus(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
sig = MagicMock(); sig.direction = MagicMock(value='LONG')
s = ScoredSignal(base_signal=sig)
s.technical_score = 65; s.volume_score = 65; s.orderflow_score = 55
s.derivatives_score = 65; s.sentiment_score = 60; s.correlation_score = 50
s.regime = 'UNKNOWN'
_, _, _, score_neutral, _ = SignalAggregator._compute_power_alignment_v2(s, htf_bias='NEUTRAL')
_, _, _, score_bullish, _ = SignalAggregator._compute_power_alignment_v2(s, htf_bias='BULLISH')
assert score_bullish > score_neutral, f'{score_bullish} should be > {score_neutral}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_htf_conflict_penalty(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
sig = MagicMock(); sig.direction = MagicMock(value='LONG')
s = ScoredSignal(base_signal=sig)
s.technical_score = 65; s.volume_score = 65; s.orderflow_score = 55
s.derivatives_score = 65; s.sentiment_score = 60; s.correlation_score = 50
s.regime = 'UNKNOWN'
_, _, _, score_neutral, _ = SignalAggregator._compute_power_alignment_v2(s, htf_bias='NEUTRAL')
_, _, _, score_bear, _ = SignalAggregator._compute_power_alignment_v2(s, htf_bias='BEARISH')
assert score_bear < score_neutral, f'{score_bear} should be < {score_neutral}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_scored_signal_v2_fields(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import ScoredSignal
s = ScoredSignal(base_signal=MagicMock())
assert hasattr(s, 'power_alignment_tier')
assert hasattr(s, 'power_alignment_score')
assert hasattr(s, 'power_alignment_boost')
assert s.power_alignment_tier == ''
assert s.power_alignment_score == 0.0
assert s.power_alignment_boost == 0
print('PASS')
""")
        assert ok, f"Failed: {out}"


# ════════════════════════════════════════════════════════════════
# TRIGGER QUALITY — ANALYZER LOGIC
# ════════════════════════════════════════════════════════════════

class TestTriggerQualityAnalyzer:

    def test_empty_triggers(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
result = trigger_quality_analyzer.analyze([])
assert result.quality_score == 0.0
assert result.quality_label == 'LOW'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_vol_confirmed_vs_unconfirmed(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
confirmed = [Trigger(name='t', category='structure', raw_strength=0.8, volume_confirmed=True)]
unconfirmed = [Trigger(name='t', category='structure', raw_strength=0.8, volume_confirmed=False)]
r_conf = trigger_quality_analyzer.analyze(confirmed)
r_unconf = trigger_quality_analyzer.analyze(unconfirmed)
assert r_conf.effective_trigger_count > r_unconf.effective_trigger_count, \\
    f'{r_conf.effective_trigger_count} should be > {r_unconf.effective_trigger_count}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_diversity_bonus(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
same = [
    Trigger(name='a', category='momentum', raw_strength=0.7, volume_confirmed=True),
    Trigger(name='b', category='momentum', raw_strength=0.6, volume_confirmed=True),
]
diverse = [
    Trigger(name='a', category='momentum', raw_strength=0.7, volume_confirmed=True),
    Trigger(name='b', category='structure', raw_strength=0.6, volume_confirmed=True),
]
r_same = trigger_quality_analyzer.analyze(same)
r_diverse = trigger_quality_analyzer.analyze(diverse)
assert r_diverse.diversity_bonus >= r_same.diversity_bonus
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_low_volume_detection(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
import numpy as np
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
triggers = [Trigger(name='t', category='structure', raw_strength=0.8)]
low_vols = np.array([200.0] * 15 + [100.0] * 4 + [5.0])
r = trigger_quality_analyzer.analyze(triggers, volumes=low_vols)
assert r.volume_context == 'LOW_VOL', f'Expected LOW_VOL, got {r.volume_context}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_climactic_volume_reversal(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
import numpy as np
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
triggers = [Trigger(name='t', category='structure', raw_strength=0.8)]
volumes = np.array([100.0] * 19 + [500.0])
closes = np.array([100.0 + i * 0.5 for i in range(20)])
closes[-1] = closes[-3] - 2.0  # Reversal
r = trigger_quality_analyzer.analyze(triggers, volumes=volumes, closes=closes, atr=1.0)
assert r.volume_context == 'CLIMACTIC', f'Expected CLIMACTIC, got {r.volume_context}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_fast_move_detection(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
import numpy as np
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
triggers = [Trigger(name='t', category='structure', raw_strength=0.8)]
volumes = np.array([100.0] * 19 + [300.0])
closes = np.array([100.0] * 19 + [101.0])
r = trigger_quality_analyzer.analyze(triggers, volumes=volumes, closes=closes, atr=1.0)
assert r.fast_move is True, f'Expected fast_move=True, got {r.fast_move}'
assert r.confidence_delta > 0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_max_useful_triggers(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.trigger_quality import trigger_quality_analyzer, Trigger
from config.constants import TriggerQuality as TQ
t_max = [Trigger(name=f't{i}', category='structure', raw_strength=0.8, volume_confirmed=True)
         for i in range(TQ.MAX_USEFUL_TRIGGERS)]
t_extra = t_max + [Trigger(name=f'e{i}', category='structure', raw_strength=0.8, volume_confirmed=True)
                    for i in range(5)]
r_max = trigger_quality_analyzer.analyze(t_max)
r_extra = trigger_quality_analyzer.analyze(t_extra)
assert abs(r_max.effective_trigger_count - r_extra.effective_trigger_count) < 0.01
print('PASS')
""")
        assert ok, f"Failed: {out}"


# ════════════════════════════════════════════════════════════════
# SPOOFING / FAKE WALL DETECTION
# ════════════════════════════════════════════════════════════════

class TestSpoofDetector:

    def test_new_walls_suspicious(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import SpoofDetector
d = SpoofDetector()
r = d.analyze_walls([(99.0, 1000.0)], [(101.0, 1000.0)], 100.0, avg_size=100.0)
assert isinstance(r.overall_spoof_score, float)
assert r.overall_spoof_score >= 0.0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_persistent_walls_trusted(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import SpoofDetector
d = SpoofDetector()
for _ in range(5):
    r = d.analyze_walls([(99.0, 500.0)], [(101.0, 500.0)], 100.0, avg_size=100.0)
assert r.overall_spoof_score < 1.0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_symmetric_walls_not_fake(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import SpoofDetector
d = SpoofDetector()
for _ in range(5):
    r = d.analyze_walls([(99.0, 500.0)], [(101.0, 500.0)], 100.0, avg_size=100.0)
assert 99.0 not in r.fake_bid_walls
assert 101.0 not in r.fake_ask_walls
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_size_anomaly_near_price_flagged(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import SpoofDetector
from config.constants import SpoofingDetection as SD
d = SpoofDetector()
wall_price = 100.0 * (1 - SD.PULL_DISTANCE_PCT * 0.5)
r = d.analyze_walls([(wall_price, 100.0 * SD.SIZE_ANOMALY_MULT + 1)], [], 100.0, avg_size=100.0)
assert r.overall_spoof_score > 0, f'Expected spoof_score > 0, got {r.overall_spoof_score}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_wall_tier_classification(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import WallAnalysis
a = WallAnalysis()
a.persistent_bid_walls = [99.0]
tier = a.get_wall_tier(99.0, 100.0)
assert 'persistent' in tier
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_orderflow_has_spoof_detector(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.orderflow import orderflow_analyzer
assert hasattr(orderflow_analyzer, 'get_spoof_detector')
d = orderflow_analyzer.get_spoof_detector()
assert d is not None
print('PASS')
""")
        assert ok, f"Failed: {out}"


# ════════════════════════════════════════════════════════════════
# VOLUME QUALITY — ANALYZER LOGIC
# ════════════════════════════════════════════════════════════════

class TestVolumeQualityAnalyzer:

    def test_normal_volume_quality(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
ohlcv = [[i*3600, 99.9, 100.3, 99.7, 100.0 + (i%3)*0.1, 1000.0] for i in range(30)]
r = volume_analyzer.assess_volume_quality(ohlcv, 100.0)
assert r.quality_label in ('LOW', 'MEDIUM', 'HIGH'), f'Got {r.quality_label}'
assert 0 <= r.quality_score <= 100
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_high_volume_breakout(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
ohlcv = [[i*3600, 100+i*0.4, 100+i*0.5+0.3, 100+i*0.5-0.3, 100+i*0.5, 1000.0] for i in range(30)]
ohlcv[-1][5] = 5000.0  # 5x volume spike
r = volume_analyzer.assess_volume_quality(ohlcv, 115.0, trade_type='breakout')
assert r.quality_score > 30
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_empty_data_returns_default(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
r = volume_analyzer.assess_volume_quality([], 100.0)
assert r.quality_score == 50.0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_insufficient_data_returns_default(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
ohlcv = [[i*3600, 99.9, 100.3, 99.7, 100.0, 1000.0] for i in range(5)]
r = volume_analyzer.assess_volume_quality(ohlcv, 100.0)
assert r.quality_score == 50.0
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_breadth_scoring(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
# Even volume
ohlcv_even = [[i*3600, 99.9, 100.3, 99.7, 100.0 + 0.01*i, 1000.0] for i in range(30)]
r_even = volume_analyzer.assess_volume_quality(ohlcv_even, 100.0)
# Uneven volume
ohlcv_uneven = [[i*3600, 99.9, 100.3, 99.7, 100.0 + 0.01*i,
                  10.0 if i%2==0 else 10000.0] for i in range(30)]
r_uneven = volume_analyzer.assess_volume_quality(ohlcv_uneven, 100.0)
assert r_uneven.breadth_score <= r_even.breadth_score, \\
    f'uneven={r_uneven.breadth_score} should be <= even={r_even.breadth_score}'
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_volume_quality_result_defaults(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import VolumeQualityResult
r = VolumeQualityResult()
assert r.quality_score == 50.0
assert r.quality_label == 'MEDIUM'
assert r.notes == []
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_low_volume_poor_quality(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from analyzers.volume import volume_analyzer
ohlcv = [[i*3600, 99.9, 100.3, 99.7, 100.0, 100.0] for i in range(30)]  # Very low vol
r = volume_analyzer.assess_volume_quality(ohlcv, 100.0)
assert r.quality_score <= 60
print('PASS')
""")
        assert ok, f"Failed: {out}"


# ════════════════════════════════════════════════════════════════
# V1 BACKWARD COMPAT
# ════════════════════════════════════════════════════════════════

class TestV1PowerAlignmentCompat:

    def test_v1_aligned(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
s = ScoredSignal(base_signal=MagicMock())
s.technical_score = 70; s.volume_score = 65; s.orderflow_score = 60
s.derivatives_score = 68; s.sentiment_score = 60; s.correlation_score = 50
aligned, reason = SignalAggregator._compute_power_alignment(s)
assert aligned is True, f'Expected aligned=True'
assert '5/5' in reason
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_v1_not_aligned(self):
        ok, out = _run_test("""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from unittest.mock import MagicMock
from signals.aggregator import SignalAggregator, ScoredSignal
s = ScoredSignal(base_signal=MagicMock())
s.technical_score = 50; s.volume_score = 50; s.orderflow_score = 50
s.derivatives_score = 50; s.sentiment_score = 50; s.correlation_score = 50
aligned, reason = SignalAggregator._compute_power_alignment(s)
assert aligned is False
assert reason == ''
print('PASS')
""")
        assert ok, f"Failed: {out}"


# ════════════════════════════════════════════════════════════════
# LIVE PATH WIRING
# ════════════════════════════════════════════════════════════════

class TestAggregatorEnhancementWiring:

    def test_process_wires_trigger_whale_and_volume_quality(self):
        ok, out = _run_test("""
import asyncio, os, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')

from analyzers.altcoin_rotation import rotation_tracker
from analyzers.derivatives import derivatives_analyzer
from analyzers.regime import Regime, regime_analyzer
from analyzers.wallet_behavior import AdvancedWhaleIntent, wallet_profiler
from signals import aggregator as agg_mod
from signals.aggregator import SignalAggregator
from strategies.base import SignalDirection, SignalResult

regime_analyzer._regime = Regime.BULL_TREND
regime_analyzer._fear_greed = 42
regime_analyzer._session = SimpleNamespace(value='LONDON')
regime_analyzer.is_killzone = lambda: (False, 0.0)
regime_analyzer.get_min_confidence_override = lambda: None
rotation_tracker.get_signal_adjustment = lambda symbol, direction: (0.0, '')
rotation_tracker.get_sector_for_symbol = lambda symbol: 'L1'
rotation_tracker.get_rotation_summary = lambda: {'hot': [], 'cold': []}
derivatives_analyzer.analyze = AsyncMock(return_value=None)
agg_mod.api.fetch_order_book = AsyncMock(return_value={})
wallet_profiler.get_advanced_intent = lambda direction: AdvancedWhaleIntent(
    intent='ACCUMULATION',
    directional_bias='BULLISH',
    confidence=0.90,
    confidence_delta=5,
    notes=['🐋 Smart accumulation detected'],
)

ohlcv = []
for i in range(30):
    base = 100 + i * 0.6
    vol = 1000.0 if i < 29 else 4200.0
    ohlcv.append([i * 3600, base - 0.2, base + 0.4, base - 0.4, base, vol])

signal = SignalResult(
    symbol='TESTUSDT',
    direction=SignalDirection.LONG,
    strategy='InstitutionalBreakout',
    confidence=84.0,
    entry_low=117.8,
    entry_high=118.2,
    stop_loss=115.5,
    tp1=121.0,
    tp2=124.0,
    tp3=127.0,
    rr_ratio=2.6,
    confluence=[
        '✅ Donchian breakout above 118.0 (20H high)',
        '✅ Volume: 3.0x average',
        '✅ ADX: 31.0 (trend strength confirmed)',
    ],
    raw_data={
        'ohlcv_1h': ohlcv,
        'vol_ratio': 3.0,
    },
)

agg = SignalAggregator()
agg._min_confidence = 0
agg._max_per_hour = 999
agg._deduplicator.check_and_mark = AsyncMock(return_value=False)
agg._persist_hourly_counter = lambda: None

scored = asyncio.run(agg.process(signal, raw_signals=[]))
assert scored is not None, 'Expected scored signal'
assert signal.raw_data['trigger_quality_label'] in ('MEDIUM', 'HIGH')
assert signal.raw_data['whale_intent'] == 'ACCUMULATION'
assert signal.raw_data['volume_quality_label'] in ('MEDIUM', 'HIGH')
assert signal.raw_data['volume_trade_type'] == 'breakout'
assert any('Whale intent:' in note for note in signal.confluence), signal.confluence
assert scored.sentiment_score > 50.0, scored.sentiment_score
print('PASS')
""")
        assert ok, f"Failed: {out}"

    def test_volume_trade_type_mapping_uses_regime_trade_type(self):
        ok, out = _run_test("""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault('OPENROUTER_API_KEY', 'test')
from signals.aggregator import SignalAggregator
from strategies.base import SignalDirection, SignalResult

signal = SignalResult(
    symbol='TESTUSDT',
    direction=SignalDirection.LONG,
    strategy='PriceAction',
    confidence=80.0,
    entry_low=100.0,
    entry_high=101.0,
    stop_loss=98.0,
    tp1=103.0,
    tp2=105.0,
    rr_ratio=2.0,
    raw_data={'regime_trade_type': 'LOCAL_CONTINUATION_LONG'},
)

trade_type = SignalAggregator._infer_volume_trade_type(signal)
assert trade_type == 'pullback', trade_type
print('PASS')
""")
        assert ok, f"Failed: {out}"
