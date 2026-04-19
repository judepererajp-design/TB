import inspect


def test_price_action_uses_configurable_key_level_and_mtf_controls():
    from strategies.price_action import PriceAction

    src = inspect.getsource(PriceAction)
    assert 'key_level_lookback' in src
    assert 'key_level_tolerance_atr' in src
    assert 'vol_confirmation_mult' in src
    assert 'reject_mtf_mismatch' in src
    assert 'mtf_mismatch_penalty' in src


def test_price_action_round_number_and_tp_ordering_guards_present():
    from strategies.price_action import PriceAction

    src = inspect.getsource(PriceAction)
    assert 'if current_price >= 1.0' in src
    assert '_exp = _log10 - 1' in src
    assert 'tp2 = max(tp2, tp1 + _tp_gap)' in src
    assert 'tp2 = min(tp2, tp1 - _tp_gap)' in src


def test_price_action_has_vol_rising_and_prior_bar_context():
    from strategies.price_action import PriceAction

    src = inspect.getsource(PriceAction)
    assert '_vol_rising' in src
    assert 'volumes[-2] > volumes[-3] > volumes[-4]' in src
    assert '_is_single_bar' in src
    assert '_prior_bar_confirms' in src
    # ATR-scaled context strength (not flat ±3)
    assert '_context_strength' in src
    assert '_context_adj' in src
    # Volume magnitude bonus inside rising-sequence block
    assert '1.5 * avg_vol' in src


def test_range_scalper_uses_bos_hold_and_absolute_stop_boundaries():
    from strategies.range_scalper import RangeScalperStrategy

    src = inspect.getsource(RangeScalperStrategy)
    assert 'range_high_abs' in src
    assert 'range_low_abs' in src
    assert 'bos_hold_bars' in src
    assert 'np.all(closes[-bos_hold_bars:]' in src
    assert 'if confidence < 45:' in src


def test_range_scalper_scaled_bounce_bonus():
    from strategies.range_scalper import RangeScalperStrategy

    src = inspect.getsource(RangeScalperStrategy)
    assert 'min(6, bounce_count * 2)' in src


def test_range_scalper_btc_breakout_gate():
    from strategies.range_scalper import RangeScalperStrategy

    src = inspect.getsource(RangeScalperStrategy)
    assert '_btc_momentum_abs' in src
    assert '_btc_breakout_thresh' in src
    # Soft graded penalty before hard block
    assert '_btc_soft_penalty' in src
    assert '_btc_soft_thresh' in src


def test_range_scalper_range_decay_detection():
    from strategies.range_scalper import RangeScalperStrategy

    src = inspect.getsource(RangeScalperStrategy)
    assert '_range_decaying' in src
    assert '_early_range' in src
    assert '_recent_range_local' in src
    assert '_atr_slow' in src


def test_reversal_divergence_uses_cached_rsi_series():
    from strategies.reversal import ReversalStrategy

    src = inspect.getsource(ReversalStrategy)
    assert '_calculate_rsi_series' in src
    assert 'rsi_series = self._calculate_rsi_series(closes, 14)' in src


def test_reversal_divergence_returns_float_strength():
    from strategies.reversal import ReversalStrategy
    import inspect as _inspect
    sig = _inspect.signature(ReversalStrategy._check_divergence)
    ann = sig.return_annotation
    # Should be annotated as float or at least not bool
    assert ann is not bool
    # Source should scale the bonus
    src = _inspect.getsource(ReversalStrategy._check_divergence)
    assert 'rsi_delta / MIN_RSI_DELTA' in src
    assert 'return 0.0' in src


def test_reversal_climactic_volume_bonus():
    from strategies.reversal import ReversalStrategy

    src = inspect.getsource(ReversalStrategy)
    assert 'vol_ratio > 3.0' in src
    assert 'Climactic volume' in src


def test_reversal_valid_regimes_excludes_volatile():
    from strategies.reversal import ReversalStrategy

    assert 'VOLATILE' not in ReversalStrategy.VALID_REGIMES
    assert 'CHOPPY' in ReversalStrategy.VALID_REGIMES


def test_reversal_has_configurable_sl_lookback_and_vol_mult():
    from strategies.reversal import ReversalStrategy

    src = inspect.getsource(ReversalStrategy)
    assert 'sl_bar_lookback' in src
    assert 'vol_confirmation_mult' in src
    assert 'allow_volatile' in src
