"""
TitanBot Pro — Named Constants
================================
Every tunable numeric literal in the codebase lives here.
No magic numbers anywhere — every threshold, penalty, buffer size,
and timing constant is defined once with a clear name and docstring.

Usage:
    from config.constants import Grading, Timing, Risk, ...

Organised by subsystem so you can find anything in seconds.

RULE: if a number appears in Python code and isn't 0, 1, or a trivial
index, it MUST be defined here (or in settings.yaml for user-tunable
values). Grep for bare numeric literals to audit compliance.
"""

# ════════════════════════════════════════════════════════════════
# 1. SIGNAL GRADING & CONFIDENCE
# ════════════════════════════════════════════════════════════════

class Grading:
    """Grade thresholds and confidence gates."""

    # Grade classification boundaries (aggregator + publisher)
    APLUS_MIN_CONF: int = 88        # A+ grade floor
    A_MIN_CONF: int = 80            # A grade floor
    B_MIN_CONF: int = 72            # B grade floor (below = C)
    NEUTRAL_SCORE: float = 50.0     # Default analyzer score (0-100 range)
    ZERO_DATA_MARGIN: float = 2.0   # Reject if ALL scores within NEUTRAL ± this margin

    # Pre-grade estimation (engine, before full scoring)
    PREGRADE_A_PLUS_THRESHOLD: int = 85
    PREGRADE_A_THRESHOLD: int = 75
    A_GRADE_ESTIMATE_RAW_CONF: int = 78   # Raw conf proxy for A-grade in daily limit

    # Adaptive confidence floor
    # Raised from 20 → 50: with only 20 samples, the 40th-percentile floor can jump
    # +8 pts in the first minute of a run, killing valid signals like DOT/AVAX/M.
    ADAPTIVE_FLOOR_MIN_HISTORY: int = 50  # Min scores before adaptive calc
    ADAPTIVE_FLOOR_PERCENTILE: int = 40   # np.percentile target
    ADAPTIVE_FLOOR_MAX_CAP: int = 68      # Never raise floor above this
    FLOOR_ABSOLUTE_MINIMUM: int = 55      # Safety — never below 55
    FLOOR_PROPOSAL_MARGIN: int = 6        # current_min - margin triggers proposal
    FLOOR_PROPOSAL_REDUCTION: int = 3     # Proposed reduction points

    # A+ path requirements (aggregator _grade_signal)
    APLUS_PATH1_DERIVATIVES_MIN: int = 65
    APLUS_PATH1_VOLUME_MIN: int = 60
    APLUS_PATH2_CONF_MIN: int = 90
    APLUS_PATH2_DERIVATIVES_MIN: int = 70
    APLUS_PATH2_VOLUME_MIN: int = 65
    APLUS_PATH3_CONF_MIN: int = 85
    APLUS_PATH3_CONFLUENCE_MULT_MIN: float = 1.15

    # Power Alignment badge — cross-category strength detection
    # Badge fires when a signal has meaningful confirmation from
    # multiple independent categories (structure + participation +
    # positioning + flow/context).  This is orthogonal to grade:
    # a B-grade signal can be power-aligned if the individual scores
    # are all solid even though overall confidence didn't reach A+.
    POWER_ALIGN_STRUCTURE_MIN: float = 55.0    # technical score
    POWER_ALIGN_VOLUME_MIN: float = 55.0       # volume score
    POWER_ALIGN_DERIVATIVES_MIN: float = 55.0  # derivatives score
    POWER_ALIGN_ORDERFLOW_MIN: float = 50.0    # orderflow score (noisier, lower bar)
    POWER_ALIGN_MIN_CATEGORIES: int = 4        # need ≥4 of 5 categories above thresholds

    # Power Alignment v2 — weighted scoring, HTF concordance, strength tiers
    # Weights for each category in the composite power score (sum to 1.0)
    POWER_ALIGN_WEIGHT_STRUCTURE: float = 0.25
    POWER_ALIGN_WEIGHT_VOLUME: float = 0.20
    POWER_ALIGN_WEIGHT_DERIVATIVES: float = 0.20
    POWER_ALIGN_WEIGHT_ORDERFLOW: float = 0.20
    POWER_ALIGN_WEIGHT_CONTEXT: float = 0.15
    # HTF concordance bonus: signal direction agrees with HTF bias
    POWER_ALIGN_HTF_CONCORDANCE_BONUS: float = 10.0
    POWER_ALIGN_HTF_CONFLICT_PENALTY: float = -8.0
    # Strength tier thresholds (composite power score 0-100)
    POWER_ALIGN_TIER_STRONG: float = 72.0      # ⚡⚡ STRONG — all dimensions firing
    POWER_ALIGN_TIER_MODERATE: float = 60.0    # ⚡ MODERATE — solid cross-category
    # Regime-aware threshold adjustments
    POWER_ALIGN_CHOPPY_PENALTY: float = 5.0    # Raise thresholds in CHOPPY regime
    POWER_ALIGN_TREND_BONUS: float = -3.0      # Lower thresholds in TREND regimes (with-trend)
    # Confidence boost for power-aligned signals
    POWER_ALIGN_CONF_BOOST_STRONG: int = 5     # +5 conf for STRONG alignment
    POWER_ALIGN_CONF_BOOST_MODERATE: int = 3   # +3 conf for MODERATE alignment

    # MTF (Multi-Timeframe) Structure Alignment
    # 4H (HTF) sets directional bias, 1H (MTF) confirms structure, 15m (LTF) times entry
    MTF_ALIGNMENT_BONUS: float = 5.0           # All three timeframes agree
    MTF_CONFLICT_PENALTY: float = -3.0         # MTF structure opposes HTF/LTF direction
    MTF_WEIGHT_HTF: float = 0.50               # 4H direction weight
    MTF_WEIGHT_MTF: float = 0.30               # 1H structure weight
    MTF_WEIGHT_LTF: float = 0.20               # Entry-timeframe weight


# ════════════════════════════════════════════════════════════════
# 2. RATE LIMITING & DEDUPLICATION
# ════════════════════════════════════════════════════════════════

class RateLimiting:
    """Rate limiting and burst throttle constants."""

    RATE_LIMIT_BYPASS_MIN_RR: float = 2.5     # RR floor for hourly cap bypass
    CLARITY_BYPASS_MIN_SCORE: int = 95        # Clarity score floor for hourly cap bypass
    CLARITY_BYPASS_MIN_CONF: int = 80         # Confidence floor for clarity bypass
    CLARITY_BYPASS_MIN_RR: float = 2.0        # RR floor for clarity bypass
    SLOT_REUSE_COOLDOWN_SECS: int = 10        # Seconds before released slot reuse
    PUBLISHER_MAX_PER_WINDOW: int = 8         # Max signals per burst window
    PUBLISHER_WINDOW_SECS: int = 600          # 10-minute burst window

    # Deduplication
    CONF_BREAKTHROUGH_PTS: float = 20.0       # Points needed to resend deduped signal
    POST_EXPIRY_BREAKTHROUGH_PTS: float = 20.0  # Post-expiry resend threshold


# ════════════════════════════════════════════════════════════════
# 3. TIMING & INTERVALS
# ════════════════════════════════════════════════════════════════

class Timing:
    """All timing constants in seconds unless noted."""

    # Engine
    WARMUP_SECS: int = 60                         # Initial regime observation period
    STRATEGY_HEALTH_ALERT_SECS: int = 3600 * 6    # 6h silence before alert

    # Post-startup extension filter
    # After warmup completes, block signals for POST_STARTUP_FILTER_SECS where the
    # recent 4-bar 1h move is ≥ POST_STARTUP_EXT_ATR_MULT × ATR (or ≥ the floor %)
    # in the signal direction.  Prevents "catch-up" signals fired at already-extended
    # prices when the bot restarts mid-move.
    POST_STARTUP_FILTER_SECS:    int   = 900   # 15 minutes post-warmup filter window
    POST_STARTUP_EXT_ATR_MULT:   float = 4.0   # ATR multiplier threshold
    POST_STARTUP_EXT_FLOOR_PCT:  float = 6.0   # Minimum extension % to trigger filter
    REGIME_MAX_AGE_SECS: int = 600                 # 10min → recompute regime
    REGIME_STALE_PENALTY_WINDOW_SECS: int = 3600   # Penalty ramp window

    # Execution engine
    EXECUTION_WATCHING_TIMEOUT_SECS: int = 14400   # 4h default fallback
    EXECUTION_ALMOST_TIMEOUT_SECS: int = 7200      # 2h almost-ready timeout
    EXECUTION_CHECK_INTERVAL_SECS: int = 15        # Price check frequency
    EXECUTION_FAST_CHECK_INTERVAL_SECS: int = 3    # Hot/near-entry polling cadence
    EXECUTION_WATCHING_BY_SETUP: dict = {
        'scalp':       5_400,     # 1.5h
        'intraday':    7_200,     # 2.0h
        'swing':      28_800,     # 8.0h
        'positional': 43_200,     # 12.0h
    }
    # Opposite-direction cooldown after publication. Faster setups can flip sooner,
    # while swing/positional setups need longer cooldown before an unsupported
    # opposite-side signal is considered trustworthy.
    OPPOSITE_SIGNAL_COOLDOWN_BY_SETUP: dict = {
        'scalp':       1_800,     # 30m
        'intraday':    5_400,     # 90m
        'swing':      14_400,     # 4h
        'positional': 28_800,     # 8h
    }

    # Outcome monitor
    OUTCOME_CHECK_INTERVAL_SECS: int = 15      # SL/TP check frequency
    PENDING_ENTRY_TIMEOUT_SECS: int = 7200     # 2h to reach entry zone
    SIGNAL_MAX_LIFETIME_SECS: int = 172800     # 48h max signal life
    API_FALLBACK_THROTTLE_SECS: float = 60.0   # Min time between API fallback fetches

    # Max hold times per setup class — prevent scalps from becoming accidental
    # position trades. Swing is 72h to allow wider ATR-scaled TPs to be reached.
    MAX_HOLD_BY_SETUP: dict = {
        'scalp':       4  * 3600,    # 4h
        'intraday':   12  * 3600,    # 12h
        'swing':      72  * 3600,    # 72h (was 48h — extended for wider TPs)
        'positional':  0,            # 0 = no limit
    }

    # Trailing stop parameters (used after TP1 hit)
    TRAIL_START_FRAC: float = 0.55   # Initial trail distance as fraction of risk
    TRAIL_TP2_FRAC: float = 0.80     # Trail tightens to this at TP2
    TRAIL_CAP_FRAC: float = 0.90     # Maximum trail tightness

    # Diagnostic engine
    DIAGNOSTIC_REPORT_INTERVAL_SECS: float = 21600.0   # 6h between reports
    EXECUTION_FUNNEL_AUDIT_SECS: float = 7200.0        # 2h audit interval
    STRATEGY_REGIME_AUDIT_SECS: float = 14400.0        # 4h audit interval
    PROPOSAL_COOLDOWN_SECS: int = 7200                 # 2h between same proposals
    SNOOZE_DEFAULT_MINUTES: int = 60                   # Default approval snooze

    # Health monitor watchdog
    WATCHDOG_INTERVAL_SECS: int = 300          # 5-minute watchdog loop

    # System
    PENDING_SIGNAL_TIMEOUT_SECS: int = 14400   # 4h pending signal timeout
    HEALTH_CHECK_MIN_UPTIME_SECS: int = 14400  # 4h before health check fires


# ════════════════════════════════════════════════════════════════
# 4. ENGINE PENALTIES & MULTIPLIERS
# ════════════════════════════════════════════════════════════════

class Penalties:
    """Confidence adjustments and multipliers applied by the engine."""

    # Counter-trend
    COUNTER_TREND_CONF_MULT: float = 0.80      # ×0.80 when HTF ADX confirms trend
    ADX_COUNTER_TREND_MIN: int = 25             # ADX threshold for counter-trend penalty

    # Time filter
    TIME_PENALTY_ALIGNED_PTS: int = 8           # Off-hours penalty (trend-aligned)
    TIME_PENALTY_COUNTER_PTS: int = 15          # Off-hours penalty (counter-trend)

    # Chop filter
    CHOP_RANGE_THRESHOLD: float = 0.55          # Chop strength → range-bound
    CHOP_MID_RANGE_PENALTY_PTS: int = 10        # Soft penalty for mid-range

    # Whale opposition
    WHALE_OPPOSED_CONF_MULT: float = 0.92       # ×0.92 when whales oppose direction

    # Market state engine
    MSE_CONF_SCALE_MIN: float = 0.5
    MSE_CONF_SCALE_MAX: float = 1.0
    MSE_PENALTY_MIN_PTS: int = 4
    MARKET_STATE_DEATH_THRESHOLD: int = 65      # Kill signal if conf < this after MSE

    # Sentiment
    SENTIMENT_DELTA_MAX: int = 3                # Max ±3 points from sentiment
    SENTIMENT_CONF_GATE: int = 70               # Only apply above this conf
    SENTIMENT_HEADLINES_GATE: int = 3            # Or if ≥3 relevant headlines
    SENTIMENT_SCORE_CENTER: int = 50             # Center of sentiment scoring
    SENTIMENT_SCORE_DIVISOR: int = 10            # Normalisation divisor

    # Sentiment extremes (aggregator scoring)
    SENTIMENT_EXTREME_FEAR_LONG_BOOST: int = 20
    SENTIMENT_MODERATE_FEAR_BOOST: int = 8
    SENTIMENT_EXTREME_GREED_LONG_PENALTY: int = 10
    SENTIMENT_EXTREME_GREED_SHORT_BOOST: int = 15
    SENTIMENT_MODERATE_GREED_BOOST: int = 5
    SENTIMENT_EXTREME_FEAR_SHORT_PENALTY: int = 10

    # Dangerous event
    DANGEROUS_EVENT_LOOKAHEAD_HOURS: int = 48
    DANGEROUS_EVENT_PENALTY_PTS: int = 8

    # Regime staleness
    REGIME_STALE_PENALTY_FLOOR: float = 0.85

    # SMC veto
    SMC_VETO_PENALTY_PTS: int = 12              # Confidence penalty for SMC conflict
    SMC_VETO_CONF_FLOOR: int = 30               # Minimum conf after veto

    # RR sanity
    RR_SANITY_CAP: float = 6.0                  # Cap display at 6R (matches risk.max_rr default)
    RR_CORRECTION_TOLERANCE: float = 0.3        # Log if verified RR differs by >0.3

    # TP clamping
    TP_NEGATIVE_FLOOR_PCT: float = 0.001        # 0.1% of entry for negative TP clamp
    TP3_MAX_ATR_MULT: float = 7.5               # Max TP3 distance = 7.5 × ATR (catches unbounded structural TPs)

    # TP1 swing-level anchoring
    TP1_SWING_DEAD_ZONE_ATR: float = 0.3        # Ignore swing levels within this × ATR of entry (noise filter)
    TP1_MAX_SWING_LEVELS: int = 5               # Number of recent swing highs/lows to consider for TP1 capping

    # Stalker
    STALKER_QUALITY_THRESHOLD: int = 80         # Min score for watchlist alert

    # Alpha model maturity
    ALPHA_BETA_MATURITY_THRESHOLD: int = 24     # Alpha+Beta sum for Bayesian maturity


class LocalRange:
    """
    Local symbol range awareness — prevents overextended TPs when a symbol is
    range-bound but BTC macro regime says BULL_TREND / BEAR_TREND.

    When the symbol's own 50-bar range is tight (< RANGE_PCT_THRESHOLD of price),
    the symbol is likely in accumulation/distribution and TP2/TP3 should not be
    pushed to trend-extension levels.
    """
    # A symbol whose 50-bar high/low range is within this % of price is "ranging"
    RANGE_PCT_THRESHOLD: float = 0.15           # 15% — tight box / accumulation
    # Soft cap: TP2 cannot exceed range_high/low by more than this fraction of range
    TP2_RANGE_OVERSHOOT: float = 0.10           # 10% beyond range boundary
    # Soft cap: TP3 cannot exceed range boundary by more than this fraction of range
    TP3_RANGE_OVERSHOOT: float = 0.25           # 25% beyond range boundary
    # Soft EQ penalty: max confidence reduction for zone mismatch in trending regimes
    SOFT_EQ_MAX_PENALTY: float = 0.10           # 10% max confidence reduction
    # Minimum chop_strength before local range caps kick in (prevents false positives
    # when the trend is strong and the range is just incidental consolidation)
    MIN_CHOP_FOR_LOCAL: float = 0.10


# ════════════════════════════════════════════════════════════════
# 5. RISK & PORTFOLIO
# ════════════════════════════════════════════════════════════════

class Portfolio:
    """Portfolio engine constants."""

    MAX_TOTAL_RISK_PCT: float = 0.06        # Max 6% of capital at risk
    MAX_POSITION_RISK_PCT: float = 0.015    # Max 1.5% per position
    MAX_POSITIONS: int = 8                  # Max concurrent positions
    MAX_SECTOR_EXPOSURE_PCT: float = 0.25   # Max 25% in one sector
    MAX_DIRECTION_IMBALANCE: float = 0.7    # Max 70% net long or short
    # Must match risk.max_correlated_positions in settings.yaml; the correlation
    # analyzer reads the YAML value directly while portfolio_engine reads this
    # constant. Keep them aligned or the two gates will disagree.
    MAX_CORRELATED_POSITIONS: int = 2       # Max positions correlation > 0.7
    MAX_SAME_DIRECTION: int = 4             # V10 correlation gate
    MAX_SAME_SECTOR: int = 2               # V10 correlation gate
    TARGET_PORTFOLIO_VOL: float = 0.02      # Target 2% daily vol
    VOL_WINDOW_DAYS: int = 30               # Days for vol calc
    DEFAULT_SYMBOL_VOLATILITY: float = 0.03 # Baseline crypto vol assumption
    MIN_RISK_ALLOCATION: float = 0.001      # Floor per position
    MIN_POSITION_SIZE_USDT: float = 25.0    # Reject undersized positions likely to fail exchange minimums
    MAX_CORRELATED_CLUSTER_SHARE: float = 0.75  # Correlated cluster cannot exceed 75% of portfolio risk budget
    # Bayesian prior win rate (0.5 = uninformative). Used as cold-start default
    # before any decisive trades are recorded. Conservative, prevents oversizing.
    PRIOR_WIN_RATE: float = 0.50

    # Kelly regime multipliers
    KELLY_MULT_VOLATILE_PANIC: float = 0.25
    KELLY_MULT_VOLATILE: float = 0.50
    KELLY_MULT_CHOPPY: float = 0.65
    KELLY_MULT_COUNTER_TREND: float = 0.50          # BEAR_TREND+LONG and BULL_TREND+SHORT
    KELLY_MULT_BULL_TREND: float = 0.85             # BULL_TREND+LONG below high-conf threshold
    BULL_TREND_HIGH_CONF_THRESHOLD: float = 0.65    # p_win for full Kelly (BULL_TREND+LONG only)

    # Kelly setup-class multipliers
    # Scalps have shorter hold times, tighter stops, and higher rate-of-loss than swings.
    # Sizing identically to swings over-allocates risk to scalp setups which fail more often
    # due to micro-structure noise. Positional setups have the widest structural backing.
    KELLY_MULT_SETUP_SCALP: float = 0.80       # Reduced: scalps are noisy / short-duration
    KELLY_MULT_SETUP_INTRADAY: float = 1.00    # Baseline — no change
    KELLY_MULT_SETUP_SWING: float = 1.05       # Slight reward: wider stop = cleaner signal
    KELLY_MULT_SETUP_POSITIONAL: float = 1.10  # Highest conviction, longest-validated setup


# ════════════════════════════════════════════════════════════════
# 6. HEALTH MONITOR
# ════════════════════════════════════════════════════════════════

class HealthMonitor:
    """Signal and system sanity check thresholds."""

    # Signal sanity
    MAX_TP_RATIO: float = 5.0          # TP must not exceed 5x entry
    MAX_SL_RATIO: float = 3.0          # SL must not exceed 3x from entry
    MAX_RR_RATIO: float = 25.0         # R/R > 25 is almost certainly wrong
    MIN_RR_RATIO: float = 0.3          # R/R < 0.3 not worth trading
    MAX_ENTRY_WIDTH: float = 0.08      # Entry zone max 8% of price
    MIN_ENTRY_WIDTH: float = 0.0001    # Entry zone min 0.01%

    # OHLCV sanity
    MAX_CANDLE_AGE_HOURS: float = 4.5  # Binance 4h candles arrive at ~3.1h
    MIN_CANDLES: int = 50              # Minimum candles required

    # Watchdog
    MAX_ASYNCIO_TASKS: int = 500       # Async task warning threshold
    MAX_PRICE_CACHE: int = 1000        # Price cache subscriber limit
    MAX_OUTCOME_ACTIVE: int = 200      # Active signal tracking limit

    # Rate limit tracking
    RATE_LIMIT_WARN: int = 5           # Warn after 5 rate limits/hour
    RATE_LIMIT_CRIT: int = 15          # Critical after 15/hour

    # Price validation
    PRICE_TOLERANCE: float = 0.30      # 30% drift tolerance
    CANDLE_ANOMALY_THRESHOLD: float = 0.05  # Wick/body anomaly ratio

    # Memory (MB) — alert at the lower threshold, warn escalates at the upper
    MEMORY_WARNING_MB: int = 800    # first (soft) warning threshold
    MEMORY_ALERT_MB: int = 1500     # hard alert / escalation threshold


# ════════════════════════════════════════════════════════════════
# 7. DIAGNOSTIC ENGINE
# ════════════════════════════════════════════════════════════════

class Diagnostics:
    """Diagnostic engine buffer sizes, thresholds, and analysis params."""

    # Buffer sizes (memory management)
    DEATH_LOG_MAX_SIZE: int = 1000
    OUTCOME_LOG_MAX_SIZE: int = 500
    TELEGRAM_LOG_MAX_SIZE: int = 200
    SIGNAL_CARD_LOG_MAX_SIZE: int = 100
    API_LATENCY_BUFFER_SIZE: int = 100
    NARRATIVE_QUALITY_LOG_MAX_SIZE: int = 50
    SCAN_TIMINGS_MAX_SIZE: int = 500
    DB_TIMINGS_MAX_SIZE: int = 200

    # Error rate
    ERROR_RATE_WINDOW_SECS: int = 60
    ERROR_RATE_MAX_ENTRIES: int = 10_000
    ERROR_RATE_CIRCUIT_BREAKER: int = 100      # Errors/min for circuit breaker

    # Telegram
    TELEGRAM_CHAR_WARNING: int = 4000          # Close to 4096 limit
    TELEGRAM_LONG_MSG_THRESHOLD: int = 3800    # Near-limit warning

    # Verdict
    VERDICT_MIN_LENGTH: int = 20               # Min chars for a useful verdict

    # Signal drought analysis
    SIGNAL_DROUGHT_WINDOW_SECS: int = 14400    # 4h window
    DROUGHT_MIN_DEATH_COUNT: int = 10
    DROUGHT_MIN_SCAN_COUNT: int = 500
    DROUGHT_STATISTICAL_MIN_DEATHS: int = 30
    KILL_REASON_DOMINANCE_PCT: int = 70        # % threshold for pattern detection

    # Dead zone (low liquidity hours UTC)
    DEAD_ZONE_START_HOUR: int = 3
    DEAD_ZONE_END_HOUR: int = 7

    # Edge case detection
    EDGE_CASE_CHECK_WINDOW_SECS: int = 3600    # 1h
    SYMBOL_REPEAT_THRESHOLD: int = 10          # Auto-exclusion trigger
    STRATEGY_ERROR_CHECK_WINDOW_SECS: int = 3600
    STRATEGY_ERROR_COUNT_THRESHOLD: int = 20
    STRATEGY_ERROR_WARNING_THRESHOLD: int = 50

    # Outcome analysis
    RECENT_OUTCOMES_WINDOW_SECS: int = 86400   # 24h
    UNDERPERF_STRATEGY_WR_FLOOR: float = 0.35
    HEALTHY_STRATEGY_WR_CEILING: float = 0.60
    CONFIDENCE_BUCKET_SIZE: int = 5
    POOR_CONF_BUCKET_WR_FLOOR: float = 0.40
    POOR_CONF_BUCKET_MAX: int = 80

    # API/system performance
    API_LATENCY_WARNING_MS: int = 2000
    API_FAILURE_RATE_WARNING: float = 0.1      # 10%
    DB_SLOW_OPERATION_MS: int = 500
    SLOW_SCAN_SYMBOL_MS: int = 5000
    SCAN_AVG_TIME_WARNING_MS: int = 3000


# ════════════════════════════════════════════════════════════════
# 8. DATABASE
# ════════════════════════════════════════════════════════════════

class Database:
    """Database retention, query, and maintenance constants."""

    # Connection pool
    POOL_READER_SIZE: int = 5                   # Concurrent reader connections
    POOL_WRITER_SIZE: int = 1                   # SQLite only supports 1 writer
    POOL_ACQUIRE_TIMEOUT_SECS: float = 10.0     # Timeout for acquiring a reader

    RETENTION_DAYS: int = 35                    # Signal retention window
    QUERY_LIMIT_BASE: int = 500                 # Base result limit
    QUERY_LIMIT_PER_HOUR: int = 20              # Scaling factor
    RECENT_EVENTS_DEFAULT_LIMIT: int = 20       # Default event query limit
    SIGNAL_MAX_AGE_RECOVERY_HOURS: int = 48     # Restart recovery window
    CLEANUP_DELETED_SIGNALS_DAYS: int = 30      # Purge deleted after 30d
    EQUITY_CURVE_DEFAULT_DAYS: int = 60         # Default equity curve window
    MONTHLY_PNL_DEFAULT_MONTHS: int = 12        # 1 year of monthly stats
    SIGNAL_NOTE_MAX_CHARS: int = 500            # User note truncation
    VACUUM_SIZE_THRESHOLD_BYTES: int = 50 * 1024 * 1024  # 50 MB
    DEFAULT_PERFORMANCE_WINDOW_DAYS: int = 30   # Performance stats lookback
    DEFAULT_QUERY_WINDOW_HOURS: int = 24        # "Recent" query window
    DEFAULT_TRAIL_PCT: float = 0.4              # Default trailing stop % for recovery

    # SQL injection protection — whitelist for dynamic column updates
    TRACKED_SIGNAL_ALLOWED_COLUMNS: frozenset = frozenset({
        'state', 'entry_price', 'be_stop', 'trail_stop', 'trail_pct',
        'pnl_r', 'max_r', 'last_price', 'last_checkin_at', 'activated_at',
        'completed_at', 'outcome', 'outcome_reason', 'updated_at',
        'message_id', 'raw_data', 'regime',
        'has_rejection_candle', 'has_structure_shift',
        'has_momentum_expansion', 'has_liquidity_reaction',
        'entry_status',
    })


# ════════════════════════════════════════════════════════════════
# 9. AGGREGATOR SCORING
# ════════════════════════════════════════════════════════════════

class AggregatorScoring:
    """Price-band thresholds and scoring-specific constants."""

    # Price-dependent entry band widths
    MICRO_PRICE_THRESHOLD: float = 0.001    # < $0.001
    SUBPENNY_THRESHOLD: float = 0.01        # < $0.01
    NORMAL_PRICE_THRESHOLD: float = 1.0     # < $1.00
    BAND_MICRO: float = 0.03               # ±3% for micro coins
    BAND_SUBPENNY: float = 0.02            # ±2% for sub-penny
    BAND_CENTS: float = 0.015              # ±1.5% for cents-range
    BAND_NORMAL: float = 0.01              # ±1% for normal coins

    # Grade-limit multiplier for A/A+ signals (daily symbol cap)
    GRADE_LIMIT_MULT_HIGH: int = 2          # A/A+ get 2× daily limit
    GRADE_LIMIT_MULT_DEFAULT: int = 1


# ════════════════════════════════════════════════════════════════
# 10. BACKTESTER
# ════════════════════════════════════════════════════════════════

class Backtester:
    """Backtester-specific constants."""

    DEFAULT_COMMISSION_PCT: float = 0.0004  # 0.04% Binance taker fee/side
    DEFAULT_SLIPPAGE_PCT: float = 0.0003    # 0.03% market impact
    CANDLE_WINDOW: int = 200                # Lookback candles for signals
    MAX_HOLD_BARS: int = 48                 # Max bars to hold
    WALKFORWARD_WINDOW: int = 30            # Trades per walk-forward window
    WALKFORWARD_STEP: int = 10              # Step size for walk-forward
    MIN_SAMPLE_SIZE: int = 20               # Minimum samples for statistics
    HIGH_VOLATILITY_THRESHOLD: float = 0.05 # ATR/price ratio
    MOMENTUM_THRESHOLD: float = 0.02        # Min price change for momentum
    MONTE_CARLO_RESAMPLES: int = 1000       # Number of MC simulations

    # ── Signal quality gates (prevent over-trading) ──────────────────
    MAX_SIGNALS_PER_BAR: int = 1            # Best signal per bar (by confidence)
    SIGNAL_COOLDOWN_BARS: int = 6           # Min bars between same-direction signals
    MIN_CONFLUENCE_COUNT: int = 2           # Min confluence factors required
    MIN_CONFIDENCE_BACKTEST: int = 75       # Stricter confidence floor for backtest

    # Bars per day by timeframe (crypto = 24/7)
    BARS_PER_DAY: dict = {
        '1m': 1440, '3m': 480, '5m': 288, '15m': 96,
        '30m': 48, '1h': 24, '2h': 12, '4h': 6,
        '6h': 4, '8h': 3, '12h': 2, '1d': 1,
    }


# ════════════════════════════════════════════════════════════════
# 11. OUTCOME MONITOR
# ════════════════════════════════════════════════════════════════

class OutcomeTracking:
    """Trailing stop and partial close defaults."""

    DEFAULT_TRAIL_PCT: float = 0.40         # 40% trailing fraction (regime-dependent)
    PARTIAL_CLOSE_AT_TP1_PCT: float = 0.50  # Close 50% at TP1
    EARLY_TRAIL_START_R: float = 0.75       # Start locking some profit before TP1 after +0.75R move
    EARLY_TRAIL_FRACTION: float = 0.35      # Lock 35% of max open profit during early-trail phase
    WICK_CONFIRM_TIMEFRAME: str = "15m"     # Candle timeframe used for stop-hunt wick validation
    DIRECTION_ACCURACY_MIN_R: float = 0.25  # Minimum favourable excursion to count direction as correct


# ════════════════════════════════════════════════════════════════
# 12. WEBSOCKET FEED
# ════════════════════════════════════════════════════════════════

class WebSocket:
    """WebSocket connection and reconnect constants."""

    PING_INTERVAL_SECS: int = 30                    # Send ping every 30s
    PONG_TIMEOUT_SECS: int = 60                     # Reconnect if no data for 60s
    RECONNECT_BASE_DELAY_SECS: float = 1.0          # Initial reconnect backoff
    RECONNECT_MAX_DELAY_SECS: float = 60.0          # Max reconnect backoff
    MAX_STREAMS_PER_CONNECTION: int = 200            # Binance per-connection limit
    BASE_URL: str = "wss://fstream.binance.com"     # Binance Futures WS endpoint
    COUNTER_WRAP: int = 1_000_000_000               # Wrap telemetry counters to prevent overflow


# ════════════════════════════════════════════════════════════════
# 13. ENTRY SCALING / DCA
# ════════════════════════════════════════════════════════════════

class EntryScaling:
    """Tiered entry and DCA configuration."""

    SCALED_TIER_PCTS: tuple = (0.30, 0.40, 0.30)   # low, mid, high
    DCA_TIER_PCTS: tuple = (0.25, 0.25, 0.25, 0.25)
    DCA_BELOW_ENTRY_FACTOR: float = 0.5             # halfway to SL


# ════════════════════════════════════════════════════════════════
# 14. CROSS-SYMBOL SIGNAL RANKING
# ════════════════════════════════════════════════════════════════

class SignalRanking:
    """Weights and limits for cross-symbol signal ranking."""

    RANK_WEIGHT_CONFIDENCE: float = 0.40
    RANK_WEIGHT_RR: float = 0.30
    RANK_WEIGHT_VOLUME: float = 0.15
    RANK_WEIGHT_REGIME: float = 0.15
    MAX_SIGNALS_PER_CYCLE: int = 3
    CORRELATION_THRESHOLD: float = 0.70   # Above this, filter correlated duplicates
    RR_NORMALIZATION_CAP: float = 5.0     # Cap RR at 5 for normalization

    # ── BTC-concentration guard ───────────────────────────────────
    # Most alts have ρ > 0.70 with BTC; the hardcoded 6-pair map only
    # catches BTC/ETH/SOL/BNB. When the ranker admits 3 "independent"
    # alts all correlated to BTC, a single BTC move drags them together.
    # These thresholds feed the BTC-concentration filter added in
    # signals/signal_ranker.py (same cycle, same direction).
    BTC_CONCENTRATION_CORR_THRESHOLD: float = 0.70  # treat ρ_BTC≥this as BTC-proxy
    BTC_CONCENTRATION_MAX_SAME_SIDE: int = 2         # 3rd same-side BTC-proxy is dropped


# ════════════════════════════════════════════════════════════════
# 15. SLIPPAGE TRACKING
# ════════════════════════════════════════════════════════════════

class SlippageTracking:
    """Live slippage tracking configuration."""

    MAX_HISTORY_SIZE: int = 1000                     # Rolling deque capacity
    ADVERSE_SLIPPAGE_PENALTY_SCALE: float = 6.0      # 1% slippage → 6% confidence reduction
    MAX_CONFIDENCE_REDUCTION: float = 0.10            # Never reduce more than 10%
    DEFAULT_WINDOW_HOURS: int = 24                    # Default lookback window
    NEUTRAL_THRESHOLD: float = 1e-4                   # ±0.01% — below this is "neutral" slippage


class FeeModel:
    """
    Cost-model gate thresholds.

    The aggregator's raw RR floor (``cfg_min_rr``) is calibrated in units of
    entry→SL distance and does NOT subtract round-trip commission + slippage.
    For a typical intraday setup with a 1.2R floor, round-trip friction of
    ~0.14% translates to ~0.1–0.3R reduction, which can flip a 1.25R setup
    into a net-losing trade. The ``fee_adjusted_rr`` helper already exists in
    ``utils.signal_guidance`` — this constant set gives the aggregator a
    canonical minimum below which a signal is rejected outright.
    """

    # A signal must clear this fee+slippage-adjusted RR floor or it is
    # structurally incapable of covering its own costs. 1.0 = break-even
    # after fees on reward vs. risk including cost adds to both sides.
    MIN_FEE_ADJUSTED_RR: float = 1.0


class ExpectedSlippage:
    """
    Predictive slippage model coefficients (see analyzers/expected_slippage.py).

    The model is intentionally first-principles, not learned, so the
    coefficients are explicit and tunable from one place. They are calibrated
    against typical Binance perp microstructure: 5 bps spread, $50k–500k
    top-of-book depth, ATR/price ~1–3 % per day on majors.

    Formula: slip = spread_bps/2/1e4 + K_IMPACT*sqrt(size/depth) + K_VOL*atr_pct
    """

    # Square-root market-impact coefficient. K=0.0006 means a trade equal in
    # size to top-book depth pays ~6 bps of impact on top of half-spread.
    K_IMPACT: float = 0.0006

    # Volatility surcharge: in fast tape (ATR/price = 3 %), add ~3 bps.
    K_VOL: float = 0.001

    # Cap on size/depth ratio so a thin quote doesn't blow up the estimate
    # to nonsense (50 % slip). 5x book = the trade walks ~5 levels.
    MAX_DEPTH_RATIO: float = 5.0

    # Lower / upper bounds on the final number.
    FLOOR_PCT: float = 0.0002          # 2 bps minimum (matches backtester default)
    CEILING_PCT: float = 0.005         # 50 bps cap

    # Default trade size used when a signal doesn't yet know its sizing.
    # Kept conservative so we don't under-estimate impact for typical fills.
    DEFAULT_SIZE_USD: float = 5000.0


# ════════════════════════════════════════════════════════════════
# 16. FUNDING RATE INTEGRATION
# ════════════════════════════════════════════════════════════════

class FundingIntegration:
    """Funding rate bridge configuration."""

    EXTREME_POSITIVE_RATE: float = 0.0003             # 0.03% per 8h cycle
    EXTREME_NEGATIVE_RATE: float = -0.0001            # -0.01%
    HIGH_POSITIVE_RATE: float = 0.0001                # 0.01%
    ALIGNED_BOOST_PTS: float = 3.0                    # Confidence boost when funding aligns
    OPPOSED_PENALTY_PTS: float = 8.0                  # Confidence penalty when funding opposes
    EXTREME_PENALTY_PTS: float = 15.0                 # Extreme funding opposition
    MAX_HISTORY_LENGTH: int = 24                      # 24 cycles (8h each on Binance)
    TREND_WINDOW: int = 3                             # Last 3 cycles for trend detection

    # ── Funding accrual tracker (cumulative funding into PnL) ─────
    # Binance perps pay funding every 8 hours. Accrued funding is
    # cumulative rate * (hours_held / cycle_hours).
    ACCRUAL_CYCLE_HOURS: float = 8.0                  # Binance perp default


class FundingCarry:
    """
    Magnitude-aware funding-carry adjustment (PR5 #3).

    The level-based penalties in ``DerivativesAnalyzer.assess_entry_validity``
    (e.g. -8 for SHORT into NEGATIVE funding) are stepwise — they don't
    distinguish a major at +0.012 % from an alt at +0.05 %, even though the
    alt's expected funding paid over the hold horizon is 4× worse.

    This converts the *actual* per-cycle rate into expected carry over the
    typical hold horizon and applies a graduated, symmetric boost/penalty.
    """

    # Default expected hold horizon for the carry calculation. Most intraday
    # signals hold 8–24 h; 16 h ≈ 2 funding cycles, a reasonable midpoint.
    HOLD_HOURS_DEFAULT: float = 16.0

    # Carry magnitudes (in basis points of notional) at which the
    # adjustment kicks in. Below the threshold the level-based logic
    # already covers it, so no double-counting.
    PENALTY_THRESHOLD_BPS: float = 5.0
    BOOST_THRESHOLD_BPS: float = 5.0

    # Confidence points per basis point of carry. 0.5 → 10 bps carry → 5 pts.
    PTS_PER_BP: float = 0.5

    # Hard caps so a single extreme funding print can't dominate confidence.
    MAX_PENALTY_PTS: float = 8.0
    MAX_BOOST_PTS: float = 6.0


# ════════════════════════════════════════════════════════════════
# 16b. EXCHANGE HEALTH GATE
# ════════════════════════════════════════════════════════════════

class ExchangeHealth:
    """
    Early gate that suppresses new signals when the exchange API is
    misbehaving. During outages, REST tickers freeze for several
    minutes — strategies that key off "current price" then trip
    breakout triggers that immediately invalidate when the feed
    catches up. Suppressing publication during these windows avoids
    catastrophic stop-outs at the moment connectivity returns.

    Health is sampled from ``data.api_client.api.get_request_stats()``
    plus a rolling consecutive-failure counter incremented by the
    aggregator before publish.
    """

    # Latency: avg latency over recent calls (ms). Above this and the
    # exchange is "DEGRADED"; double this and it is "UNHEALTHY".
    DEGRADED_AVG_LATENCY_MS: float = 3000.0
    UNHEALTHY_AVG_LATENCY_MS: float = 8000.0

    # Error rate: total_errors / total_requests for the lifetime of
    # the api client. Above this we consider the API unstable.
    DEGRADED_ERROR_RATE: float = 0.05    # 5 %
    UNHEALTHY_ERROR_RATE: float = 0.15   # 15 %

    # Consecutive-failure counter (incremented by the gate when
    # api.is_healthy is False, reset when True). When this trips we
    # block regardless of latency/error-rate.
    UNHEALTHY_CONSEC_FAILS: int = 3

    # After a recovery (api.is_healthy flips back to True), keep the
    # gate "DEGRADED" for this many seconds so a single ticker recovery
    # doesn't immediately re-open the floodgate while feeds are still
    # stabilising.
    RECOVERY_GRACE_SECS: int = 60


# ════════════════════════════════════════════════════════════════
# 16c. STABLECOIN DEPEG GUARD
# ════════════════════════════════════════════════════════════════

class StablecoinDepeg:
    """
    Reject signals quoted in a depegged stablecoin. When USDT trades
    at 0.985 instead of 1.000, every USDT-denominated chart is biased
    by ~150 bps — entries, stops and targets are all systematically
    wrong, and "breakouts" are often just stable-coin re-pegging.

    Thresholds are absolute distance from $1.00 in fraction terms,
    so 0.005 = 50 bps. Fail-open: if no quote is available the gate
    treats it as healthy (we should not block trading on a missing
    sanity check).
    """

    # 50 bps wobble from $1.00 → DEGRADED (allow but log)
    # 200 bps wobble → UNHEALTHY (block new signals)
    DEGRADED_DEVIATION: float = 0.005
    UNHEALTHY_DEVIATION: float = 0.020

    # How long a fetched stablecoin price is reused before re-fetching.
    PRICE_CACHE_SECS: int = 60

    # Stablecoins to monitor. The aggregator checks the *quote* asset
    # of each signal against this map.
    MONITORED_QUOTES: tuple = ("USDT", "USDC", "DAI", "FDUSD", "TUSD")


# ════════════════════════════════════════════════════════════════
# 16d. REGIME-TRANSITION HYSTERESIS
# ════════════════════════════════════════════════════════════════

class RegimeHysteresis:
    """
    The 2-cycle confirmation in ``RegimeAnalyzer._update_regime`` is
    a *pre-commit* hysteresis (don't flip until the next regime is
    confirmed). It does NOT cover the post-commit case where the
    bot has already switched regimes but the market has not yet
    settled — the first 5–15 min after a regime flip see the worst
    fakeouts as price re-tests the prior regime's structures.

    This adds a *post-commit* hysteresis: for ``TRANSITION_HYSTERESIS_SECS``
    after the regime flip is committed, the aggregator's min-confidence
    floor is bumped by ``TRANSITION_CONF_FLOOR_BUMP`` points so only
    very-high-quality setups publish during the noisy window.
    """

    TRANSITION_HYSTERESIS_SECS: int = 600         # 10 min
    TRANSITION_CONF_FLOOR_BUMP: int = 5           # +5 pts on min_confidence


# ════════════════════════════════════════════════════════════════
# 16e. FUNDING Z-SCORE
# ════════════════════════════════════════════════════════════════

class FundingZScore:
    """
    Per-asset z-score of current funding vs. its own recent history.

    The existing absolute-level classifier (``EXTREMELY_HIGH``/``NEGATIVE``)
    can't tell that BTC funding of 0.012 % is unremarkable while ALT
    funding of 0.012 % is the highest reading of the week — every alt
    has a different baseline. The z-score normalises by each asset's
    own running mean/std so the same threshold catches the *relative*
    extreme on any symbol.

    History size is intentionally larger than the delta classifier
    (5 samples) because z-score is meaningless on tiny samples.
    """

    HISTORY_MAX: int = 30                       # ~4 hours at 8-min poll cadence
    MIN_SAMPLES_FOR_Z: int = 8                  # below this we return 0.0 (insufficient)

    # |z| ≥ this → extreme. Standard ±2σ tail (~5 % of observations).
    EXTREME_Z: float = 2.0
    # |z| ≥ this → very extreme (~1 % tail). Used for stronger signals.
    VERY_EXTREME_Z: float = 3.0

    # Floor on std-dev to prevent a flat history from blowing up z-scores
    # (e.g. all-zeros after a fresh deploy). 0.0001 pp = noise level.
    MIN_STD: float = 1e-4


# ════════════════════════════════════════════════════════════════
# 16f. LIQUIDITY-SWEEP ESTIMATOR
# ════════════════════════════════════════════════════════════════

class LiqSweep:
    """
    Probability that price is set up to sweep nearby resting liquidity
    before continuing in the signal direction. Two cheap proxies are
    combined:

      * proximity to a recent swing high/low (resting stops cluster
        just beyond local extremes)
      * proximity to a round number (resting stops cluster on
        psychologically obvious price levels)

    A LONG entry whose stop sits just *below* a recent swing low or
    a round number is at high risk of being swept first, then
    rallying — a classic "stop hunt then real move" pattern.
    """

    # How close (as a fraction of price) to the swing/round level
    # the stop must be before we consider it "in the sweep zone".
    PROXIMITY_PCT: float = 0.005                 # 0.5 % of price

    # Look-back bars on whatever timeframe is supplied for swing detection.
    SWING_LOOKBACK_BARS: int = 20

    # Round-number granularities by price magnitude. We pick the
    # smallest granularity whose absolute value is ≥ 1 % of price
    # so the level is meaningful (e.g. on BTC at $60 000 the round
    # numbers we care about are $1 000 ticks; on a $0.05 alt they
    # are $0.001 ticks).
    ROUND_NUMBER_FRACTIONS: tuple = (1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)

    # Maximum size of a "meaningful" round-number granularity, expressed
    # as a fraction of price. We pick the largest power-of-10 slice ≤ this
    # cap so the chosen level is materially close enough to the stop to be
    # a credible hunt target.
    ROUND_NUMBER_MAX_FRACTION_OF_PRICE: float = 0.05

    # Probability weights for the two proxies (must sum ≤ 1.0).
    SWING_WEIGHT: float = 0.6
    ROUND_WEIGHT: float = 0.4

    # Confidence penalty applied at full sweep probability (=1.0).
    # Linearly scaled by the actual probability.
    MAX_CONFIDENCE_PENALTY: float = 4.0          # points
    # Probability above which we *log* the warning (no hard block).
    WARN_PROBABILITY: float = 0.6


# ════════════════════════════════════════════════════════════════
# 16g. WEEKEND CHOP / SESSION SPREAD
# ════════════════════════════════════════════════════════════════

class WeekendChop:
    """
    Weekend liquidity is 20–30 % lower with materially wider spreads
    on most perp venues. The existing ``signals.time_filter.TimeFilter``
    already applies grade-aware confidence penalties on Saturday/Sunday,
    but two effects are not modelled:

      1. Spread expectation — execution gate sees the *snapshot* spread
         from `volume_quality`, but a 5-bps quote on Saturday morning
         can blow out to 25 bps in a sweep. The filter now publishes
         a session-aware multiplier the aggregator can apply when no
         live spread sample is available.
      2. Aggregate min-confidence floor — alongside the per-signal
         penalty, raise the *floor* during the worst weekend windows
         so even high-grade signals must clear a higher bar.
    """

    # Multipliers applied to spread_bps based on the active session.
    # 1.0 = no change, 1.5 = expect 50 % wider than observed.
    SPREAD_MULT_KILLZONE: float = 1.0            # London/NY open — best quotes
    SPREAD_MULT_OFF_SESSION: float = 1.1
    SPREAD_MULT_ASIA: float = 1.2
    SPREAD_MULT_DEAD_ZONE: float = 1.3
    SPREAD_MULT_SATURDAY: float = 1.5
    SPREAD_MULT_SUNDAY_EARLY: float = 1.7        # Sunday 00:00–08:00 UTC

    # Aggregator min-confidence floor bump during weekend chop.
    # Stacks on top of TimeFilter's per-signal penalty.
    WEEKEND_CONF_FLOOR_BUMP: int = 3
    # Sunday-early window gets a bigger bump (worst liquidity all week).
    SUNDAY_EARLY_CONF_FLOOR_BUMP: int = 5


# ════════════════════════════════════════════════════════════════
# 16h. PARTIAL-FILL TRACKER
# ════════════════════════════════════════════════════════════════

class PartialFill:
    """
    Bookkeeping for partial fills.

    The slippage tracker today records the *last* fill price as if it
    were the entry. When a 10 000-USD order fills in three legs at
    increasingly worse prices, the recorded slippage understates
    reality. The partial-fill tracker computes a true VWAP across all
    legs of the same `signal_id` and feeds that into the slippage
    tracker once the fill is complete.
    """

    # Maximum partial fills retained per signal (most exchanges cap
    # at well below this; we keep generous slack for adversarial
    # microstructure that fragments aggressively).
    MAX_LEGS_PER_SIGNAL: int = 64

    # How long to retain a closed signal's fill record before purge.
    RETENTION_HOURS: int = 48


# ════════════════════════════════════════════════════════════════
# 17. ENRICHMENT PIPELINE (additive confidence guardrails)
# ════════════════════════════════════════════════════════════════

class Enrichment:
    """
    Guards against the additive confidence problem identified in
    architectural review: 7 new institutional-grade analyzers each
    contribute ±10–15 pts.  Without an aggregate cap the theoretical
    max enrichment is ±75 — far too much.

    Hierarchy:
      1. Valuation pre-gate  — blocks bullish enrichment when overvalued
      2. Vol regime dampener — scales down in noisy environments
      3. Correlated-signal diminishing returns (inside onchain_analytics)
      4. Aggregate cap       — hard clamp on total enrichment delta
    """

    # Aggregate cap across ALL 7 new modules (onchain, stablecoin, mining,
    # network, volatility, wallet, leverage).  Individual module caps remain.
    AGGREGATE_DELTA_CAP: int = 25                     # max(−25, min(+25, total))

    # Valuation pre-gate: when on-chain says OVERVALUED, LONG enrichment
    # deltas are clamped to this ceiling (0 = no positive boost allowed).
    OVERVALUED_LONG_DELTA_CEIL: int = 0
    # Mirror: when UNDERVALUED, SHORT enrichment deltas are clamped.
    UNDERVALUED_SHORT_DELTA_CEIL: int = 0

    # Volatility regime dampener: in HIGH_VOL or EXTREME regimes, multiply
    # all enrichment deltas by this factor (slower signals less reliable
    # when volatility is elevated).
    HIGH_VOL_DAMPENER: float = 0.60                   # keep 60% of delta

    # Correlated sub-signal diminishing returns inside onchain_analytics.
    # When MVRV, SOPR, NUPL all fire in the same direction:
    CORR_SECOND_SIGNAL_MULT: float = 0.60             # 2nd correlated → 60%
    CORR_THIRD_SIGNAL_MULT: float = 0.40              # 3rd+ correlated → 40%


# ════════════════════════════════════════════════════════════════
# 17b. ENSEMBLE VOTER
# ════════════════════════════════════════════════════════════════

class EnsembleVoter:
    """
    Thresholds for the rules-based ensemble voter that decides
    SUPPRESS / REDUCE / BOOST / PASS on each signal.
    """

    # Weighted score below this → HARD SUPPRESS (signal killed).
    # Was -3.0 — too aggressive; good signals with mild disagreement
    # (e.g. 3 sources oppose weakly) were killed.  -4.5 requires
    # stronger conviction to suppress.
    SUPPRESS_WEIGHTED_THRESHOLD: float = -4.5

    # Weighted score below this → REDUCE (confidence penalty).
    REDUCE_WEIGHTED_THRESHOLD: float = -1.5

    # Weighted score above this → BOOST (confidence bonus).
    BOOST_WEIGHTED_THRESHOLD: float = 3.0

    # High-confidence override: if signal.confidence >= this AND
    # rr_ratio >= OVERRIDE_MIN_RR, downgrade SUPPRESS → REDUCE
    # instead of killing the signal outright.
    SUPPRESS_OVERRIDE_MIN_CONF: int = 85
    SUPPRESS_OVERRIDE_MIN_RR: float = 2.5


# ════════════════════════════════════════════════════════════════
# 18. ADAPTIVE ENSEMBLE WEIGHTS
# ════════════════════════════════════════════════════════════════

class AdaptiveWeights:
    """
    Self-learning ensemble weight system.  Queries outcome data per
    regime and computes score = (win_rate × avg_return) / drawdown.
    Rolling 30d/60d blended windows with minimum sample thresholds.
    """

    # Minimum completed trades before a source's adaptive weight is used.
    # Below this, fall back to the static _WEIGHTS_BY_REGIME.
    MIN_SAMPLES: int = 10

    # Rolling window sizes (days) — blended 60% recent / 40% longer-term
    SHORT_WINDOW_DAYS: int = 30
    LONG_WINDOW_DAYS: int = 60
    SHORT_WINDOW_BLEND: float = 0.60          # recent window contribution
    LONG_WINDOW_BLEND: float = 0.40           # longer-term contribution

    # Weight clamps — adaptive weights are bounded to prevent runaway
    WEIGHT_FLOOR: float = 0.3                 # Never drop below 0.3×
    WEIGHT_CEIL: float = 2.5                  # Never exceed 2.5×

    # Drawdown floor — prevents division by zero / inflate on zero-DD
    DRAWDOWN_FLOOR: float = 0.10

    # How often to recalculate weights (seconds)
    RECALC_INTERVAL_SECS: int = 3600 * 4      # Every 4 hours

    # Default score when no data (neutral = 1.0)
    DEFAULT_SCORE: float = 1.0

    # Stability factor — penalises sources whose score differs widely
    # between the 30d and 60d windows (high variance = unreliable).
    # stability = 1 - (|short_score - long_score| / max(short, long, 0.1))
    # Final weight is multiplied by max(stability, STABILITY_FLOOR).
    STABILITY_FLOOR: float = 0.40             # Never penalise more than 60%


# ════════════════════════════════════════════════════════════════
# 19. FLOW ACCELERATION & SHOCK DETECTION
# ════════════════════════════════════════════════════════════════

class FlowAcceleration:
    """
    Second-derivative tracking on stablecoin and wallet flows.
    Detects velocity, acceleration, and shock events.
    """

    # Rolling window for velocity/acceleration (number of data points)
    VELOCITY_WINDOW: int = 7

    # Shock detection: current > SHOCK_MULT × rolling_avg triggers SHOCK
    SHOCK_MULT: float = 2.5

    # Require SHOCK_CONFIRM_INTERVALS consecutive intervals to confirm
    SHOCK_CONFIRM_INTERVALS: int = 2

    # Confidence deltas from acceleration signals
    ACCEL_BULLISH_DELTA: int = 4              # Accelerating inflow → +4 LONG
    ACCEL_BEARISH_DELTA: int = 4              # Accelerating outflow → -4 LONG
    SHOCK_BULLISH_DELTA: int = 6              # Confirmed inflow shock → +6 LONG
    SHOCK_BEARISH_DELTA: int = 6              # Confirmed outflow shock → -6 LONG

    # Whale acceleration thresholds
    WHALE_ACCEL_MIN_EVENTS: int = 5           # Min events for acceleration calc
    WHALE_SHOCK_USD_THRESHOLD: float = 5_000_000  # $5M shock floor
    WHALE_ACCEL_USD_THRESHOLD: float = 1_000_000  # $1M change = significant acceleration

    # Rolling average floor to prevent division by near-zero
    MIN_ROLLING_AVG_FLOOR: float = 0.01

    # Low-volume dampener — reduce acceleration weight in thin markets
    # to avoid noise-driven signals.  Volume < threshold → multiply delta
    # by LOW_VOLUME_DAMPENER.
    LOW_VOLUME_DAMPENER: float = 0.50         # Halve delta in thin markets
    LOW_VOLUME_THRESHOLD_USD: float = 500_000_000  # $500M 24h vol minimum


# ════════════════════════════════════════════════════════════════
# 20. CONTEXT-AWARE VOLATILITY COMPRESSION TIMER
# ════════════════════════════════════════════════════════════════

class VolCompressionTimer:
    """
    Enhanced volatility compression timing that combines duration
    with OI buildup, funding neutrality, and leverage zone context.
    Turns passive timing into context-aware breakout probability.
    """

    # Minimum compression days before context scoring activates
    MIN_COMPRESSION_DAYS: int = 3

    # Context factor weights (must sum to 1.0)
    WEIGHT_DURATION: float = 0.40           # Compression duration weight
    WEIGHT_OI_BUILDUP: float = 0.25         # OI trend weight
    WEIGHT_FUNDING_NEUTRAL: float = 0.20    # Funding neutrality weight
    WEIGHT_LEVERAGE_ZONE: float = 0.15      # Leverage buildup weight

    # Duration scoring: days_in_regime maps to 0-1 score
    DURATION_HALF_LIFE_DAYS: int = 7        # 7 days → 0.5 score
    DURATION_MAX_SCORE: float = 0.95

    # OI buildup detection
    OI_BUILDING_THRESHOLD: float = 5.0      # OI change >5% = building

    # Funding neutrality band (annualized rate %)
    FUNDING_NEUTRAL_LOW: float = -5.0       # Below = bearish pressure
    FUNDING_NEUTRAL_HIGH: float = 5.0       # Above = bullish pressure

    # Context-enhanced breakout probability boost
    CONTEXT_BOOST_THRESHOLD: float = 0.60   # Min context score for boost
    CONTEXT_DELTA_BONUS: int = 3            # Extra delta when context high

    # Directional bias: combine whale intent + flow direction to hint
    # which way the breakout is most likely (LONG, SHORT, or NEUTRAL).
    DIRECTIONAL_BIAS_WEIGHT_WHALE: float = 0.60   # Whale intent weight
    DIRECTIONAL_BIAS_WEIGHT_FLOW: float = 0.40    # Stablecoin flow weight
    DIRECTIONAL_BIAS_THRESHOLD: float = 0.30      # Min score to have a bias
    DIRECTIONAL_BIAS_DELTA: int = 2               # Bonus when direction aligns


# ════════════════════════════════════════════════════════════════
# 21. CASCADE PRESSURE SCORING
# ════════════════════════════════════════════════════════════════

class CascadePressure:
    """
    Enhanced cascade detection: pressure = (cluster_size / distance) × momentum.
    Ranks cascade opportunities and requires momentum confirmation.
    """

    # Momentum window (number of price snapshots)
    MOMENTUM_WINDOW: int = 12

    # Momentum threshold: price must be moving toward cluster
    MOMENTUM_TOWARD_THRESHOLD: float = 0.3  # >0.3% move toward cluster per interval

    # Pressure score thresholds
    PRESSURE_CRITICAL: float = 50.0         # pressure score ≥ 50 → CRITICAL
    PRESSURE_HIGH: float = 20.0             # ≥ 20 → HIGH
    PRESSURE_MEDIUM: float = 8.0            # ≥ 8 → MEDIUM

    # Minimum cluster size to consider (USD)
    MIN_CLUSTER_USD: float = 2_500_000      # $2.5M minimum

    # Distance floor to prevent division by near-zero
    DISTANCE_FLOOR_PCT: float = 0.1         # 0.1% minimum distance

    # Cluster persistence: how long a cluster exists before confidence boost
    PERSISTENCE_MIN_MINS: int = 10           # Cluster must persist ≥10 min
    PERSISTENCE_BOOST_MINS: int = 30         # Full boost at 30 min
    PERSISTENCE_MAX_BONUS: float = 1.5       # Up to 1.5× pressure boost

    # Pressure delta bonuses (on top of existing cascade deltas)
    PRESSURE_CRITICAL_DELTA: int = 5        # Momentum-confirmed critical cascade
    PRESSURE_HIGH_DELTA: int = 3            # Momentum-confirmed high cascade


# ════════════════════════════════════════════════════════════════
# 22. REGIME TRANSITION EARLY WARNING
# ════════════════════════════════════════════════════════════════

class RegimeTransition:
    """
    4-factor regime transition detection:
    (1) ADX declining, (2) vol compressing, (3) flow divergence,
    (4) OI/price exhaustion.
    Requires multi-signal agreement to reduce false positives.
    """

    # ADX decline detection
    ADX_DECLINE_THRESHOLD: float = 3.0      # ADX must drop ≥3 pts from recent peak
    ADX_PEAK_LOOKBACK: int = 5              # Cycles to look back for ADX peak

    # Vol compression detection (uses vol regime from volatility_structure)
    VOL_COMPRESS_REGIMES: tuple = ("COMPRESSED", "LOW")

    # Flow divergence: stablecoin inflow slowing
    FLOW_DECEL_THRESHOLD: float = -0.5      # Acceleration < -0.5% = slowing

    # Minimum factors required for transition warning
    MIN_FACTORS_FOR_WARNING: int = 2         # At least 2 of 4 factors
    MIN_EXTERNAL_INPUTS_FOR_WARNING_BACKSTOP: int = 2  # Below this, rely on internal backstop

    # Exhaustion detection (4th factor): OI rising but price stagnant
    EXHAUSTION_OI_THRESHOLD: float = 5.0     # OI must rise ≥5%
    EXHAUSTION_PRICE_THRESHOLD: float = 1.0  # Price must be within ±1%

    # Transition warning confidence penalty
    WARNING_CONFIDENCE_PENALTY: int = 5      # Reserved for future direct confidence reduction

    # Warning enrichment delta (used in engine enrichment pipeline)
    WARNING_ENRICHMENT_DELTA: int = -4       # Penalize signals during transition

    # Risk manager: additional Kelly reduction during transition warning
    WARNING_KELLY_REDUCTION: float = 0.70    # 70% of normal Kelly when transition active

    # News amplification for active technical transitions
    NEWS_AMPLIFIER_MAX_MULT: float = 1.35    # Cap transition penalty amplification
    NEWS_AMPLIFIER_SENSITIVITY: float = 0.50  # How strongly net-news score scales transition risk


# ════════════════════════════════════════════════════════════════
# 22b. VOL-OF-VOL SIGNAL SCORING
# ════════════════════════════════════════════════════════════════

class VolOfVol:
    """Vol-of-vol (volatility clustering) thresholds for signal scoring."""
    # When vol_of_vol exceeds this, apply penalty (erratic vol = harder to trade)
    HIGH_THRESHOLD: float = 0.15            # Stddev of recent RV changes
    # When vol_of_vol is very low, slight bonus (stable regime = cleaner signals)
    LOW_THRESHOLD: float = 0.04
    # Confidence deltas
    HIGH_PENALTY: int = -2                  # Penalty for high clustering
    LOW_BONUS: int = 1                      # Small bonus for stable vol


# ════════════════════════════════════════════════════════════════
# 22c. F&G EXTREME THRESHOLDS (shared by regime + HTF guardrail)
# ════════════════════════════════════════════════════════════════

class FearGreedThresholds:
    """Centralized Fear & Greed extreme thresholds.

    Previously hardcoded as 20/80 in regime.py and htf_guardrail.py
    while constants.py defined 25/75. Now single source of truth.
    """
    EXTREME_FEAR: int = 25                  # Below this = extreme fear
    EXTREME_GREED: int = 75                 # Above this = extreme greed


# ════════════════════════════════════════════════════════════════
# 22d. STABLECOIN-LINKED HTF THRESHOLD ADJUSTMENT
# ════════════════════════════════════════════════════════════════

class StablecoinHTF:
    """Dynamic HTF ADX threshold adjustment based on stablecoin flow."""
    # When stablecoin outflow is strong, raise ADX threshold (more restrictive)
    OUTFLOW_ADX_BOOST: int = 3              # Raise from 25 → 28
    # When stablecoin inflow is strong, lower ADX threshold (more permissive)
    INFLOW_ADX_REDUCTION: int = 2           # Lower from 25 → 23
    # Base ADX threshold (used when stablecoin signal is neutral)
    BASE_ADX_THRESHOLD: int = 25


# ════════════════════════════════════════════════════════════════
# 22e. VOLATILE_PANIC POSITION REVIEW
# ════════════════════════════════════════════════════════════════

class PanicPositionReview:
    """Constants for automatic position review on VOLATILE_PANIC."""
    # Tighten SL to breakeven for positions that have hit TP1
    TIGHTEN_POST_TP1: bool = True
    # Kelly multiplier for NEW signals during VOLATILE_PANIC.
    # This replaces (not multiplies) the normal KELLY_MULT_VOLATILE_PANIC (0.25).
    # 0.15 = 15% of full Kelly → even smaller than the standard 25% panic sizing.
    PANIC_NEW_SIGNAL_KELLY_MULT: float = 0.15


# ════════════════════════════════════════════════════════════════
# 23. DYNAMIC R:R FLOOR BY CONFIDENCE
# ════════════════════════════════════════════════════════════════

class DynamicRR:
    """
    When confidence is exceptionally high, the rigid RR floor hurts
    Expected Value.  EV(99% × 1.8R) = 1.78R >> EV(68% × 2.5R) = 1.70R.
    Allow the floor to scale down for high-confidence signals.
    """

    HIGH_CONF_THRESHOLD: float = 95.0       # Confidence must be ≥ 95 for discount
    HIGH_CONF_RR_DISCOUNT: float = 0.75     # Multiply floor by 0.75 (e.g. 2.0 → 1.5)


# ════════════════════════════════════════════════════════════════
# 24. OHLCV DATA QUALITY COOLDOWN
# ════════════════════════════════════════════════════════════════

class OHLCVCooldown:
    """
    Symbols that repeatedly fail OHLCV quality checks are auto-excluded
    for a cooldown period, preventing wasted API calls every scan cycle.
    """

    FAIL_THRESHOLD: int = 3                 # Consecutive failures before cooldown
    COOLDOWN_SECS: int = 1800               # 30 min cooldown before retrying


# ════════════════════════════════════════════════════════════════
# 25. STRATEGY KEY MAP
# ════════════════════════════════════════════════════════════════

# Maps strategy class names to settings.yaml section keys.
# Shared by core/engine.py and backtester/engine.py so both use
# the same key when checking cfg.is_strategy_enabled().
# ════════════════════════════════════════════════════════════════
# NEWS INTELLIGENCE CONSTANTS (Project Inheritance)
# ════════════════════════════════════════════════════════════════

class NewsIntelligence:
    """Constants for FCN-inherited news intelligence features."""

    # PRE-WORK 1: News gating confidence threshold
    # Raised from 0.15 to 0.40 to reduce false positives that block valid trades.
    NEWS_GATING_MIN_CONFIDENCE: float = 0.40

    # PRE-WORK 2: Time decay for headline signals
    # Formula: weight *= exp(-DECAY_LAMBDA * age_minutes)
    # Lambda=0.08 → ~50% weight at 9 min, ~30% at 15 min, ~1% at 60 min
    HEADLINE_DECAY_LAMBDA: float = 0.08
    HEADLINE_MAX_AGE_MINUTES: int = 90    # Ignore headlines older than this

    # Event-type-aware decay: slow-burn events (regulatory, ETF, macro, scheduled)
    # decay at a gentler rate because markets digest them over hours, not minutes.
    # Lambda=0.03 → ~64% weight at 15 min, ~17% at 60 min, ~5% at 100 min.
    HEADLINE_DECAY_LAMBDA_SLOW: float = 0.03
    HEADLINE_MAX_AGE_MINUTES_SLOW: int = 180   # 3 hours for slow-burn events

    # Event types that use the slower decay (case-insensitive keyword match)
    SLOW_DECAY_EVENT_KEYWORDS: tuple = (
        "etf", "sec", "regulation", "regulatory", "fed ", "fomc",
        "cpi", "inflation", "interest rate", "halving", "approval",
        "legislation", "congress", "executive order", "ban",
    )

    # Feature 1: Source credibility
    SOURCE_CREDIBILITY_FLOOR: float = 0.80
    SOURCE_CREDIBILITY_CEILING: float = 1.00

    # Feature 2: Clickbait detection
    CLICKBAIT_WEIGHT_PENALTY: float = 0.50   # sentiment_weight *= 0.5 if clickbait
    CLICKBAIT_SCORE_THRESHOLD: float = 0.50  # Score above this = clickbait

    # Feature 3: Title deduplication
    TITLE_DEDUP_SIMILARITY_THRESHOLD: float = 0.85  # Jaccard ngram threshold

    # Feature 4: Headline evolution tracking
    HEADLINE_EVOLUTION_SIMILARITY_THRESHOLD: float = 0.55  # Lowered from 0.70 to catch more mutations
    HEADLINE_EVOLUTION_CONFIDENCE_PENALTY: float = 0.60  # Multiply by this (40% reduction)

    # Feature 5: On-chain to news correlation
    CORRELATION_SAME_EVENT_BOOST: float = 1.05  # Correlated = same event
    CORRELATION_INDEPENDENT_BOOST: float = 1.15  # Independent confirmation
    CORRELATION_FULL_WINDOW_MINUTES: int = 20
    CORRELATION_PARTIAL_WINDOW_MINUTES: int = 45
    CORRELATION_PARTIAL_WEIGHT: float = 0.50
    UNEXPLAINED_ONCHAIN_PENALTY: float = 0.90  # Reduce confidence for unexplained moves

    # Feature 7: Fear & Greed regime overlay
    # Delegate to FearGreedThresholds so regime/htf_guardrail/sentiment stay
    # aligned on a single source of truth and can't silently drift apart.
    FEAR_GREED_EXTREME_FEAR_THRESHOLD: int = FearGreedThresholds.EXTREME_FEAR
    FEAR_GREED_EXTREME_GREED_THRESHOLD: int = FearGreedThresholds.EXTREME_GREED
    FEAR_GREED_MAX_ADJUSTMENT_PCT: float = 0.15   # ±15% max threshold adjustment (was ±10%)

    # Feature 8: Narrative tracker
    NARRATIVE_MIN_ARTICLES: int = 3
    NARRATIVE_VELOCITY_RISING_THRESHOLD: float = 2.0  # 2x avg = rising
    NARRATIVE_VELOCITY_FADING_THRESHOLD: float = 0.5  # 0.5x avg = fading
    CONTEXTUAL_CONFIDENCE_CAP: float = 0.20  # Max ±20% from all contextual signals combined

    # Feature 9: Pump/dump detection
    PUMP_DUMP_PRICE_CHANGE_THRESHOLD: float = 7.0     # 7% price change (lowered from 10%)
    PUMP_DUMP_VOLUME_CHANGE_THRESHOLD: float = 200.0  # 200% volume change (lowered from 300%)
    PUMP_DUMP_WINDOW_MINUTES: int = 15
    # If a correlated news event explains the move, downgrade risk by one tier.
    PUMP_DUMP_NEWS_EXEMPT_WINDOW_MINUTES: int = 30

    # BTC News Intelligence: severity-gated blocking
    # Only hard-block LONGs/SHORTs when classification confidence reaches HIGH
    # severity (≥0.65).  MEDIUM-severity events (0.40–0.65) still reduce
    # confidence and position size but let signals through so good trades
    # aren't lost on headlines the market ultimately shrugs off.
    HARD_BLOCK_MIN_CONFIDENCE: float = 0.65

    # BTC News Intelligence: confidence-scaled event TTL
    # Scale event duration by confidence so weaker classifications expire
    # sooner.  Floor at 50% of the base TTL to keep minimum protection.
    # e.g. 50% conf MACRO_RISK_OFF → 4h × 0.50 = 2h instead of full 4h.
    TTL_CONFIDENCE_FLOOR: float = 0.50

    # Mixed signal detection — when a headline contains BOTH risk-off
    # (geopolitical, macro) AND capital-inflow signals (institutional buying,
    # Saylor, ETF inflows).  These conflicting forces create volatility but
    # NOT pure downside, so the bot should adjust (reduce size, lower conf)
    # rather than hard-block LONGs.
    # Example: "Saylor buys BTC despite Iran-US talks collapse"
    MIXED_SIGNAL_CONFIDENCE_MULT: float = 0.85   # vs 0.70 for pure risk-off
    MIXED_SIGNAL_SIZE_MULT: float = 0.75          # vs 0.50 for pure risk-off
    MIXED_SIGNAL_TTL_SCALE: float = 0.60          # 60% of base TTL (shorter window)
    MIXED_SIGNAL_MIN_BULLISH_CONF: float = 0.15   # Min confidence for bullish match to count as conflict

    # Headline timeliness — gate alerts for stale news
    # Headlines older than STALE_HEADLINE_ALERT_GATE_MINUTES won't fire
    # Telegram risk-event alerts.  They still feed sentiment scoring
    # (where TIME_DECAY handles weight reduction), but we don't want the
    # bot sending "urgent" alerts for news that broke hours ago.
    STALE_HEADLINE_ALERT_GATE_MINUTES: int = 45   # Don't fire alerts for news older than this
    # Confidence decay for stale headlines that still pass the gate:
    # conf *= max(floor, 1 - (age_minutes / gate) * (1 - floor))
    # e.g. 30-min-old headline with gate=45 → conf × 0.56
    STALE_HEADLINE_CONFIDENCE_FLOOR: float = 0.40  # Minimum retained confidence fraction

    # ── Feature 12: Exponential news decay curve ───────────────
    # Replace linear staleness decay with exp(-age / tau).
    # tau controls half-life: at age=tau, multiplier ≈ 0.37.
    # Fast-burn news (hacks, liquidations) decays quickly;
    # slow-burn events (ETF, regulation) decay gently.
    NEWS_DECAY_TAU_FAST_MINUTES: float = 60.0     # τ for fast events → 0.37 at 60m
    NEWS_DECAY_TAU_SLOW_MINUTES: float = 150.0    # τ for slow events → 0.37 at 150m
    NEWS_DECAY_FLOOR: float = 0.10                 # Minimum multiplier (never fully zero)

    # ── Feature 13: News vs Price reaction scoring ─────────────
    # After news is classified, wait REACTION_DELAY minutes then
    # compare BTC price move direction vs expected direction.
    # If market contradicts news → reduce context confidence.
    REACTION_DELAY_MINUTES: float = 15.0     # Wait this long before checking reaction
    REACTION_CONFIRM_MULT: float = 1.10      # Boost confidence if market confirms
    REACTION_CONTRADICT_MULT: float = 0.50   # Halve confidence if market contradicts
    REACTION_NEUTRAL_MULT: float = 0.85      # Slight reduction if market shrugs
    REACTION_MIN_MOVE_PCT: float = 0.5       # Price must move ≥ this % to count

    # Event replacement ratio — when a new classification has a DIFFERENT
    # event type from the currently active context, require that its confidence
    # is at least this fraction of the active context's confidence before
    # replacing.  Prevents a weak BTC_TECHNICAL headline from clobbering an
    # active EXCHANGE_EVENT / MACRO_RISK_OFF block simply because the type
    # changed.  Same-type updates still replace when conf >= current.conf.
    EVENT_REPLACE_CONF_RATIO: float = 0.85

    # ── Feature 14: Multi-news conflict resolver ───────────────
    # Aggregate all recent headlines into a single net_news_score.
    # score_per_headline = direction_sign × confidence × decay × source_weight
    # Net score interpretation:
    NET_SCORE_BULLISH_THRESHOLD: float = 0.30   # > this = bullish bias
    NET_SCORE_BEARISH_THRESHOLD: float = -0.30  # < this = bearish bias
    # Between thresholds = mixed/neutral
    NET_SCORE_BULLISH_CONF_BOOST: float = 1.05   # Boost signal conf when net bullish
    NET_SCORE_BEARISH_CONF_PENALTY: float = 0.90  # Penalize signal conf when net bearish
    NET_SCORE_MIXED_SIZE_MULT: float = 0.85       # Reduce position size in mixed regime
    HEADLINE_HISTORY_MAX_AGE_MINUTES: float = 120.0  # Track headlines for 2 hours
    HEADLINE_HISTORY_MAX_SIZE: int = 50              # Max tracked headlines

    # Feature 11: DB retry queue safety limits
    DB_RETRY_MAX_QUEUE_SIZE: int = 1000       # Max queued records before dropping oldest
    DB_RETRY_MAX_RETRIES_PER_RECORD: int = 10 # Max retries before dead-lettering a record
    DB_RETRY_QUEUE_ALERT_THRESHOLD: int = 100 # Alert when queue exceeds this size


# ════════════════════════════════════════════════════════════════
# TRIGGER QUALITY SCORING
# ════════════════════════════════════════════════════════════════

class TriggerQuality:
    """
    Trigger quality > trigger quantity.
    More triggers ≠ better moves. What matters is:
      1. Quality of each trigger (volume-confirmed, regime-aligned)
      2. Diversity of trigger sources (not redundant)
      3. Volume context (high vol in low-liquidity = different from high vol in deep market)
    """
    # Diminishing returns: Nth trigger adds DIMINISH_BASE × DIMINISH_DECAY^(N-1) value
    DIMINISH_BASE: float = 1.0         # First trigger = full value
    DIMINISH_DECAY: float = 0.65       # Each additional trigger worth 65% of prior
    MAX_USEFUL_TRIGGERS: int = 5       # Beyond 5, additional triggers add negligible value

    # Volume-confirmed trigger: trigger fires with volume > threshold = higher quality
    VOL_CONFIRMED_BONUS: float = 1.3   # 1.3× weight for volume-confirmed triggers
    VOL_UNCONFIRMED_PENALTY: float = 0.6  # 0.6× weight for triggers without volume

    # Trigger diversity bonus: triggers from different categories (structure/momentum/volume)
    DIVERSITY_BONUS_PER_CATEGORY: float = 0.10  # +10% per unique category
    MAX_DIVERSITY_BONUS: float = 0.30  # Cap at +30%

    # Fast-move detection: volume surge + price velocity > thresholds
    FAST_MOVE_VOL_SPIKE_MIN: float = 2.5    # Volume must be 2.5× average
    FAST_MOVE_PRICE_VEL_ATR: float = 0.5    # Price must move > 0.5 ATR per bar
    FAST_MOVE_CONF_BONUS: int = 4            # +4 confidence for fast-move setup

    # Low-volume environment detection
    LOW_VOL_PERCENTILE: float = 0.25    # Bottom 25% of 20-bar volume = low-vol
    LOW_VOL_TRIGGER_DISCOUNT: float = 0.50  # Halve trigger value in low-vol
    LOW_VOL_WARNING_NOTE: str = "⚠️ Low-volume environment — triggers less reliable"

    # High-volume isn't always better — context matters
    # Climactic volume (extreme spike at end of move) = exhaustion, not continuation
    CLIMACTIC_VOL_MULT: float = 4.0    # 4× average = potentially climactic
    CLIMACTIC_REVERSAL_PENALTY: float = 0.5  # Halve trigger value if climactic + reversal bar

    # Trigger sequence detection (ordered event chains)
    # High-quality entries follow predictable sequences:
    #   sweep → BOS → retest (best), BOS → retest (good), random (neutral)
    SEQUENCE_SWEEP_BOS_RETEST_BONUS: float = 0.15   # Triple-sequence bonus
    SEQUENCE_BOS_RETEST_BONUS: float = 0.08          # Double-sequence bonus
    SEQUENCE_MIN_TRIGGERS: int = 2                   # Need ≥2 triggers for sequence detection

    # Quality score thresholds for final classification
    QUALITY_HIGH: float = 0.75    # Trigger quality composite ≥ 0.75 = HIGH
    QUALITY_MEDIUM: float = 0.45  # ≥ 0.45 = MEDIUM
    # Below MEDIUM = LOW quality


# ════════════════════════════════════════════════════════════════
# SPOOFING & FAKE WALL DETECTION
# ════════════════════════════════════════════════════════════════

class SpoofingDetection:
    """
    Detect and discount fake/spoofed order book walls.
    Whales place large orders to manipulate sentiment then pull them.
    """
    # Snapshot tracking: compare walls across consecutive snapshots
    SNAPSHOT_INTERVAL_SECS: float = 30.0     # Check order book every 30s
    MAX_SNAPSHOTS: int = 10                  # Keep 10 snapshots (5 min of history)

    # Wall persistence: real walls persist; fake walls disappear
    MIN_PERSISTENCE_SNAPSHOTS: int = 3       # Must appear in ≥3 consecutive snapshots
    PERSISTENCE_CONFIDENCE: float = 0.85     # Persistent wall confidence

    # Pull pattern: wall existed then vanished when price approached
    PULL_DISTANCE_PCT: float = 0.005   # Wall pulled when price within 0.5%
    PULL_PENALTY: float = 0.90         # 90% chance it was spoofing if pulled near price

    # Size anomaly: walls significantly larger than surrounding levels
    SIZE_ANOMALY_MULT: float = 8.0     # 8× average = suspicious
    SIZE_ANOMALY_DISCOUNT: float = 0.60  # Discount suspiciously large walls by 40%

    # Known spoofing signatures
    # Symmetric walls (both sides at same distance) = likely market maker, not spoof
    SYMMETRIC_WALL_TOLERANCE: float = 0.10  # Within 10% distance of each other
    SYMMETRIC_WALL_BOOST: float = 1.2       # Boost confidence for symmetric (real MM)

    # Fake wall classification thresholds
    FAKE_WALL_CONFIDENCE: float = 0.70  # ≥ 0.70 confidence it's fake → discount heavily
    REAL_WALL_CONFIDENCE: float = 0.30  # ≤ 0.30 → treat as real

    # Absorption detection: price hits wall repeatedly without breaking through
    # This is the strongest form of liquidity confirmation — real intent
    ABSORPTION_MIN_TOUCHES: int = 3             # Min price touches at wall level
    ABSORPTION_TOUCH_DISTANCE_PCT: float = 0.003  # Price within 0.3% of wall = "touch"
    ABSORPTION_LOOKBACK_SECS: float = 300.0     # Look back 5 min for touches
    ABSORPTION_CONF_BONUS: int = 3              # Confidence bonus for absorption detected

    # Wall strength tiers
    WALL_TIER_MEGA: float = 10.0       # 10× average = mega wall
    WALL_TIER_LARGE: float = 6.0       # 6× average
    WALL_TIER_MODERATE: float = 4.0    # 4× average (existing threshold)


# ════════════════════════════════════════════════════════════════
# WHALE INTENT CLASSIFICATION
# ════════════════════════════════════════════════════════════════

class WhaleIntent:
    """
    Advanced whale intent classification beyond simple buy/sell.
    """
    # Intent taxonomy
    INTENT_ACCUMULATION = "ACCUMULATION"     # Steady buying, holding
    INTENT_DISTRIBUTION = "DISTRIBUTION"     # Steady selling, offloading
    INTENT_REBALANCING = "REBALANCING"       # Both sides, neutral
    INTENT_MARKET_MAKING = "MARKET_MAKING"   # Tight spread, both sides
    INTENT_DIRECTIONAL = "DIRECTIONAL"       # Strong one-sided flow
    INTENT_HEDGING = "HEDGING"               # Counter-position to derivatives

    # Multi-source correlation weights
    SOURCE_WEIGHT_ORDERBOOK: float = 0.30    # Real-time but noisy
    SOURCE_WEIGHT_ONCHAIN: float = 0.35      # Slower but more reliable
    SOURCE_WEIGHT_DERIVATIVES: float = 0.20  # OI changes, funding
    SOURCE_WEIGHT_DEPOSIT: float = 0.15      # Exchange inflows/outflows

    # Intent confidence decay
    INTENT_DECAY_TAU_SECS: float = 1800.0    # 30-min half-life for intent signals
    INTENT_MIN_CONFIDENCE: float = 0.20      # Floor — never fully ignore

    # Market-maker detection (as opposed to directional whale)
    MM_SPREAD_RATIO_MAX: float = 0.003  # Buy/sell within 0.3% spread = MM
    MM_SIZE_SYMMETRY_MIN: float = 0.7   # Buy/sell volumes within 30% = MM

    # Whale coordination thresholds (enhanced)
    COORD_TIME_WINDOW_SECS: float = 600.0    # 10-min window
    COORD_MIN_PARTICIPANTS: int = 3          # ≥3 distinct sources
    COORD_USD_THRESHOLD: float = 500_000.0   # Combined $500k+ to qualify

    # Confidence deltas for intent signals
    INTENT_CONF_ACCUMULATION: int = 5        # Strong bullish conviction
    INTENT_CONF_DISTRIBUTION: int = -5       # Strong bearish conviction
    INTENT_CONF_DIRECTIONAL_WITH: int = 4    # Directional whale with-trend
    INTENT_CONF_DIRECTIONAL_AGAINST: int = -6  # Directional whale counter-trend
    INTENT_CONF_HEDGING: int = -2            # Hedging suggests caution
    INTENT_CONF_MM: int = 0                  # Market-making = noise, ignore

    # Historical behavior profiling: track past whale patterns
    # "This wallet usually buys before pumps" → higher trust
    HISTORICAL_MIN_EVENTS: int = 10          # Min events to form a profile
    HISTORICAL_CONSISTENCY_BONUS: float = 0.20  # Max bonus for consistent directional pattern
    HISTORICAL_DECAY_DAYS: float = 30.0      # Older events count less (30-day half-life)


# ════════════════════════════════════════════════════════════════
# VOLUME QUALITY ASSESSMENT
# ════════════════════════════════════════════════════════════════

class VolumeQuality:
    """
    High volume ≠ always better. Context determines if volume is bullish or bearish.
    """
    # Climactic volume (exhaustion) detection
    CLIMACTIC_MULT: float = 3.5        # Volume > 3.5× 20-bar avg = potentially climactic
    CLIMACTIC_WITH_REVERSAL: str = "EXHAUSTION"  # Climactic + reversal bar = exhaustion
    CLIMACTIC_WITH_TREND: str = "BREAKOUT"       # Climactic + trend bar = genuine breakout

    # Participation breadth: is volume from many participants or few large orders?
    # Measured by comparing tick count to volume — high volume from few ticks = whale-driven
    TICK_VOLUME_RATIO_LOW: float = 0.3    # Low tick/vol ratio = few large orders
    TICK_VOLUME_RATIO_HIGH: float = 0.7   # High tick/vol ratio = broad participation

    # Spread-adjusted volume: volume in wide-spread markets is less meaningful
    SPREAD_WIDE_THRESHOLD_BPS: float = 15.0  # > 15 bps spread = wide
    SPREAD_WIDE_VOL_DISCOUNT: float = 0.70   # Discount volume by 30% in wide spreads

    # Dry volume (low volume but price moves): distribution or thin market
    DRY_VOL_PRICE_MOVE_ATR: float = 0.3    # Price moved > 0.3 ATR
    DRY_VOL_THRESHOLD_MULT: float = 0.5    # But volume < 0.5× average
    DRY_VOL_PENALTY: int = -3              # Confidence penalty for dry moves

    # Volume confirmation windows
    CONFIRM_WINDOW_BARS: int = 3    # Volume should confirm within 3 bars of trigger
    CONFIRM_MIN_MULT: float = 1.5   # Confirmation requires 1.5× average volume

    # Quality composite score weights
    QUALITY_WEIGHT_MAGNITUDE: float = 0.25    # How much volume
    QUALITY_WEIGHT_TREND: float = 0.20        # Volume increasing or decreasing
    QUALITY_WEIGHT_CONTEXT: float = 0.25      # Is it climactic, breakout, or normal
    QUALITY_WEIGHT_BREADTH: float = 0.15      # Participation breadth
    QUALITY_WEIGHT_SPREAD: float = 0.15       # Spread-adjusted quality

    # Quality score thresholds
    QUALITY_SCORE_HIGH: float = 70.0    # ≥ 70 = high quality volume
    QUALITY_SCORE_LOW: float = 35.0     # ≤ 35 = low quality (suspect)

    # Context-dependent evaluation
    # Breakout: need HIGH volume + trend bar
    BREAKOUT_VOL_MIN_MULT: float = 2.0  # 2× average minimum for breakout confirmation
    # Range trading: moderate volume is fine, high volume = potential breakout
    RANGE_VOL_HIGH_WARNING: float = 2.5  # Warn about potential range break on high volume
    # Low-volume doesn't always mean bad — pullbacks should have decreasing volume
    PULLBACK_VOL_DECREASE_GOOD: bool = True  # Decreasing vol on pullback = healthy

    # Intrabar delta estimation (buy vs sell volume)
    # Uses close position within bar range: (close-low)/(high-low) approximates buy fraction
    # This is the standard tick-rule approximation when trade-level data isn't available
    DELTA_STRONG_THRESHOLD: float = 0.70     # Delta > 70% = strong buyer/seller control
    DELTA_WEAK_THRESHOLD: float = 0.30       # Delta < 30% = opposition in control
    DELTA_CONF_BONUS: int = 2                # Confidence bonus for aligned delta
    DELTA_CONF_PENALTY: int = -2             # Penalty for opposing delta


class ExecutionGate:
    """
    Execution Quality Gate — separates "good setup" from "good trade."

    The bot's existing pipeline evaluates *setup quality* (confluence, regime,
    patterns, R:R).  But a great setup can still be a bad *trade* if execution
    conditions are poor: dead-zone session, wide spread, low trigger quality,
    whale conflict, premium entry, or weak volume.

    Previously each of these applied independent soft penalties (a few confidence
    points each).  A strong-confluence signal could survive all of them.  This
    gate evaluates execution factors *as a group* — when too many stack up, the
    signal is hard-blocked regardless of confidence.

    Design principles:
      1. Each factor scores 0-100 (100 = perfect execution conditions).
      2. Weighted composite → single Execution Score (0-100).
      3. Hard-block threshold: score < HARD_BLOCK_THRESHOLD → NO TRADE.
      4. A+ override: truly elite setups (grade A+, conf ≥ 90) get a relaxed
         threshold, but are NOT exempt. Even the best setup fails in zero
         liquidity.
      5. Adds "NO_TRADE" as an explicit outcome, logged with full breakdown.
    """

    # ── Factor weights (must sum to 1.0) ──────────────────────────────
    W_SESSION:          float = 0.20   # Session quality (killzone vs dead zone)
    W_TRIGGER_QUALITY:  float = 0.20   # Trigger quality score from analyzer
    W_SPREAD:           float = 0.15   # Bid-ask spread (liquidity proxy)
    W_WHALE_ALIGNMENT:  float = 0.15   # Whale intent alignment with direction
    W_ENTRY_POSITION:   float = 0.15   # Premium/discount vs direction
    W_VOLUME_ENV:       float = 0.15   # Volume environment quality

    # ── Hard-block thresholds ─────────────────────────────────────────
    HARD_BLOCK_THRESHOLD:    float = 35.0   # Below this → NO TRADE
    APLUS_BLOCK_THRESHOLD:   float = 25.0   # A+ signals get relaxed threshold
    SOFT_PENALTY_THRESHOLD:  float = 50.0   # Below this → heavy confidence penalty
    SOFT_PENALTY_MULT:       float = 0.85   # Multiplier when in soft penalty zone
    EXECUTION_HISTORY_MAX_ENTRIES: int = 10  # Cap stored execution assessments per signal

    # ── Session scoring ───────────────────────────────────────────────
    SESSION_SCORE_KILLZONE:  float = 100.0  # London/NY open, London close
    SESSION_SCORE_NORMAL:    float = 70.0   # Standard hours (not killzone, not dead)
    SESSION_SCORE_ASIA:      float = 55.0   # Asia session — lower liquidity
    SESSION_SCORE_WEEKEND:   float = 40.0   # Saturday/Sunday
    SESSION_SCORE_DEAD_ZONE: float = 15.0   # 03:00-07:00 UTC dead zone

    # ── Trigger quality mapping ───────────────────────────────────────
    # Maps trigger_quality_score (0.0-1.0) to execution score (0-100)
    TRIGGER_SCORE_MULT:      float = 100.0  # Direct linear map

    # ── Spread scoring ────────────────────────────────────────────────
    SPREAD_SCORE_TIGHT:      float = 100.0  # Spread < 10 bps
    SPREAD_TIGHT_BPS:        float = 10.0
    SPREAD_SCORE_NORMAL:     float = 70.0   # 10-30 bps
    SPREAD_NORMAL_BPS:       float = 30.0
    SPREAD_SCORE_WIDE:       float = 35.0   # 30-100 bps
    SPREAD_WIDE_BPS:         float = 100.0
    SPREAD_SCORE_EXTREME:    float = 10.0   # > 100 bps

    # ── Whale alignment scoring ───────────────────────────────────────
    WHALE_ALIGNED_SCORE:     float = 100.0  # Whales confirm direction
    WHALE_NEUTRAL_SCORE:     float = 60.0   # No whale data or neutral
    WHALE_OPPOSING_SCORE:    float = 15.0   # Whales oppose direction (severe)

    # ── Entry position scoring ────────────────────────────────────────
    # How well the entry aligns with smart-money logic
    ENTRY_DISCOUNT_LONG:     float = 100.0  # LONG in discount = ideal
    ENTRY_PREMIUM_SHORT:     float = 100.0  # SHORT in premium = ideal
    ENTRY_EQ_ZONE:           float = 50.0   # At equilibrium = mediocre
    ENTRY_WRONG_ZONE:        float = 20.0   # LONG in premium / SHORT in discount
    ENTRY_TREND_BYPASS:      float = 70.0   # Trend regime bypasses EQ rules

    # ── Volume environment scoring ────────────────────────────────────
    VOLUME_BREAKOUT_SCORE:   float = 100.0  # Breakout volume
    VOLUME_ABOVE_AVG_SCORE:  float = 80.0   # Above average
    VOLUME_NORMAL_SCORE:     float = 60.0   # Normal
    VOLUME_LOW_SCORE:        float = 25.0   # Low volume
    VOLUME_CLIMACTIC_SCORE:  float = 30.0   # Climactic (exhaustion risk)

    # ── Stacking penalty ──────────────────────────────────────────────
    # When multiple factors are simultaneously bad, apply extra penalty
    BAD_FACTOR_THRESHOLD:    float = 40.0   # Factor score below this = "bad"
    STACKING_PENALTY_PER:    float = 5.0    # Extra penalty per bad factor beyond 2
    MAX_STACKING_PENALTY:    float = 15.0   # Cap on stacking penalty

    # ── Kill Combos (non-negotiable hard blocks) ──────────────────────
    # Specific combinations of bad factors that ALWAYS block, regardless
    # of composite score.  These bypass the weighted scoring entirely.
    # Even a great killzone + high-confidence signal dies if a kill combo fires.
    #
    # Rationale: scoring averages can be gamed by one strong factor masking
    # several weak ones.  Kill combos enforce non-linear "one bad combo kills all."

    # KC1: Terrible trigger + low volume = no edge, no confirmation
    KC_TRIGGER_QUALITY_FLOOR:     float = 0.25   # Trigger quality below this
    KC_VOLUME_LOW_LABEL:          str   = "LOW_VOL"  # Volume context label

    # KC2: Extreme whale opposition = fighting institutional flow
    KC_WHALE_OPPOSITION_RATIO:    float = 0.85   # Buy ratio threshold for opposition
    # A+ signals can survive whale opposition (Wyckoff UTAD = whales distributing)
    KC_WHALE_OPPOSITION_APLUS_EXEMPT: bool = True

    # KC3: Dead zone + wide spread = zero liquidity
    KC_SPREAD_WIDE_BPS:           float = 50.0   # Spread threshold for kill combo
    # Dead zone already scored, but dead + wide spread = absolute kill

    # KC4: Wrong-zone entry + low trigger = worst possible timing
    # LONG in premium OR SHORT in discount with no momentum confirmation

    # KC5: Dead zone + low trigger + low volume = triple liquidity death
    # (subsumes KC1 when session is also dead)

    # ── Near-miss tracking ────────────────────────────────────────────
    # Blocked signals within the near-miss band are flagged for post-block
    # outcome tracking.  This detects cases where the gate was too strict
    # (false negatives) and drives adaptive feedback.
    NEAR_MISS_LOWER:            float = 25.0   # Score floor for near-miss (below = clear block)
    NEAR_MISS_UPPER:            float = 50.0   # Score ceiling for near-miss (aligns with soft-penalty boundary; 50+ is a clear pass)
    NEAR_MISS_MAX_TRACKED:      int   = 200    # Max near-misses to track (deque)
    NEAR_MISS_OUTCOME_WINDOW:   int   = 3600   # Seconds to monitor price after block (1 hour)
    NEAR_MISS_TP_CHECK_PERCENT: float = 1.5    # % price move toward TP = "would have won"

    # ── Execution Gate Feedback Metrics ────────────────────────────────
    # Track accuracy of block/pass decisions against actual outcomes.
    FEEDBACK_MIN_SAMPLE:        int   = 20     # Min trades before computing rates
    FEEDBACK_WINDOW_HOURS:      int   = 168    # 7 days rolling window (for future time-windowed metrics)

    # ── Adaptive block thresholds ───────────────────────────────────────────
    # Only applied when the gate receives structured execution context.
    # Bounds keep the gate from becoming effectively disabled (<20) or
    # unrealistically strict (>50) when multiple context adjustments stack.
    ADAPTIVE_THRESHOLD_MIN:         float = 20.0
    ADAPTIVE_THRESHOLD_MAX:         float = 50.0
    ADAPTIVE_KILLZONE_ADJ:          float = -5.0
    ADAPTIVE_DEAD_ZONE_ADJ:         float = 10.0
    ADAPTIVE_WEEKEND_ADJ:           float = 5.0
    ADAPTIVE_HIGH_VOL_ADJ:          float = -5.0
    ADAPTIVE_LOW_VOL_ADJ:           float = 5.0

    # ── Signal Promotion (Block → Re-activate) ──────────────────────────
    # Near-misses with improving execution conditions can be promoted.
    PROMOTION_ENABLED:              bool  = True
    PROMOTION_SCORE_THRESHOLD:      float = 50.0   # Must exceed this on re-evaluation
    PROMOTION_MAX_AGE_SECS:         int   = 1800   # 30 min max age for promotion
    PROMOTION_MAX_PER_CYCLE:        int   = 1      # Max promotions per scan cycle

    # ── Confidence Calibration (Feedback Loop) ──────────────────────────
    # Adjust gate threshold based on recent false-negative rate.
    CALIBRATION_ENABLED:            bool  = True
    CALIBRATION_FNR_HIGH:           float = 0.20   # >20% false-neg → gate too strict
    CALIBRATION_FNR_LOW:            float = 0.05   # <5% false-neg → gate OK or too loose
    CALIBRATION_LOOSEN_STEP:        float = -2.0   # Lower threshold by 2 when too strict
    CALIBRATION_TIGHTEN_STEP:       float = 1.0    # Raise threshold by 1 when too loose
    CALIBRATION_INTERVAL_SECS:      int   = 3600   # Re-calibrate every hour

    # ── Signal Decay ─────────────────────────────────────────────────────
    # Confidence degrades as a signal ages (staleness penalty).
    DECAY_ENABLED:                  bool  = True
    DECAY_GRACE_PERIOD_SECS:       int   = 900     # 15 min grace — no decay
    DECAY_RATE_PER_HOUR:           float = 2.0     # Lose 2% confidence per hour after grace
    DECAY_MAX_PENALTY:             float = 15.0    # Max total decay penalty (%)
    DECAY_MIN_CONFIDENCE:          float = 50.0    # Floor — never decay below this

    # ── Macro Semantic Guards ────────────────────────────────────────────
    # Higher-order logic that blocks signals contradicting HTF context.
    SEMANTIC_TREND_GUARD:           bool  = True    # Block SHORT in strong bullish trend etc.
    SEMANTIC_VOLATILITY_GUARD:      bool  = True    # Warn on tight-stop scalps in extreme vol
    SEMANTIC_EXTREME_VOL_LABELS:    tuple = ("EXTREME", "CRISIS")  # Vol regimes to guard
    COUNTERTREND_LOCAL_STRENGTH_MIN: int = 2
    COUNTERTREND_PULLBACK_STRENGTH_MIN: int = 3
    COUNTERTREND_POSITIONING_SCORE_MIN: float = 65.0
    COUNTERTREND_OI_CONFIRM_MIN: float = 8.0
    COUNTERTREND_FUNDING_POSITIVE_EXTREME: float = 0.03
    COUNTERTREND_FUNDING_NEGATIVE_EXTREME: float = -0.01


# ════════════════════════════════════════════════════════════════
# X7. ANALYZER INFRASTRUCTURE (PR-1 of analyzer remediation plan)
# ════════════════════════════════════════════════════════════════
# The classes below extract magic numbers from individual analyzer
# modules into a single auditable place. They intentionally only hold
# *defaults* — each analyzer may still accept an explicit override from
# settings.yaml. The goal is that no new magic numbers enter the
# analyzers/ folder; everything routes through here.
#
# Each class is documented with the analyzer(s) that consume it and a
# citation to the current call-site so auditors can track drift.


class EnergyModel:
    """Defaults for analyzers/energy_model.py (entry-energy scoring).

    The energy model scores how much directional "fuel" a candidate
    trade has left, combining momentum, participation, and positioning.
    All weights sum to 1.0 and are normalised before use.
    """

    # Weights assigned to the four energy components. Must sum to 1.0.
    WEIGHT_MOMENTUM: float = 0.35
    WEIGHT_PARTICIPATION: float = 0.25
    WEIGHT_POSITIONING: float = 0.25
    WEIGHT_FLOW: float = 0.15

    # Energy thresholds (0–100 scale).
    HIGH_ENERGY: float = 70.0        # Above → strong entry
    LOW_ENERGY: float = 35.0         # Below → de-prioritise
    EXHAUSTED: float = 15.0          # Below → block (no fuel left)

    # Half-life in bars for the EWMA used to smooth the raw component scores.
    EWMA_HALFLIFE_BARS: int = 8

    # Minimum number of bars before the energy score is considered valid.
    MIN_BARS_FOR_SCORE: int = 30


class MarketBrain:
    """Defaults for analyzers/market_brain.py (regime + confluence orchestrator)."""

    # Confluence buckets and the confidence boosts they grant.
    CONFLUENCE_STRONG_MIN: int = 4    # ≥4 confirming analyzers → strong boost
    CONFLUENCE_MODERATE_MIN: int = 2  # ≥2 → moderate boost
    CONFLUENCE_STRONG_PTS: int = 6    # Confidence points added at STRONG
    CONFLUENCE_MODERATE_PTS: int = 3  # Confidence points added at MODERATE

    # Conflict detection: if N analyzers disagree directionally with the
    # proposed trade, penalise confidence.
    CONFLICT_THRESHOLD: int = 3       # ≥3 disagreeing → penalty
    CONFLICT_PENALTY_PTS: int = 10

    # Regime stability gate — don't boost confluence when regime just flipped.
    REGIME_STABILITY_MIN_SECS: int = 900   # 15 min
    REGIME_FLIP_COOLDOWN_PENALTY: int = 5

    # Cache TTL for the aggregate market-brain score.
    CACHE_TTL_SECS: int = 60


class NoTradeZones:
    """Defaults for analyzers/no_trade_zones.py (hard-block regions)."""

    # ATR-ratio gate: if current ATR > (ATR_20d * ratio), block — the
    # market has entered an atypical volatility regime. Overridable
    # per-regime downstream (PR-3).
    ATR_BLOCK_RATIO_CHOPPY: float = 2.5
    ATR_BLOCK_RATIO_TREND: float = 3.5
    ATR_BLOCK_RATIO_VOLATILE: float = 5.0

    # Multi-bar confirmation: how many consecutive bars must the ATR
    # ratio be out-of-band before the block fires. Prevents single-tick
    # flash-spikes from freezing the scanner.
    CONFIRMATION_BARS: int = 3

    # Minimum ATR% below which markets are too illiquid to trade.
    MIN_ATR_PCT: float = 0.10   # 0.10 % of price

    # BTC hard-block window around macro events (seconds either side).
    BTC_EVENT_WINDOW_SECS: int = 1800   # ±30 min

    # Price-level no-trade band (round-number magnet). Block trades
    # within this fraction of ATR from a major round number.
    ROUND_NUMBER_ATR_DISTANCE: float = 0.25


class Correlation:
    """Defaults for analyzers/correlation.py (cross-asset correlation / beta)."""

    # Cache TTL (seconds) — cited at analyzers/correlation.py:39
    CACHE_TTL_SECS: int = 1800

    # Lookback window (bars) for rolling correlation / beta.
    LOOKBACK_BARS: int = 168            # 7 days of 1h bars

    # Correlation regime classifications (absolute value).
    STRONG_THRESHOLD: float = 0.70
    MODERATE_THRESHOLD: float = 0.40

    # Beta sanity clamp: prevents a single-sample or mis-aligned series
    # from producing β=42. Anything outside is replaced with the
    # clamped value and flagged. (Wired in PR-3.)
    BETA_CLAMP_MIN: float = -5.0
    BETA_CLAMP_MAX: float = 5.0

    # Minimum sample count before a beta is considered trustworthy.
    MIN_SAMPLES: int = 50


class ParabolicDetector:
    """Defaults for analyzers/parabolic_detector.py (exhaustion detection)."""

    # Cache TTL — cited at analyzers/parabolic_detector.py:47
    CACHE_TTL_SECS: int = 120

    # ROC / acceleration defaults (cited at analyzers/parabolic_detector.py:53-54).
    ROC_PERIOD_BARS: int = 10
    ACCEL_THRESHOLD: float = 0.005

    # Directional threshold for classifying ROC as UP vs DOWN vs FLAT.
    DIR_ROC_THRESHOLD: float = 0.01

    # Exhaustion sub-scores (must sum ≤ 1.0 after PR-2 normalisation).
    EXHAUST_RSI_OVERBOUGHT: float = 0.30
    EXHAUST_RANGE_EXPANSION: float = 0.25
    EXHAUST_RANGE_COMPRESSION: float = 0.15
    EXHAUST_VOLUME_CLIMAX: float = 0.35
    EXHAUST_VOLUME_DRYUP: float = 0.15

    # Number of exhaustion signals that must fire for is_exhausted=True.
    EXHAUSTION_MIN_SIGNALS: int = 2

    # Near-zero price guard — below this, abort (penny-token instability).
    MIN_PRICE_USD: float = 1e-8


class Sentiment:
    """Defaults for analyzers/sentiment.py (Fear & Greed + news sentiment).

    The canonical Fear & Greed extreme thresholds live in
    ``FearGreedThresholds`` (see above) — do not duplicate them.
    """

    # API response bounds; enforced at ingest to clamp upstream errors.
    API_MIN: int = 0
    API_MAX: int = 100

    # EWMA decay for the rolling sentiment series.
    EWMA_DECAY: float = 0.2          # heavier weight on fresh reads

    # Sentiment velocity → confidence adjustment.
    # cap = min(VELOCITY_CAP_PTS, abs(vel) * VELOCITY_PTS_PER_UNIT).
    # Sign is applied by caller based on direction/alignment.
    VELOCITY_CAP_PTS: float = 8.0
    VELOCITY_PTS_PER_UNIT: float = 4.0

    # Refresh cadence for the external Fear & Greed API.
    REFRESH_SECS: int = 900          # 15 min (matches provider SLA)

    # Percentile-history depth used for contextualising current reading.
    HISTORY_MAX_DAYS: int = 365


class BTCDominance:
    """Defaults for analyzers/btc_dominance.py (altcoin rotation gate).

    Mirror of the module-level constants at analyzers/btc_dominance.py:34-46
    so future consumers have a single import path. The analyzer
    definitions are kept for backward compatibility during PR-6 migration.
    """

    CHECK_INTERVAL_SECS: int = 600
    SHARP_RISE_PCT: float = 2.0
    SHARP_FALL_PCT: float = -2.0
    MODERATE_RISE_PCT: float = 1.0
    MODERATE_FALL_PCT: float = -1.0

    ALT_LONG_PENALTY_SHARP: int = -15
    ALT_LONG_PENALTY_MODERATE: int = -8
    ALT_LONG_BOOST_SHARP: int = 10
    ALT_LONG_BOOST_MODERATE: int = 5
    BTC_SHORT_BOOST: int = 10
    BTC_SHORT_PENALTY: int = -5

    # 24h off-by-one window for the rolling change calculation
    # (PR-2 bug-fix target): the sampler should return the reading at
    # now - 86400 ± WINDOW_TOLERANCE_SECS.
    WINDOW_TARGET_SECS: int = 86400
    WINDOW_TOLERANCE_SECS: int = 300   # ±5 min

    # Bounded-history cap (PR-6 bounded-state target).
    HISTORY_MAX_POINTS: int = 2048


class Equilibrium:
    """Defaults for analyzers/equilibrium.py (value-area equilibrium)."""

    # Mirrors analyzers/equilibrium.py:57-58.
    DEAD_ZONE_PCT_DEFAULT: float = 0.10
    DEAD_ZONE_PCT_CHOPPY: float = 0.07

    # PR-4 MAD fallback: when σ of the distribution is below this, use
    # MAD instead so a single outlier doesn't collapse the bands.
    SIGMA_MAD_FALLBACK_THRESHOLD: float = 1e-6

    # Minimum chop regime score before equilibrium rules bind.
    MIN_CHOP_FOR_RULES: float = 0.40

    # Lookback (bars) for the value-area calculation.
    VALUE_AREA_LOOKBACK_BARS: int = 100

    # Value-area width in standard deviations from the POC.
    VALUE_AREA_SIGMA: float = 0.7        # ~70 % VA on Gaussian


class OrderFlow:
    """Defaults for analyzers/orderflow.py (CVD / absorption / block-trade)."""

    # Daily CVD reset (PR-4 target): UTC hour at which cumulative delta
    # is zeroed. 0 (UTC midnight) matches Binance settlement boundaries.
    CVD_RESET_UTC_HOUR: int = 0

    # Absorption flag: bar marked as absorption when |delta/volume|
    # exceeds this ratio *and* price doesn't move in the direction of
    # the delta (i.e. big buying, no higher price → hidden seller).
    ABSORPTION_DELTA_RATIO: float = 0.7
    ABSORPTION_PRICE_TOL_ATR: float = 0.1

    # Block-trade filter: single prints above this USD notional are
    # tagged as block trades and surfaced separately from retail flow.
    BLOCK_TRADE_USD_MIN: float = 250_000.0

    # Delta smoothing — EWMA span (bars) for the CVD slope calculation.
    DELTA_EWMA_SPAN: int = 20

    # Minimum bars of history before orderflow signals are emitted.
    MIN_BARS: int = 30


class LeverageMapper:
    """Defaults for analyzers/leverage_mapper.py (cascade / effective-leverage).

    Mirrors analyzers/leverage_mapper.py:33-35.
    """

    EFFECTIVE_LEVERAGE_EXTREME: float = 0.40
    EFFECTIVE_LEVERAGE_HIGH: float = 0.30
    CASCADE_MIN_USD: float = 5_000_000.0

    # Cluster dwell time (seconds) before a cascade cluster is
    # promoted from "forming" to "confirmed". Prevents single-tick
    # mis-classification.
    CLUSTER_DWELL_SECS: int = 120

    # Cap on leverage penalty (points).
    MAX_PENALTY_PTS: float = 15.0

    # Cache TTL for the mapped output.
    CACHE_TTL_SECS: int = 30


class Liquidation:
    """Defaults for analyzers/liquidation_analyzer.py.

    Mirrors analyzers/liquidation_analyzer.py:48-51 and extends with the
    PR-4 dedup / reconnect-replay targets.
    """

    OI_REFRESH_SECS: int = 300
    LIQ_REFRESH_SECS: int = 600
    EXCHANGE_STAGGER_SECS: float = 1.5
    WALL_MIN_USD: float = 1_000_000.0

    # ATR-scaled bin size for the liquidation heatmap. Bin = ATR / N.
    HEATMAP_BIN_ATR_DIV: float = 4.0

    # Bounded dedup buffer ((exchange, id) → timestamp).
    DEDUP_BUFFER_SIZE: int = 10_000
    DEDUP_TTL_SECS: int = 3600

    # Reconnect-replay window: on WS reconnect, re-fetch the last N
    # seconds of liquidations via REST to avoid gaps.
    REPLAY_WINDOW_SECS: int = 120


class InstitutionalFlow:
    """Defaults for analyzers/institutional_flow.py.

    Mirrors analyzers/institutional_flow.py:56-61.
    """

    FAST_INTERVAL_SECS: int = 300
    SLOW_INTERVAL_SECS: int = 3600
    COT_INTERVAL_SECS: int = 86400
    HISTORY_DAYS: int = 90

    # Percentile window defaults.  30d is currently copied from 90d
    # in the analyzer — PR-4 fixes this. Defining separately here so the
    # fix can reference a named constant.
    PCTILE_30D_WINDOW: int = 30
    PCTILE_90D_WINDOW: int = 90

    # Holiday-aware lookback: when N holidays fall in the last
    # HISTORY_DAYS, extend the window by HOLIDAY_EXTEND_DAYS so the
    # percentile isn't biased by missing prints.
    HOLIDAY_EXTEND_DAYS: int = 3


class Volume:
    """Defaults for analyzers/volume.py (volume / participation scoring).

    Additional thresholds live in ``VolumeQuality`` above — this class
    holds the infrastructure-level defaults (not signal-grading).
    """

    # Log-z-score floor (PR-4): small-floor replacement for zero volume
    # before taking log. Prevents ``log(0) = -inf`` from poisoning the
    # distribution.
    LOG_FLOOR: float = 1.0      # 1 contract / $1 notional

    # Session-of-day detrending: bucket trading hours into N slices and
    # compute z-score within the slice.  24 = hourly buckets.
    SESSION_BUCKETS: int = 24

    # Lookback for computing the z-score baseline.
    BASELINE_BARS: int = 240      # 10 days of 1h bars

    # Minimum samples before a z-score is considered reliable.
    MIN_SAMPLES: int = 30


class VolatilityStructure:
    """Defaults for analyzers/volatility_structure.py.

    Mirrors analyzers/volatility_structure.py:41-48.
    """

    RV_REFRESH_SECS: int = 300
    IV_REFRESH_SECS: int = 1800

    VOL_EXTREME_HIGH_PCTILE: int = 90
    VOL_HIGH_PCTILE: int = 75
    VOL_LOW_PCTILE: int = 25
    VOL_EXTREME_LOW_PCTILE: int = 10

    # ATR-timeframe alignment: RV is computed on 1h bars; if the caller
    # asks about a different TF, scale by sqrt(ratio).
    RV_BASE_TF_MINUTES: int = 60

    # GARCH(1,1) fallback coefficients (PR-4 adds the real estimator).
    # Used as a graceful degradation when the arch library is missing
    # or has too few samples to fit.
    GARCH_OMEGA: float = 0.00001
    GARCH_ALPHA: float = 0.08
    GARCH_BETA: float = 0.90
    GARCH_MIN_SAMPLES: int = 200


STRATEGY_KEY_MAP: dict[str, str] = {
    'SmartMoneyConcepts': 'smc',
    'InstitutionalBreakout': 'breakout',
    'ExtremeReversal': 'reversal',
    'MeanReversion': 'mean_reversion',
    'PriceAction': 'price_action',
    'Momentum': 'momentum',
    'Ichimoku': 'ichimoku',
    'ElliottWave': 'elliott_wave',
    'FundingRateArb': 'funding_arb',
    'RangeScalper': 'range_scalper',
    'WyckoffAccDist': 'wyckoff',
    'HarmonicPattern': 'harmonic',
    'GeometricPattern': 'geometric',
    # Legacy aliases for backward compat with persisted signal data
    'MomentumContinuation': 'momentum',
    'IchimokuCloud': 'ichimoku',
    'Wyckoff': 'wyckoff',
    'HarmonicDetector': 'harmonic',
    'GeometricPatterns': 'geometric',
}

# ════════════════════════════════════════════════════════════════
# X6. STRATEGY VALID-REGIME MAPPING — single source of truth
# ════════════════════════════════════════════════════════════════
# Centralised here so that adding a new Regime value (e.g. BULL_EUPHORIA)
# requires updating exactly one place.  Each strategy's VALID_REGIMES class
# attribute is derived from this dict.
#
# Key  = strategy `name` attribute (same as STRATEGY_FILE_MAP above)
# Value = frozenset of regime strings that the strategy is permitted to run in
#
# Regime strings must match analyzers.regime.Regime enum values:
#   BULL_TREND, BEAR_TREND, CHOPPY, VOLATILE, VOLATILE_PANIC, NEUTRAL

STRATEGY_VALID_REGIMES: "dict[str, frozenset]" = {
    "InstitutionalBreakout": frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE"}),
    "ElliottWave":           frozenset({"BULL_TREND", "BEAR_TREND"}),
    "FundingRateArb":        frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}),
    "Ichimoku":              frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE"}),
    "MeanReversion":         frozenset({"CHOPPY"}),
    "Momentum":              frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE"}),
    "PriceAction":           frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}),
    "ExtremeReversal":       frozenset({"CHOPPY"}),
    "SmartMoneyConcepts":    frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE", "VOLATILE_PANIC", "CHOPPY", "UNKNOWN"}),
    "WyckoffAccDist":        frozenset({"BULL_TREND", "BEAR_TREND", "VOLATILE", "CHOPPY", "UNKNOWN"}),
}
