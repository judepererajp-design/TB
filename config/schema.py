"""
TitanBot Pro — Config Schema Validation
========================================
Pure-Python validation layer using stdlib dataclasses.
No external dependencies (no pydantic, no attrs).

Each config section has a dataclass with a ``validate()`` method that
returns a list of error strings (empty list = valid).

Top-level entry point::

    from config.schema import validate_config
    is_valid, errors = validate_config(raw_dict)
"""

import dataclasses as _dc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from config.constants import Portfolio

# ══════════════════════════════════════════════════════════════
# VALIDATION BOUNDARY CONSTANTS
# ══════════════════════════════════════════════════════════════
# Acceptable ranges for user-tunable settings.
# Referenced by name in error messages — no magic numbers.

# Risk
MIN_RISK_PER_TRADE: float = 0.001
MAX_RISK_PER_TRADE: float = 0.05
MIN_ACCOUNT_SIZE: float = 100.0
MAX_ACCOUNT_SIZE: float = 10_000_000.0
MIN_KELLY_FRACTION: float = 0.01
MAX_KELLY_FRACTION: float = 1.0
MIN_POSITION_PCT: float = 0.001
MAX_POSITION_PCT: float = 1.0
MIN_DAILY_LOSS_PCT: float = 0.001
MAX_DAILY_LOSS_PCT: float = 1.0

# Scan intervals (seconds)
MIN_SCAN_INTERVAL: int = 10
MAX_SCAN_INTERVAL: int = 3600

# Logging
VALID_LOG_LEVELS: frozenset = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# Exchange
MIN_TIMEOUT_MS: int = 1000
MAX_TIMEOUT_MS: int = 120_000
MIN_RETRY_ATTEMPTS: int = 0
MAX_RETRY_ATTEMPTS: int = 20
MIN_RETRY_DELAY: float = 0.1
MAX_RETRY_DELAY: float = 60.0

# System
MIN_WORKERS: int = 1
MAX_WORKERS: int = 100
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 1000
MIN_LOG_BYTES: int = 1024
MAX_LOG_BYTES: int = 1_073_741_824  # 1 GiB

# Telegram
VALID_PARSE_MODES: frozenset = frozenset({"HTML", "Markdown", "MarkdownV2"})
MIN_SIGNALS_PER_HOUR: int = 1
MAX_SIGNALS_PER_HOUR: int = 100

# Cache TTL (seconds)
MIN_CACHE_TTL: int = 1
MAX_CACHE_TTL: int = 86400  # 24 hours

# Aggregator
MIN_CONFIDENCE_FLOOR: int = 0
MAX_CONFIDENCE_CEILING: int = 100
WEIGHT_SUM_TARGET: float = 1.0
WEIGHT_SUM_TOLERANCE: float = 0.01
MIN_EWMA_DECAY: float = 0.0
MAX_EWMA_DECAY: float = 1.0

EXPECTED_WEIGHT_KEYS: frozenset = frozenset(
    {"technical", "volume", "orderflow", "derivatives", "sentiment", "correlation"}
)

# Scanner tier bounds
MIN_TIER_SYMBOLS: int = 1
MAX_TIER_SYMBOLS: int = 500
MIN_VOLUME_24H: float = 0.0

# Global / AI mode
VALID_AI_MODES: frozenset = frozenset({"full", "off", "diagnosis_only"})

# Database pool
MIN_DB_POOL_SIZE: int = 1
MAX_DB_POOL_SIZE: int = 50
MIN_DB_TIMEOUT: float = 0.1
MAX_DB_TIMEOUT: float = 300.0

# Governance
MIN_WIN_RATE: float = 0.0
MAX_WIN_RATE: float = 1.0

# Time filters
MIN_HOUR_UTC: int = 0
MAX_HOUR_UTC: int = 23

# Strategy confidence base
MIN_CONFIDENCE_BASE: int = 0
MAX_CONFIDENCE_BASE: int = 100

# Valid set of strategy keys expected in the ``strategies`` section.
# Keep in sync with the strategies registered in signals/aggregator.py.
EXPECTED_STRATEGY_KEYS: frozenset = frozenset({
    "smc", "breakout", "reversal", "mean_reversion", "price_action",
    "momentum", "ichimoku", "elliott_wave", "funding_arb", "range_scalper",
    "wyckoff", "harmonic", "geometric",
})

# Valid set of analyzer keys expected in the ``analyzers`` section.
EXPECTED_ANALYZER_KEYS: frozenset = frozenset({
    "volume", "orderflow", "derivatives", "regime",
    "rotation", "sentiment", "correlation",
})


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════


def _check_type(section: str, key: str, value: Any, expected: type) -> List[str]:
    """Return an error if *value* is not an instance of *expected*.

    ``int`` is accepted where ``float`` is expected (e.g. 5 for 5.0).
    """
    if expected is float and isinstance(value, (int, float)):
        return []
    if not isinstance(value, expected):
        return [
            f"{section}.{key}: expected {expected.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        ]
    return []


def _check_range(
    section: str,
    key: str,
    value: Any,
    lo: Any = None,
    hi: Any = None,
) -> List[str]:
    """Return an error if *value* is outside [lo, hi]."""
    errors: List[str] = []
    if lo is not None and value < lo:
        errors.append(f"{section}.{key}={value} is below minimum {lo}")
    if hi is not None and value > hi:
        errors.append(f"{section}.{key}={value} is above maximum {hi}")
    return errors


def _check_required(section: str, data: dict, keys: List[str]) -> List[str]:
    """Return errors for any missing required keys in *data*."""
    return [
        f"{section}: missing required field '{k}'"
        for k in keys
        if k not in data
    ]


# ══════════════════════════════════════════════════════════════
# SECTION DATACLASSES
# ══════════════════════════════════════════════════════════════


@dataclass
class SystemConfig:
    """Validates the ``system`` section."""

    log_level: str = "INFO"
    tier1_interval: int = 120
    tier2_interval: int = 300
    tier3_interval: int = 900
    async_workers: int = 10
    batch_size: int = 10
    max_symbols: int = 200
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "system"

        # log_level
        if self.log_level not in VALID_LOG_LEVELS:
            errors.append(
                f"{sec}.log_level='{self.log_level}' is not valid; "
                f"expected one of {sorted(VALID_LOG_LEVELS)}"
            )

        # Tier intervals — type, range, and ordering
        for name, val in [
            ("tier1_interval", self.tier1_interval),
            ("tier2_interval", self.tier2_interval),
            ("tier3_interval", self.tier3_interval),
        ]:
            errors.extend(_check_type(sec, name, val, int))
            if isinstance(val, int):
                errors.extend(
                    _check_range(sec, name, val, MIN_SCAN_INTERVAL, MAX_SCAN_INTERVAL)
                )

        if self.tier1_interval > self.tier2_interval:
            errors.append(
                f"{sec}: tier1_interval ({self.tier1_interval}) "
                f"should not exceed tier2_interval ({self.tier2_interval})"
            )
        if self.tier2_interval > self.tier3_interval:
            errors.append(
                f"{sec}: tier2_interval ({self.tier2_interval}) "
                f"should not exceed tier3_interval ({self.tier3_interval})"
            )

        # Workers / batch
        errors.extend(_check_type(sec, "async_workers", self.async_workers, int))
        errors.extend(
            _check_range(sec, "async_workers", self.async_workers, MIN_WORKERS, MAX_WORKERS)
        )
        errors.extend(_check_type(sec, "batch_size", self.batch_size, int))
        errors.extend(
            _check_range(sec, "batch_size", self.batch_size, MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        )

        # Log rotation
        errors.extend(_check_type(sec, "log_max_bytes", self.log_max_bytes, int))
        errors.extend(
            _check_range(sec, "log_max_bytes", self.log_max_bytes, MIN_LOG_BYTES, MAX_LOG_BYTES)
        )

        return errors


@dataclass
class ExchangeConfig:
    """Validates the ``exchange`` section."""

    name: str = "binance"
    timeout_ms: int = 30000
    retry_attempts: int = 5
    retry_delay: float = 3.0

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "exchange"

        errors.extend(_check_type(sec, "name", self.name, str))

        errors.extend(_check_type(sec, "timeout_ms", self.timeout_ms, int))
        errors.extend(
            _check_range(sec, "timeout_ms", self.timeout_ms, MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)
        )

        errors.extend(_check_type(sec, "retry_attempts", self.retry_attempts, int))
        errors.extend(
            _check_range(
                sec, "retry_attempts", self.retry_attempts,
                MIN_RETRY_ATTEMPTS, MAX_RETRY_ATTEMPTS,
            )
        )

        errors.extend(_check_type(sec, "retry_delay", self.retry_delay, float))
        errors.extend(
            _check_range(
                sec, "retry_delay", self.retry_delay,
                MIN_RETRY_DELAY, MAX_RETRY_DELAY,
            )
        )

        return errors


@dataclass
class TelegramConfig:
    """Validates the ``telegram`` section."""

    bot_token: str = ""
    chat_id: str = ""
    max_signals_per_hour: int = 12
    parse_mode: str = "HTML"
    send_signals: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "telegram"

        errors.extend(
            _check_type(sec, "max_signals_per_hour", self.max_signals_per_hour, int)
        )
        errors.extend(
            _check_range(
                sec, "max_signals_per_hour", self.max_signals_per_hour,
                MIN_SIGNALS_PER_HOUR, MAX_SIGNALS_PER_HOUR,
            )
        )

        if self.parse_mode not in VALID_PARSE_MODES:
            errors.append(
                f"{sec}.parse_mode='{self.parse_mode}' is not valid; "
                f"expected one of {sorted(VALID_PARSE_MODES)}"
            )

        return errors


@dataclass
class RiskConfig:
    """Validates the ``risk`` section."""

    risk_per_trade: float = 0.007
    max_risk_per_trade: float = 0.012
    min_risk_per_trade: float = 0.003
    account_size: float = 5000.0
    account_balance: float = 5000.0
    min_rr: float = 2.0
    max_rr: float = 6.0
    target_rr: float = 2.5
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_concurrent_signals: int = 4
    max_correlated_positions: int = 2
    max_daily_loss_pct: float = 0.03

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "risk"

        # ── risk_per_trade bounds ────────────────────────────────
        errors.extend(_check_type(sec, "risk_per_trade", self.risk_per_trade, float))
        errors.extend(
            _check_range(
                sec, "risk_per_trade", self.risk_per_trade,
                MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE,
            )
        )

        # max >= default >= min ordering
        errors.extend(
            _check_type(sec, "max_risk_per_trade", self.max_risk_per_trade, float)
        )
        if self.max_risk_per_trade < self.risk_per_trade:
            errors.append(
                f"{sec}: max_risk_per_trade ({self.max_risk_per_trade}) "
                f"must be >= risk_per_trade ({self.risk_per_trade})"
            )

        errors.extend(
            _check_type(sec, "min_risk_per_trade", self.min_risk_per_trade, float)
        )
        if self.min_risk_per_trade > self.risk_per_trade:
            errors.append(
                f"{sec}: min_risk_per_trade ({self.min_risk_per_trade}) "
                f"must be <= risk_per_trade ({self.risk_per_trade})"
            )

        # ── Account size ─────────────────────────────────────────
        errors.extend(_check_type(sec, "account_size", self.account_size, float))
        errors.extend(
            _check_range(
                sec, "account_size", self.account_size,
                MIN_ACCOUNT_SIZE, MAX_ACCOUNT_SIZE,
            )
        )

        # ── R:R ordering ─────────────────────────────────────────
        if self.min_rr > self.max_rr:
            errors.append(
                f"{sec}: min_rr ({self.min_rr}) must be <= max_rr ({self.max_rr})"
            )
        if self.target_rr < self.min_rr:
            errors.append(
                f"{sec}: target_rr ({self.target_rr}) should be >= "
                f"min_rr ({self.min_rr})"
            )

        # ── Kelly fraction ───────────────────────────────────────
        errors.extend(_check_type(sec, "kelly_fraction", self.kelly_fraction, float))
        errors.extend(
            _check_range(
                sec, "kelly_fraction", self.kelly_fraction,
                MIN_KELLY_FRACTION, MAX_KELLY_FRACTION,
            )
        )

        # ── Position sizing ──────────────────────────────────────
        errors.extend(
            _check_type(sec, "max_position_pct", self.max_position_pct, float)
        )
        errors.extend(
            _check_range(
                sec, "max_position_pct", self.max_position_pct,
                MIN_POSITION_PCT, MAX_POSITION_PCT,
            )
        )

        # ── Daily loss cap ───────────────────────────────────────
        errors.extend(
            _check_type(sec, "max_daily_loss_pct", self.max_daily_loss_pct, float)
        )
        errors.extend(
            _check_range(
                sec, "max_daily_loss_pct", self.max_daily_loss_pct,
                MIN_DAILY_LOSS_PCT, MAX_DAILY_LOSS_PCT,
            )
        )

        # ── Concurrent signals vs portfolio constant ─────────────
        if self.max_concurrent_signals > Portfolio.MAX_POSITIONS:
            errors.append(
                f"{sec}: max_concurrent_signals ({self.max_concurrent_signals}) "
                f"exceeds Portfolio.MAX_POSITIONS ({Portfolio.MAX_POSITIONS})"
            )

        return errors


@dataclass
class ScannerConfig:
    """Validates the ``scanning`` section."""

    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    tier3: dict = field(default_factory=dict)
    quote_currency: str = "USDT"

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "scanning"

        for tier_name, tier_data in [
            ("tier1", self.tier1),
            ("tier2", self.tier2),
            ("tier3", self.tier3),
        ]:
            if not isinstance(tier_data, dict):
                errors.append(
                    f"{sec}.{tier_name}: expected a mapping, "
                    f"got {type(tier_data).__name__}"
                )
                continue

            max_sym = tier_data.get("max_symbols")
            if max_sym is not None:
                if not isinstance(max_sym, int):
                    errors.append(
                        f"{sec}.{tier_name}.max_symbols: expected int, "
                        f"got {type(max_sym).__name__}"
                    )
                else:
                    errors.extend(
                        _check_range(
                            f"{sec}.{tier_name}", "max_symbols", max_sym,
                            MIN_TIER_SYMBOLS, MAX_TIER_SYMBOLS,
                        )
                    )

            min_vol = tier_data.get("min_volume_24h")
            if min_vol is not None:
                errors.extend(
                    _check_type(f"{sec}.{tier_name}", "min_volume_24h", min_vol, float)
                )
                if isinstance(min_vol, (int, float)):
                    errors.extend(
                        _check_range(
                            f"{sec}.{tier_name}", "min_volume_24h", min_vol,
                            MIN_VOLUME_24H,
                        )
                    )

        # Volume ordering: tier1 >= tier2 >= tier3
        t1v = self.tier1.get("min_volume_24h", 0) if isinstance(self.tier1, dict) else 0
        t2v = self.tier2.get("min_volume_24h", 0) if isinstance(self.tier2, dict) else 0
        t3v = self.tier3.get("min_volume_24h", 0) if isinstance(self.tier3, dict) else 0
        if t1v < t2v:
            errors.append(
                f"{sec}: tier1.min_volume_24h ({t1v}) should be >= "
                f"tier2.min_volume_24h ({t2v})"
            )
        if t2v < t3v:
            errors.append(
                f"{sec}: tier2.min_volume_24h ({t2v}) should be >= "
                f"tier3.min_volume_24h ({t3v})"
            )

        return errors


@dataclass
class AggregatorConfig:
    """Validates the ``aggregator`` section."""

    min_confidence: int = 60
    weights: dict = field(default_factory=dict)
    adaptation: dict = field(default_factory=dict)
    max_signals_per_symbol: int = 3
    dedup_window_minutes: int = 180

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "aggregator"

        # ── min_confidence ───────────────────────────────────────
        errors.extend(_check_type(sec, "min_confidence", self.min_confidence, int))
        errors.extend(
            _check_range(
                sec, "min_confidence", self.min_confidence,
                MIN_CONFIDENCE_FLOOR, MAX_CONFIDENCE_CEILING,
            )
        )

        # ── weights ──────────────────────────────────────────────
        if not isinstance(self.weights, dict):
            errors.append(
                f"{sec}.weights: expected a mapping, "
                f"got {type(self.weights).__name__}"
            )
        else:
            missing_keys = EXPECTED_WEIGHT_KEYS - set(self.weights.keys())
            if missing_keys:
                errors.append(
                    f"{sec}.weights: missing keys {sorted(missing_keys)}"
                )

            total = 0.0
            for k, v in self.weights.items():
                if not isinstance(v, (int, float)):
                    errors.append(
                        f"{sec}.weights.{k}: expected float, "
                        f"got {type(v).__name__}"
                    )
                else:
                    total += float(v)

            if abs(total - WEIGHT_SUM_TARGET) > WEIGHT_SUM_TOLERANCE:
                errors.append(
                    f"{sec}.weights sum to {total:.4f}, "
                    f"must be {WEIGHT_SUM_TARGET} "
                    f"(±{WEIGHT_SUM_TOLERANCE})"
                )

        # ── adaptation ───────────────────────────────────────────
        if isinstance(self.adaptation, dict) and self.adaptation:
            decay = self.adaptation.get("ewma_decay")
            if decay is not None:
                errors.extend(
                    _check_type(sec, "adaptation.ewma_decay", decay, float)
                )
                if isinstance(decay, (int, float)):
                    errors.extend(
                        _check_range(
                            sec, "adaptation.ewma_decay", decay,
                            MIN_EWMA_DECAY, MAX_EWMA_DECAY,
                        )
                    )

        return errors


@dataclass
class CacheConfig:
    """Validates the ``cache`` section."""

    ohlcv_ttl: int = 60
    derivatives_ttl: int = 30
    orderflow_ttl: int = 3
    regime_ttl: int = 300
    sentiment_ttl: int = 1800
    universe_ttl: int = 3600

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "cache"

        for name in (
            "ohlcv_ttl",
            "derivatives_ttl",
            "orderflow_ttl",
            "regime_ttl",
            "sentiment_ttl",
            "universe_ttl",
        ):
            val = getattr(self, name)
            errors.extend(_check_type(sec, name, val, int))
            if isinstance(val, int):
                errors.extend(
                    _check_range(sec, name, val, MIN_CACHE_TTL, MAX_CACHE_TTL)
                )

        return errors


@dataclass
class GlobalConfig:
    """Validates the ``global`` section."""

    debug_mode: bool = False
    test_mode: bool = False
    paper_trading: bool = True
    timezone: str = "UTC"
    version: str = ""
    openrouter_api_key: str = ""
    ai_mode: str = "full"
    hypertracker_api_key: str = ""
    hypertracker_watched_wallets: list = field(default_factory=list)

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "global"

        errors.extend(_check_type(sec, "debug_mode",   self.debug_mode,   bool))
        errors.extend(_check_type(sec, "test_mode",    self.test_mode,    bool))
        errors.extend(_check_type(sec, "paper_trading", self.paper_trading, bool))
        errors.extend(_check_type(sec, "timezone",     self.timezone,     str))
        errors.extend(_check_type(sec, "ai_mode",      self.ai_mode,      str))

        if isinstance(self.ai_mode, str) and self.ai_mode not in VALID_AI_MODES:
            errors.append(
                f"{sec}.ai_mode='{self.ai_mode}' is not valid; "
                f"expected one of {sorted(VALID_AI_MODES)}"
            )

        if not isinstance(self.hypertracker_watched_wallets, list):
            errors.append(
                f"{sec}.hypertracker_watched_wallets: expected list, "
                f"got {type(self.hypertracker_watched_wallets).__name__}"
            )

        return errors


@dataclass
class DatabaseConfig:
    """Validates the ``database`` section."""

    path: str = "data/titanbot.db"
    wal_mode: bool = True
    pool_size: int = 3
    timeout: float = 5.0
    enable_backup: bool = True
    backup_interval: int = 86400
    backup_path: str = "data/backups/"

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "database"

        errors.extend(_check_type(sec, "path", self.path, str))
        errors.extend(_check_type(sec, "wal_mode", self.wal_mode, bool))
        errors.extend(_check_type(sec, "enable_backup", self.enable_backup, bool))
        errors.extend(_check_type(sec, "backup_path", self.backup_path, str))

        errors.extend(_check_type(sec, "pool_size", self.pool_size, int))
        if isinstance(self.pool_size, int):
            errors.extend(
                _check_range(sec, "pool_size", self.pool_size,
                             MIN_DB_POOL_SIZE, MAX_DB_POOL_SIZE)
            )

        errors.extend(_check_type(sec, "timeout", self.timeout, float))
        if isinstance(self.timeout, (int, float)):
            errors.extend(
                _check_range(sec, "timeout", self.timeout,
                             MIN_DB_TIMEOUT, MAX_DB_TIMEOUT)
            )

        errors.extend(_check_type(sec, "backup_interval", self.backup_interval, int))
        if isinstance(self.backup_interval, int) and self.backup_interval < 0:
            errors.append(
                f"{sec}.backup_interval={self.backup_interval} must be >= 0"
            )

        return errors


@dataclass
class GovernanceConfig:
    """Validates the ``governance`` section."""

    outcome_tracking: bool = True
    report_windows: list = field(default_factory=lambda: [7, 30, 90])
    auto_disable: dict = field(default_factory=dict)
    health_check_interval: int = 60
    unhealthy_threshold: int = 3

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "governance"

        errors.extend(_check_type(sec, "outcome_tracking", self.outcome_tracking, bool))

        if not isinstance(self.report_windows, list):
            errors.append(
                f"{sec}.report_windows: expected list, "
                f"got {type(self.report_windows).__name__}"
            )
        else:
            for i, w in enumerate(self.report_windows):
                if not isinstance(w, int) or w <= 0:
                    errors.append(
                        f"{sec}.report_windows[{i}]={w!r} must be a positive int"
                    )

        if isinstance(self.auto_disable, dict) and self.auto_disable:
            enabled = self.auto_disable.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"{sec}.auto_disable.enabled: expected bool, "
                    f"got {type(enabled).__name__}"
                )
            wr = self.auto_disable.get("min_win_rate")
            if wr is not None:
                if not isinstance(wr, (int, float)):
                    errors.append(
                        f"{sec}.auto_disable.min_win_rate: expected float, "
                        f"got {type(wr).__name__}"
                    )
                else:
                    errors.extend(
                        _check_range(sec, "auto_disable.min_win_rate", float(wr),
                                     MIN_WIN_RATE, MAX_WIN_RATE)
                    )
            ms = self.auto_disable.get("min_signals")
            if ms is not None and (not isinstance(ms, int) or ms < 0):
                errors.append(
                    f"{sec}.auto_disable.min_signals={ms!r} must be a non-negative int"
                )
            rd = self.auto_disable.get("re_enable_after_days")
            if rd is not None and (not isinstance(rd, int) or rd < 0):
                errors.append(
                    f"{sec}.auto_disable.re_enable_after_days={rd!r} "
                    f"must be a non-negative int"
                )

        errors.extend(
            _check_type(sec, "health_check_interval", self.health_check_interval, int)
        )
        if isinstance(self.health_check_interval, int) and self.health_check_interval <= 0:
            errors.append(
                f"{sec}.health_check_interval={self.health_check_interval} must be > 0"
            )

        errors.extend(
            _check_type(sec, "unhealthy_threshold", self.unhealthy_threshold, int)
        )
        if isinstance(self.unhealthy_threshold, int) and self.unhealthy_threshold <= 0:
            errors.append(
                f"{sec}.unhealthy_threshold={self.unhealthy_threshold} must be > 0"
            )

        return errors


@dataclass
class TimeFiltersConfig:
    """Validates the ``time_filters`` section."""

    dead_zone_start_utc: int = 3
    dead_zone_end_utc: int = 7
    enable_dead_zone: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "time_filters"

        errors.extend(_check_type(sec, "enable_dead_zone", self.enable_dead_zone, bool))

        for name, val in [
            ("dead_zone_start_utc", self.dead_zone_start_utc),
            ("dead_zone_end_utc",   self.dead_zone_end_utc),
        ]:
            errors.extend(_check_type(sec, name, val, int))
            if isinstance(val, int):
                errors.extend(
                    _check_range(sec, name, val, MIN_HOUR_UTC, MAX_HOUR_UTC)
                )

        return errors


@dataclass
class PatternsConfig:
    """Validates the ``patterns`` section (harmonic / geometric / wyckoff)."""

    harmonic: dict = field(default_factory=dict)
    geometric: dict = field(default_factory=dict)
    wyckoff: dict = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "patterns"

        for name, data in [
            ("harmonic",  self.harmonic),
            ("geometric", self.geometric),
            ("wyckoff",   self.wyckoff),
        ]:
            if not isinstance(data, dict):
                errors.append(
                    f"{sec}.{name}: expected a mapping, got {type(data).__name__}"
                )
                continue
            enabled = data.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"{sec}.{name}.enabled: expected bool, "
                    f"got {type(enabled).__name__}"
                )

        # Harmonic-specific numeric bounds
        if isinstance(self.harmonic, dict):
            tol = self.harmonic.get("tolerance")
            if tol is not None:
                if not isinstance(tol, (int, float)):
                    errors.append(
                        f"{sec}.harmonic.tolerance: expected float, "
                        f"got {type(tol).__name__}"
                    )
                elif not (0.0 < float(tol) <= 1.0):
                    errors.append(
                        f"{sec}.harmonic.tolerance={tol} must be in (0, 1]"
                    )

        # Geometric-specific
        if isinstance(self.geometric, dict):
            ct = self.geometric.get("confidence_threshold")
            if ct is not None:
                if not isinstance(ct, int):
                    errors.append(
                        f"{sec}.geometric.confidence_threshold: expected int, "
                        f"got {type(ct).__name__}"
                    )
                else:
                    errors.extend(
                        _check_range(sec, "geometric.confidence_threshold", ct,
                                     MIN_CONFIDENCE_FLOOR, MAX_CONFIDENCE_CEILING)
                    )

        return errors


class StrategiesConfig:
    """Validates the ``strategies`` section.

    Each sub-strategy is a mapping. This validator enforces structural
    invariants (``enabled`` is bool, ``confidence_base`` in 0-100,
    ``min_volume_24h`` non-negative) without asserting every strategy-specific
    numeric parameter — those are validated at the strategy level.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: Dict[str, Any] = dict(kwargs)

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "strategies"

        unknown = set(self._data.keys()) - EXPECTED_STRATEGY_KEYS
        if unknown:
            errors.append(
                f"{sec}: unknown strategy keys {sorted(unknown)}; "
                f"expected subset of {sorted(EXPECTED_STRATEGY_KEYS)}"
            )

        for name, cfg in self._data.items():
            if not isinstance(cfg, dict):
                errors.append(
                    f"{sec}.{name}: expected a mapping, got {type(cfg).__name__}"
                )
                continue

            enabled = cfg.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"{sec}.{name}.enabled: expected bool, "
                    f"got {type(enabled).__name__}"
                )

            cb = cfg.get("confidence_base")
            if cb is not None:
                if not isinstance(cb, int):
                    errors.append(
                        f"{sec}.{name}.confidence_base: expected int, "
                        f"got {type(cb).__name__}"
                    )
                else:
                    errors.extend(
                        _check_range(
                            f"{sec}.{name}", "confidence_base", cb,
                            MIN_CONFIDENCE_BASE, MAX_CONFIDENCE_BASE,
                        )
                    )

            mv = cfg.get("min_volume_24h")
            if mv is not None:
                if not isinstance(mv, (int, float)):
                    errors.append(
                        f"{sec}.{name}.min_volume_24h: expected number, "
                        f"got {type(mv).__name__}"
                    )
                elif float(mv) < 0:
                    errors.append(
                        f"{sec}.{name}.min_volume_24h={mv} must be >= 0"
                    )

            tf = cfg.get("timeframe")
            if tf is not None and not isinstance(tf, str):
                errors.append(
                    f"{sec}.{name}.timeframe: expected str, got {type(tf).__name__}"
                )
            tfs = cfg.get("timeframes")
            if tfs is not None:
                if not isinstance(tfs, list) or not all(isinstance(x, str) for x in tfs):
                    errors.append(
                        f"{sec}.{name}.timeframes: expected list of str"
                    )

        return errors


class AnalyzersConfig:
    """Validates the ``analyzers`` section.

    Performs structural checks only: each analyzer is a mapping, and an
    ``enabled`` flag (if present) is a bool.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: Dict[str, Any] = dict(kwargs)

    def validate(self) -> List[str]:
        errors: List[str] = []
        sec = "analyzers"

        unknown = set(self._data.keys()) - EXPECTED_ANALYZER_KEYS
        if unknown:
            errors.append(
                f"{sec}: unknown analyzer keys {sorted(unknown)}; "
                f"expected subset of {sorted(EXPECTED_ANALYZER_KEYS)}"
            )

        for name, cfg in self._data.items():
            if not isinstance(cfg, dict):
                errors.append(
                    f"{sec}.{name}: expected a mapping, got {type(cfg).__name__}"
                )
                continue
            enabled = cfg.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"{sec}.{name}.enabled: expected bool, "
                    f"got {type(enabled).__name__}"
                )
            ttl = cfg.get("cache_ttl")
            if ttl is not None:
                if not isinstance(ttl, int):
                    errors.append(
                        f"{sec}.{name}.cache_ttl: expected int, "
                        f"got {type(ttl).__name__}"
                    )
                else:
                    errors.extend(
                        _check_range(
                            f"{sec}.{name}", "cache_ttl", ttl,
                            MIN_CACHE_TTL, MAX_CACHE_TTL,
                        )
                    )

        # Sentiment extreme thresholds ordering
        sent = self._data.get("sentiment")
        if isinstance(sent, dict):
            lo = sent.get("extreme_fear_threshold")
            hi = sent.get("extreme_greed_threshold")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo >= hi:
                errors.append(
                    f"{sec}.sentiment: extreme_fear_threshold ({lo}) must be < "
                    f"extreme_greed_threshold ({hi})"
                )

        return errors


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL VALIDATOR
# ══════════════════════════════════════════════════════════════

# (yaml_section_name, dataclass_class, required_fields_within_section)
_SECTION_REGISTRY: List[Tuple[str, type, List[str]]] = [
    ("global",     GlobalConfig,     []),
    ("system",     SystemConfig,     ["log_level", "tier1_interval"]),
    ("exchange",   ExchangeConfig,   ["name", "timeout_ms"]),
    ("telegram",   TelegramConfig,   []),
    ("risk",       RiskConfig,       ["risk_per_trade"]),
    ("scanning",   ScannerConfig,    []),
    ("strategies", StrategiesConfig, []),
    ("analyzers",  AnalyzersConfig,  []),
    ("aggregator", AggregatorConfig, ["min_confidence", "weights"]),
    ("database",   DatabaseConfig,   []),
    ("patterns",   PatternsConfig,   []),
    ("governance", GovernanceConfig, []),
    ("cache",      CacheConfig,      []),
    ("time_filters", TimeFiltersConfig, []),
]


def _build_dataclass(cls: type, data: dict) -> Any:
    """Instantiate *cls* using only the keys it declares as fields.

    Classes that are not dataclasses (or declare no fields) are treated as
    "open schema" and receive the full section dict verbatim so they can
    validate the whole mapping.
    """
    if _dc.is_dataclass(cls):
        field_names = {f.name for f in _dc.fields(cls)}
        if not field_names:
            return cls(**data)
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
    # Non-dataclass: pass everything
    return cls(**data)


def validate_config(raw: dict) -> Tuple[bool, List[str]]:
    """Validate a raw settings dict against all section schemas.

    Returns
    -------
    (is_valid, error_messages)
        ``is_valid`` is ``True`` when *error_messages* is empty.
    """
    all_errors: List[str] = []

    for section_name, cls, required_fields in _SECTION_REGISTRY:
        section_data = raw.get(section_name)

        if section_data is None:
            if required_fields:
                all_errors.append(f"Missing config section: '{section_name}'")
            continue

        if not isinstance(section_data, dict):
            all_errors.append(
                f"Section '{section_name}' should be a mapping, "
                f"got {type(section_data).__name__}"
            )
            continue

        # Check that required fields exist inside the section
        all_errors.extend(
            _check_required(section_name, section_data, required_fields)
        )

        # Build the dataclass and run its validate() method
        try:
            instance = _build_dataclass(cls, section_data)
            all_errors.extend(instance.validate())
        except (TypeError, ValueError) as exc:
            all_errors.append(f"{section_name}: failed to parse — {exc}")

    return (len(all_errors) == 0, all_errors)
