"""
TitanBot Pro — Config Loader
============================
Singleton that loads settings.yaml once and provides typed access
to every setting in the system. All modules import from here.

Usage:
    from config.loader import cfg
    interval = cfg.system.tier1_interval
    min_conf = cfg.aggregator.min_confidence
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - tiny compatibility shim
    def load_dotenv(*args, **kwargs):
        return False

from config.schema import validate_config

logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()


class ConfigNode:
    """
    Wraps a dict so you can access keys with dot notation.
    cfg.system.tier1_interval instead of cfg['system']['tier1_interval']
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        return self._data


class Config:
    """
    Master config object. Loaded once at startup.
    All settings are accessible via dot notation.
    """

    _instance: Optional['Config'] = None
    _config_path: Path = Path(__file__).parent / "settings.yaml"

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    @property
    def global_cfg(self):
        """Safe access to 'global' section (which is a Python reserved word)"""
        return getattr(self, 'global', ConfigNode({}))

    def _load(self):
        """Load and parse settings.yaml"""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Settings file not found: {self._config_path}\n"
                f"Expected at: {self._config_path.absolute()}"
            )

        with open(self._config_path, 'r') as f:
            raw = yaml.safe_load(f)

        if not raw:
            raise ValueError("settings.yaml is empty or invalid")

        # Inject environment variables into config
        self._raw = self._inject_env(raw)

        # Create dot-accessible nodes for each top-level section
        for section, data in self._raw.items():
            if isinstance(data, dict):
                setattr(self, section, ConfigNode(data))
            else:
                setattr(self, section, data)

        logger.info(f"Config loaded from {self._config_path}")

    def _inject_env(self, raw: Dict) -> Dict:
        """Replace ENV-sourced values with actual environment variables.

        All secrets MUST come from environment variables — never from YAML.
        YAML stores placeholders like ${VAR_NAME}; this method resolves them.
        Missing required keys log a clear warning instead of silently sending
        empty strings to external APIs.
        """

        def _resolve(value: Any) -> Any:
            """Recursively resolve ${VAR} placeholders in string values.

            Supports both exact matches (``"${VAR}"``) and embedded
            interpolation (``"https://${HOST}:${PORT}/api"``). Missing
            variables resolve to the empty string with a one-time warning
            so we never silently ship a literal ``${...}`` to an external
            service.
            """
            if isinstance(value, str) and "${" in value:
                missing: list = []

                def _sub(match: "re.Match[str]") -> str:
                    var_name = match.group(1)
                    resolved = os.getenv(var_name, "")
                    if not resolved:
                        missing.append(var_name)
                    return resolved

                new_value = re.sub(r"\$\{(\w+)\}", _sub, value)
                for var_name in missing:
                    logger.warning(
                        f"Config: env var '{var_name}' is not set. "
                        f"Add it to your .env file. Functionality requiring "
                        f"this key will be disabled."
                    )
                return new_value
            if isinstance(value, dict):
                return {k: _resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_resolve(i) for i in value]
            return value

        raw = _resolve(raw)

        # Telegram — always from env, never from YAML
        if 'telegram' in raw:
            raw['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', '')
            raw['telegram']['chat_id']   = os.getenv('TELEGRAM_CHAT_ID', '')
            _market = os.getenv('TELEGRAM_MARKET_CHAT_ID', '')
            if _market:
                raw['telegram']['market_chat_id'] = _market
            _admin = os.getenv('TELEGRAM_ADMIN_CHAT_ID', '')
            if _admin:
                raw['telegram']['admin_chat_id'] = _admin
            admin_ids_str = os.getenv('TELEGRAM_ADMIN_IDS', '')
            _valid_ids = []
            for x in admin_ids_str.split(','):
                x = x.strip()
                if not x:
                    continue
                # Accept negative IDs (group chats are negative in Telegram).
                try:
                    _valid_ids.append(int(x))
                except ValueError:
                    logger.warning(
                        f"TELEGRAM_ADMIN_IDS: skipping non-numeric entry '{x}'"
                    )
            raw['telegram']['admin_ids'] = _valid_ids

        # AI / Intelligence API keys — injected from env, overriding YAML placeholders
        if 'global' in raw:
            _or_key = os.getenv('OPENROUTER_API_KEY', raw['global'].get('openrouter_api_key', ''))
            raw['global']['openrouter_api_key'] = _or_key if _or_key and not _or_key.startswith('${') else ''

            # HyperTracker: no API key needed, uses free Hyperliquid public API
            raw['global']['hypertracker_api_key'] = ''

        # Dashboard auth token
        raw['_dashboard_auth_token'] = os.getenv('DASHBOARD_AUTH_TOKEN', '')

        return raw

    def reload(self):
        """Hot-reload config without restart"""
        self._loaded = False
        self._load()
        logger.info("Config reloaded")

    def get_strategy_config(self, strategy_name: str) -> Optional[ConfigNode]:
        """Get config for a specific strategy by name"""
        strategies = getattr(self, 'strategies', None)
        if strategies:
            return getattr(strategies, strategy_name.lower(), None)
        return None

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        strategy_cfg = self.get_strategy_config(strategy_name)
        if strategy_cfg is None:
            return False
        return getattr(strategy_cfg, 'enabled', False)

    def validate(self) -> bool:
        """Validate critical config values.

        Raises RuntimeError for required secrets that are absent at startup —
        an empty TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID means the bot cannot
        function and should fail fast rather than run silently broken.
        """
        errors = []

        # ── Schema validation (type / range / structure checks) ──────────
        schema_ok, schema_errors = validate_config(self._raw)
        if not schema_ok:
            for err in schema_errors:
                logger.error(f"Schema validation: {err}")
            errors.extend(schema_errors)

        # ── Required secrets — fail fast with RuntimeError ───────────────
        telegram_cfg = getattr(self, 'telegram', None)
        if telegram_cfg is None:
            raise RuntimeError(
                "settings.yaml is missing a 'telegram' section. "
                "TitanBot cannot start without Telegram credentials."
            )
        bot_token = telegram_cfg.get('bot_token', '')
        chat_id   = telegram_cfg.get('chat_id', '')
        if not bot_token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is not set. "
                "Add it to your .env file before starting TitanBot."
            )
        if not chat_id:
            raise RuntimeError(
                "TELEGRAM_CHAT_ID is not set. "
                "Add it to your .env file before starting TitanBot."
            )

        # ── TELEGRAM_CHAT_ID format — must be an integer (optionally negative) ──
        if not re.match(r'^-?\d+$', str(chat_id).strip()):
            errors.append(
                f"TELEGRAM_CHAT_ID '{chat_id}' is not a valid Telegram chat ID "
                f"(expected a numeric value, e.g. -1001234567890)"
            )

        # ── Risk per trade bounds ────────────────────────────────────────
        risk_cfg = getattr(self, 'risk', None)
        if risk_cfg is not None:
            rpt = float(risk_cfg.get('risk_per_trade', 0.007))
            if not (0.001 <= rpt <= 0.05):
                errors.append(
                    f"risk_per_trade={rpt:.4f} is out of bounds [0.001, 0.05]. "
                    f"Typical values: 0.005–0.015 (0.5%–1.5% of account)."
                )
            max_rpt = float(risk_cfg.get('max_risk_per_trade', rpt * 2))
            if max_rpt < rpt:
                errors.append(
                    f"max_risk_per_trade={max_rpt:.4f} < risk_per_trade={rpt:.4f}"
                )

        # ── Symbol list must be non-empty ────────────────────────────────
        # NOTE: the section is ``scanning`` (plural tiers), not ``scanner``.
        # Every consumer in the repo uses ``cfg.scanning.tier{1,2,3}``.
        scanning_cfg = getattr(self, 'scanning', None)
        if scanning_cfg is not None:
            total_caps = 0
            any_tier_enabled = False
            for tier_name in ('tier1', 'tier2', 'tier3'):
                tier = getattr(scanning_cfg, tier_name, None)
                if tier is None:
                    continue
                enabled = bool(tier.get('enabled', False))
                cap = int(tier.get('max_symbols', 0) or 0)
                if enabled:
                    any_tier_enabled = True
                    total_caps += cap
            if not any_tier_enabled:
                errors.append(
                    "scanning: no tier is enabled — the bot will not scan any symbols."
                )
            elif total_caps == 0:
                errors.append(
                    "scanning: every enabled tier has max_symbols=0 — "
                    "the bot will not scan any symbols."
                )

        # ── Aggregator weights must sum to ~1.0 ──────────────────────────
        agg_cfg = getattr(self, 'aggregator', None)
        if agg_cfg is not None:
            weights = getattr(agg_cfg, 'weights', None)
            if weights is not None:
                total = sum([
                    float(getattr(weights, k, 0) or 0) for k in
                    ['technical', 'volume', 'orderflow', 'derivatives',
                     'sentiment', 'correlation']
                ])
                if abs(total - 1.0) > 0.01:
                    errors.append(
                        f"Aggregator weights sum to {total:.3f}, must be 1.0"
                    )

        if errors:
            for err in errors:
                logger.error(f"Config validation error: {err}")
            return False

        logger.info("Config validation passed")
        return True


# ── Singleton instance ─────────────────────────────────────
# All modules import this single instance:
#   from config.loader import cfg
cfg = Config()
