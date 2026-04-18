"""
TitanBot Pro — Feature Flags
==============================
Every Project Inheritance feature is off by default and switchable
without redeployment. Flags can be overridden via environment variables
(TITANBOT_FF_<FLAG_NAME>=1).

Usage:
    from config.feature_flags import ff
    if ff.is_enabled("SOURCE_CREDIBILITY"):
        ...

Shadow mode:
    When a feature is in shadow mode, the logic executes but its output
    is logged only — it does NOT affect live signals or confidence scores.
    Shadow mode is enabled by setting the flag to "shadow" instead of True.
"""

import logging
import os
from typing import Dict, Literal

try:
    from dotenv import load_dotenv
    # Defensive: ensure .env is loaded even if feature_flags is imported before
    # config.loader (which also calls load_dotenv). Without this, every
    # TITANBOT_FF_* override silently falls back to the default because os.getenv
    # sees nothing. load_dotenv is idempotent and cheap.
    load_dotenv()
except ImportError:  # pragma: no cover - tiny compatibility shim
    pass

logger = logging.getLogger(__name__)

FlagState = Literal["off", "shadow", "live"]


class FeatureFlags:
    """
    Central feature flag registry for Project Inheritance features.
    All flags default to 'off'. Override via env vars:
        TITANBOT_FF_SOURCE_CREDIBILITY=live
        TITANBOT_FF_CLICKBAIT_FILTER=shadow
    """

    # Default flag states — all OFF until explicitly enabled
    _DEFAULTS: Dict[str, FlagState] = {
        "SOURCE_CREDIBILITY":       "off",
        "CLICKBAIT_FILTER":         "off",
        "TITLE_DEDUP":              "off",
        "HEADLINE_EVOLUTION":       "off",
        "NARRATIVE_TRACKER":        "off",
        "FEAR_GREED":               "off",
        "TIME_DECAY":               "off",
        "ONCHAIN_NEWS_CORRELATION": "off",
        "WHALE_INTENT":             "off",
        "PUMP_DUMP_DETECTION":      "off",
        "DEGRADATION_ENGINE":       "off",
        "POWER_ALIGNMENT":          "off",
        "SIGNAL_VALIDATOR":         "off",
    }

    def __init__(self):
        self._flags: Dict[str, FlagState] = {}
        self._load()

    def _load(self):
        """Load flags from defaults, overridden by environment variables."""
        for flag_name, default in self._DEFAULTS.items():
            env_key = f"TITANBOT_FF_{flag_name}"
            env_val = os.getenv(env_key, "").strip().lower()
            if env_val in ("live", "1", "true", "on"):
                self._flags[flag_name] = "live"
            elif env_val == "shadow":
                self._flags[flag_name] = "shadow"
            else:
                self._flags[flag_name] = default

        # Log all flag states on startup
        for flag_name, state in sorted(self._flags.items()):
            logger.info(f"🏁 Feature flag {flag_name}: {state}")

    def is_enabled(self, flag_name: str) -> bool:
        """Returns True if feature is 'live' (not shadow, not off)."""
        return self._flags.get(flag_name, "off") == "live"

    def is_shadow(self, flag_name: str) -> bool:
        """Returns True if feature is in shadow mode."""
        return self._flags.get(flag_name, "off") == "shadow"

    def is_active(self, flag_name: str) -> bool:
        """Returns True if feature should execute (live OR shadow)."""
        return self._flags.get(flag_name, "off") in ("live", "shadow")

    def get_state(self, flag_name: str) -> FlagState:
        """Get the current state of a flag."""
        return self._flags.get(flag_name, "off")

    def set_state(self, flag_name: str, state: FlagState):
        """Dynamically change a flag state (for testing or runtime control)."""
        old = self._flags.get(flag_name, "off")
        self._flags[flag_name] = state
        logger.info(f"🏁 Feature flag {flag_name}: {old} → {state}")

    def all_states(self) -> Dict[str, FlagState]:
        """Return a copy of all flag states."""
        return dict(self._flags)


# Singleton instance
ff = FeatureFlags()
