"""
TitanBot Pro — Keyboards v6 (Apple-like UX)
============================================
Design principles:
  - One home (menu:main). Every screen has a back path.
  - Signal cards edit in-place — no reply popups.
  - panel:{signal_id}:{action} callback format for all inline panels.
  - Grade-aware buttons: C=minimal, B=basic+levels button, B+=all, A/A+=all+profile.
  - All features reachable from menus — no text-command-only functions.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def _b(label, cb): return InlineKeyboardButton(label, callback_data=cb)
def _u(label, url): return InlineKeyboardButton(label, url=url)
def _mk(*rows): return InlineKeyboardMarkup(list(rows))
def _home(): return [_b("🏠 Main Menu", "menu:main")]
def _back(dest, label="↩ Back"): return _b(label, dest)


def _nav(back=None, refresh=None, search=True, more=None):
    row = []
    if back:
        row.append(_back(back))
    row.append(_b("🏠 Home", "menu:main"))
    if refresh:
        row.append(_b("🔄 Refresh", refresh))
    if search:
        row.append(_b("🔎 Search", "menu:search"))
    if more:
        row.append(_b("⋯ More", more))
    return row


# ── TradingView URL helper ────────────────────────────────────────────────────

def _tv_url(symbol: str) -> str:
    tv = symbol.replace("/USDT", "USDT").replace("/", "")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv}PERP"


# ── MAIN MENU ─────────────────────────────────────────────────────────────────

def main_menu():
    return _mk(
        [_b("📊 Dashboard", "menu:dashboard"), _b("🎯 Signals", "menu:signals")],
        [_b("📡 Markets", "menu:market"), _b("👁 Watchlist", "menu:watchlist")],
        [_b("📈 Performance", "menu:performance"), _b("🩺 Bot Health", "menu:bot_health")],
        [_b("🔎 Search", "menu:search"), _b("⚙️ Settings", "menu:settings")],
        [_b("🔄 Force Scan", "bot:scan"), _b("❓ Help", "menu:help")],
    )


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

def dashboard_menu(paused=False):
    tl = "▶️ Resume" if paused else "⏸ Pause"
    tc = "bot:resume" if paused else "bot:pause_ask"
    return _mk(
        [_b("📌 Overview", "menu:dashboard"), _b("📡 Regime", "dash:regime")],
        [_b("🔄 Refresh", "dash:refresh"), _b("🔄 Force Scan", "bot:scan")],
        [_b(tl, tc), _b("🩺 Bot Health", "menu:bot_health")],
        _nav(search=True, more="menu:market"),
    )


# ── MARKET ────────────────────────────────────────────────────────────────────

def market_menu():
    return _mk(
        [_b("📊 Full Digest", "dash:market_full"), _b("📈 Regime", "dash:regime")],
        [_b("🌍 Alt Season", "dash:altseason"), _b("😨 Fear & Greed", "dash:fg")],
        [_b("🐋 Whales", "dash:whales_ht"), _b("💧 Whale Flow", "dash:whales")],
        [_b("🔄 Rotation", "dash:rotation"), _b("📰 Top News", "dash:news_feed")],
        _nav(back="menu:dashboard", refresh="dash:market_full"),
    )


# ── SIGNALS ───────────────────────────────────────────────────────────────────

def signals_menu():
    return _mk(
        [_b("🟢 Live Trades", "sig:status:live"), _b("🆕 New Setups", "sig:list:24")],
        [_b("⏳ Pending", "sig:filter:pending"), _b("✅ Closed", "sig:status:closed")],
        [_b("⏭ Skipped", "sig:status:skipped"), _b("⬆️ Upgrades", "sig:upgrades")],
        [_b("🏆 Ranking", "perf:ranking"), _b("🔎 Lookup", "search:mode:any")],
        _nav(back="menu:dashboard", refresh="menu:signals"),
    )


# ── PERFORMANCE ───────────────────────────────────────────────────────────────

def performance_menu():
    return _mk(
        [_b("7 Days", "perf:7"), _b("30 Days", "perf:30"), _b("All Time", "perf:all")],
        [_b("🏆 Strategy Ranking", "perf:ranking"), _b("📼 Replay History", "replay:prompt")],
        [_b("📤 Export Info", "perf:export"), _b("🔎 Search", "menu:search")],
        _nav(back="menu:dashboard", refresh="menu:performance"),
    )

def performance_period():
    return performance_menu()


# ── WATCHLIST ─────────────────────────────────────────────────────────────────

def watchlist_menu():
    return _mk(
        [_b("👁 View Symbols", "wl:view"), _b("🔄 Refresh", "wl:refresh")],
        _nav(back="menu:dashboard", refresh="wl:refresh"),
    )

def watchlist_symbols(symbols, page=0):
    page_syms = symbols[page * 9:(page + 1) * 9]
    rows = []
    row = []
    for sym in page_syms:
        short = sym.replace("/USDT", "").replace("USDT", "")[:8]
        row.append(_b(short, f"wl:snap:{sym}"))
        if len(row) == 3:
            rows.append(row); row = []
    if row:
        rows.append(row)
    nav = []
    if page > 0:
        nav.append(_b("◀ Prev", f"wl:page:{page - 1}"))
    if len(symbols) > (page + 1) * 9:
        nav.append(_b("Next ▶", f"wl:page:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append(_home())
    return InlineKeyboardMarkup(rows)

def watchlist_snapshot_actions(symbol):
    return _mk(
        [_b("📊 Chart",      f"chart:sym:{symbol}"), _b("⚡ Scan Now",  f"force:sym:{symbol}")],
        [_b("👀 Prioritise", f"wl:watch:{symbol}"),   _b("🚫 Remove",    f"wl:ignore:{symbol}")],
        [_back("menu:watchlist"), _b("🏠 Main", "menu:main")],
    )

def watchlist_actions(symbol):
    return watchlist_snapshot_actions(symbol)

def whale_actions(symbol):
    return _mk([_b("📊 Chart", f"chart:sym:{symbol}"), _b("⚡ Scan", f"force:sym:{symbol}")])


# ── SETTINGS ──────────────────────────────────────────────────────────────────

def settings_menu():
    return _mk(
        [_b("⚖️ Risk",       "cfg:risk"),         _b("🧠 Strategies", "cfg:strategies")],
        [_b("📡 Scanning",   "cfg:scanning"),      _b("🔔 Alerts",     "cfg:notifications")],
        [_b("🔄 Reload Config", "cfg:reload"), _b("🩺 Bot Health", "menu:bot_health")],
        _nav(back="menu:dashboard"),
    )

def config_menu():
    return settings_menu()

def risk_settings_menu():
    return _mk(
        [_b("✏️ Account Size",   "edit:account_size")],
        [_b("✏️ Risk Per Trade", "edit:risk_per_trade")],
        [_b("✏️ Min Confidence", "edit:min_confidence")],
        [_b("✏️ Max Daily Loss", "edit:max_daily_loss_pct")],
        [_b("✏️ Min R/R",        "edit:min_rr")],
        [_back("menu:settings"), _b("🏠 Main", "menu:main")],
    )

_ABBREV = {
    "SmartMoneyConcepts":"SMC","InstitutionalBreakout":"Breakout",
    "Momentum":"Momentum","MomentumContinuation":"Momentum",
    "ExtremeReversal":"Reversal",
    "MeanReversion":"Mean Rev","FundingRateArb":"Funding",
    "Ichimoku":"Ichimoku","IchimokuCloud":"Ichimoku","ElliottWave":"Elliott",
    "RangeScalper":"Scalper","PriceAction":"PA",
    "WyckoffAccDist":"Wyckoff","Wyckoff":"Wyckoff",
    "HarmonicPattern":"Harmonic","GeometricPattern":"Geometric",
    "smc":"SMC","breakout":"Breakout","reversal":"Reversal",
    "mean_reversion":"Mean Rev","price_action":"PA","momentum":"Momentum",
    "ichimoku":"Ichimoku","elliott_wave":"Elliott","funding_arb":"Funding",
    "wyckoff":"Wyckoff","harmonic":"Harmonic","geometric":"Geometric",
    "range_scalper":"Scalper",
}

def strategy_menu(strategies):
    rows = []
    pair = []
    for name, enabled in strategies.items():
        icon  = "✅" if enabled else "❌"
        short = _ABBREV.get(name, name[:10])
        pair.append(_b(f"{icon} {short}", f"strat:toggle:{name}"))
        if len(pair) == 2:
            rows.append(pair); pair = []
    if pair:
        rows.append(pair)
    rows.append([_back("menu:settings"), _b("🏠 Main", "menu:main")])
    return InlineKeyboardMarkup(rows)

def back_to_settings():
    return _mk([_back("menu:settings"), _b("🏠 Main", "menu:main")])

def back_to_main():
    return _mk(_home())


def bot_health_menu():
    return _mk(
        [_b("📌 Overview", "health:overview"), _b("📡 Scans", "health:scans")],
        [_b("📱 Telegram", "health:telegram"), _b("🤖 AI", "health:ai")],
        [_b("🛡️ Circuit", "health:circuit"), _b("🧾 Incidents", "health:events")],
        [_b("⏸ Pause", "bot:pause_ask"), _b("▶️ Resume", "bot:resume")],
        _nav(back="menu:dashboard", refresh="health:refresh"),
    )


def search_menu():
    return _mk(
        [_b("🔢 Signal ID", "search:mode:signal"), _b("🪙 Symbol", "search:mode:symbol")],
        [_b("🧠 Strategy", "search:mode:strategy"), _b("📝 Notes", "search:mode:notes")],
        [_b("🩺 Health/Event", "search:mode:event"), _b("✨ Universal", "search:mode:any")],
        _nav(back="menu:dashboard", refresh="menu:search", search=False),
    )


def search_results_menu(results: dict):
    rows = []
    for sig in results.get("signals", [])[:5]:
        rows.append([_b(f"#{sig.get('id')} {sig.get('symbol', '?')}", f"search:open_signal:{sig.get('id')}")])
    for note in results.get("notes", [])[:3]:
        sid = note.get("signal_id")
        if sid:
            rows.append([_b(f"📝 Note #{sid}", f"search:open_signal:{sid}")])
    rows.append(_nav(back="menu:search", refresh="menu:search", search=False))
    return InlineKeyboardMarkup(rows)


def search_signal_actions(signal_id, symbol=""):
    rows = [
        [_b("📊 Summary", f"sig:summary:{signal_id}"), _b("🔍 Explain", f"sig:explain:{signal_id}")],
    ]
    if symbol:
        rows.append([_b("🪙 This Symbol", f"search:quick:symbol:{symbol}")])
    rows.append(_nav(back="menu:search", refresh=f"search:open_signal:{signal_id}", search=False))
    return InlineKeyboardMarkup(rows)


# ── SIGNAL CARDS ─────────────────────────────────────────────────────────────
#
# Callback format for all inline panels: panel:{signal_id}:{action}
# Actions: market | logic | metrics | profile | plan | live | targets | exit | back
#
# signal_card()       — SETUP state (grade-aware)
# signal_panel_back() — while a panel is shown inline (Back restores card)
# signal_in_trade()   — ENTRY_ACTIVE / TRADE_ACTIVE / TP1_HIT state
# signal_closed()     — CLOSED state

def signal_card(signal_id, symbol="", grade="A"):
    """
    Grade-aware setup card keyboard.

    C    — Market context + Chart + Dismiss (informational, no levels)
    B    — Market + Logic + Full Levels button + Metrics + Chart + Dismiss
    B+   — Market + Logic + Metrics + Trade Plan + Chart + Dismiss
    A/A+ — Full suite + Profile + Dismiss
    All grades now include: ✅ Mark as Taken button (FIX 6B)
    """
    sid = signal_id
    tv  = _tv_url(symbol)

    if grade == "C":
        return _mk(
            [_b("🌍 Market Context", f"panel:{sid}:market"), _u("📈 Chart", tv)],
            [_b("🧾 Details", f"sig:summary:{sid}"), _b("🔎 Symbol", f"search:quick:symbol:{symbol}")],
            [_b("✅ Mark as Taken", f"taken:{sid}"), _b("✖️ Dismiss", f"dismiss:{sid}")],
        )
    elif grade == "B":
        return _mk(
            [_b("🌍 Market", f"panel:{sid}:market"), _b("💡 Logic", f"panel:{sid}:logic"), _u("📈 Chart", tv)],
            [_b("🎯 Full Levels",  f"panel:{sid}:plan"),    _b("📊 Metrics", f"panel:{sid}:metrics")],
            [_b("🧾 Details", f"sig:summary:{sid}"), _b("🔎 Symbol", f"search:quick:symbol:{symbol}")],
            [_b("✅ Mark as Taken", f"taken:{sid}"), _b("✖️ Dismiss", f"dismiss:{sid}")],
        )
    elif grade == "B+":
        return _mk(
            [_b("🌍 Market", f"panel:{sid}:market"), _b("💡 Logic", f"panel:{sid}:logic"), _b("📊 Metrics", f"panel:{sid}:metrics")],
            [_b("📋 Trade Plan", f"panel:{sid}:plan"), _u("📈 Chart", tv)],
            [_b("🧾 Details", f"sig:summary:{sid}"), _b("🔎 Symbol", f"search:quick:symbol:{symbol}")],
            [_b("✅ I'm In", f"taken:{sid}:enter"), _b("⏭ Skip", f"panel:skip:{sid}"), _b("✖️ Dismiss", f"dismiss:{sid}")],
        )
    else:  # A / A+
        return _mk(
            [_b("🌍 Market", f"panel:{sid}:market"), _b("💡 Logic", f"panel:{sid}:logic"), _b("📊 Metrics", f"panel:{sid}:metrics")],
            [_b("📋 Trade Plan", f"panel:{sid}:plan"), _b("🧬 Profile", f"panel:{sid}:profile"), _u("📈 Chart", tv)],
            [_b("🧾 Details", f"sig:summary:{sid}"), _b("🔎 Symbol", f"search:quick:symbol:{symbol}")],
            [_b("✅ I'm In", f"taken:{sid}:enter"), _b("⏭ Skip", f"panel:skip:{sid}"), _b("✖️ Dismiss", f"dismiss:{sid}")],
        )


def signal_panel_back(signal_id, symbol="", back_label="← Back to Signal"):
    """Shown while a panel is displayed inline. Back restores the card."""
    return _mk(
        [_b(back_label, f"panel:{signal_id}:back")],
        [_u("📈 Chart", _tv_url(symbol))],
    )


def signal_in_trade(signal_id, symbol="", confirmed=False):
    """
    Active trade keyboard (ENTRY_ACTIVE / TRADE_ACTIVE / TP1_HIT).
    confirmed=True: user already pressed I'm In — no confirm button shown.
    confirmed=False: user hasn't confirmed — show I'm In button.
    """
    sid = signal_id
    if confirmed:
        # Already confirmed — full management keyboard with thesis check
        return _mk(
            [_b("📊 P&L", f"panel:{sid}:live"), _b("🎯 Targets", f"panel:{sid}:targets"), _u("📈 Chart", _tv_url(symbol))],
            [_b("🌍 Market", f"panel:{sid}:market"), _b("💡 Logic", f"panel:{sid}:logic"), _b("🚪 Exit Rules", f"panel:{sid}:exit")],
            [_b("🔍 Check Signal", f"panel:{sid}:check")],
        )
    else:
        # Not yet confirmed — show I'm In as primary CTA
        return _mk(
            [_b("✅ I'm In — Start Tracking", f"taken:{sid}:enter"),
             _b("⏭ Not Taken", f"panel:skip:{sid}")],
            [_b("📊 P&L", f"panel:{sid}:live"), _b("🎯 Targets", f"panel:{sid}:targets"), _u("📈 Chart", _tv_url(symbol))],
            [_b("🌍 Market", f"panel:{sid}:market"), _b("💡 Logic", f"panel:{sid}:logic"), _b("🚪 Exit Rules", f"panel:{sid}:exit")],
        )


def signal_closed(signal_id):
    """Closed trade — minimal review buttons."""
    sid = signal_id
    return _mk(
        [_b("📊 Review Trade", f"panel:{sid}:metrics"), _b("🧠 Why It Worked", f"panel:{sid}:logic")],
        [_b("🔍 AI Explanation", f"sig:explain:{sid}"), _b("🧾 Details", f"sig:summary:{sid}")],
    )


# ── Backward-compat wrappers (existing call sites keep working unchanged) ─────

def signal_setup(signal_id, symbol="", tf="", grade="A"):
    return signal_card(signal_id, symbol, grade)

def signal_actions(signal_id, symbol="", tf=""):
    return signal_card(signal_id, symbol, "A")

def signal_entry_active(signal_id, symbol="", tf=""):
    return signal_in_trade(signal_id, symbol)

def panel_close(panel_message_id: int):
    """Legacy: close button for old reply-message panels."""
    return _mk([_b("✖️ Close", f"panel:close:{panel_message_id}")])

def stage_approaching(signal_id, symbol=""):
    return _mk([
        _u("📈 Chart", _tv_url(symbol)),
        _b("❌ Cancel Signal", f"sig:cancel:{signal_id}"),
    ])

def stage_execute(signal_id, symbol=""):
    """ENTER NOW card — shown when all triggers confirmed."""
    return _mk(
        [_b("✅ I'm In — Start Tracking", f"taken:{signal_id}:enter"),
         _b("⏭ Skipped", f"panel:skip:{signal_id}")],
        [_u("📈 Chart", _tv_url(symbol))],
    )


# ── BOT CONTROL ───────────────────────────────────────────────────────────────

def pause_confirm():
    return _mk([_b("✅ Yes, Pause", "bot:pause_confirm"), _b("↩ Cancel", "bot:cancel")])

def resume_confirm():
    return _mk([_b("✅ Yes, Resume", "bot:resume_confirm"), _b("↩ Cancel", "bot:cancel")])

def confirm_resume():
    return _mk([_b("✅ Override CB", "cb:resume_confirm"), _b("↩ Stay Paused", "cb:keep_paused")])

def status_actions():
    return _mk(
        [_b("🔄 Refresh",     "dash:refresh"),    _b("📡 Market",    "menu:market")],
        [_b("⏸ Pause",        "bot:pause_ask"),    _b("🔄 Scan",      "bot:scan")],
        [_b("📈 Performance", "menu:performance"), _b("🏠 Main",      "menu:main")],
    )

def control_panel(paused=False):
    """Admin panel redirects to unified main menu."""
    return main_menu()


# ── OUTCOME ───────────────────────────────────────────────────────────────────

def outcome_result(signal_id):
    return _mk([
        _b("✅ Win",  f"outcome:{signal_id}:WIN"),
        _b("🔴 Loss", f"outcome:{signal_id}:LOSS"),
        _b("➡️ BE",   f"outcome:{signal_id}:BREAKEVEN"),
    ])

def skip_reason(signal_id):
    return _mk(
        [_b("Already in trade", f"skip_reason:{signal_id}:in_trade"),
         _b("Missed entry",     f"skip_reason:{signal_id}:missed")],
        [_b("Don't like setup", f"skip_reason:{signal_id}:dislike"),
         _b("Too risky",        f"skip_reason:{signal_id}:risky")],
    )


# ── HELP ──────────────────────────────────────────────────────────────────────

def help_menu():
    return _mk(
        [_b("📊 Signal Grades",   "help:grades"),        _b("📋 Signal Types",   "help:signal_types")],
        [_b("📈 Reading Cards",   "help:reading_cards"),  _b("⚙️ Settings Guide","help:settings")],
        [_b("📱 All Commands",    "help:commands"),        _b("🔢 Grade Actions",  "help:grade_actions")],
        _home(),
    )


# ── SINGLETON ─────────────────────────────────────────────────────────────────

class Keyboards:
    main_menu             = staticmethod(main_menu)
    dashboard_menu        = staticmethod(dashboard_menu)
    market_menu           = staticmethod(market_menu)
    signals_menu          = staticmethod(signals_menu)
    performance_menu      = staticmethod(performance_menu)
    performance_period    = staticmethod(performance_period)
    watchlist_menu        = staticmethod(watchlist_menu)
    watchlist_symbols     = staticmethod(watchlist_symbols)
    watchlist_actions     = staticmethod(watchlist_actions)
    watchlist_snapshot_actions = staticmethod(watchlist_snapshot_actions)
    whale_actions         = staticmethod(whale_actions)
    settings_menu         = staticmethod(settings_menu)
    bot_health_menu       = staticmethod(bot_health_menu)
    search_menu           = staticmethod(search_menu)
    search_results_menu   = staticmethod(search_results_menu)
    search_signal_actions = staticmethod(search_signal_actions)
    config_menu           = staticmethod(config_menu)
    risk_settings_menu    = staticmethod(risk_settings_menu)
    strategy_menu         = staticmethod(strategy_menu)
    back_to_settings      = staticmethod(back_to_settings)
    back_to_main          = staticmethod(back_to_main)
    signal_card           = staticmethod(signal_card)
    signal_panel_back     = staticmethod(signal_panel_back)
    signal_in_trade       = staticmethod(signal_in_trade)
    signal_closed         = staticmethod(signal_closed)
    signal_setup          = staticmethod(signal_setup)
    signal_actions        = staticmethod(signal_actions)
    signal_entry_active   = staticmethod(signal_entry_active)
    panel_close           = staticmethod(panel_close)
    stage_approaching     = staticmethod(stage_approaching)
    stage_execute         = staticmethod(stage_execute)
    status_actions        = staticmethod(status_actions)
    pause_confirm         = staticmethod(pause_confirm)
    resume_confirm        = staticmethod(resume_confirm)
    confirm_resume        = staticmethod(confirm_resume)
    control_panel         = staticmethod(control_panel)
    outcome_result        = staticmethod(outcome_result)
    skip_reason           = staticmethod(skip_reason)
    help_menu             = staticmethod(help_menu)

keyboards = Keyboards()
