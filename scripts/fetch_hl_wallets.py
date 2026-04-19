#!/usr/bin/env python3
"""
TitanBot Pro — Hyperliquid Leaderboard Wallet Fetcher
======================================================
Fetches the top-performing wallets from:
  1. Hyperliquid's public leaderboard (all-time PnL ranking)
  2. HyperTracker's Money Printer cohort (highest win rate traders)

Then writes the combined, deduplicated list to settings.yaml under
hypertracker_watched_wallets.

Run once to seed your wallet list, then re-run weekly to refresh:
    python scripts/fetch_hl_wallets.py

Requirements: requests (pip install requests)
              OR works with stdlib urllib as fallback
"""

import json
import sys
import time
import os
import re
from pathlib import Path

# ── Try requests, fall back to urllib ────────────────────────
try:
    import requests
    def _get(url, headers=None, json_body=None, timeout=15):
        if json_body:
            r = requests.post(url, json=json_body, headers=headers or {}, timeout=timeout)
        else:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
except ImportError:
    import urllib.request
    import urllib.error
    def _get(url, headers=None, json_body=None, timeout=15):
        body = json.dumps(json_body).encode() if json_body else None
        req = urllib.request.Request(url, data=body, headers=headers or {},
                                     method="POST" if body else "GET")
        if body:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

ROOT = Path(__file__).parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def _is_eth_address(addr: str) -> bool:
    """Return True only for valid-looking Ethereum hex addresses (0x + 40 hex chars)."""
    return (
        bool(addr)
        and addr.startswith("0x")
        and len(addr) == 42
        and all(c in "0123456789abcdefABCDEF" for c in addr[2:])
    )

# ── Load API key from settings.yaml ──────────────────────────
def load_api_key():
    content = SETTINGS_PATH.read_text()
    m = re.search(r'hypertracker_api_key:\s*["\']([^"\']+)["\']', content)
    if m and m.group(1) not in ("", "YOUR_KEY_HERE"):
        return m.group(1)
    return None

# ── Source 1: Hyperliquid public leaderboard ─────────────────
def fetch_hl_leaderboard(top_n=50):
    """
    Fetches top traders from Hyperliquid's stats API.
    The leaderboard is at stats-data.hyperliquid.xyz, NOT the info endpoint.
    Returns list of {address, pnl} dicts sorted by all-time PnL.
    """
    print("📡 Fetching Hyperliquid leaderboard...")
    try:
        # Correct URL: stats-data endpoint, not api endpoint
        data = _get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard")

        wallets = []
        # Response is {"leaderboardRows": [{...}, ...]}
        entries = (
            data.get("leaderboardRows") or
            data.get("rows") or
            (data if isinstance(data, list) else [])
        )

        for entry in entries[:top_n]:
            addr = (
                entry.get("ethAddress") or
                entry.get("address") or
                entry.get("user", "")
            )
            if not _is_eth_address(addr):
                continue

            # Extract all-time PnL from windowPerformances
            pnl = 0.0
            perfs = entry.get("windowPerformances", [])
            for perf in perfs:
                if isinstance(perf, list) and len(perf) >= 2:
                    if perf[0] == "allTime":
                        try:
                            pnl = float(perf[1].get("pnl", 0))
                        except Exception:
                            pass
                elif isinstance(perf, dict) and perf.get("window") == "allTime":
                    try:
                        pnl = float(perf.get("pnl", 0))
                    except Exception:
                        pass

            wallets.append({
                "address": addr.lower(),
                "source": "hl_leaderboard",
                "pnl": pnl,
            })

        # Sort by PnL descending
        wallets.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"  ✅ Got {len(wallets)} addresses from HL leaderboard")
        if wallets:
            return wallets[:top_n]
        print("  ⚠️  HL leaderboard returned no usable wallet rows; trying fallback")

    except Exception as e:
        print(f"  ⚠️  HL leaderboard fetch failed: {e}")

    # Fallback: try the info endpoint with the correct type for clearinghouse state
    print("  ↳ Trying fallback: HL /info leaderboard type...")
    try:
        data = _get(
            "https://api.hyperliquid.xyz/info",
            json_body={"type": "leaderboard", "req": {"timeWindow": "allTime"}},
        )
        entries = data if isinstance(data, list) else data.get("leaderboardRows", [])
        wallets = []
        for entry in entries[:top_n]:
            addr = entry.get("ethAddress") or entry.get("address", "")
            if addr and _is_eth_address(addr):
                wallets.append({"address": addr.lower(), "source": "hl_leaderboard", "pnl": 0.0})
        if wallets:
            print(f"  ✅ Got {len(wallets)} addresses from HL /info fallback")
            return wallets
    except Exception as e2:
        print(f"  ⚠️  HL /info fallback also failed: {e2}")

    return []

# ── Source 2: HyperTracker money_printer cohort ───────────────
def fetch_ht_money_printers(api_key, top_n=30):
    """
    Fetches top wallet addresses from HyperTracker's Money Printer cohort.
    These are the highest all-time PnL + win rate traders on Hyperliquid.
    """
    print("📡 Fetching HyperTracker Money Printer wallets...")
    try:
        # Get positions endpoint filtered for money_printer cohort
        data = _get(
            "https://ht-api.coinmarketman.com/api/external/positions",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        wallets = []
        entries = data.get("data") or data.get("results") or []
        seen = set()
        for pos in entries[:top_n * 3]:  # grab more since many per wallet
            addr = pos.get("address") or pos.get("wallet", "")
            if addr and addr.lower() not in seen:
                seen.add(addr.lower())
                wallets.append({
                    "address": addr.lower(),
                    "source": "ht_money_printer",
                    "pnl": float(
                        pos.get("pnl") or
                        pos.get("total_pnl") or
                        pos.get("realizedPnl") or
                        pos.get("unrealizedPnl") or
                        0
                    ),
                })
            if len(wallets) >= top_n:
                break
        print(f"  ✅ Got {len(wallets)} addresses from HyperTracker")
        return wallets
    except Exception as e:
        print(f"  ⚠️  HyperTracker fetch failed: {e}")
        return []

# ── Source 3: HyperTracker leaderboard endpoint ───────────────
def fetch_ht_leaderboard(api_key, top_n=20):
    """Try HyperTracker's own leaderboard endpoint."""
    print("📡 Fetching HyperTracker leaderboard...")
    endpoints = [
        "/api/external/leaderboard",
        "/api/external/traders",
        "/api/external/money_printer/wallets",
        "/api/external/segments/money_printer/wallets",
    ]
    for ep in endpoints:
        try:
            data = _get(
                f"https://ht-api.coinmarketman.com{ep}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            entries = data.get("data") or data.get("wallets") or data.get("results") or (data if isinstance(data, list) else [])
            wallets = []
            seen = set()
            for entry in entries[:top_n]:
                addr = (entry.get("address") or entry.get("wallet") or entry.get("ethAddress") or "").lower()
                if _is_eth_address(addr) and addr not in seen:
                    seen.add(addr)
                    wallets.append({
                        "address": addr,
                        "source": "ht_leaderboard",
                        "pnl": float(entry.get("pnl") or entry.get("total_pnl") or 0),
                    })
            if wallets:
                print(f"  ✅ Got {len(wallets)} from {ep}")
                return wallets
        except Exception:
            continue
    print("  ⚠️  No leaderboard endpoint worked")
    return []

# ── Known public whale addresses (manually curated) ──────────
# These are publicly documented high-PnL Hyperliquid traders.
# Sourced from public leaderboard screenshots and community posts.
KNOWN_WHALES = [
    # Format: (address, label)
    # These are placeholder addresses — replace with real ones from the leaderboard
    # Once you run this script with network access, it will auto-populate from the API
]

# ── Write to settings.yaml ────────────────────────────────────
def write_wallets_to_settings(addresses):
    content = SETTINGS_PATH.read_text()
    # Build YAML list
    if not addresses:
        wallet_yaml = "    []   # no wallets found — check your API key and network"
    else:
        items = "\n".join(f'    - "{addr}"' for addr in addresses)
        wallet_yaml = f"    # Auto-fetched {time.strftime('%Y-%m-%d %H:%M')} — {len(addresses)} wallets\n{items}"

    # Replace existing list.
    # Anchored multiline pattern: matches the key at column 0 followed by
    # either an inline value ([]) or any number of indented list/comment lines.
    pattern = r'(?m)^hypertracker_watched_wallets:[ \t]*(?:\[\][^\n]*)?\n(?:[ \t][^\n]*\n)*'
    replacement = f"hypertracker_watched_wallets:\n{wallet_yaml}\n"

    new_content = re.sub(pattern, replacement, content, count=1)
    if new_content == content:
        # Fallback: line-by-line replacement that is careful to stop skipping
        # only when it sees a non-indented, non-list, non-comment line so it
        # cannot accidentally delete content from other YAML sections.
        lines = content.split("\n")
        out = []
        skip_until_next_key = False
        for line in lines:
            if "hypertracker_watched_wallets:" in line and not line.startswith(" "):
                out.append("hypertracker_watched_wallets:")
                out.append(wallet_yaml)
                skip_until_next_key = True
                continue
            if skip_until_next_key:
                stripped = line.strip()
                # Stop skipping when we hit a non-indented key (new YAML section).
                # Allow blank lines, comments, and list items to pass through
                # as they're part of the current list block.
                if stripped and not stripped.startswith("-") and not stripped.startswith("#") and not line.startswith(" ") and not line.startswith("\t"):
                    skip_until_next_key = False
                    out.append(line)
                # Otherwise silently drop the old list line
                continue
            out.append(line)
        new_content = "\n".join(out)

    SETTINGS_PATH.write_text(new_content)
    print(f"\n✅ Written {len(addresses)} wallet addresses to {SETTINGS_PATH}")


def main():
    print("=" * 60)
    print("TitanBot — Hyperliquid Whale Wallet Fetcher")
    print("=" * 60)

    api_key = load_api_key()

    # Collect from all sources
    all_wallets = []

    # Source 1: Hyperliquid public leaderboard (no key needed)
    hl_wallets = fetch_hl_leaderboard(top_n=50)
    all_wallets.extend(hl_wallets)

    # Source 2 & 3: HyperTracker (key required)
    if api_key:
        ht_wallets = fetch_ht_money_printers(api_key, top_n=30)
        all_wallets.extend(ht_wallets)
        ht_lb = fetch_ht_leaderboard(api_key, top_n=25)
        all_wallets.extend(ht_lb)
    else:
        print("ℹ️  HyperTracker API key not set; skipping HyperTracker-only wallet sources")

    # Deduplicate, keeping first occurrence (HL leaderboard ranked by PnL)
    seen = set()
    unique = []
    for w in all_wallets:
        addr = w["address"].lower()
        if _is_eth_address(addr) and addr not in seen:
            seen.add(addr)
            unique.append(addr)

    # Top 50 max (100 req/day free tier — 50 wallets × 2 polls/day)
    final = unique[:50]

    print(f"\n📋 Total unique whale wallets: {len(final)}")
    for i, addr in enumerate(final[:10], 1):
        print(f"   {i:2d}. {addr[:10]}...{addr[-6:]}")
    if len(final) > 10:
        print(f"   ... and {len(final)-10} more")

    if final:
        write_wallets_to_settings(final)
        print("\n🚀 Restart TitanBot to begin tracking these wallets.")
        print("   HyperTracker will poll positions every 30 minutes.")
    else:
        print("\n⚠️  No wallets found. Check network and API key.")
        print("   You can also manually add addresses to config/settings.yaml")
        print("   under hypertracker_watched_wallets:")

    print("=" * 60)


if __name__ == "__main__":
    main()
