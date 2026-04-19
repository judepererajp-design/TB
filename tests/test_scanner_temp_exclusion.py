from scanner.scanner import Scanner, SymbolState, Tier


def test_temporary_exclusion_skips_due_without_removing_symbol(monkeypatch):
    scanner = Scanner()
    scanner._symbols["ETH/USDT"] = SymbolState(symbol="ETH/USDT", tier=Tier.TIER1, last_scan=0.0)

    monkeypatch.setattr("scanner.scanner.time.time", lambda: 1_000.0)
    scanner.temporarily_exclude_symbol("ETH/USDT", duration_secs=3600, reason="test")

    assert "ETH/USDT" in scanner.get_all_symbols()
    assert scanner.is_temporarily_excluded("ETH/USDT") is True
    assert "ETH/USDT" not in scanner.get_due_symbols()


def test_temporary_exclusion_expires_and_symbol_becomes_due(monkeypatch):
    scanner = Scanner()
    scanner._symbols["ETH/USDT"] = SymbolState(symbol="ETH/USDT", tier=Tier.TIER1, last_scan=0.0)

    monkeypatch.setattr("scanner.scanner.time.time", lambda: 1_000.0)
    scanner.temporarily_exclude_symbol("ETH/USDT", duration_secs=60, reason="test")

    monkeypatch.setattr("scanner.scanner.time.time", lambda: 1_061.0)
    assert scanner.is_temporarily_excluded("ETH/USDT") is False
    assert "ETH/USDT" in scanner.get_due_symbols()
