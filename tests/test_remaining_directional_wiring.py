import sys
import types


if "utils.formatting" not in sys.modules:
    _fmt_mod = types.ModuleType("utils.formatting")
    _fmt_mod.fmt_price = lambda value: str(value)
    _fmt_mod.fmt_price_raw = lambda value: str(value)
    sys.modules["utils.formatting"] = _fmt_mod
else:
    _fmt_mod = sys.modules["utils.formatting"]
    if not hasattr(_fmt_mod, "fmt_price"):
        _fmt_mod.fmt_price = lambda value: str(value)
    if not hasattr(_fmt_mod, "fmt_price_raw"):
        _fmt_mod.fmt_price_raw = lambda value: str(value)


from signals.whale_aggregator import WhaleAggregator


def test_recent_whale_dominant_side_long():
    agg = WhaleAggregator()
    agg.add_event({"symbol": "BTCUSDT", "side": "buy", "order_usd": 900000, "price": 100000, "qty": 9})
    agg.add_event({"symbol": "BTCUSDT", "side": "buy", "order_usd": 600000, "price": 100100, "qty": 6})
    agg.add_event({"symbol": "BTCUSDT", "side": "sell", "order_usd": 300000, "price": 99900, "qty": 3})

    assert agg.get_recent_dominant_side(symbol="BTCUSDT", max_age_secs=300) == "LONG"


def test_recent_whale_dominant_side_short():
    agg = WhaleAggregator()
    agg.add_event({"symbol": "ETHUSDT", "side": "sell", "order_usd": 700000, "price": 3000, "qty": 200})
    agg.add_event({"symbol": "ETHUSDT", "side": "sell", "order_usd": 500000, "price": 2995, "qty": 170})
    agg.add_event({"symbol": "ETHUSDT", "side": "buy", "order_usd": 200000, "price": 3005, "qty": 65})

    assert agg.get_recent_dominant_side(symbol="ETHUSDT", max_age_secs=300) == "SHORT"


def test_recent_whale_dominant_side_requires_clear_bias():
    agg = WhaleAggregator()
    agg.add_event({"symbol": "SOLUSDT", "side": "buy", "order_usd": 500000, "price": 150, "qty": 3333})
    agg.add_event({"symbol": "SOLUSDT", "side": "sell", "order_usd": 480000, "price": 149, "qty": 3221})

    assert agg.get_recent_dominant_side(symbol="SOLUSDT", max_age_secs=300) == ""
