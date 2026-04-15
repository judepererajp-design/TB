"""
TitanBot Pro — Smart Price Formatter
======================================
Formats prices with appropriate decimal places based on value.

$95,000.50  → BTC-level (2 decimals)
$172.44     → SOL-level (2 decimals)
$0.2157     → Mid-cap (4 decimals)
$0.0040     → Small-cap (6 decimals)
$0.00000123 → Micro-cap (8 decimals)

Used across: formatter, whale aggregator, invalidation monitor, bot, etc.
"""


def fmt_price(price: float, prefix: str = "$") -> str:
    """
    Smart price formatting — auto-scales decimal places to price magnitude.
    
    Args:
        price: The price value
        prefix: Currency prefix (default "$")
    
    Returns:
        Formatted price string like "$95,000.50" or "$0.004012"
    """
    if price is None or price == 0:
        return f"{prefix}0.00"
    
    abs_price = abs(price)
    
    if abs_price >= 1000:
        return f"{prefix}{price:,.2f}"
    elif abs_price >= 1:
        return f"{prefix}{price:,.4f}"
    elif abs_price >= 0.01:
        return f"{prefix}{price:,.6f}"
    elif abs_price >= 0.0001:
        return f"{prefix}{price:,.8f}"
    else:
        return f"{prefix}{price:.10f}"


def fmt_price_raw(price: float) -> str:
    """Same as fmt_price but no $ prefix — for use in code/monospace blocks"""
    return fmt_price(price, prefix="")
