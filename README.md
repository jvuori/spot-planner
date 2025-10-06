# Spot Planner

A high-performance Python library for finding optimal time periods in price sequences. Perfect for spot price analysis, cost optimization, and resource planning.

## What is Spot Planner?

Spot Planner helps you identify the most cost-effective periods in a sequence of prices. Whether you're analyzing electricity spot prices, cloud computing costs, or any time-series pricing data, this library efficiently finds periods that meet your criteria.

### Key Features

- 🚀 **High Performance**: Core algorithm implemented in Rust for maximum speed
- 🐍 **Python Native**: Seamless Python integration with automatic fallback
- 📊 **Flexible Criteria**: Find periods based on price thresholds, duration, and gaps
- 🔧 **Easy Integration**: Simple API that works with any price sequence
- ⚡ **Zero Dependencies**: No external dependencies required

## Installation

### Using uv (Recommended)

```bash
uv add spot-planner
```

### Using pip

```bash
pip install spot-planner
```

## Quick Start

```python
from decimal import Decimal
from spot_planner import get_cheapest_periods

# Example: Find cheapest electricity periods
prices = [
    Decimal("50"),  # 6 AM - expensive
    Decimal("40"),  # 7 AM - moderate
    Decimal("30"),  # 8 AM - cheap
    Decimal("20"),  # 9 AM - very cheap
    Decimal("45"),  # 10 AM - expensive again
]

# Find 2 cheapest periods with price under 35
result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("35"),
    min_selections=2,
    min_consecutive_selections=1,
    max_gap_between_periods_between_periods=1,
    max_gap_between_periods_from_start=1
)

print(result)  # [2, 3] - periods starting at index 2 and 3
```

## Use Cases

### Electricity Spot Price Analysis

```python
# Find cheapest 3-hour periods for running high-power equipment
cheap_periods = get_cheapest_periods(
    prices=hourly_prices,
    low_price_threshold=Decimal("0.05"),  # 5 cents per kWh
    min_selections=3,
    min_consecutive_selections=3,  # 3-hour minimum
    max_gap_between_periods=2      # Allow 2-hour gaps between periods
)
```

### Cloud Computing Cost Optimization

```python
# Find most cost-effective periods for batch processing
optimal_windows = get_cheapest_periods(
    prices=aws_spot_prices,
    low_price_threshold=Decimal("0.10"),  # $0.10 per hour
    min_selections=5,
    min_consecutive_selections=4,  # 4-hour processing windows
    max_gap_between_periods=1      # Minimal gaps between windows
)
```

### Resource Planning

```python
# Plan maintenance windows during low-cost periods
maintenance_slots = get_cheapest_periods(
    prices=resource_costs,
    low_price_threshold=budget_threshold,
    min_selections=2,
    min_consecutive_selections=8,  # 8-hour maintenance windows
    max_gap_between_periods=0      # No gaps allowed
)
```

## API Reference

### `get_cheapest_periods(prices, low_price_threshold, min_selections, min_consecutive_selections=1, max_gap_between_periods=0, max_gap_from_start=0)`

Find the cheapest periods in a price sequence.

**Parameters:**

- `prices` (List[Decimal]): Sequence of prices to analyze
- `low_price_threshold` (Decimal): Maximum price for valid periods
- `min_selections` (int): Number of periods to find
- `min_consecutive_selections` (int, optional): Minimum period length. Defaults to 1.
- `max_gap_between_periods` (int, optional): Maximum gap between periods. Defaults to 0.
- `max_gap_from_start` (int, optional): Maximum gap from start to first period. Defaults to 0.

**Returns:**

- `List[int]`: Starting indices of the cheapest periods

**Raises:**

- `ValueError`: If parameters are invalid or no valid periods found

## Performance

Spot Planner uses Rust for the core algorithm, providing significant performance improvements over pure Python implementations:

- **10-100x faster** than naive Python approaches
- **Memory efficient** with minimal allocations
- **Automatic fallback** to Python implementation if Rust module unavailable

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
