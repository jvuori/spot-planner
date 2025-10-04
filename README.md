# Spot Timer

Algorithm for finding the cheapest periods in a sequence of prices.

## Features

- **High Performance**: Core algorithm implemented in Rust using PyO3
- **Python Compatibility**: Drop-in replacement with Python fallback
- **Easy Installation**: Standard pip installation process
- **Comprehensive Testing**: Full test suite maintained

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Usage

```python
from decimal import Decimal
from spot_timer.main import get_cheapest_periods

price_data = [Decimal("50"), Decimal("40"), Decimal("30"), Decimal("20")]
result = get_cheapest_periods(
    price_data=price_data,
    price_threshold=Decimal("35"),
    desired_count=2,
    min_period=1,
    max_gap=1,
    max_start_gap=1
)
print(result)  # [2, 3]
```

## Development

This project uses:

- **Rust** for the core algorithm implementation
- **PyO3** for Python bindings
- **Maturin** for building Python extensions
- **pytest** for testing

### Building

```bash
# Install maturin
pip install maturin

# Build in development mode
maturin develop

# Build wheel for distribution
maturin build
```

### Testing

```bash
python -m pytest tests/ -v
```

## Architecture

The project maintains backward compatibility by:

1. Using Rust for the core algorithm (better performance)
2. Providing Python fallback if Rust module is unavailable
3. Keeping the same Python API and test suite
4. Supporting standard pip installation

The Rust implementation is automatically used when available, with seamless fallback to the original Python implementation if needed.
