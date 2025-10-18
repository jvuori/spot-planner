#!/usr/bin/env python3

import sys
import time
from decimal import Decimal

# Import the Rust module directly
try:
    from src.spot_planner import spot_planner as rust_module

    print("Using Rust implementation")
except ImportError:
    print("Rust module not available")
    sys.exit(1)


def test_minimal_rust():
    """Test the Rust function directly with minimal input"""
    prices = [Decimal("1.0"), Decimal("2.0"), Decimal("3.0"), Decimal("4.0")]
    low_price_threshold = Decimal("2.5")
    min_selections = 2
    min_consecutive_selections = 1
    max_consecutive_selections = 2
    max_gap_between_periods = 1
    max_gap_from_start = 1

    print("Testing minimal case...")
    start_time = time.time()

    try:
        result = rust_module.get_cheapest_periods(
            [str(p) for p in prices],
            str(low_price_threshold),
            min_selections,
            min_consecutive_selections,
            max_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )

        end_time = time.time()
        print(f"Minimal case success! Result: {result}")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        return True

    except Exception as e:
        end_time = time.time()
        print(f"Minimal case error after {end_time - start_time:.3f} seconds: {e}")
        return False


def test_original_rust():
    """Test the original case with Python wrapper"""
    from src.spot_planner.main import get_cheapest_periods

    prices = [
        Decimal("3.17"),
        Decimal("2.98"),
        Decimal("3.702"),
        Decimal("5.067"),
        Decimal("4.19"),
        Decimal("4.692"),
        Decimal("4.493"),
        Decimal("3.813"),
        Decimal("4.902"),
        Decimal("3.559"),
        Decimal("2.396"),
        Decimal("1.758"),
        Decimal("2.599"),
        Decimal("1.531"),
        Decimal("1.25"),
        Decimal("1.134"),
        Decimal("1.167"),
        Decimal("1.077"),
        Decimal("0.583"),
        Decimal("0.404"),
        Decimal("0.508"),
        Decimal("0.473"),
        Decimal("0.4"),
        Decimal("0.341"),
    ]

    low_price_threshold = Decimal("2.554812749003984063745019920")
    min_selections = 12
    min_consecutive_selections = 1
    max_consecutive_selections = 1
    max_gap_between_periods = 18
    max_gap_from_start = 17

    print("\nTesting original case with Python wrapper...")
    start_time = time.time()

    try:
        result = get_cheapest_periods(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )

        end_time = time.time()
        print(f"Original case success! Result: {result}")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        return True

    except Exception as e:
        end_time = time.time()
        print(f"Original case error after {end_time - start_time:.3f} seconds: {e}")
        return False


if __name__ == "__main__":
    # Set a timeout
    import signal

    def timeout_handler(signum, frame):
        print("\nTimeout!")
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    try:
        test_minimal_rust()
        test_original_rust()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        signal.alarm(0)
