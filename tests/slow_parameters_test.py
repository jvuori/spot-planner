import time
from decimal import Decimal

import pytest

from spot_planner.main import get_cheapest_periods


def test_slow_parameters():
    """Test the specific parameters that are causing performance issues."""
    prices = [
        Decimal("0.31"),
        Decimal("0.444"),
        Decimal("0.317"),
        Decimal("0.283"),
        Decimal("0.299"),
        Decimal("0.344"),
        Decimal("0.4"),
        Decimal("1.224"),
        Decimal("1.49"),
        Decimal("2.452"),
        Decimal("2.524"),
        Decimal("1.523"),
        Decimal("1.53"),
        Decimal("1.491"),
        Decimal("1.481"),
        Decimal("1.688"),
        Decimal("1.21"),
        Decimal("1.15"),
        Decimal("1.167"),
        Decimal("0.963"),
        Decimal("0.505"),
        Decimal("0.429"),
        Decimal("0.399"),
        Decimal("0.417"),
    ]

    low_price_threshold = Decimal("1.140812749003984063745019920")
    min_selections = 12
    min_consecutive_periods = 1
    max_gap_between_periods = 4
    max_gap_from_start = 4
    aggressive = False

    print(f"Testing with {len(prices)} prices")
    print(f"Low price threshold: {low_price_threshold}")
    print(f"Min selections: {min_selections}")
    print(f"Min consecutive periods: {min_consecutive_periods}")
    print(
        f"Gap constraints: max_gap_between_periods={max_gap_between_periods}, max_gap_from_start={max_gap_from_start}"
    )
    print(f"Aggressive mode: {aggressive}")

    # Count cheap items
    cheap_items = [p for p in prices if p <= low_price_threshold]
    print(
        f"Cheap items (<= {low_price_threshold}): {len(cheap_items)} out of {len(prices)}"
    )
    print(f"Cheap items: {[float(p) for p in cheap_items]}")

    start_time = time.time()

    try:
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=low_price_threshold,
            min_selections=min_selections,
            min_consecutive_periods=min_consecutive_periods,
            max_gap_between_periods=max_gap_between_periods,
            max_gap_from_start=max_gap_from_start,
            aggressive=aggressive,
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Result: {result}")
        print(f"Result length: {len(result)}")

        # Verify the result
        assert len(result) >= min_selections, (
            f"Result has {len(result)} items, but min_selections is {min_selections}"
        )

        # Check that all selected prices are valid
        selected_prices = [prices[i] for i in result]
        print(f"Selected prices: {[float(p) for p in selected_prices]}")

        # Verify consecutive requirements
        if len(result) > 1:
            result_sorted = sorted(result)
            consecutive_count = 1
            max_consecutive = 1
            min_consecutive_found = float("inf")
            for i in range(1, len(result_sorted)):
                if result_sorted[i] == result_sorted[i - 1] + 1:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    min_consecutive_found = min(
                        min_consecutive_found, consecutive_count
                    )
                    consecutive_count = 1

            # Check the last block
            min_consecutive_found = min(min_consecutive_found, consecutive_count)

            print(f"Max consecutive run: {max_consecutive}")
            print(f"Min consecutive run: {min_consecutive_found}")

            # Only check that minimum consecutive requirement is met
            assert min_consecutive_found >= min_consecutive_periods, (
                f"Min consecutive run {min_consecutive_found} is less than min_consecutive_periods {min_consecutive_periods}"
            )

        # Check gap constraints
        if len(result) > 1:
            result_sorted = sorted(result)
            for i in range(1, len(result_sorted)):
                gap = result_sorted[i] - result_sorted[i - 1] - 1
                assert gap <= max_gap_between_periods, (
                    f"Gap {gap} between {result_sorted[i - 1]} and {result_sorted[i]} exceeds max_gap_between_periods {max_gap_between_periods}"
                )

            # Check gap from start
            assert result_sorted[0] <= max_gap_from_start, (
                f"First item at index {result_sorted[0]} exceeds max_gap_from_start {max_gap_from_start}"
            )

            # Check gap from end
            gap_from_end = len(prices) - 1 - result_sorted[-1]
            assert gap_from_end <= max_gap_between_periods, (
                f"Gap from end {gap_from_end} exceeds max_gap_between_periods {max_gap_between_periods}"
            )

        print("Test passed!")

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Error: {e}")
        raise


@pytest.mark.slow
def test_slow_parameters_with_timing():
    """Same test but marked as slow for CI purposes."""
    test_slow_parameters()
