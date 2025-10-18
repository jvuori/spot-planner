#!/usr/bin/env python3

import itertools
import time
from decimal import Decimal


def test_validation_speed():
    """Test how fast the validation function is"""
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

    price_items = list(enumerate(prices))

    # Test validation speed
    print("Testing validation speed...")
    start_time = time.time()

    valid_count = 0
    total_count = 0

    for combination in itertools.combinations(price_items, min_selections):
        total_count += 1
        if total_count > 10000:  # Test first 10k combinations
            break

        # Test the validation logic
        if not combination:
            continue

        indices = [index for index, _ in combination]

        # Check max_gap_from_start first (fastest check)
        if indices[0] > max_gap_from_start:
            continue

        # Check start gap
        if indices[0] > max_gap_between_periods:
            continue

        # Check gaps between consecutive indices and min_consecutive_selections in single pass
        block_length = 1
        valid = True
        for i in range(1, len(indices)):
            gap = indices[i] - indices[i - 1] - 1
            if gap > max_gap_between_periods:
                valid = False
                break

            if indices[i] == indices[i - 1] + 1:
                block_length += 1
            else:
                if block_length < min_consecutive_selections:
                    valid = False
                    break
                block_length = 1

        if valid and block_length >= min_consecutive_selections:
            # Check end gap
            if (len(price_items) - 1 - indices[-1]) <= max_gap_between_periods:
                valid_count += 1
                if valid_count == 1:
                    print(f"First valid combination: {indices}")

    end_time = time.time()
    print(
        f"Validated {total_count} combinations in {end_time - start_time:.3f} seconds"
    )
    print(f"Found {valid_count} valid combinations")
    print(
        f"Validation rate: {total_count / (end_time - start_time):.0f} combinations/second"
    )


if __name__ == "__main__":
    test_validation_speed()
