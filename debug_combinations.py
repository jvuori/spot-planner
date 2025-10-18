#!/usr/bin/env python3

import itertools
import time
from decimal import Decimal


def analyze_combination_complexity():
    """Analyze the combination complexity for the problematic input"""
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
    cheap_items = [(i, p) for i, p in price_items if p <= low_price_threshold]

    print(f"Total items: {len(price_items)}")
    print(f"Cheap items: {len(cheap_items)}")
    print(f"Min selections: {min_selections}")

    # Calculate dynamic consecutive selections
    cheap_percentage = len(cheap_items) / len(price_items)
    print(f"Cheap percentage: {cheap_percentage:.1%}")

    if cheap_percentage > 0.8:
        actual_consecutive_selections = min_consecutive_selections
    else:
        # Calculate dynamic consecutive selections
        min_selections_percentage = min_selections / len(price_items)
        if min_selections_percentage <= 0.25:
            base_consecutive = min_consecutive_selections
        elif min_selections_percentage >= 0.75:
            base_consecutive = max_consecutive_selections
        else:
            interpolation_factor = (min_selections_percentage - 0.25) / (0.75 - 0.25)
            base_consecutive = int(
                min_consecutive_selections
                + interpolation_factor
                * (max_consecutive_selections - min_consecutive_selections)
            )

        gap_factor = min(max_gap_between_periods / 10.0, 1.0)
        gap_adjustment = int(
            gap_factor * (max_consecutive_selections - min_consecutive_selections)
        )

        actual_consecutive_selections = max(
            min_consecutive_selections,
            min(base_consecutive + gap_adjustment, max_consecutive_selections),
        )

    print(f"Actual consecutive selections: {actual_consecutive_selections}")

    # Count combinations for each size
    total_combinations = 0
    for current_count in range(min_selections, len(price_items) + 1):
        combinations_count = len(
            list(itertools.combinations(price_items, current_count))
        )
        total_combinations += combinations_count
        print(f"C({len(price_items)},{current_count}) = {combinations_count:,}")

    print(f"Total combinations to check: {total_combinations:,}")

    # Test a small sample to see how many are valid
    print("\nTesting first 1000 combinations of size 12...")
    valid_count = 0
    start_time = time.time()

    for i, combination in enumerate(
        itertools.combinations(price_items, min_selections)
    ):
        if i >= 1000:
            break

        # Check if valid (simplified check)
        indices = [idx for idx, _ in combination]

        # Check max_gap_from_start
        if indices[0] > max_gap_from_start:
            continue

        # Check start gap
        if indices[0] > max_gap_between_periods:
            continue

        # Check gaps and consecutive runs
        block_length = 1
        valid = True
        for j in range(1, len(indices)):
            gap = indices[j] - indices[j - 1] - 1
            if gap > max_gap_between_periods:
                valid = False
                break

            if indices[j] == indices[j - 1] + 1:
                block_length += 1
            else:
                if block_length < actual_consecutive_selections:
                    valid = False
                    break
                block_length = 1

        if valid and block_length >= actual_consecutive_selections:
            # Check end gap
            if (len(price_items) - 1 - indices[-1]) <= max_gap_between_periods:
                valid_count += 1

    end_time = time.time()
    print(f"Valid combinations in first 1000: {valid_count}")
    print(f"Time for 1000 combinations: {end_time - start_time:.3f} seconds")

    if valid_count > 0:
        estimated_time = (total_combinations / 1000) * (end_time - start_time)
        print(f"Estimated time for all combinations: {estimated_time:.1f} seconds")


if __name__ == "__main__":
    analyze_combination_complexity()
