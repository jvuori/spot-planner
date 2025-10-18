#!/usr/bin/env python3

import itertools
import time
from decimal import Decimal


def is_valid_combination(
    combination,
    min_consecutive_selections,
    max_gap_between_periods,
    max_gap_from_start,
    full_length,
):
    if not combination:
        return False

    # Items are already sorted, so indices are in order
    indices = [index for index, _ in combination]

    # Check max_gap_from_start first (fastest check)
    if indices[0] > max_gap_from_start:
        return False

    # Check start gap
    if indices[0] > max_gap_between_periods:
        return False

    # Check gaps between consecutive indices and min_consecutive_selections in single pass
    block_length = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            if block_length < min_consecutive_selections:
                return False
            block_length = 1

    # Check last block min_consecutive_selections
    if block_length < min_consecutive_selections:
        return False

    # Check end gap
    if (full_length - 1 - indices[-1]) > max_gap_between_periods:
        return False

    return True


def debug_algorithm():
    """Debug the algorithm step by step"""
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
    print(f"Max gap between periods: {max_gap_between_periods}")
    print(f"Max gap from start: {max_gap_from_start}")

    # Calculate actual consecutive selections
    cheap_percentage = len(cheap_items) / len(price_items)
    if cheap_percentage > 0.8:
        actual_consecutive_selections = min_consecutive_selections
    else:
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

    # Test combinations starting from min_selections
    found = False
    start_time = time.time()

    for current_count in range(min_selections, len(price_items) + 1):
        print(f"\nTrying combinations of size {current_count}...")
        combination_count = 0
        valid_count = 0

        for combination in itertools.combinations(price_items, current_count):
            combination_count += 1

            if combination_count % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  Checked {combination_count:,} combinations in {elapsed:.1f}s")

            if is_valid_combination(
                combination,
                actual_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
                len(price_items),
            ):
                valid_count += 1
                if valid_count == 1:
                    indices = [idx for idx, _ in combination]
                    print(f"  First valid combination: {indices}")
                    found = True
                    break

        print(f"  Total combinations checked: {combination_count:,}")
        print(f"  Valid combinations found: {valid_count}")

        if found:
            break

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print(f"Found valid combination: {found}")


if __name__ == "__main__":
    debug_algorithm()
