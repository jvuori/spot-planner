#!/usr/bin/env python3
"""
Debug script for custom_test_data scenario to identify constraint violations.
"""

import sys
from decimal import Decimal
from spot_planner import get_cheapest_periods
from spot_planner.two_phase import _validate_full_selection

# Monkey patch the validation function to print detailed diagnostics
original_validate = _validate_full_selection

def debug_validate_full_selection(
    selected_indices: list[int],
    total_length: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> bool:
    """Debug version that prints detailed constraint analysis."""
    print(f"\n=== CONSTRAINT VALIDATION DEBUG ===")
    print(f"Total periods: {total_length}")
    print(f"Selected indices: {selected_indices}")
    print(f"Selections: {len(selected_indices)}")
    print(f"min_consecutive_periods: {min_consecutive_periods}")
    print(f"max_gap_between_periods: {max_gap_between_periods}")
    print(f"max_gap_from_start: {max_gap_from_start}")

    if not selected_indices:
        print("❌ FAIL: No periods selected")
        return False

    indices = sorted(selected_indices)

    # Check max_gap_from_start
    if indices[0] > max_gap_from_start:
        print(f"❌ FAIL: First selection at index {indices[0]}, but max_gap_from_start is {max_gap_from_start}")
        return False
    else:
        print(f"✓ PASS: First selection at index {indices[0]} <= max_gap_from_start {max_gap_from_start}")

    # Check gap from start (also constrained by max_gap_between_periods)
    if indices[0] > max_gap_between_periods:
        print(f"❌ FAIL: First selection at index {indices[0]}, but max_gap_between_periods is {max_gap_between_periods}")
        return False
    else:
        print(f"✓ PASS: First selection at index {indices[0]} <= max_gap_between_periods {max_gap_between_periods}")

    # Check gap at end
    end_gap = total_length - 1 - indices[-1]
    if end_gap > max_gap_between_periods:
        print(f"❌ FAIL: End gap of {end_gap} periods after index {indices[-1]}, but max_gap_between_periods is {max_gap_between_periods}")
        return False
    else:
        print(f"✓ PASS: End gap {end_gap} <= max_gap_between_periods {max_gap_between_periods}")

    # Check gaps between selections and consecutive block lengths
    print(f"\nAnalyzing consecutive runs:")
    block_length = 1
    consecutive_runs = []
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            print(f"❌ FAIL: Gap of {gap} periods between index {indices[i-1]} and {indices[i]}, but max_gap_between_periods is {max_gap_between_periods}")
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            # End of a block
            consecutive_runs.append((indices[i - block_length], block_length))
            print(f"  Block ending at {indices[i-1]}: length {block_length} {'✓' if block_length >= min_consecutive_periods else '✗'}")
            if block_length < min_consecutive_periods:
                print(f"❌ FAIL: Consecutive run starting at index {indices[i - block_length]} has length {block_length}, but min_consecutive_periods is {min_consecutive_periods}")
                return False
            block_length = 1

    # Final block
    consecutive_runs.append((indices[len(indices) - block_length], block_length))
    print(f"  Final block starting at {indices[len(indices) - block_length]}: length {block_length} {'✓' if block_length >= min_consecutive_periods else '✗'}")
    if block_length < min_consecutive_periods:
        print(f"❌ FAIL: Final consecutive run starting at index {indices[len(indices) - block_length]} has length {block_length}, but min_consecutive_periods is {min_consecutive_periods}")
        return False

    print(f"✓ PASS: All constraints met")
    return True

# Apply monkey patch
import spot_planner.two_phase
spot_planner.two_phase._validate_full_selection = debug_validate_full_selection

# Custom test data prices (from visualize_results.py)
prices = [
    Decimal("4.78"),
    Decimal("4.252"),
    Decimal("3.869"),
    Decimal("3.721"),
    Decimal("3.792"),
    Decimal("3.697"),
    Decimal("3.593"),
    Decimal("3.476"),
    Decimal("7.152"),
    Decimal("4.211"),
    Decimal("4.133"),
    Decimal("3.687"),
    Decimal("4.084"),
    Decimal("3.875"),
    Decimal("3.66"),
    Decimal("3.503"),
    Decimal("3.712"),
    Decimal("3.637"),
    Decimal("3.335"),
    Decimal("3.048"),
    Decimal("2.896"),
    Decimal("3.182"),
    Decimal("3.131"),
    Decimal("3.119"),
    Decimal("2.727"),
    Decimal("2.938"),
    Decimal("3.195"),
    Decimal("3.488"),
    Decimal("2.6"),
    Decimal("3.028"),
    Decimal("3.559"),
    Decimal("4.321"),
    Decimal("2.301"),
    Decimal("3.21"),
    Decimal("4.29"),
    Decimal("5.699"),
    Decimal("4.5"),
    Decimal("6.147"),
    Decimal("8.161"),
    Decimal("9.924"),
    Decimal("5.945"),
    Decimal("6.764"),
    Decimal("7.731"),
    Decimal("8.374"),
    Decimal("7.501"),
    Decimal("7.962"),
    Decimal("9.852"),
    Decimal("10.999"),
    Decimal("6.248"),
    Decimal("7.505"),
    Decimal("9.999"),
    Decimal("10.825"),
    Decimal("8.566"),
    Decimal("8.603"),
    Decimal("8.221"),
    Decimal("8.346"),
    Decimal("7.832"),
    Decimal("8.211"),
    Decimal("7.507"),
    Decimal("7.253"),
    Decimal("11.3"),
    Decimal("11.517"),
    Decimal("12.154"),
    Decimal("13.057"),
    Decimal("11.606"),
    Decimal("12.523"),
    Decimal("14.322"),
    Decimal("15.832"),
    Decimal("12.529"),
    Decimal("13.219"),
    Decimal("14.048"),
    Decimal("14.923"),
    Decimal("13.182"),
    Decimal("14.6"),
    Decimal("14.995"),
    Decimal("14.643"),
    Decimal("15.967"),
    Decimal("16.106"),
    Decimal("15.394"),
    Decimal("15.005"),
    Decimal("14.696"),
    Decimal("14.18"),
    Decimal("13.969"),
    Decimal("13.987"),
    Decimal("14.071"),
    Decimal("13.576"),
    Decimal("13.501"),
    Decimal("12.638"),
    Decimal("9.999"),
    Decimal("8.164"),
    Decimal("7.12"),
    Decimal("6.717"),
    Decimal("7.925"),
    Decimal("10.565"),
    Decimal("10.636"),
    Decimal("9.747"),
]

params = {
    "low_price_threshold": Decimal("4.665812749003984"),
    "min_selections": 34,
    "min_consecutive_periods": 4,
    "max_gap_between_periods": 24,
    "max_gap_from_start": 22,
    "aggressive": False,
}

print("Running custom_test_data scenario...")

try:
    selected = get_cheapest_periods(
        prices=prices,
        aggressive=params["aggressive"],
        low_price_threshold=params["low_price_threshold"],
        min_selections=params["min_selections"],
        min_consecutive_periods=params["min_consecutive_periods"],
        max_gap_between_periods=params["max_gap_between_periods"],
        max_gap_from_start=params["max_gap_from_start"],
    )
    print(f"\nSUCCESS: Selected {len(selected)} periods")
except ValueError as e:
    print(f"\nFAILED: {e}")
