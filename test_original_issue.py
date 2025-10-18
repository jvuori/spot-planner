#!/usr/bin/env python3
"""
Standalone test for the original issue scenario.
Tests that both Rust and Python implementations work correctly and select the first 3 items.
"""

import sys
sys.path.insert(0, 'src')

from decimal import Decimal
from spot_planner.main import _get_cheapest_periods_python, get_cheapest_periods


def test_original_issue_scenario():
    """Test the original issue scenario with detailed validation."""
    print("=== Testing Original Issue Scenario ===")
    
    # Original input data that was causing the issue
    prices = [
        Decimal("2.232"),
        Decimal("2.4"),
        Decimal("2.599"),
        Decimal("2.768"),
        Decimal("2.6"),
        Decimal("3.336"),
        Decimal("3.5"),
        Decimal("3.349"),
        Decimal("3.148"),
        Decimal("2.625"),
        Decimal("2.51"),
        Decimal("3.992"),
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
    ]
    low_price_threshold = Decimal("4.355812749003984063745019920")
    min_selections = 16
    min_consecutive_selections = 4
    max_consecutive_selections = 8
    max_gap_between_periods = 18
    max_gap_from_start = 16

    print(f"Input parameters:")
    print(f"  Prices: {len(prices)} items")
    print(f"  Low price threshold: {low_price_threshold}")
    print(f"  Min selections: {min_selections}")
    print(f"  Min consecutive selections: {min_consecutive_selections}")
    print(f"  Max consecutive selections: {max_consecutive_selections}")
    print(f"  Max gap between periods: {max_gap_between_periods}")
    print(f"  Max gap from start: {max_gap_from_start}")
    
    # Count cheap items
    cheap_items = [i for i, p in enumerate(prices) if p <= low_price_threshold]
    print(f"  Cheap items: {len(cheap_items)}/{len(prices)} ({len(cheap_items)/len(prices)*100:.1f}%)")
    print(f"  First 3 items prices: {[prices[i] for i in [0, 1, 2]]}")
    print(f"  First 3 items below threshold: {all(prices[i] <= low_price_threshold for i in [0, 1, 2])}")

    # Get results from both implementations
    print(f"\n=== Running Implementations ===")
    
    try:
        rust_result = get_cheapest_periods(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
        print(f"✓ Rust implementation succeeded")
    except Exception as e:
        print(f"✗ Rust implementation failed: {e}")
        return False

    try:
        python_result = _get_cheapest_periods_python(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
        print(f"✓ Python implementation succeeded")
    except Exception as e:
        print(f"✗ Python implementation failed: {e}")
        return False

    # Results should be identical
    print(f"\n=== Comparing Results ===")
    print(f"Rust result:   {rust_result}")
    print(f"Python result: {python_result}")
    print(f"Length: Rust={len(rust_result)}, Python={len(python_result)}")
    
    if rust_result != python_result:
        print(f"✗ Results are different!")
        return False
    else:
        print(f"✓ Results are identical")

    # Both implementations should select the first 3 items (the original issue)
    first_three_selected = all(i in rust_result for i in [0, 1, 2])
    print(f"\n=== Validating First 3 Items ===")
    print(f"First 3 items [0, 1, 2] selected: {first_three_selected}")
    
    if not first_three_selected:
        print(f"✗ First 3 items not selected!")
        print(f"  Selected items: {rust_result}")
        print(f"  First 3 prices: {[prices[i] for i in [0, 1, 2]]}")
        print(f"  All below threshold: {all(prices[i] <= low_price_threshold for i in [0, 1, 2])}")
        return False
    else:
        print(f"✓ First 3 items are selected!")

    # Verify the result has the correct length
    if len(rust_result) != min_selections:
        print(f"✗ Result length {len(rust_result)} != expected {min_selections}")
        return False
    else:
        print(f"✓ Result has correct length: {len(rust_result)}")

    # Verify all selected items are valid indices
    if not all(0 <= i < len(prices) for i in rust_result):
        print(f"✗ Invalid indices in result: {rust_result}")
        return False
    else:
        print(f"✓ All indices are valid")

    # Verify the result is sorted (indices should be in order)
    if rust_result != sorted(rust_result):
        print(f"✗ Result indices not sorted: {rust_result}")
        return False
    else:
        print(f"✓ Result indices are sorted")

    # Calculate and verify cost metrics
    total_cost = sum(prices[i] for i in rust_result)
    avg_cost = total_cost / len(rust_result)
    cheap_count = sum(1 for i in rust_result if prices[i] <= low_price_threshold)

    print(f"\n=== Cost Analysis ===")
    print(f"Total cost: {total_cost}")
    print(f"Average cost: {avg_cost:.3f}")
    print(f"Cheap items: {cheap_count}/{len(rust_result)} ({cheap_count/len(rust_result)*100:.1f}%)")
    
    # Show which items were selected
    print(f"\n=== Selected Items ===")
    for i in rust_result:
        price = prices[i]
        is_cheap = price <= low_price_threshold
        print(f"  [{i:2d}] {price:8.3f} {'(cheap)' if is_cheap else '(expensive)'}")

    print(f"\n🎉 All tests passed! Both implementations work correctly!")
    return True


if __name__ == "__main__":
    success = test_original_issue_scenario()
    sys.exit(0 if success else 1)
