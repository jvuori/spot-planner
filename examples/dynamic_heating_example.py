#!/usr/bin/env python3
"""
Example demonstrating get_cheapest_periods usage
with different heating requirements.
"""

import os
import sys
from decimal import Decimal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spot_planner.main import get_cheapest_periods


def demonstrate_heating_example():
    """Demonstrate get_cheapest_periods usage for different heating scenarios."""

    # Sample price data (24 hours of hourly prices)
    prices = [
        Decimal("45.2"),
        Decimal("42.1"),
        Decimal("38.5"),
        Decimal("35.2"),  # 0-3
        Decimal("32.8"),
        Decimal("28.9"),
        Decimal("25.4"),
        Decimal("22.1"),  # 4-7
        Decimal("19.8"),
        Decimal("18.2"),
        Decimal("16.5"),
        Decimal("15.1"),  # 8-11
        Decimal("14.8"),
        Decimal("16.2"),
        Decimal("18.9"),
        Decimal("22.4"),  # 12-15
        Decimal("26.1"),
        Decimal("29.8"),
        Decimal("33.5"),
        Decimal("37.2"),  # 16-19
        Decimal("41.8"),
        Decimal("44.5"),
        Decimal("47.1"),
        Decimal("49.8"),  # 20-23
    ]

    print("=== Heating Period Selection Example ===\n")

    # Scenario 1: Low heating requirement (summer scenario)
    print("🌞 LOW HEATING SCENARIO (Summer - low heating)")
    print("-" * 50)
    print("Min selections: 6 (25% of 24 prices - summer)")
    print("Min consecutive periods: 2")

    low_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=6,
        min_consecutive_periods=2,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {low_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in low_periods]}")
    print()

    # Scenario 2: High heating requirement (winter scenario)
    print("❄️ HIGH HEATING SCENARIO (Winter - high heating)")
    print("-" * 50)
    print("Min selections: 18 (75% of 24 prices - winter)")
    print("Min consecutive periods: 2")

    high_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=18,
        min_consecutive_periods=2,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {high_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in high_periods]}")
    print()

    # Scenario 3: Medium heating requirement
    print("🌡️ MEDIUM HEATING SCENARIO")
    print("-" * 50)
    print("Min selections: 12 (50% of 24 prices)")
    print("Min consecutive periods: 2")

    medium_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=12,
        min_consecutive_periods=2,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {medium_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in medium_periods]}")
    print()

    # Scenario 4: Large gaps scenario
    print("⏰ LARGE GAPS SCENARIO")
    print("-" * 50)
    print("Min selections: 9 (37.5% of 24 prices)")
    print("Min consecutive periods: 2")
    print("Max gap between periods: 8 (large gap)")

    gap_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=9,
        min_consecutive_periods=2,
        max_gap_between_periods=8,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {gap_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in gap_periods]}")
    print()

    # Show the algorithm explanation
    print("🔧 ALGORITHM EXPLANATION")
    print("-" * 50)
    print("The algorithm selects periods based on:")
    print("1. min_consecutive_periods: Minimum consecutive periods required for each block")
    print("2. min_selections: Minimum total number of periods to select")
    print("3. max_gap_between_periods: Maximum gap allowed between selected periods")
    print("4. max_gap_from_start: Maximum gap from start before first selection")
    print("5. Cost optimization: Prioritizes cheapest periods while meeting constraints")
    print()
    print("The algorithm:")
    print("- Finds the cheapest combination that meets all constraints")
    print("- Tries adding cheap items (below threshold) to improve the solution")
    print("- Adaptively increases min_selections if no valid solution is found")


if __name__ == "__main__":
    demonstrate_heating_example()
