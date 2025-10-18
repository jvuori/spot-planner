#!/usr/bin/env python3
"""
Example demonstrating dynamic min_consecutive_selections calculation
based on historical heating data and heating requirements.
"""

import os
import sys
from decimal import Decimal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spot_planner.main import (
    calculate_dynamic_consecutive_selections,
    get_cheapest_periods,
)


def demonstrate_dynamic_calculation():
    """Demonstrate how the dynamic calculation works for different scenarios."""

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

    print("=== Dynamic Heating Requirements Calculator ===\n")

    # Scenario 1: Low percentage (summer scenario)
    print("🌞 LOW PERCENTAGE SCENARIO (Summer - low heating)")
    print("-" * 50)
    low_consecutive = calculate_dynamic_consecutive_selections(
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        min_selections=6,  # 25% of 24 prices - low percentage
        total_prices=24,
        max_gap_between_periods=3,
    )
    print("Min selections: 6 (25% of 24 prices - summer)")
    print("Min consecutive: 2, Max consecutive: 5")
    print(f"Calculated consecutive_selections: {low_consecutive}")

    low_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=6,
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {low_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in low_periods]}")
    print()

    # Scenario 2: High percentage (winter scenario)
    print("❄️ HIGH PERCENTAGE SCENARIO (Winter - high heating)")
    print("-" * 50)
    high_consecutive = calculate_dynamic_consecutive_selections(
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        min_selections=18,  # 75% of 24 prices - high percentage
        total_prices=24,
        max_gap_between_periods=3,
    )
    print("Min selections: 18 (75% of 24 prices - winter)")
    print("Min consecutive: 2, Max consecutive: 5")
    print(f"Calculated consecutive_selections: {high_consecutive}")

    high_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=18,
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {high_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in high_periods]}")
    print()

    # Scenario 3: Medium percentage (interpolation)
    print("🌡️ MEDIUM PERCENTAGE SCENARIO (Interpolation)")
    print("-" * 50)
    medium_consecutive = calculate_dynamic_consecutive_selections(
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        min_selections=12,  # 50% of 24 prices - medium percentage
        total_prices=24,
        max_gap_between_periods=3,
    )
    print("Min selections: 12 (50% of 24 prices)")
    print("Min consecutive: 2, Max consecutive: 5")
    print(f"Calculated consecutive_selections: {medium_consecutive}")

    medium_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=12,
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {medium_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in medium_periods]}")
    print()

    # Scenario 4: Gap adjustment scenario
    print("⏰ GAP ADJUSTMENT SCENARIO (Large gaps)")
    print("-" * 50)
    gap_consecutive = calculate_dynamic_consecutive_selections(
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        min_selections=9,  # 37.5% of 24 prices
        total_prices=24,
        max_gap_between_periods=8,  # Large gap
    )
    print("Min selections: 9 (37.5% of 24 prices)")
    print("Min consecutive: 2, Max consecutive: 5")
    print("Max gap between periods: 8 (large gap)")
    print(f"Calculated consecutive_selections: {gap_consecutive}")

    gap_periods = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20.0"),
        min_selections=9,
        min_consecutive_selections=2,
        max_consecutive_selections=5,
        max_gap_between_periods=8,
        max_gap_from_start=2,
    )
    print(f"Selected periods: {gap_periods}")
    print(f"Selected prices: {[float(prices[i]) for i in gap_periods]}")
    print()

    # Show the algorithm explanation
    print("🔧 ALGORITHM EXPLANATION")
    print("-" * 50)
    print("The dynamic calculation considers:")
    print("1. Percentage of min_selections relative to total prices")
    print("   - < 25% of total prices → use min_consecutive_selections")
    print("   - > 75% of total prices → use max_consecutive_selections")
    print("   - 25-75% → linear interpolation between min and max")
    print("2. Gap between heating periods")
    print("   - Larger gaps → push result closer to max_consecutive_selections")
    print("3. Cost optimization")
    print("   - Still prioritizes cheapest periods while meeting heating requirements")
    print()
    print("Formula: base_consecutive + gap_adjustment")
    print("Where base_consecutive is determined by percentage rules")
    print("And gap_adjustment = gap_factor * (max - min)")
    print(
        "Result is bounded by min_consecutive_selections and max_consecutive_selections"
    )


if __name__ == "__main__":
    demonstrate_dynamic_calculation()
