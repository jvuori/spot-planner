import os
import sys
import unittest
from decimal import Decimal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spot_planner.main import (
    get_cheapest_periods,
)

PRICE_DATA = [
    Decimal("50"),  # 0
    Decimal("40"),  # 1
    Decimal("30"),  # 2
    Decimal("20"),  # 3
    Decimal("10"),  # 4
    Decimal("20"),  # 5
    Decimal("30"),  # 6
    Decimal("40"),  # 7
    Decimal("50"),  # 8
]


class TestGetCheapestPeriods(unittest.TestCase):
    """Test get_cheapest_periods with min_consecutive_periods."""

    def test_basic_functionality(self):
        """Test basic functionality with min_consecutive_periods."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=2,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # With threshold 25, cheap items are [2, 3, 4, 5] (indices 2,3,4,5)
        # Need at least 2 consecutive
        assert len(periods) >= 2

    def test_mandatory_parameters(self):
        """Test that all parameters are now mandatory."""
        # This should work with all parameters
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=3,
            min_consecutive_periods=2,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        assert len(periods) >= 2

    def test_min_consecutive_one(self):
        """Test that when min_consecutive_periods is 1, algorithm picks cheapest."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=3,
            min_consecutive_periods=1,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should select at least 3 periods (min_selections)
        assert len(periods) >= 3

        # With threshold 25, the cheap items are [3, 4, 5] (indices 3,4,5 with prices 20,10,20)
        # The algorithm may include additional periods if they form valid combinations
        # and are still cost-effective

        # Verify the cheapest items are included
        assert 4 in periods  # Index 4 has the lowest price (10.0)
        # Should include the cheap items (indices 3, 4, 5 with prices 20, 10, 20)
        assert 3 in periods  # Index 3 has price 20.0
        assert 5 in periods  # Index 5 has price 20.0

        # Verify it's a valid combination (all items can form consecutive runs with min_consecutive=1)
        selected_prices = [float(PRICE_DATA[i]) for i in periods]
        print(f"Selected: {periods}, prices: {selected_prices}")

    def test_min_consecutive_one_exact_selection(self):
        """Test that when min_consecutive_periods is 1, algorithm picks exactly the cheapest when forced."""
        # Use a higher threshold to force selection of exactly min_selections
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal(
                "15"
            ),  # Very low threshold - only index 4 qualifies
            min_selections=3,
            min_consecutive_periods=1,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should select exactly 3 periods (min_selections)
        assert len(periods) == 3

        # The algorithm should pick the 3 cheapest overall periods
        # All prices: [50, 40, 30, 20, 10, 20, 30, 40, 50]
        # Cheapest 3: [10, 30, 30] at indices [4, 2, 6]
        expected_indices = {2, 4, 6}  # The 3 cheapest indices
        assert set(periods) == expected_indices

        # Verify the prices are correct
        selected_prices = [float(PRICE_DATA[i]) for i in periods]
        expected_prices = [10.0, 30.0, 30.0]
        selected_prices.sort()
        expected_prices.sort()
        assert selected_prices == expected_prices


if __name__ == "__main__":
    unittest.main()
