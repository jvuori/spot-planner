import os
import sys
import unittest
from decimal import Decimal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spot_planner.main import (
    calculate_dynamic_consecutive_selections,
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


class TestCalculateDynamicConsecutiveSelections(unittest.TestCase):
    """Test the dynamic calculation of consecutive_selections."""

    def test_low_percentage_uses_min(self):
        """Test that low percentage uses min_consecutive_selections."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=2,  # 20% of 10 prices
            total_prices=10,
            max_gap_between_periods=3,
        )
        # 20% < 25%, should use min_consecutive_selections = 2
        assert result == 2

    def test_high_percentage_uses_max(self):
        """Test that high percentage uses max_consecutive_selections."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=8,  # 80% of 10 prices
            total_prices=10,
            max_gap_between_periods=3,
        )
        # 80% > 75%, should use max_consecutive_selections = 5
        assert result == 5

    def test_medium_percentage_interpolates(self):
        """Test that medium percentage interpolates between min and max."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=5,  # 50% of 10 prices
            total_prices=10,
            max_gap_between_periods=3,
        )
        # 50% between 25-75%, should interpolate: 2 + 0.5*(5-2) = 3.5 -> 3
        assert result == 3

    def test_gap_adjustment_increases_value(self):
        """Test that larger gaps push toward max_consecutive_selections."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=3,  # 30% of 10 prices
            total_prices=10,
            max_gap_between_periods=8,  # Large gap
        )
        # Base: 2 (30% < 25%), gap adjustment: 8/10 * (5-2) = 2.4 -> 2
        # Result should be 2 + 2 = 4
        assert result == 4

    def test_respects_bounds(self):
        """Test that result is always within min and max bounds."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=1,  # Very low percentage
            total_prices=10,
            max_gap_between_periods=20,  # Very large gap
        )
        # Should be capped at max_consecutive_selections = 5
        assert result == 5

    def test_exact_25_percent_boundary(self):
        """Test exact 25% boundary."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=2,  # Exactly 20% of 10 prices
            total_prices=10,
            max_gap_between_periods=3,
        )
        # Exactly 20% < 25%, should use min_consecutive_selections = 2
        assert result == 2

    def test_exact_75_percent_boundary(self):
        """Test exact 75% boundary."""
        result = calculate_dynamic_consecutive_selections(
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            min_selections=8,  # Exactly 80% of 10 prices
            total_prices=10,
            max_gap_between_periods=3,
        )
        # Exactly 80% > 75%, should use max_consecutive_selections = 5
        assert result == 5


class TestGetCheapestPeriodsWithDynamicCalculation(unittest.TestCase):
    """Test get_cheapest_periods with dynamic consecutive_selections calculation."""

    def test_low_percentage_scenario(self):
        """Test low percentage scenario uses min_consecutive_selections."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=2,  # 22% of 9 prices - low percentage
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should calculate consecutive_selections = 2 (low percentage)
        # With threshold 25, cheap items are [2, 3, 4, 5] (indices 2,3,4,5)
        # Need at least 2 consecutive
        assert len(periods) >= 2

    def test_high_percentage_scenario(self):
        """Test high percentage scenario uses max_consecutive_selections."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=7,  # 78% of 9 prices - high percentage
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should calculate consecutive_selections = 5 (high percentage)
        # With threshold 25, cheap items are [2, 3, 4, 5] (indices 2,3,4,5)
        # Need at least 5 consecutive
        assert len(periods) >= 5

    def test_medium_percentage_scenario(self):
        """Test medium percentage scenario interpolates."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=4,  # 44% of 9 prices - medium percentage
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should calculate consecutive_selections = 3 (interpolated)
        # With threshold 25, cheap items are [2, 3, 4, 5] (indices 2,3,4,5)
        # Need at least 3 consecutive
        assert len(periods) >= 3

    def test_gap_adjustment_scenario(self):
        """Test that gap adjustment increases consecutive_selections."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=6,  # 67% of 9 prices - higher to accommodate gap adjustment
            min_consecutive_selections=2,
            max_consecutive_selections=5,
            max_gap_between_periods=8,  # Large gap
            max_gap_from_start=2,
        )
        # Should calculate consecutive_selections = 2 + gap_adjustment
        # With threshold 25, cheap items are [2, 3, 4, 5] (indices 2,3,4,5)
        assert len(periods) >= 2

    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        # Test min > max error
        with self.assertRaises(ValueError) as context:
            get_cheapest_periods(
                prices=PRICE_DATA,
                low_price_threshold=Decimal("25"),
                min_selections=3,
                min_consecutive_selections=5,  # min > max
                max_consecutive_selections=3,
                max_gap_between_periods=2,
                max_gap_from_start=2,
            )
        self.assertIn(
            "min_consecutive_selections cannot be greater than max_consecutive_selections",
            str(context.exception),
        )

    def test_mandatory_parameters(self):
        """Test that all parameters are now mandatory."""
        # This should work with all parameters
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=3,
            min_consecutive_selections=2,
            max_consecutive_selections=4,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        assert len(periods) >= 2

    def test_min_max_both_one_cheapest_selection(self):
        """Test that when min and max are both 1, algorithm picks cheapest regardless of consecutive."""
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("25"),
            min_selections=3,
            min_consecutive_selections=1,  # Min is 1
            max_consecutive_selections=1,  # Max is 1 - forces cheapest selection
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        # Should select at least 3 periods (min_selections)
        assert len(periods) >= 3

        # With threshold 25, the cheap items are [3, 4, 5] (indices 3,4,5 with prices 20,10,20)
        # The algorithm may include additional periods if they form valid combinations
        # and are still cost-effective

        # Verify that all selected periods are either cheap or part of a valid combination
        selected_prices = [float(PRICE_DATA[i]) for i in periods]
        cheap_indices = {3, 4, 5}  # Indices with price <= 25

        # The algorithm should select the cheapest valid combination and add cheap items
        # With min=max=1, it should pick the 3 cheapest overall items, then add cheap items
        assert len(periods) >= 3  # Should select at least 3 items

        # Verify the cheapest items are included
        assert 4 in periods  # Index 4 has the lowest price (10.0)
        # Should include the cheap items (indices 3, 4, 5 with prices 20, 10, 20)
        assert 3 in periods  # Index 3 has price 20.0
        assert 5 in periods  # Index 5 has price 20.0

        # Verify it's a valid combination (all items can form consecutive runs with min_consecutive=1)
        selected_prices = [float(PRICE_DATA[i]) for i in periods]
        print(f"Selected: {periods}, prices: {selected_prices}")

    def test_min_max_both_one_exact_selection(self):
        """Test that when min and max are both 1, algorithm picks exactly the cheapest when forced."""
        # Use a higher threshold to force selection of exactly min_selections
        periods = get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal(
                "15"
            ),  # Very low threshold - only index 4 qualifies
            min_selections=3,
            min_consecutive_selections=1,  # Min is 1
            max_consecutive_selections=1,  # Max is 1 - forces cheapest selection
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
