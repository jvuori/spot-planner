"""
Comprehensive test suite for corner cases and invalid parameter combinations.

This test suite covers:
1. Invalid parameter combinations that should throw exceptions
2. Real-world edge cases and corner scenarios
3. Boundary conditions and extreme values
4. Error handling and validation
"""

from decimal import Decimal

import pytest

from spot_planner.main import get_cheapest_periods


class TestInvalidParameterCombinations:
    """Test cases for invalid parameter combinations that should throw exceptions."""

    def test_desired_count_greater_than_total_items(self):
        """Test that desired_count > len(price_data) throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError,
            match="desired_count cannot be greater than total number of items",
        ):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=5,  # More than 3 items
                min_period=1,
                max_gap=1,
                max_start_gap=1,
            )

    def test_desired_count_zero(self):
        """Test that desired_count = 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="desired_count must be greater than 0"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=0,
                min_period=1,
                max_gap=1,
                max_start_gap=1,
            )

    def test_desired_count_negative(self):
        """Test that desired_count < 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="desired_count must be greater than 0"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=-1,
                min_period=1,
                max_gap=1,
                max_start_gap=1,
            )

    def test_min_period_greater_than_desired_count(self):
        """Test that min_period > desired_count throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="min_period cannot be greater than desired_count"
        ):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=3,  # Greater than desired_count
                max_gap=1,
                max_start_gap=1,
            )

    def test_min_period_zero(self):
        """Test that min_period = 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_period must be greater than 0"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=0,
                max_gap=1,
                max_start_gap=1,
            )

    def test_min_period_negative(self):
        """Test that min_period < 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_period must be greater than 0"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=-1,
                max_gap=1,
                max_start_gap=1,
            )

    def test_max_gap_negative(self):
        """Test that max_gap < 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_gap must be greater than or equal to 0"
        ):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=1,
                max_gap=-1,
                max_start_gap=1,
            )

    def test_max_start_gap_negative(self):
        """Test that max_start_gap < 0 throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_start_gap must be greater than or equal to 0"
        ):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=1,
                max_gap=1,
                max_start_gap=-1,
            )

    def test_max_start_gap_greater_than_max_gap(self):
        """Test that max_start_gap > max_gap throws ValueError."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_start_gap must be less than or equal to max_gap"
        ):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=1,
                max_gap=1,
                max_start_gap=2,  # Greater than max_gap
            )

    def test_empty_price_data(self):
        """Test that empty price_data throws ValueError."""
        with pytest.raises(ValueError, match="price_data cannot be empty"):
            get_cheapest_periods(
                price_data=[],
                price_threshold=Decimal("25"),
                desired_count=1,
                min_period=1,
                max_gap=1,
                max_start_gap=1,
            )


class TestRealWorldCornerCases:
    """Test cases for real-world corner cases and edge scenarios."""

    def test_single_price_item(self):
        """Test with only one price item."""
        price_data = [Decimal("15")]

        # Should return the single item if it meets criteria
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("20"),
            desired_count=1,
            min_period=1,
            max_gap=0,
            max_start_gap=0,
        )
        assert result == [0]

    def test_single_price_item_above_threshold(self):
        """Test with single price item above threshold."""
        price_data = [Decimal("25")]

        # Should return the single item even if above threshold (desired_count=1)
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("20"),
            desired_count=1,
            min_period=1,
            max_gap=0,
            max_start_gap=0,
        )
        assert result == [0]

    def test_two_price_items(self):
        """Test with exactly two price items."""
        price_data = [Decimal("10"), Decimal("30")]

        # Both items below threshold, should return both
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert result == [0, 1]

    def test_all_prices_identical(self):
        """Test with all prices being identical."""
        price_data = [Decimal("15")] * 10

        # All prices identical and below threshold
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("20"),
            desired_count=5,
            min_period=1,
            max_gap=2,
            max_start_gap=2,
        )
        assert (
            len(result) == 10
        )  # Should return all items since all are below threshold
        assert result == list(range(10))

    def test_all_prices_above_threshold(self):
        """Test with all prices above threshold but desired_count < total."""
        price_data = [Decimal("50"), Decimal("60"), Decimal("70")]

        # All prices above threshold, but we want only 2
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("40"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 2
        assert result == [0, 1]  # Should return first 2 (cheapest)

    def test_very_small_price_differences(self):
        """Test with very small price differences."""
        price_data = [
            Decimal("10.0001"),
            Decimal("10.0002"),
            Decimal("10.0003"),
            Decimal("10.0004"),
            Decimal("10.0005"),
        ]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("10.0003"),
            desired_count=3,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 4  # All items below/equal threshold
        assert result == [0, 1, 2, 3]

    def test_negative_prices(self):
        """Test with negative prices (realistic for electricity spot prices)."""
        price_data = [
            Decimal("-0.5"),
            Decimal("-0.3"),
            Decimal("-0.1"),
            Decimal("0.1"),
            Decimal("0.3"),
        ]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("0.0"),
            desired_count=3,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 4  # All items below/equal threshold
        assert result == [0, 1, 2, 3]

    def test_zero_prices(self):
        """Test with zero prices."""
        price_data = [Decimal("0"), Decimal("0"), Decimal("10"), Decimal("20")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("5"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 3  # All items below/equal threshold
        assert result == [0, 1, 2]

    def test_very_large_numbers(self):
        """Test with very large price numbers."""
        price_data = [
            Decimal("999999.99"),
            Decimal("999999.98"),
            Decimal("999999.97"),
        ]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("999999.98"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 2
        assert set(result) == {1, 2}  # Last 2 items (cheapest)

    def test_extreme_gap_constraints(self):
        """Test with extreme gap constraints."""
        price_data = [Decimal("10")] * 20

        # Very strict gap constraints
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("15"),
            desired_count=5,
            min_period=1,
            max_gap=0,  # No gaps allowed
            max_start_gap=0,
        )
        assert len(result) == 20  # All items below threshold
        assert result == list(range(20))

    def test_very_loose_gap_constraints(self):
        """Test with very loose gap constraints."""
        price_data = [Decimal("10")] * 10

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("15"),
            desired_count=3,
            min_period=1,
            max_gap=10,  # Very loose
            max_start_gap=10,
        )
        assert len(result) == 10  # All items below threshold
        assert result == list(range(10))

    def test_min_period_equals_desired_count(self):
        """Test when min_period equals desired_count."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,
            min_period=2,  # Same as desired_count
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 2
        assert result == [0, 1]  # Consecutive items

    def test_max_gap_zero(self):
        """Test with max_gap = 0 (no gaps allowed)."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        # This should raise an error due to impossible constraints
        with pytest.raises(ValueError, match="No combination found"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=1,
                max_gap=0,  # No gaps
                max_start_gap=0,
            )

    def test_max_start_gap_zero(self):
        """Test with max_start_gap = 0 (must start from beginning)."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=0,  # Must start from index 0
        )
        assert len(result) == 3  # All items below/equal threshold
        assert result == [0, 1, 2]

    def test_impossible_constraints(self):
        """Test with impossible constraints that should fail gracefully."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        # This should work because desired_count=3 equals total items
        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=3,
            min_period=3,
            max_gap=0,  # No gaps allowed
            max_start_gap=0,
        )
        assert result == [0, 1, 2]  # All items


class TestBoundaryConditions:
    """Test cases for boundary conditions and extreme values."""

    def test_desired_count_equals_total_items(self):
        """Test when desired_count equals total number of items."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=3,  # Same as total items
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert result == [0, 1, 2]  # All items

    def test_desired_count_equals_cheap_items_count(self):
        """Test when desired_count equals number of cheap items."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,  # Same as cheap items count
            min_period=1,
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 3  # Should return all items since all are below threshold
        assert result == [0, 1, 2]

    def test_min_period_equals_one(self):
        """Test with min_period = 1 (minimum valid value)."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,
            min_period=1,  # Minimum valid value
            max_gap=1,
            max_start_gap=1,
        )
        assert len(result) == 2

    def test_max_gap_equals_zero(self):
        """Test with max_gap = 0 (no gaps allowed)."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        # This should raise an error due to impossible constraints
        with pytest.raises(ValueError, match="No combination found"):
            get_cheapest_periods(
                price_data=price_data,
                price_threshold=Decimal("25"),
                desired_count=2,
                min_period=1,
                max_gap=0,  # No gaps
                max_start_gap=0,
            )

    def test_max_start_gap_equals_zero(self):
        """Test with max_start_gap = 0 (must start from beginning)."""
        price_data = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            price_data=price_data,
            price_threshold=Decimal("25"),
            desired_count=2,
            min_period=1,
            max_gap=1,
            max_start_gap=0,  # Must start from 0
        )
        assert len(result) == 2
        assert result[0] == 0  # Must start from 0
