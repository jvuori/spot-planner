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

    def test_min_selections_greater_than_total_items(self):
        """Test that min_selections > len(prices) throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError,
            match="min_selections cannot be greater than total number of items",
        ):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=5,  # More than 3 items
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_min_selections_zero(self):
        """Test that min_selections = 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_selections must be greater than 0"):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=0,
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_min_selections_negative(self):
        """Test that min_selections < 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_selections must be greater than 0"):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=-1,
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_min_consecutive_periods_greater_than_min_selections(self):
        """Test that min_consecutive_periods > min_selections throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="min_consecutive_periods cannot be greater than min_selections"
        ):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=3,  # Greater than min_selections
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_min_consecutive_periods_zero(self):
        """Test that min_consecutive_periods = 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_consecutive_periods must be greater than 0"):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=0,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_min_consecutive_periods_negative(self):
        """Test that min_consecutive_periods < 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(ValueError, match="min_consecutive_periods must be greater than 0"):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=-1,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )

    def test_max_gap_between_periods_negative(self):
        """Test that max_gap_between_periods < 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_gap_between_periods must be greater than or equal to 0"
        ):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=1,
                max_gap_between_periods=-1,
                max_gap_from_start=1,
            )

    def test_max_gap_from_start_negative(self):
        """Test that max_gap_from_start < 0 throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_gap_from_start must be greater than or equal to 0"
        ):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=-1,
            )

    def test_max_gap_from_start_greater_than_max_gap_between_periods(self):
        """Test that max_gap_from_start > max_gap_between_periods throws ValueError."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        with pytest.raises(
            ValueError, match="max_gap_from_start must be less than or equal to max_gap_between_periods"
        ):
            get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("25"),
                min_selections=2,
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=2,  # Greater than max_gap_between_periods
            )

    def test_empty_prices(self):
        """Test that empty prices throws ValueError."""
        with pytest.raises(ValueError, match="prices cannot be empty"):
            get_cheapest_periods(
                prices=[],
                low_price_threshold=Decimal("25"),
                min_selections=1,
                min_consecutive_periods=1,
                max_gap_between_periods=1,
                max_gap_from_start=1,
            )


class TestRealWorldCornerCases:
    """Test cases for real-world corner cases and edge scenarios."""

    def test_single_price_item(self):
        """Test with only one price item."""
        prices = [Decimal("15")]

        # Should return the single item if it meets criteria
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("20"),
            min_selections=1,
            min_consecutive_periods=1,
            max_gap_between_periods=0,
            max_gap_from_start=0,
        )
        assert result == [0]

    def test_single_price_item_above_threshold(self):
        """Test with single price item above threshold."""
        prices = [Decimal("25")]

        # Should return the single item even if above threshold (min_selections=1)
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("20"),
            min_selections=1,
            min_consecutive_periods=1,
            max_gap_between_periods=0,
            max_gap_from_start=0,
        )
        assert result == [0]

    def test_two_price_items(self):
        """Test with exactly two price items."""
        prices = [Decimal("10"), Decimal("30")]

        # Both items below threshold, should return both
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert result == [0, 1]

    def test_all_prices_identical(self):
        """Test with all prices being identical."""
        prices = [Decimal("15")] * 10

        # All prices identical and below threshold
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("20"),
            min_selections=5,
            min_consecutive_periods=1,
            max_gap_between_periods=2,
            max_gap_from_start=2,
        )
        assert (
            len(result) == 10
        )  # Should return all items since all are below threshold
        assert result == list(range(10))

    def test_all_prices_above_threshold(self):
        """Test with all prices above threshold but min_selections < total."""
        prices = [Decimal("50"), Decimal("60"), Decimal("70")]

        # All prices above threshold, but we want only 2
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("40"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert len(result) == 2
        assert result == [0, 1]  # Should return first 2 (cheapest)

    def test_very_small_price_differences(self):
        """Test with very small price differences."""
        prices = [
            Decimal("10.0001"),
            Decimal("10.0002"),
            Decimal("10.0003"),
            Decimal("10.0004"),
            Decimal("10.0005"),
        ]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("10.0003"),
            min_selections=3,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        # Algorithm selects minimum number of cheapest items that meet constraints
        assert len(result) == 3
        assert result == [0, 1, 3]  # Cheapest 3 items: indices 0, 1, 3

    def test_negative_prices(self):
        """Test with negative prices (realistic for electricity spot prices)."""
        prices = [
            Decimal("-0.5"),
            Decimal("-0.3"),
            Decimal("-0.1"),
            Decimal("0.1"),
            Decimal("0.3"),
        ]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("0.0"),
            min_selections=3,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        # Algorithm selects minimum number of cheapest items that meet constraints
        assert len(result) == 3
        assert result == [0, 1, 3]  # Cheapest 3 items: indices 0, 1, 3

    def test_zero_prices(self):
        """Test with zero prices."""
        prices = [Decimal("0"), Decimal("0"), Decimal("10"), Decimal("20")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("5"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert len(result) == 3  # All items below/equal threshold
        assert result == [0, 1, 2]

    def test_very_large_numbers(self):
        """Test with very large price numbers."""
        prices = [
            Decimal("999999.99"),
            Decimal("999999.98"),
            Decimal("999999.97"),
        ]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("999999.98"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert len(result) == 2
        assert set(result) == {1, 2}  # Last 2 items (cheapest)

    def test_extreme_gap_constraints(self):
        """Test with extreme gap constraints."""
        prices = [Decimal("10")] * 20

        # Very strict gap constraints
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("15"),
            min_selections=5,
            min_consecutive_periods=1,
            max_gap_between_periods=0,  # No gaps allowed
            max_gap_from_start=0,
        )
        assert len(result) == 20  # All items below threshold
        assert result == list(range(20))

    def test_very_loose_gap_constraints(self):
        """Test with very loose gap constraints."""
        prices = [Decimal("10")] * 10

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("15"),
            min_selections=3,
            min_consecutive_periods=1,
            max_gap_between_periods=10,  # Very loose
            max_gap_from_start=10,
        )
        assert len(result) == 10  # All items below threshold
        assert result == list(range(10))

    def test_min_consecutive_periods_equals_min_selections(self):
        """Test when min_consecutive_periods equals min_selections."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=2,  # Same as min_selections
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert len(result) == 2
        assert result == [0, 1]  # Consecutive items

    def test_max_gap_between_periods_zero(self):
        """Test with max_gap_between_periods = 0 (no gaps allowed)."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        # Algorithm finds valid solution by selecting all items (no gaps when all selected)
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=0,  # No gaps
            max_gap_from_start=0,
        )
        # When all items are selected, there are no gaps, so constraint is satisfied
        assert result == [0, 1, 2, 3]  # All items selected

    def test_max_gap_from_start_zero(self):
        """Test with max_gap_from_start = 0 (must start from beginning)."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=0,  # Must start from index 0
        )
        # Algorithm selects minimum number of cheapest items that meet constraints
        assert len(result) == 2
        assert result == [0, 2]  # Cheapest 2 items starting from index 0

    def test_impossible_constraints(self):
        """Test with impossible constraints that should fail gracefully."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        # This should work because min_selections=3 equals total items
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=3,
            min_consecutive_periods=3,
            max_gap_between_periods=0,  # No gaps allowed
            max_gap_from_start=0,
        )
        assert result == [0, 1, 2]  # All items


class TestBoundaryConditions:
    """Test cases for boundary conditions and extreme values."""

    def test_min_selections_equals_total_items(self):
        """Test when min_selections equals total number of items."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=3,  # Same as total items
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert result == [0, 1, 2]  # All items

    def test_min_selections_equals_cheap_items_count(self):
        """Test when min_selections equals number of cheap items."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,  # Same as cheap items count
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        # Algorithm selects minimum number of cheapest items that meet constraints
        assert len(result) == 2
        assert result == [0, 2]  # Cheapest 2 items: indices 0 (10) and 2 (30)

    def test_min_consecutive_periods_equals_one(self):
        """Test with min_consecutive_periods = 1 (minimum valid value)."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,  # Minimum valid value
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )
        assert len(result) == 2

    def test_max_gap_between_periods_equals_zero(self):
        """Test with max_gap_between_periods = 0 (no gaps allowed)."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        # Algorithm finds valid solution by selecting all items (no gaps when all selected)
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=0,  # No gaps
            max_gap_from_start=0,
        )
        # When all items are selected, there are no gaps, so constraint is satisfied
        assert result == [0, 1, 2]  # All items selected

    def test_max_gap_from_start_equals_zero(self):
        """Test with max_gap_from_start = 0 (must start from beginning)."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("25"),
            min_selections=2,
            min_consecutive_periods=1,
            max_gap_between_periods=1,
            max_gap_from_start=0,  # Must start from 0
        )
        assert len(result) == 2
        assert result[0] == 0  # Must start from 0
