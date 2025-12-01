"""
Tests for the extended algorithm that handles longer price sequences (> 28 items).

The extended algorithm uses a two-phase approach:
1. Rough planning with averaged groups
2. Fine-grained planning with boundary-aware constraint handling
"""

from decimal import Decimal

import pytest

from spot_planner import get_cheapest_periods
from spot_planner.main import (
    _ChunkBoundaryState,
    _calculate_chunk_boundary_state,
    _get_cheapest_periods_extended,
    _repair_selection,
    _validate_full_selection,
)


class TestExtendedAlgorithmBasics:
    """Test basic functionality of the extended algorithm."""

    def test_handles_96_items(self):
        """Test that 96 items can be processed without error."""
        # 96 hourly prices (4 days)
        prices = [Decimal(str(i % 24 + 1)) for i in range(96)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=24,
            min_consecutive_periods=2,
            max_gap_between_periods=4,
            max_gap_from_start=4,
        )

        assert len(result) >= 24
        assert all(0 <= idx < 96 for idx in result)
        assert result == sorted(result)  # Should be sorted

    def test_handles_48_items(self):
        """Test that 48 items (2 days) can be processed."""
        prices = [Decimal(str((i % 12) + 1)) for i in range(48)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("6"),
            min_selections=12,
            min_consecutive_periods=2,
            max_gap_between_periods=3,
            max_gap_from_start=3,
        )

        assert len(result) >= 12
        assert all(0 <= idx < 48 for idx in result)

    def test_prefers_cheap_items(self):
        """Test that the algorithm prefers cheap items in longer sequences."""
        # Create a pattern where some items are clearly cheaper
        prices = []
        for i in range(96):
            if i % 4 == 0:  # Every 4th item is cheap
                prices.append(Decimal("1"))
            else:
                prices.append(Decimal("10"))

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("2"),
            min_selections=20,
            min_consecutive_periods=1,
            max_gap_between_periods=10,
            max_gap_from_start=10,
        )

        # Count how many selected items are cheap (price = 1)
        cheap_selected = sum(1 for idx in result if prices[idx] == Decimal("1"))

        # Should prefer cheap items
        assert cheap_selected > len(result) // 2

    def test_respects_min_selections(self):
        """Test that min_selections is respected for long sequences."""
        prices = [Decimal(str(i + 1)) for i in range(60)]

        for min_sel in [10, 20, 30]:
            result = get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("30"),
                min_selections=min_sel,
                min_consecutive_periods=2,
                max_gap_between_periods=5,
                max_gap_from_start=5,
            )

            assert len(result) >= min_sel


class TestConstraintValidation:
    """Test that constraints are validated correctly across the full sequence."""

    def test_validate_full_selection_valid(self):
        """Test validation of a valid selection."""
        # Valid selection: [0,1,2, 5,6,7, 10,11] for total_length=12
        selected = [0, 1, 2, 5, 6, 7, 10, 11]

        assert _validate_full_selection(
            selected_indices=selected,
            total_length=12,
            min_consecutive_periods=2,
            max_gap_between_periods=3,
            max_gap_from_start=0,
        )

    def test_validate_full_selection_gap_too_large(self):
        """Test validation fails when gap is too large."""
        selected = [0, 1, 10, 11]  # Gap of 8 between 1 and 10

        assert not _validate_full_selection(
            selected_indices=selected,
            total_length=12,
            min_consecutive_periods=2,
            max_gap_between_periods=3,  # Max gap is 3, but we have 8
            max_gap_from_start=0,
        )

    def test_validate_full_selection_consecutive_too_short(self):
        """Test validation fails when consecutive block is too short."""
        selected = [0, 1, 5, 6, 7]  # First block is 2, need 3

        assert not _validate_full_selection(
            selected_indices=selected,
            total_length=12,
            min_consecutive_periods=3,
            max_gap_between_periods=5,
            max_gap_from_start=0,
        )

    def test_validate_full_selection_start_gap_too_large(self):
        """Test validation fails when gap from start is too large."""
        selected = [5, 6, 7, 8]  # Starts at 5, max_gap_from_start is 2

        assert not _validate_full_selection(
            selected_indices=selected,
            total_length=12,
            min_consecutive_periods=2,
            max_gap_between_periods=5,
            max_gap_from_start=2,
        )


class TestBoundaryStateCalculation:
    """Test boundary state calculation between chunks."""

    def test_boundary_state_ended_with_selected(self):
        """Test state when chunk ends with selected items."""
        chunk_selected = [0, 1, 2, 7, 8, 9]  # Ends at 9 (last item of 10-item chunk)
        chunk_length = 10

        state = _calculate_chunk_boundary_state(chunk_selected, chunk_length)

        assert state.ended_with_selected is True
        assert state.trailing_selected_count == 3  # 7, 8, 9 are consecutive at end
        assert state.trailing_unselected_count == 0

    def test_boundary_state_ended_with_unselected(self):
        """Test state when chunk ends with unselected items."""
        chunk_selected = [0, 1, 2, 5, 6, 7]  # Last selected is 7, chunk has 10 items
        chunk_length = 10

        state = _calculate_chunk_boundary_state(chunk_selected, chunk_length)

        assert state.ended_with_selected is False
        assert state.trailing_selected_count == 0
        assert state.trailing_unselected_count == 2  # Items 8, 9 are unselected

    def test_boundary_state_empty_selection(self):
        """Test state when no items are selected."""
        chunk_selected = []
        chunk_length = 10

        state = _calculate_chunk_boundary_state(chunk_selected, chunk_length)

        assert state.ended_with_selected is False
        assert state.trailing_selected_count == 0
        assert state.trailing_unselected_count == 10


class TestConstraintsAcrossBoundaries:
    """Test that constraints are properly maintained across chunk boundaries."""

    def test_max_gap_between_periods_across_chunks(self):
        """Test that max_gap_between_periods is respected across chunk boundaries."""
        # Create prices where cheap items are spread out
        prices = [Decimal("10")] * 96  # All expensive
        # Add cheap items with specific pattern
        for i in [5, 6, 7, 25, 26, 27, 45, 46, 47, 65, 66, 67, 85, 86, 87]:
            prices[i] = Decimal("1")

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("2"),
            min_selections=12,
            min_consecutive_periods=3,
            max_gap_between_periods=20,
            max_gap_from_start=10,
        )

        # Verify no gap exceeds max_gap_between_periods
        sorted_result = sorted(result)
        for i in range(1, len(sorted_result)):
            gap = sorted_result[i] - sorted_result[i - 1] - 1
            assert gap <= 20, f"Gap of {gap} exceeds max of 20 between {sorted_result[i-1]} and {sorted_result[i]}"

    def test_min_consecutive_periods_across_chunks(self):
        """Test that min_consecutive_periods blocks are valid across entire sequence."""
        prices = [Decimal(str(i % 10 + 1)) for i in range(96)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("5"),
            min_selections=24,
            min_consecutive_periods=4,
            max_gap_between_periods=8,
            max_gap_from_start=8,
        )

        # Verify all consecutive blocks have at least min_consecutive_periods
        sorted_result = sorted(result)
        block_length = 1
        for i in range(1, len(sorted_result)):
            if sorted_result[i] == sorted_result[i - 1] + 1:
                block_length += 1
            else:
                # End of block
                assert block_length >= 4, f"Block of length {block_length} is less than min 4"
                block_length = 1

        # Check final block (only if not at the end of sequence)
        if sorted_result[-1] < 95:  # Not at the very end
            assert block_length >= 4, f"Final block of length {block_length} is less than min 4"


class TestRepairSelection:
    """Test the repair function that fixes invalid selections."""

    def test_repair_adds_missing_items_for_gap(self):
        """Test that repair adds items to fix gaps that are too large."""
        prices = [Decimal(str(i + 1)) for i in range(20)]
        selected = [0, 1, 2, 15, 16, 17]  # Gap of 12 between 2 and 15

        repaired = _repair_selection(
            selected=selected,
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=6,
            min_consecutive_periods=3,
            max_gap_between_periods=5,
            max_gap_from_start=0,
        )

        # Verify gap is now within limits
        sorted_repaired = sorted(repaired)
        for i in range(1, len(sorted_repaired)):
            gap = sorted_repaired[i] - sorted_repaired[i - 1] - 1
            assert gap <= 5

    def test_repair_extends_short_blocks(self):
        """Test that repair extends blocks that are too short."""
        prices = [Decimal(str(i + 1)) for i in range(20)]
        selected = [0, 1, 5, 6]  # First block has 2, need 3

        repaired = _repair_selection(
            selected=selected,
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=4,
            min_consecutive_periods=3,
            max_gap_between_periods=5,
            max_gap_from_start=0,
        )

        # Verify blocks are now long enough
        sorted_repaired = sorted(repaired)
        block_length = 1
        for i in range(1, len(sorted_repaired)):
            if sorted_repaired[i] == sorted_repaired[i - 1] + 1:
                block_length += 1
            else:
                assert block_length >= 3, f"Block of {block_length} still too short"
                block_length = 1


class TestRealWorldScenarios:
    """Test with realistic price patterns."""

    def test_typical_electricity_day_pattern(self):
        """Test with a typical daily electricity price pattern repeated over 4 days."""
        # Typical pattern: cheap at night, expensive during day
        daily_pattern = [
            Decimal("0.03"),  # 00:00 - cheap
            Decimal("0.03"),  # 01:00
            Decimal("0.03"),  # 02:00
            Decimal("0.03"),  # 03:00
            Decimal("0.04"),  # 04:00
            Decimal("0.05"),  # 05:00
            Decimal("0.08"),  # 06:00 - morning peak starts
            Decimal("0.12"),  # 07:00
            Decimal("0.15"),  # 08:00 - morning peak
            Decimal("0.12"),  # 09:00
            Decimal("0.10"),  # 10:00
            Decimal("0.08"),  # 11:00
            Decimal("0.08"),  # 12:00 - midday
            Decimal("0.09"),  # 13:00
            Decimal("0.10"),  # 14:00
            Decimal("0.12"),  # 15:00
            Decimal("0.15"),  # 16:00 - evening peak starts
            Decimal("0.18"),  # 17:00
            Decimal("0.20"),  # 18:00 - evening peak
            Decimal("0.18"),  # 19:00
            Decimal("0.12"),  # 20:00
            Decimal("0.08"),  # 21:00
            Decimal("0.05"),  # 22:00
            Decimal("0.04"),  # 23:00
        ]

        # 4 days = 96 hours
        prices = daily_pattern * 4

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("0.06"),
            min_selections=24,  # 6 hours per day of heating
            min_consecutive_periods=2,
            max_gap_between_periods=8,
            max_gap_from_start=4,
        )

        assert len(result) >= 24

        # Verify result is valid
        assert _validate_full_selection(
            result, 96, 2, 8, 4
        ), "Result should be valid"

        # Most selections should be during cheap hours (0-5, 21-23)
        cheap_hours = set()
        for day in range(4):
            for hour in [0, 1, 2, 3, 4, 5, 21, 22, 23]:
                cheap_hours.add(day * 24 + hour)

        cheap_selections = sum(1 for idx in result if idx in cheap_hours)
        assert cheap_selections > len(result) // 2, "Should prefer cheap hours"

    def test_variable_length_sequences(self):
        """Test various sequence lengths around the 28-item boundary."""
        for length in [30, 40, 50, 60, 72, 96]:
            prices = [Decimal(str((i % 10) + 1)) for i in range(length)]

            result = get_cheapest_periods(
                prices=prices,
                low_price_threshold=Decimal("5"),
                min_selections=length // 4,
                min_consecutive_periods=2,
                max_gap_between_periods=5,
                max_gap_from_start=3,
            )

            assert len(result) >= length // 4
            assert _validate_full_selection(
                result, length, 2, 5, 3
            ), f"Result for length {length} should be valid"


class TestEdgeCases:
    """Test edge cases for the extended algorithm."""

    def test_all_prices_below_threshold(self):
        """Test when all prices are below threshold in long sequence."""
        prices = [Decimal("1")] * 50

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=20,
            min_consecutive_periods=2,
            max_gap_between_periods=5,
            max_gap_from_start=2,
        )

        assert len(result) >= 20
        assert _validate_full_selection(result, 50, 2, 5, 2)

    def test_all_prices_above_threshold(self):
        """Test when all prices are above threshold in long sequence."""
        prices = [Decimal("100")] * 50

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=20,
            min_consecutive_periods=2,
            max_gap_between_periods=5,
            max_gap_from_start=2,
        )

        assert len(result) >= 20
        assert _validate_full_selection(result, 50, 2, 5, 2)

    def test_exactly_28_items_uses_direct_algorithm(self):
        """Test that exactly 28 items uses the direct brute-force algorithm."""
        prices = [Decimal(str(i + 1)) for i in range(28)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("14"),
            min_selections=10,
            min_consecutive_periods=2,
            max_gap_between_periods=3,
            max_gap_from_start=2,
        )

        assert len(result) >= 10

    def test_29_items_uses_extended_algorithm(self):
        """Test that 29 items uses the extended algorithm."""
        prices = [Decimal(str(i + 1)) for i in range(29)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("15"),
            min_selections=10,
            min_consecutive_periods=2,
            max_gap_between_periods=3,
            max_gap_from_start=2,
        )

        assert len(result) >= 10
        assert _validate_full_selection(result, 29, 2, 3, 2)

    def test_tight_constraints_long_sequence(self):
        """Test with tight constraints that might be hard to satisfy."""
        prices = [Decimal(str((i % 20) + 1)) for i in range(80)]

        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("10"),
            min_selections=30,
            min_consecutive_periods=3,
            max_gap_between_periods=4,
            max_gap_from_start=2,
        )

        assert len(result) >= 30
        assert _validate_full_selection(result, 80, 3, 4, 2)


class TestAggressiveVsConservative:
    """Test aggressive vs conservative mode for extended algorithm."""

    def test_aggressive_mode_minimizes_average_cost(self):
        """Test that aggressive mode tends to minimize average cost."""
        prices = [Decimal(str(i % 10 + 1)) for i in range(50)]

        result_aggressive = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("5"),
            min_selections=15,
            min_consecutive_periods=2,
            max_gap_between_periods=5,
            max_gap_from_start=3,
            aggressive=True,
        )

        assert len(result_aggressive) >= 15
        assert _validate_full_selection(result_aggressive, 50, 2, 5, 3)

    def test_conservative_mode_prefers_cheap_items(self):
        """Test that conservative mode prefers more cheap items."""
        prices = [Decimal(str(i % 10 + 1)) for i in range(50)]

        result_conservative = get_cheapest_periods(
            prices=prices,
            low_price_threshold=Decimal("5"),
            min_selections=15,
            min_consecutive_periods=2,
            max_gap_between_periods=5,
            max_gap_from_start=3,
            aggressive=False,
        )

        assert len(result_conservative) >= 15
        assert _validate_full_selection(result_conservative, 50, 2, 5, 3)

