"""Test cases for spot_planner library behavior.

These tests verify the external spot_planner library correctly handles
the min_consecutive_selections parameter.
"""

from decimal import Decimal

import spot_planner


class TestSpotPlannerMinConsecutive:
    """Test min_consecutive_selections enforcement in spot_planner."""

    def test_min_consecutive_selections_bug(self):
        """Test that spot_planner respects min_consecutive_selections parameter.

        This test reproduces a bug where spot_planner returns isolated single
        selections even when min_consecutive_selections=4.

        Bug: Index 8 is returned as an isolated selection (run length of 1),
        which violates min_consecutive_selections=4.

        Expected behavior: All returned indices should be part of consecutive
        runs of at least min_consecutive_selections length.
        """
        # Real price data that exposes the bug
        prices = [
            Decimal("5.960079948696822"),  # 0  ─┐
            Decimal("5.642847881739181"),  # 1   │
            Decimal("5.528618141464342"),  # 2   │
            Decimal("7.361589786044312"),  # 3   ├─ Run of 7 ✓
            Decimal("7.325805656981777"),  # 4   │
            Decimal("7.283237132689745"),  # 5   │
            Decimal("7.832182285751291"),  # 6  ─┘
            Decimal("8.508621453707589"),  # 7    (gap)
            Decimal("8.048035862254752"),  # 8  ─  Run of 1 ✗ BUG!
            Decimal("8.327459705541802"),  # 9    (gap)
            Decimal("8.114418158869745"),  # 10   (gap)
            Decimal("8.22749259589503"),  # 11 ─┐
            Decimal("7.867492963544045"),  # 12  │
            Decimal("7.714255331533462"),  # 13  │
            Decimal("7.699835731078247"),  # 14  │
            Decimal("6.841730331230942"),  # 15  │
            Decimal("6.638664293795596"),  # 16  ├─ Run of 13 ✓
            Decimal("6.857518473352007"),  # 17  │
            Decimal("7.070519560849684"),  # 18  │
            Decimal("5.4446702778272185"),  # 19  │
            Decimal("5.345626417536506"),  # 20  │
            Decimal("5.773645683572311"),  # 21  │
            Decimal("5.505239461522056"),  # 22  │
            Decimal("4.315909235478207"),  # 23 ─┘
        ]

        low_price_threshold = Decimal("8.080049881693729063745019920")
        min_selections = 4
        min_consecutive_selections = 4
        max_gap_between_periods = 22
        max_gap_from_start = 7

        indices = spot_planner.get_cheapest_periods(
            prices=prices,
            low_price_threshold=low_price_threshold,
            min_selections=min_selections,
            min_consecutive_selections=min_consecutive_selections,
            max_gap_between_periods=max_gap_between_periods,
            max_gap_from_start=max_gap_from_start,
        )

        # Analyze consecutive runs
        runs = []
        if indices:
            current_run = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:
                    current_run.append(indices[i])
                else:
                    runs.append(current_run)
                    current_run = [indices[i]]
            runs.append(current_run)

        # Verify all runs meet min_consecutive_selections
        for run in runs:
            run_length = len(run)
            assert run_length >= min_consecutive_selections, (
                f"Run [{run[0]}..{run[-1]}] has length {run_length} "
                f"which is less than min_consecutive_selections={min_consecutive_selections}. "
                f"Full indices: {indices}"
            )

        # Expected: index 8 should NOT be in the result
        # because it forms a run of only 1 selection
        assert 8 not in indices, (
            f"Index 8 should not be included as it forms a run shorter than "
            f"min_consecutive_selections={min_consecutive_selections}. "
            f"Actual indices: {indices}"
        )

    def test_min_consecutive_selections_simple_case(self):
        """Test min_consecutive_selections with a simple case."""
        # Simple case: 10 prices, all cheap
        prices = [Decimal(str(i)) for i in range(10)]
        low_price_threshold = Decimal("100.0")  # All prices are below this
        min_selections = 4
        min_consecutive_selections = 4
        max_gap_between_periods = 10
        max_gap_from_start = 10

        indices = spot_planner.get_cheapest_periods(
            prices=prices,
            low_price_threshold=low_price_threshold,
            min_selections=min_selections,
            min_consecutive_selections=min_consecutive_selections,
            max_gap_between_periods=max_gap_between_periods,
            max_gap_from_start=max_gap_from_start,
        )

        # Should get at least min_selections
        assert len(indices) >= min_selections

        # Verify all runs meet min_consecutive_selections
        if indices:
            runs = []
            current_run = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:
                    current_run.append(indices[i])
                else:
                    runs.append(current_run)
                    current_run = [indices[i]]
            runs.append(current_run)

            for run in runs:
                assert len(run) >= min_consecutive_selections, (
                    f"Run [{run[0]}..{run[-1]}] has length {len(run)} "
                    f"which is less than min_consecutive_selections={min_consecutive_selections}"
                )
