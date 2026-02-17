"""
Regression test for gap constraint violation bug (2026-02-17).

Bug: With 120 price periods where indices 40-116 are expensive, spot_planner
returned a selection with a 62-period gap (>44=max_gap), violating its own
max_gap_between_periods constraint.

Root causes:
1. Chunked rough planning (len(averages)>28) did not bridge cross-chunk gaps.
   Rough groups 9→28 had gap=18 > rough_max_gap=11, so fine-grained chunks
   4-7 (indices 56-112) all got target=0.
2. Fine-grained bridge step added only ONE block per gap. With a 66-period
   gap, one block of 4 reduced it to 62, still violating max_gap=44.

Fix:
1. After chunked rough planning, iteratively bridge gaps > rough_max_gap.
2. Changed single-pass bridge to inner while-loop that keeps adding blocks
   until the gap is within constraints.
"""

from decimal import Decimal

from spot_planner import two_phase


def _validate_selection(
    selected: list[int],
    n: int,
    min_consecutive: int,
    max_gap: int,
    max_gap_from_start: int,
) -> None:
    """Assert that a selection satisfies all constraints."""
    assert selected, "Selection must not be empty"
    indices = sorted(selected)

    # Check start gap
    assert indices[0] <= max_gap_from_start, (
        f"First selection at {indices[0]} exceeds max_gap_from_start={max_gap_from_start}"
    )

    # Check consecutive blocks and inter-block gaps
    block_len = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        assert gap <= max_gap, (
            f"Gap {gap} at {indices[i - 1]}->{indices[i]} > max_gap={max_gap}"
        )
        if indices[i] == indices[i - 1] + 1:
            block_len += 1
        else:
            assert block_len >= min_consecutive, (
                f"Block ending at {indices[i - 1]} has length {block_len} < {min_consecutive}"
            )
            block_len = 1
    # Last block
    assert block_len >= min_consecutive, (
        f"Last block has length {block_len} < {min_consecutive}"
    )


# Real-world price data from 2026-02-17 morning (30 hours / 120 periods).
# Prices c/kWh per 15-minute slot. Cheap region [0-49], expensive [40-116],
# then cheaper again at end [110-119].
PRICES_2026_02_17 = [
    12.842,
    12.497,
    11.312,
    10.21,
    11.292,
    10.195,
    8.721,
    7.324,
    6.66,
    6.829,
    7.245,
    7.569,
    6.765,
    7.776,
    8.739,
    8.534,
    8.678,
    8.178,
    7.686,
    7.549,
    7.927,
    8.172,
    8.045,
    7.91,
    4.846,
    5.312,
    5.734,
    6.248,
    5.41,
    6.149,
    6.809,
    7.335,
    6.467,
    7.28,
    7.477,
    8.4,
    6.972,
    7.452,
    8.265,
    9.046,
    8.198,
    9.148,
    9.437,
    9.582,
    9.672,
    9.692,
    10.098,
    10.762,
    10.065,
    11.146,
    12.931,
    15.058,
    14.831,
    17.306,
    19.214,
    20.599,
    20.638,
    20.963,
    19.368,
    18.039,
    16.009,
    16.307,
    16.941,
    18.975,
    18.479,
    17.971,
    18.797,
    18.032,
    15.009,
    14.269,
    13.534,
    13.245,
    15.21,
    15.0,
    14.281,
    13.508,
    13.792,
    13.611,
    13.582,
    13.615,
    13.605,
    13.612,
    13.526,
    13.88,
    13.025,
    13.49,
    14.301,
    15.0,
    13.492,
    14.327,
    15.0,
    16.423,
    16.804,
    18.384,
    18.913,
    18.944,
    18.563,
    18.201,
    16.424,
    13.907,
    16.382,
    14.281,
    12.795,
    12.392,
    13.494,
    12.603,
    12.097,
    11.557,
    12.605,
    12.0,
    11.975,
    11.148,
    11.47,
    10.893,
    10.621,
    10.503,
    10.43,
    10.353,
    10.263,
    10.101,
]


class TestGapBridge20260217:
    """Tests for the 2026-02-17 gap constraint violation regression."""

    def _prices(self) -> list[Decimal]:
        return [Decimal(str(p)) for p in PRICES_2026_02_17]

    def test_proportional_min_selections_does_not_violate_gap(self) -> None:
        """
        Proportionally scaled min_selections=32 with max_gap=44 must not
        produce a selection that violates max_gap.

        This was the primary failure: spot_planner raised ValueError with
        'gap at 49->112 = 62 > max_gap_between_periods=44'.
        """
        prices = self._prices()
        n = len(prices)
        min_selections = 32
        min_consecutive = 4
        max_gap = 44

        result = two_phase.get_cheapest_periods_extended(
            prices=prices,
            low_price_threshold=Decimal("7.686"),
            min_selections=min_selections,
            min_consecutive_periods=min_consecutive,
            max_gap_between_periods=max_gap,
            max_gap_from_start=max_gap,
            aggressive=False,
        )

        assert len(result) >= min_selections
        _validate_selection(result, n, min_consecutive, max_gap, max_gap)

    def test_full_min_selections_does_not_violate_gap(self) -> None:
        """
        Full (unscaled) min_selections=52 with max_gap=44 must not violate gap.
        """
        prices = self._prices()
        n = len(prices)
        min_selections = 52
        min_consecutive = 4
        max_gap = 44

        result = two_phase.get_cheapest_periods_extended(
            prices=prices,
            low_price_threshold=Decimal("7.686"),
            min_selections=min_selections,
            min_consecutive_periods=min_consecutive,
            max_gap_between_periods=max_gap,
            max_gap_from_start=max_gap,
            aggressive=False,
        )

        assert len(result) >= min_selections
        _validate_selection(result, n, min_consecutive, max_gap, max_gap)

    def test_no_gap_exceeds_max_gap(self) -> None:
        """At 30h of price data, no gap between selected periods should exceed 11h."""
        prices = self._prices()
        n = len(prices)
        min_consecutive = 4
        max_gap = 44  # 11 hours in 15-minute periods

        result = two_phase.get_cheapest_periods_extended(
            prices=prices,
            low_price_threshold=Decimal("7.686"),
            min_selections=32,
            min_consecutive_periods=min_consecutive,
            max_gap_between_periods=max_gap,
            max_gap_from_start=max_gap,
            aggressive=False,
        )

        indices = sorted(result)
        for i in range(1, len(indices)):
            gap = indices[i] - indices[i - 1] - 1
            assert gap <= max_gap, (
                f"Selection violates max_gap: gap={gap} at "
                f"{indices[i - 1]}->{indices[i]} > max_gap={max_gap}"
            )

    def test_all_blocks_meet_min_consecutive(self) -> None:
        """All consecutive blocks in result must have >= min_consecutive periods."""
        prices = self._prices()
        min_consecutive = 4

        result = two_phase.get_cheapest_periods_extended(
            prices=prices,
            low_price_threshold=Decimal("7.686"),
            min_selections=32,
            min_consecutive_periods=min_consecutive,
            max_gap_between_periods=44,
            max_gap_from_start=44,
            aggressive=False,
        )

        indices = sorted(result)
        block_len = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                block_len += 1
            else:
                assert block_len >= min_consecutive, (
                    f"Block ending at {indices[i - 1]} has length {block_len} < {min_consecutive}"
                )
                block_len = 1
        assert block_len >= min_consecutive, (
            f"Last block has length {block_len} < {min_consecutive}"
        )
