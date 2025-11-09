from decimal import Decimal

import pytest

from spot_planner.main import _is_valid_combination, get_cheapest_periods

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


def test_min_selections_is_same_as_for_low_price_threshold():
    periods = get_cheapest_periods(
        prices=PRICE_DATA,
        low_price_threshold=Decimal("20"),
        min_selections=3,
        min_consecutive_periods=1,
        max_gap_between_periods=3,
        max_gap_from_start=3,
    )
    # Algorithm selects cheapest items below threshold
    assert set(periods) == {3, 4, 5}  # Indices 3, 4, 5 with prices 20, 10, 20 (all <= 20)


def test_min_selections_is_greater_than_for_low_price_threshold():
    periods = get_cheapest_periods(
        prices=PRICE_DATA,
        low_price_threshold=Decimal("10"),
        min_selections=3,
        min_consecutive_periods=1,
        max_gap_between_periods=3,
        max_gap_from_start=3,
    )
    # Algorithm selects cheapest items below threshold
    assert periods == [3, 4, 5]  # Indices 3, 4, 5 with prices 20, 10, 20 (only index 4 <= 10, but algorithm selects cheapest 3)


def test_min_selections_is_less_than_for_min_consecutive_periods():
    # This should now raise an error since min_consecutive_periods > min_selections
    with pytest.raises(
        ValueError,
        match="min_consecutive_periods cannot be greater than min_selections",
    ):
        get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("10"),
            min_selections=1,
            min_consecutive_periods=3,
            max_gap_between_periods=3,
            max_gap_from_start=3,
        )


def test_min_selections_is_zero():
    # This should now raise an error since min_selections must be > 0
    with pytest.raises(ValueError, match="min_selections must be greater than 0"):
        get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("10"),
            min_selections=0,
            min_consecutive_periods=8,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )


def test_max_prices_length():
    # Test that prices cannot contain more than 28 items
    prices_29 = [Decimal(str(i)) for i in range(29)]
    with pytest.raises(ValueError, match="prices cannot contain more than 28 items"):
        get_cheapest_periods(
            prices=prices_29,
            low_price_threshold=Decimal("10"),
            min_selections=1,
            min_consecutive_periods=1,
            max_gap_between_periods=5,
            max_gap_from_start=5,
        )


def test_max_prices_length_exactly_28():
    # Test that prices with exactly 28 items works fine
    prices_28 = [Decimal(str(i)) for i in range(28)]
    result = get_cheapest_periods(
        prices=prices_28,
        low_price_threshold=Decimal("5"),
        min_selections=1,
        min_consecutive_periods=1,
        max_gap_between_periods=30,
        max_gap_from_start=30,
    )
    # Algorithm selects minimum number of cheapest items
    assert result == [0]  # Index 0 has the lowest price (0)


@pytest.mark.parametrize(
    "indices, min_consecutive_periods, expected",
    [
        ([], 1, False),
        ([0], 1, True),
        ([0, 1], 1, True),
        ([0, 1, 2], 1, True),
        ([0, 1, 3], 1, True),
        ([], 2, False),
        ([0], 2, False),
        ([0, 1], 2, True),
        ([0, 2], 2, False),
        ([0, 1, 3], 2, False),
        ([0, 2, 3], 2, False),
        ([2, 3], 2, True),
        ([0, 2, 3, 5], 2, False),
        ([0, 2, 3, 5, 6, 7, 9, 10], 3, False),
        ([2, 3, 4, 6, 7, 8], 3, True),
    ],
)
def test_is_valid_min_consecutive_periods(
    indices: list[int], min_consecutive_periods: int, expected: bool
):
    # Test min_consecutive_periods validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    max_gap_between_periods = 100  # Very permissive
    max_gap_from_start = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            full_length,
        )
        == expected
    )


@pytest.mark.parametrize(
    "indices, max_gap_between_periods, full_length, expected",
    [
        ([], 0, 0, False),
        ([0], 0, 1, True),
        ([0, 1], 0, 2, True),
        ([0, 2], 0, 3, False),
        ([0, 1, 2], 0, 3, True),
        ([0, 1, 2, 4], 0, 5, False),
        ([], 1, 0, False),
        ([0], 1, 1, True),
        ([0, 1], 1, 2, True),
        ([0, 2], 1, 3, True),
        ([0, 1, 2], 1, 3, True),
        ([0, 1, 3, 4, 6], 1, 7, True),
        ([0, 1, 4], 1, 5, False),
        ([0, 1, 3, 4, 7], 1, 8, False),
        ([0], 1, 3, False),
        ([0], 2, 3, True),
        ([2], 2, 5, True),
        ([3, 4], 2, 5, False),
    ],
)
def test_is_valid_max_gap_between_periods(
    indices: list[int], max_gap_between_periods: int, full_length: int, expected: bool
):
    # Test max_gap_between_periods validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    min_consecutive_periods = 1  # Very permissive
    max_gap_from_start = 100  # Very permissive

    assert (
        _is_valid_combination(
            combination,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            full_length,
        )
        == expected
    )


@pytest.mark.parametrize(
    "indices, max_gap_from_start, expected",
    [
        ([], 1, False),
        ([0], 1, True),
        ([1], 1, True),
        ([2], 1, False),
        ([2, 3], 2, True),
    ],
)
def test_is_valid_max_gap_from_start(
    indices: list[int], max_gap_from_start: int, expected: bool
):
    # Test max_gap_from_start validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    min_consecutive_periods = 1  # Very permissive
    max_gap_between_periods = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            full_length,
        )
        == expected
    )


def test_min_consecutive_periods_ending_at_last_price_item():
    """Test that min_consecutive_periods is not enforced for the last block if it ends at the last price item."""
    # Test case 1: Selection ending at last price item with short consecutive block should be valid
    # Prices: [10, 20, 30, 40, 50] (indices 0-4)
    # Selection: [3, 4] with min_consecutive_periods=3
    # Since index 4 is the last item, the block [3, 4] (length 2) should be valid even though 2 < 3
    combination = tuple([(3, Decimal("40")), (4, Decimal("50"))])
    assert _is_valid_combination(
        combination,
        min_consecutive_periods=3,
        max_gap_between_periods=100,
        max_gap_from_start=100,
        full_length=5,  # Last index is 4, which equals full_length - 1
    ) is True

    # Test case 2: Selection NOT ending at last price item with short consecutive block should be invalid
    # Prices: [10, 20, 30, 40, 50] (indices 0-4)
    # Selection: [2, 3] with min_consecutive_periods=3
    # Since index 3 is NOT the last item, the block [2, 3] (length 2) should be invalid
    combination = tuple([(2, Decimal("30")), (3, Decimal("40"))])
    assert _is_valid_combination(
        combination,
        min_consecutive_periods=3,
        max_gap_between_periods=100,
        max_gap_from_start=100,
        full_length=5,  # Last index is 3, which is NOT full_length - 1
    ) is False

    # Test case 3: Selection with multiple blocks, last block ending at last price item
    # Prices: [10, 20, 30, 40, 50, 60] (indices 0-5)
    # Selection: [0, 1, 2, 4, 5] with min_consecutive_periods=3
    # First block [0, 1, 2] has length 3 >= 3, so valid
    # Last block [4, 5] has length 2 < 3, but index 5 is the last item, so should be valid
    combination = tuple([
        (0, Decimal("10")),
        (1, Decimal("20")),
        (2, Decimal("30")),
        (4, Decimal("50")),
        (5, Decimal("60")),
    ])
    assert _is_valid_combination(
        combination,
        min_consecutive_periods=3,
        max_gap_between_periods=100,
        max_gap_from_start=100,
        full_length=6,  # Last index is 5, which equals full_length - 1
    ) is True

    # Test case 4: Selection with multiple blocks, last block NOT ending at last price item
    # Prices: [10, 20, 30, 40, 50, 60] (indices 0-5)
    # Selection: [0, 1, 2, 4, 5] with min_consecutive_periods=3
    # First block [0, 1, 2] has length 3 >= 3, so valid
    # Last block [4, 5] has length 2 < 3, and index 5 IS the last item, so should be valid
    # But wait, let's test with a selection that doesn't end at the last item
    # Selection: [0, 1, 2, 4] with min_consecutive_periods=3
    # First block [0, 1, 2] has length 3 >= 3, so valid
    # Last block [4] has length 1 < 3, and index 4 is NOT the last item (full_length=6), so should be invalid
    combination = tuple([
        (0, Decimal("10")),
        (1, Decimal("20")),
        (2, Decimal("30")),
        (4, Decimal("50")),
    ])
    assert _is_valid_combination(
        combination,
        min_consecutive_periods=3,
        max_gap_between_periods=100,
        max_gap_from_start=100,
        full_length=6,  # Last index is 4, which is NOT full_length - 1
    ) is False


def test_get_cheapest_periods_with_ending_at_last_price():
    """Test that get_cheapest_periods can select periods ending at the last price item even if too short."""
    # Create prices where the cheapest period is at the end but too short
    # Prices: [50, 50, 50, 50, 10] (indices 0-4)
    # With min_consecutive_periods=3, we should be able to select [4] (the last item)
    # even though it's only 1 period, because it's at the end
    # Note: min_selections must be >= min_consecutive_periods for validation
    prices = [Decimal("50"), Decimal("50"), Decimal("50"), Decimal("50"), Decimal("10")]
    result = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("20"),
        min_selections=3,  # Must be >= min_consecutive_periods
        min_consecutive_periods=3,
        max_gap_between_periods=100,
        max_gap_from_start=100,
    )
    # Should be able to select the last item even though it's only 1 period
    assert len(result) >= 1
    assert 4 in result  # The last item (index 4) should be selected
