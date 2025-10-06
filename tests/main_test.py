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
        min_consecutive_selections=1,
        max_gap_between_periods=3,
        max_gap_from_start=3,
    )
    assert set(periods) == {3, 4, 5}  # Should be these 3 items, order may vary


def test_min_selections_is_greater_than_for_low_price_threshold():
    periods = get_cheapest_periods(
        prices=PRICE_DATA,
        low_price_threshold=Decimal("10"),
        min_selections=3,
        min_consecutive_selections=1,
        max_gap_between_periods=3,
        max_gap_from_start=3,
    )
    assert periods == [3, 4, 5]


def test_min_selections_is_less_than_for_min_consecutive_selections():
    # This should now raise an error since min_consecutive_selections > min_selections
    with pytest.raises(ValueError, match="min_consecutive_selections cannot be greater than min_selections"):
        get_cheapest_periods(
            prices=PRICE_DATA,
            low_price_threshold=Decimal("10"),
            min_selections=1,
            min_consecutive_selections=3,
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
            min_consecutive_selections=8,
            max_gap_between_periods=1,
            max_gap_from_start=1,
        )


@pytest.mark.parametrize(
    "indices, min_consecutive_selections, expected",
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
def test_is_valid_min_consecutive_selections(indices: list[int], min_consecutive_selections: int, expected: bool):
    # Test min_consecutive_selections validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    max_gap_between_periods = 100  # Very permissive
    max_gap_from_start = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination, min_consecutive_selections, max_gap_between_periods, max_gap_from_start, full_length
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
    min_consecutive_selections = 1  # Very permissive
    max_gap_from_start = 100  # Very permissive

    assert (
        _is_valid_combination(
            combination, min_consecutive_selections, max_gap_between_periods, max_gap_from_start, full_length
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
def test_is_valid_max_gap_from_start(indices: list[int], max_gap_from_start: int, expected: bool):
    # Test max_gap_from_start validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    min_consecutive_selections = 1  # Very permissive
    max_gap_between_periods = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination, min_consecutive_selections, max_gap_between_periods, max_gap_from_start, full_length
        )
        == expected
    )


def test_performance():
    prices = [Decimal(f"{i}") for i in range(24)]
    get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("10"),
        min_selections=12,
        min_consecutive_selections=1,
        max_gap_between_periods=1,
        max_gap_from_start=1,
    )
