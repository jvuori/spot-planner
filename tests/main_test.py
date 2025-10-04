from decimal import Decimal

import pytest

from spot_timer.main import _is_valid_combination, get_cheapest_periods

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


def test_desired_count_is_same_as_for_price_threshold():
    periods = get_cheapest_periods(
        price_data=PRICE_DATA,
        price_threshold=Decimal("20"),
        desired_count=3,
        min_period=1,
        max_gap=3,
        max_start_gap=3,
    )
    assert periods == [3, 4, 5]


def test_desired_count_is_greater_than_for_price_threshold():
    periods = get_cheapest_periods(
        price_data=PRICE_DATA,
        price_threshold=Decimal("10"),
        desired_count=3,
        min_period=1,
        max_gap=3,
        max_start_gap=3,
    )
    assert periods == [3, 4, 5]


def test_desired_count_is_less_than_for_min_period():
    periods = get_cheapest_periods(
        price_data=PRICE_DATA,
        price_threshold=Decimal("10"),
        desired_count=1,
        min_period=3,
        max_gap=3,
        max_start_gap=3,
    )
    assert periods == [3, 4, 5]


def test_desired_count_is_zero():
    periods = get_cheapest_periods(
        price_data=PRICE_DATA,
        price_threshold=Decimal("10"),
        desired_count=0,
        min_period=8,
        max_gap=1,
        max_start_gap=1,
    )
    assert periods == [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize(
    "indices, min_period, expected",
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
def test_is_valid_min_period(indices: list[int], min_period: int, expected: bool):
    # Test min_period validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    max_gap = 100  # Very permissive
    max_start_gap = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination, min_period, max_gap, max_start_gap, full_length
        )
        == expected
    )


@pytest.mark.parametrize(
    "indices, max_gap, full_length, expected",
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
def test_is_valid_max_gap(
    indices: list[int], max_gap: int, full_length: int, expected: bool
):
    # Test max_gap validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    min_period = 1  # Very permissive
    max_start_gap = 100  # Very permissive

    assert (
        _is_valid_combination(
            combination, min_period, max_gap, max_start_gap, full_length
        )
        == expected
    )


@pytest.mark.parametrize(
    "indices, max_start_gap, expected",
    [
        ([], 1, False),
        ([0], 1, True),
        ([1], 1, True),
        ([2], 1, False),
        ([2, 3], 2, True),
    ],
)
def test_is_valid_max_start_gap(indices: list[int], max_start_gap: int, expected: bool):
    # Test max_start_gap validation by setting other constraints to be permissive
    combination = tuple([(index, Decimal("47")) for index in indices])
    min_period = 1  # Very permissive
    max_gap = 100  # Very permissive
    full_length = max(indices) + 10 if indices else 10  # Large enough

    assert (
        _is_valid_combination(
            combination, min_period, max_gap, max_start_gap, full_length
        )
        == expected
    )


def test_performance():
    price_data = [Decimal(f"{i}") for i in range(24)]
    get_cheapest_periods(
        price_data=price_data,
        price_threshold=Decimal("10"),
        desired_count=12,
        min_period=1,
        max_gap=1,
        max_start_gap=1,
    )
