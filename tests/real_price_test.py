from decimal import Decimal

from spot_planner.main import get_cheapest_periods

ALL_CHEAP_PRICES = [
    Decimal("0.021"),
    Decimal("0.04925"),
    Decimal("0.00675"),
    Decimal("-0.00025"),
    Decimal("-0.00225"),
    Decimal("-0.002"),
    Decimal("0.001"),
    Decimal("0.0005"),
    Decimal("0.05525"),
    Decimal("0.09625"),
    Decimal("0.129"),
    Decimal("0.13"),
    Decimal("0.12825"),
    Decimal("0.13975"),
    Decimal("0.2035"),
    Decimal("0.20125"),
    Decimal("0.26925"),
    Decimal("0.3105"),
    Decimal("0.3865"),
    Decimal("0.454"),
    Decimal("0.526"),
    Decimal("0.4945"),
    Decimal("0.4815"),
    Decimal("0.49425"),
]


def test_whole_day_cheap():
    periods = get_cheapest_periods(
        prices=ALL_CHEAP_PRICES,
        low_price_threshold=Decimal("0.7973127490039840637554282614"),
        min_selections=24,
        min_consecutive_periods=1,
            max_consecutive_periods=8,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert periods == list(range(24))


def test_whole_day_cheap_with_low_min_selections():
    periods = get_cheapest_periods(
        prices=ALL_CHEAP_PRICES,
        low_price_threshold=Decimal("0.7973127490039840637554282614"),
        min_selections=2,
        min_consecutive_periods=1,
            max_consecutive_periods=8,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert (
        len(periods) == 24
    )  # Should return all 24 items since all are below threshold
    assert periods == list(range(24))


def test_whole_day_desired():
    periods = get_cheapest_periods(
        prices=ALL_CHEAP_PRICES,
        low_price_threshold=Decimal("-1.0"),
        min_selections=24,
        min_consecutive_periods=1,
            max_consecutive_periods=8,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert periods == list(range(24))


PRICES_2025_10_07 = [
    13.9926225,
    9.67824625,
    9.1320075,
    8.75582125,
    9.15867625,
    9.0880825,
    10.777626249999999,
    12.655733750000001,
    21.66412375,
    26.508423750000002,
    19.3251175,
    14.12282875,
    11.440579999999999,
    10.266841249999999,
    9.579728750000001,
    9.60577,
    9.34818125,
    9.88783125,
    8.21679875,
    5.122596250000001,
    1.11663625,
    0.6513450000000001,
    0.53714,
    0.45682,
]


def test_2025_10_07():
    periods = get_cheapest_periods(
        prices=[Decimal(price) for price in PRICES_2025_10_07],
        low_price_threshold=Decimal("1.651"),
        min_selections=3,
        min_consecutive_periods=1,
            max_consecutive_periods=8,
        max_gap_between_periods=8,
        max_gap_from_start=8,
    )
    assert periods == [5, 14, 20, 21, 22, 23]


PRICES_2025_10_02 = [
    7.7609200000000005,
    11.10329875,
    10.0330975,
    8.25068375,
    6.6885225,
    7.90116625,
    11.782881249999999,
    11.956698750000001,
    14.733072499999999,
    13.031920000000001,
    12.16063625,
    11.82429625,
    10.84351375,
    11.75715375,
    10.060707500000001,
    11.56137375,
    13.01968375,
    18.089256250000002,
    21.096549999999997,
    16.9104975,
    8.4085,
    2.39108875,
    1.9170125,
    1.0046275,
]


def test_2025_10_02():
    periods = get_cheapest_periods(
        prices=[Decimal(price) for price in PRICES_2025_10_02],
        low_price_threshold=Decimal("8.761"),
        min_selections=5,
        min_consecutive_periods=1,
            max_consecutive_periods=8,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert periods == [0, 3, 4, 5, 11, 14, 20, 21, 22, 23]
