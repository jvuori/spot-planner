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
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    # Algorithm selects cheapest items that meet constraints
    assert periods == [4, 10, 15, 21, 22, 23]  # Cheapest 6 items below threshold


PRICES_2026_02_15_HANG_TEST = [
    5.153, 4.999, 5.0, 4.702, 4.151, 3.499, 7.894, 7.279, 6.365, 5.77,
    5.771, 5.511, 5.0, 4.999, 4.999, 4.5, 4.295, 4.029, 4.04, 4.101,
    4.114, 4.3, 4.516, 4.999, 4.999, 5.0, 5.0, 5.746, 6.12, 6.889,
    7.656, 8.471, 10.477, 9.962, 7.271, 9.37, 10.168, 10.3, 9.103, 9.29,
    9.259, 9.587, 9.053, 9.251, 9.444, 9.255, 10.216, 9.161, 8.773, 14.999,
    8.762, 11.004, 11.829, 11.999, 14.999, 13.193, 8.898, 6.911, 12.999, 8.852,
    8.135, 7.413, 8.239, 7.322, 6.999, 7.616, 6.787, 7.276, 7.608, 8.327,
    8.181, 8.212, 8.355, 8.425, 9.527, 10.501, 9.899, 9.334, 9.008, 8.361,
    7.712, 7.225, 7.552, 6.295, 5.033, 4.3, 3.5, 3.219, 3.164, 2.999,
    3.219, 3.187, 3.029, 2.977, 2.947, 2.796, 2.644, 2.471, 2.688, 2.524,
    2.392, 2.313,
]


def test_2026_02_15_hang_test():
    """
    Test case from 2026-02-15 that potentially causes hung planning.
    
    Client reported: spot_planner never completes and consumes 100% CPU.
    
    Input parameters:
    - 98 price points
    - low_price_threshold=6.295
    - min_selections=45 (need to select 45 periods)
    - min_consecutive_periods=4 (periods must be in chunks of at least 4)
    - max_gap_between_periods=36 (max gap between selections)
    - max_gap_from_start=36 (max gap from start)
    """
    periods = get_cheapest_periods(
        prices=[Decimal(price) for price in PRICES_2026_02_15_HANG_TEST],
        low_price_threshold=Decimal("6.295"),
        min_selections=45,
        min_consecutive_periods=4,
        max_gap_between_periods=36,
        max_gap_from_start=36,
    )
    # Test that the algorithm completes without hanging
    assert isinstance(periods, list)
    assert len(periods) >= 45  # Should have at least min_selections
