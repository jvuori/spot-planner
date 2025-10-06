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
        min_consecutive_selections=1,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert periods == list(range(24))


def test_whole_day_cheap_with_low_min_selections():
    periods = get_cheapest_periods(
        prices=ALL_CHEAP_PRICES,
        low_price_threshold=Decimal("0.7973127490039840637554282614"),
        min_selections=2,
        min_consecutive_selections=1,
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
        min_consecutive_selections=1,
        max_gap_between_periods=5,
        max_gap_from_start=5,
    )
    assert periods == list(range(24))
