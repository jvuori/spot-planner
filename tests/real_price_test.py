from decimal import Decimal

from spot_planner.main import get_cheapest_periods


def test_whole_day_cheap():
    periods = get_cheapest_periods(
        price_data=[
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
        ],
        price_threshold=Decimal("0.7973127490039840637554282614"),
        desired_count=24,
        min_period=1,
        max_gap=5,
        max_start_gap=5,
    )
    assert periods == list(range(24))
