"""Tests for the final pass that drops surplus above-threshold run edges.

Found in production (mlp daemon, 2026-10-07 21:00): rough planning selected whole
4-item groups, so the cheap morning run 0-11 was extended with the rising prices
12-15 and the mid-day bridge was 4 items although min_consecutive_periods was 2.
34 items were returned where 28 satisfy every constraint, shortening the pause
over the expensive hours from 9.5 h to 8.5 h.
"""

from decimal import Decimal

from spot_planner import get_cheapest_periods, two_phase

THRESHOLD = Decimal("0.08")

# fmt: off
PRICES_2026_10_07 = [
    4.565, 3.117, 3.431, 3.657, 4.189, 3.136, 3.757, 4.396, 5.008, 4.044, 4.858,
    5.868, 8.707, 8.142, 10.653, 11.515, 11.733, 12.395, 13.613, 15.312, 15.637,
    16.215, 16.418, 16.567, 16.526, 18.758, 18.425, 18.149, 17.254, 17.878,
    16.624, 16.086, 15.428, 16.107, 14.822, 13.995, 13.291, 13.862, 13.536,
    12.961, 12.682, 12.296, 12.191, 12.122, 12.129, 12.129, 12.123, 12.028,
    11.77, 11.393, 11.367, 11.34, 11.519, 11.231, 11.909, 12.116, 13.692, 12.827,
    12.118, 13.534, 16.128, 14.392, 11.358, 13.0, 12.05, 13.0, 12.094, 12.101,
    10.804, 10.72, 10.027, 9.094, 8.573, 9.23, 8.43, 7.637, 7.022, 7.363, 7.343,
    6.478, 4.999, 4.999, 3.303, 3.0, 2.636, 3.0, 2.72, 2.626, 2.53,
]
# fmt: on


def _d(values: list[str]) -> list[Decimal]:
    return [Decimal(v) for v in values]


def test_surplus_expensive_edge_removed():
    prices = _d(["0.05"] * 4 + ["0.20", "0.30"] + ["0.50"] * 4)
    selected = [0, 1, 2, 3, 4, 5]

    result = two_phase._trim_expensive_run_edges(selected, prices, THRESHOLD, 4, 2, 8, 0)

    assert result == [0, 1, 2, 3]


def test_trim_stops_at_min_selections():
    prices = _d(["0.05"] * 4 + ["0.20", "0.30"] + ["0.50"] * 4)
    selected = [0, 1, 2, 3, 4, 5]

    result = two_phase._trim_expensive_run_edges(selected, prices, THRESHOLD, 5, 2, 8, 0)

    assert result == [0, 1, 2, 3, 4]


def test_cheap_edge_never_removed():
    prices = _d(["0.05"] * 6 + ["0.50"] * 4)
    selected = [0, 1, 2, 3, 4, 5]

    result = two_phase._trim_expensive_run_edges(selected, prices, THRESHOLD, 2, 2, 8, 0)

    assert result == selected


def test_trim_that_breaks_constraints_rejected():
    # Run 6-7 is the bridge that keeps both gaps within max_gap=4, and it is
    # already at min_consecutive_periods=2: nothing may be removed.
    prices = _d(["0.05"] * 2 + ["0.50"] * 4 + ["0.20", "0.20"] + ["0.50"] * 4 + ["0.05"] * 2)
    selected = [0, 1, 6, 7, 12, 13]

    result = two_phase._trim_expensive_run_edges(selected, prices, THRESHOLD, 4, 2, 4, 0)

    assert result == selected


def test_invalid_selection_returned_unchanged():
    prices = _d(["0.20", "0.03", "0.30"])
    selected = [1, 2]  # run shorter than min_consecutive_periods=4

    assert two_phase._trim_expensive_run_edges(selected, prices, THRESHOLD, 1, 4, 2, 2) == selected


def test_production_2026_10_07_no_heating_on_rising_prices():
    prices = [Decimal(str(p)) for p in PRICES_2026_10_07]
    threshold = Decimal("7.637")

    result = get_cheapest_periods(prices, threshold, 26, 2, 47, 47)

    assert two_phase._validate_full_selection(result, len(prices), 2, 47, 47)
    cheap = [i for i, p in enumerate(prices) if p <= threshold]
    assert set(cheap) <= set(result)
    # The 63-item expensive block 12-74 exceeds max_gap=47, so exactly one
    # minimum-length bridge is needed in it and nothing else.
    expensive = [i for i in result if prices[i] > threshold]
    assert len(expensive) == 2
    assert expensive[1] == expensive[0] + 1
    assert len(result) == 28
