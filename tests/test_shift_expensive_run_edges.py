"""Tests for the final pass that swaps an expensive run edge for a cheaper neighbour.

Found in the cheap_day visualization: chunk-local planning padded the cheap run
30-32 with the 0.230 spike at index 33 (the chunk's last item) although index 29
(0.183) on the run's other side was cheaper and equally valid.
"""

from decimal import Decimal

from spot_planner import get_cheapest_periods, two_phase

THRESHOLD = Decimal("0.08")


def _d(values: list[str]) -> list[Decimal]:
    return [Decimal(v) for v in values]


def test_expensive_edge_replaced_by_cheaper_neighbour():
    prices = _d(["0.05", "0.05", "0.05", "0.05", "0.18", "0.03", "0.03", "0.03", "0.23"])
    selected = [0, 1, 2, 3, 5, 6, 7, 8]

    result = two_phase._shift_expensive_run_edges(selected, prices, THRESHOLD, 4, 8, 0)

    assert result == [0, 1, 2, 3, 4, 5, 6, 7]


def test_cheap_edge_never_removed():
    # Shifting would drop the below-threshold item 8 for the cheaper item 4,
    # losing a cheap item: must not happen.
    prices = _d(["0.05", "0.05", "0.05", "0.05", "0.01", "0.03", "0.03", "0.03", "0.07"])
    selected = [0, 1, 2, 3, 5, 6, 7, 8]

    result = two_phase._shift_expensive_run_edges(selected, prices, THRESHOLD, 4, 8, 0)

    assert result == selected


def test_shift_that_breaks_constraints_rejected():
    # Shifting run 8-11 right (drop 0.20, add 0.10) would widen the gap after
    # index 3 from 4 to 5 > max_gap=4, so the selection must stay as is.
    prices = _d(["0.05"] * 4 + ["0.30"] * 4 + ["0.20", "0.03", "0.03", "0.03", "0.10"])
    selected = [0, 1, 2, 3, 8, 9, 10, 11]

    result = two_phase._shift_expensive_run_edges(selected, prices, THRESHOLD, 4, 4, 0)

    assert result == selected


def test_invalid_selection_returned_unchanged():
    prices = _d(["0.20", "0.03", "0.30"])
    selected = [1, 2]  # run shorter than min_consecutive_periods=4

    assert two_phase._shift_expensive_run_edges(selected, prices, THRESHOLD, 4, 2, 2) == selected


def test_long_sequence_uses_cheaper_filler():
    # 40 items -> extended algorithm. Cheap run 20-22 needs a 4th item: 19 (0.18)
    # is cheaper than 23 (0.23), so 23 must not be selected.
    prices = _d(["0.05"] * 8 + ["0.20"] * 11 + ["0.18", "0.03", "0.03", "0.03", "0.23"] + ["0.20"] * 8 + ["0.05"] * 8)

    result = get_cheapest_periods(prices, THRESHOLD, 16, 4, 12, 0)

    assert 23 not in result
    assert two_phase._validate_full_selection(result, len(prices), 4, 12, 0)
