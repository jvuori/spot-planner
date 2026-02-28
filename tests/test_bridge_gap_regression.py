"""Regression test for bridge block placement creating new gap violations.

Bug: The large-gap bridge loop placed the cheapest consecutive block anywhere
in the gap. If the cheapest block was near the far end of the gap, it satisfied
the gap from block_end to next_run, but created a NEW violation from prev_idx
to block_start — which was never rechecked.

Fix: Constrain block start so that block_start - prev_idx - 1 <= max_gap_between_periods
(i.e., start <= max_gap_between_periods since gap_start = prev_idx + 1).

Ticket/crash: ValueError: gap at 62->85 = 22 > max_gap_between_periods=20
"""

from decimal import Decimal

import pytest
from spot_planner import get_cheapest_periods


def _max_gap(indices: list[int], total_length: int) -> int:
    """Return the maximum gap (start gap, between-slot gaps, end gap)."""
    if not indices:
        return total_length
    gaps = [indices[0]]  # gap from time-0 to first selection
    for i in range(1, len(indices)):
        gaps.append(indices[i] - indices[i - 1] - 1)
    gaps.append(total_length - 1 - indices[-1])  # gap from last selection to end
    return max(gaps)


def test_bridge_does_not_create_new_gap_violation():
    """The bridge must place the block within max_gap of the previous run.

    Scenario (101 prices):
      - prices[0..31]:  1.0  — cheap, selected by assembly (run A)
      - prices[32..51]: 9.0  — expensive, gap 20 between run A and run B
      - prices[52..62]: 1.0  — cheap, selected by assembly (run B)
      - prices[63..84]: 8.0  — expensive (large gap 63→90 = 27 > max_gap=20)
      - prices[85..88]: 5.0  — cheapest 4-item window in gap, but starts at
                               gap-offset 22 > max_gap=20
      - prices[89]:     8.0  — still expensive
      - prices[90..100]:1.0  — cheap, selected by assembly (run C)

    Without the fix the bridge picks block 85-88 (cheapest), leaves gap
    62→85=22 > max_gap=20 and raises ValueError.

    With the fix the bridge constrains start ≤ max_gap=20, picks block 83-86
    (cheapest within the allowed window), gap 62→83=20 ✓.
    """
    prices = (
        [Decimal("1.0")] * 32   # 0-31: at/below threshold
        + [Decimal("9.0")] * 20  # 32-51: expensive
        + [Decimal("1.0")] * 11  # 52-62: at/below threshold
        + [Decimal("8.0")] * 22  # 63-84: expensive
        + [Decimal("5.0")] * 4   # 85-88: cheapest 4-block, but offset 22 in gap
        + [Decimal("8.0")] * 1   # 89: expensive
        + [Decimal("1.0")] * 11  # 90-100: at/below threshold
    )
    assert len(prices) == 101

    result = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("4.448"),
        min_selections=40,
        min_consecutive_periods=4,
        max_gap_between_periods=20,
        max_gap_from_start=20,
        aggressive=False,
    )

    assert len(result) >= 40, f"Expected >=40 selections, got {len(result)}"

    max_g = _max_gap(result, len(prices))
    assert max_g <= 20, (
        f"Gap constraint violated: max gap = {max_g} > 20. "
        f"Selected indices: {result}"
    )

    # First selection must be within max_gap_from_start=20
    assert result[0] <= 20, (
        f"First selection {result[0]} exceeds max_gap_from_start=20"
    )
