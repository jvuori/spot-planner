"""Regression test for the 2026-03-01 customer crash.

Bug: When chunk 0 of the extended algorithm had no rough-plan selections and
was skipped (target=0), subsequent chunks computed their adjusted_max_gap_start
using max_gap_between_periods instead of max_gap_from_start. This allowed the
first selection to be placed beyond max_gap_from_start.

Example: 88 prices, max_gap_from_start=27, max_gap_between_periods=32.
Chunk 0 (indices 0-16) had target=0. Chunk 1 computed:
  adjusted_max_gap_start = max(0, 32 - 17) = 15    # WRONG
  → first selection at index 30 > 27 → ValueError

Fix: when no selections have been made yet (all_selected is empty), use
max_gap_from_start as the base for gap adjustment, not max_gap_between_periods:
  adjusted_max_gap_start = max(0, 27 - 17) = 10    # CORRECT
  → first selection at index 19 ≤ 27 ✓
"""

from decimal import Decimal

from spot_planner import get_cheapest_periods

# Exact prices from the 2026-03-01 production crash.
CRASH_PRICES = [
    Decimal("7.279"),
    Decimal("7.596"),
    Decimal("8.647"),
    Decimal("9.728"),
    Decimal("8.528"),
    Decimal("7.826"),
    Decimal("8.059"),
    Decimal("9.215"),
    Decimal("8.443"),
    Decimal("8.962"),
    Decimal("8.484"),
    Decimal("9.1"),
    Decimal("9.209"),
    Decimal("9.995"),
    Decimal("10.104"),
    Decimal("11.838"),
    Decimal("11.647"),
    Decimal("9.99"),
    Decimal("7.95"),
    Decimal("7.227"),
    Decimal("8.015"),
    Decimal("7.949"),
    Decimal("7.377"),
    Decimal("6.999"),
    Decimal("8.452"),
    Decimal("8.63"),
    Decimal("7.665"),
    Decimal("6.999"),
    Decimal("11.006"),
    Decimal("8.801"),
    Decimal("7.17"),
    Decimal("5.166"),
    Decimal("8.148"),
    Decimal("8.24"),
    Decimal("7.602"),
    Decimal("7.28"),
    Decimal("8.0"),
    Decimal("8.011"),
    Decimal("7.298"),
    Decimal("7.095"),
    Decimal("8.587"),
    Decimal("8.197"),
    Decimal("8.428"),
    Decimal("7.751"),
    Decimal("8.946"),
    Decimal("8.74"),
    Decimal("8.786"),
    Decimal("8.876"),
    Decimal("7.563"),
    Decimal("8.057"),
    Decimal("8.136"),
    Decimal("7.743"),
    Decimal("7.28"),
    Decimal("6.946"),
    Decimal("7.28"),
    Decimal("7.407"),
    Decimal("4.499"),
    Decimal("6.269"),
    Decimal("7.633"),
    Decimal("12.665"),
    Decimal("8.045"),
    Decimal("9.279"),
    Decimal("9.345"),
    Decimal("9.404"),
    Decimal("9.447"),
    Decimal("9.473"),
    Decimal("9.52"),
    Decimal("9.528"),
    Decimal("9.587"),
    Decimal("8.605"),
    Decimal("8.916"),
    Decimal("8.118"),
    Decimal("8.15"),
    Decimal("7.755"),
    Decimal("7.252"),
    Decimal("6.439"),
    Decimal("8.137"),
    Decimal("7.726"),
    Decimal("7.336"),
    Decimal("6.65"),
    Decimal("7.293"),
    Decimal("6.617"),
    Decimal("6.036"),
    Decimal("5.591"),
    Decimal("6.762"),
    Decimal("6.211"),
    Decimal("6.004"),
    Decimal("5.407"),
]


def _validate_constraints(
    indices: list[int],
    total_length: int,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[str]:
    """Return list of constraint violation messages (empty = all OK)."""
    violations: list[str] = []
    if not indices:
        return ["No selections"]
    if len(indices) < min_selections:
        violations.append(
            f"selections {len(indices)} < min_selections {min_selections}"
        )
    if indices[0] > max_gap_from_start:
        violations.append(
            f"first_selection {indices[0]} > max_gap_from_start {max_gap_from_start}"
        )
    end_gap = total_length - 1 - indices[-1]
    if end_gap > max_gap_between_periods:
        violations.append(
            f"end gap {end_gap} > max_gap_between_periods {max_gap_between_periods}"
        )
    run_start = indices[0]
    run_len = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            violations.append(
                f"gap {indices[i-1]}->{indices[i]} = {gap} > "
                f"max_gap_between_periods {max_gap_between_periods}"
            )
        if indices[i] == indices[i - 1] + 1:
            run_len += 1
        else:
            if run_len < min_consecutive_periods:
                violations.append(
                    f"run [{run_start}-{run_start + run_len - 1}] "
                    f"len={run_len} < min_consecutive_periods {min_consecutive_periods}"
                )
            run_start = indices[i]
            run_len = 1
    if run_len < min_consecutive_periods:
        violations.append(
            f"run [{run_start}-{run_start + run_len - 1}] "
            f"len={run_len} < min_consecutive_periods {min_consecutive_periods}"
        )
    return violations


def test_crash_2026_03_01():
    """Exact production parameters from the crash.

    88 prices, max_gap_from_start=27. Chunk 0 (indices 0-16) is all expensive
    (above threshold 7.563). The algorithm must still place the first selection
    within index 27, using cheap items at indices 19, 22, 23, 27.
    """
    result = get_cheapest_periods(
        prices=CRASH_PRICES,
        low_price_threshold=Decimal("7.563"),
        min_selections=29,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=27,
    )

    violations = _validate_constraints(
        result,
        total_length=len(CRASH_PRICES),
        min_selections=29,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=27,
    )
    assert not violations, (
        f"Constraint violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )


def test_crash_2026_03_01_first_selection_within_gap():
    """Specifically verify first selection respects max_gap_from_start.

    This is the exact constraint that was violated before the fix.
    """
    result = get_cheapest_periods(
        prices=CRASH_PRICES,
        low_price_threshold=Decimal("7.563"),
        min_selections=29,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=27,
    )

    assert result[0] <= 27, (
        f"First selection at index {result[0]} exceeds max_gap_from_start=27. "
        f"Cheap items exist at indices 19, 22, 23, 27 (all ≤ threshold 7.563)."
    )
