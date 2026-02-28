"""Regression test for the 2026-02-27 customer crash.

Customer experienced a crash loop (500+ restarts) caused by a spot_planner bug
where the gap-filling algorithm failed when:
  - The price distribution has a long expensive stretch (> max_gap periods)
  - min_selections is low enough that the algorithm doesn't naturally select
    enough periods in the expensive region
  - The bridge block placement picked the cheapest block too far from the
    previous run, creating a new gap violation

Root cause: positions 34-89 (56 periods) are all expensive (above threshold
6.985), creating a gap that exceeds max_gap=32. The bridge algorithm needed
to insert blocks within this stretch but previously could place them anywhere,
potentially creating new gap violations.

Fixed by constraining bridge block start position to be within max_gap of
the previous run end (commit: "Fix bridge block placement causing gap
violations").
"""

from decimal import Decimal

from spot_planner import get_cheapest_periods

# Exact prices from the 2026-02-27 production crash.
# 103 prices from 4.204 to 12.078 c/kWh, with a 56-period expensive stretch
# at positions 34-89 that triggers the spot_planner bug.
CRASH_PRICES = [
    Decimal("7.115"),
    Decimal("5.923"),
    Decimal("5.512"),
    Decimal("6.028"),
    Decimal("5.746"),
    Decimal("5.472"),
    Decimal("5.227"),
    Decimal("7.95"),
    Decimal("7.756"),
    Decimal("7.636"),
    Decimal("7.411"),
    Decimal("7.281"),
    Decimal("7.337"),
    Decimal("6.955"),
    Decimal("6.641"),
    Decimal("7.548"),
    Decimal("6.989"),
    Decimal("6.767"),
    Decimal("6.573"),
    Decimal("6.694"),
    Decimal("6.572"),
    Decimal("6.492"),
    Decimal("6.519"),
    Decimal("6.687"),
    Decimal("6.0"),
    Decimal("6.543"),
    Decimal("6.882"),
    Decimal("6.974"),
    Decimal("6.89"),
    Decimal("7.009"),
    Decimal("4.95"),
    Decimal("4.204"),
    Decimal("4.259"),
    Decimal("4.282"),
    Decimal("7.956"),
    Decimal("8.65"),
    Decimal("8.68"),
    Decimal("8.009"),
    Decimal("7.766"),
    Decimal("8.86"),
    Decimal("7.296"),
    Decimal("8.366"),
    Decimal("9.434"),
    Decimal("8.696"),
    Decimal("9.909"),
    Decimal("10.003"),
    Decimal("9.608"),
    Decimal("8.405"),
    Decimal("8.155"),
    Decimal("8.141"),
    Decimal("8.162"),
    Decimal("9.196"),
    Decimal("9.345"),
    Decimal("8.834"),
    Decimal("9.107"),
    Decimal("9.038"),
    Decimal("8.964"),
    Decimal("9.002"),
    Decimal("9.005"),
    Decimal("8.891"),
    Decimal("8.354"),
    Decimal("8.989"),
    Decimal("10.674"),
    Decimal("8.837"),
    Decimal("8.678"),
    Decimal("9.548"),
    Decimal("9.501"),
    Decimal("7.191"),
    Decimal("7.432"),
    Decimal("7.831"),
    Decimal("12.078"),
    Decimal("7.906"),
    Decimal("7.955"),
    Decimal("9.894"),
    Decimal("9.844"),
    Decimal("9.541"),
    Decimal("9.799"),
    Decimal("10.08"),
    Decimal("10.614"),
    Decimal("9.745"),
    Decimal("9.647"),
    Decimal("10.009"),
    Decimal("9.053"),
    Decimal("10.766"),
    Decimal("8.721"),
    Decimal("7.943"),
    Decimal("7.191"),
    Decimal("10.154"),
    Decimal("7.813"),
    Decimal("7.005"),
    Decimal("5.48"),
    Decimal("9.041"),
    Decimal("7.511"),
    Decimal("7.132"),
    Decimal("5.937"),
    Decimal("7.584"),
    Decimal("6.985"),
    Decimal("6.732"),
    Decimal("5.874"),
    Decimal("7.641"),
    Decimal("6.404"),
    Decimal("6.141"),
    Decimal("5.434"),
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
            f"start gap {indices[0]} > max_gap_from_start {max_gap_from_start}"
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


def test_crash_2026_02_27_max_gap_from_start_0():
    """Exact production parameters that caused the crash loop.

    Customer had max_gap_from_start=0 (incomplete-chunk override).
    The 56-period expensive stretch (positions 34-89) must be bridged
    without creating new gap violations.
    """
    result = get_cheapest_periods(
        prices=CRASH_PRICES,
        low_price_threshold=Decimal("6.985"),
        min_selections=32,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=0,
    )

    violations = _validate_constraints(
        result,
        total_length=len(CRASH_PRICES),
        min_selections=32,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=0,
    )
    assert not violations, (
        f"Constraint violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )


def test_crash_2026_02_27_max_gap_from_start_32():
    """Same crash data with max_gap_from_start=32.

    Customer also confirmed this variant crashes, proving the issue is in
    the bridge algorithm, not the incomplete-chunk override.
    """
    result = get_cheapest_periods(
        prices=CRASH_PRICES,
        low_price_threshold=Decimal("6.985"),
        min_selections=32,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=32,
    )

    violations = _validate_constraints(
        result,
        total_length=len(CRASH_PRICES),
        min_selections=32,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=32,
    )
    assert not violations, (
        f"Constraint violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )


def test_crash_2026_02_27_higher_min_selections():
    """Customer's workaround: min_selections=50 forces enough bridge blocks.

    This always worked even before the fix. Verify it still works and that
    the result is at least as good (cheaper or equal) as the lower
    min_selections variant.
    """
    result = get_cheapest_periods(
        prices=CRASH_PRICES,
        low_price_threshold=Decimal("6.985"),
        min_selections=50,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=32,
    )

    assert len(result) >= 50, f"Expected >=50 selections, got {len(result)}"

    violations = _validate_constraints(
        result,
        total_length=len(CRASH_PRICES),
        min_selections=50,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=32,
    )
    assert not violations, (
        f"Constraint violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )
