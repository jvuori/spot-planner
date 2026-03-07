import logging as _logging
from decimal import Decimal
from typing import Sequence

# Import two-phase algorithm which handles both short and long sequences
from spot_planner import two_phase

_LOG = _logging.getLogger(__name__)

# Maximum number of consecutive cheap-item groups that _try_add_cheap_items will
# enumerate via bitmask.  Each additional group doubles the search space:
# 2^20 ≈ 1 M iterations is an acceptable upper bound; anything beyond that is a
# sign of unexpected input and we raise immediately instead of hanging forever.
_MAX_CHEAP_GROUPS = 20


def get_cheapest_periods(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int = 0,
    max_gap_from_start: int = 0,
    aggressive: bool = True,
) -> list[int]:
    """
    Find optimal periods in a price sequence based on cost and timing constraints.

    This algorithm selects periods (indices) from a price sequence to minimize cost
    while satisfying various timing constraints. The algorithm prioritizes periods
    with prices at or below the threshold, but still respects all constraints.

    For sequences longer than 28 items, uses a two-phase approach:
    1. Rough planning: Creates averages of 4-item groups and runs brute-force
       on the averages to determine approximate selection distribution.
    2. Fine-grained planning: Processes actual prices in 20-item chunks,
       using the rough plan as a guide, with boundary-aware constraint handling.

    Args:
        prices: Sequence of prices for each period. Each element represents the
               price for one time period (e.g., hourly, 15-minute intervals).
               Can handle sequences of any length (uses extended algorithm for > 28).
        low_price_threshold: Price threshold below/equal to which periods are
                           preferentially selected. Periods with price <= threshold
                           will be included if they can form valid consecutive runs
                           meeting the consecutive_selections constraint.
        min_selections: Desired minimum number of periods to select. If no valid
                       combination is found, this will be incremented by 1 until
                       a valid solution is found.
        min_consecutive_periods: Minimum consecutive periods required for each
                                consecutive block. This is enforced as a minimum
                                for every consecutive block.
        max_gap_between_periods: Maximum number of periods allowed between selected
                               periods. Controls the maximum downtime between operating
                               periods. Set to 0 to require consecutive selections only.
        max_gap_from_start: Maximum number of periods from the beginning before the
                          first selection must occur. Controls how long we can wait
                          before starting operations.
        aggressive: If True (default), minimizes average cost per selected period.
                   If False, maximizes number of cheap items while respecting constraints.

    Returns:
        List of indices representing the selected periods, sorted by index.
        The indices correspond to positions in the input prices sequence.

    Raises:
        ValueError: If the input parameters are invalid or no valid combination
                   can be found that satisfies all constraints.

    Examples:
        >>> prices = [Decimal('0.05'), Decimal('0.08'), Decimal('0.12'), Decimal('0.06')]
        >>> get_cheapest_periods(prices, Decimal('0.10'), 2, 1, 1, 1)
        [0, 1, 3]  # Selects periods 0, 1, 3 (all <= 0.10 and form valid runs)

    Note:
        The algorithm uses adaptive retry logic:
        1. Find cheapest main combination that meets constraints
        2. Try adding cheap items to that combination
        3. If no valid solution found, increment min_selections and retry
    """
    # Basic validation
    if not prices:
        msg = "prices cannot be empty"
        raise ValueError(msg)

    if min_selections <= 0:
        msg = "min_selections must be greater than 0"
        raise ValueError(msg)

    if min_selections > len(prices):
        msg = "min_selections cannot be greater than total number of items"
        raise ValueError(msg)

    if min_consecutive_periods <= 0:
        msg = "min_consecutive_periods must be greater than 0"
        raise ValueError(msg)

    if min_consecutive_periods > min_selections:
        msg = "min_consecutive_periods cannot be greater than min_selections"
        raise ValueError(msg)

    if max_gap_between_periods < 0:
        msg = "max_gap_between_periods must be greater than or equal to 0"
        raise ValueError(msg)

    if max_gap_from_start < 0:
        msg = "max_gap_from_start must be greater than or equal to 0"
        raise ValueError(msg)

    if max_gap_from_start > max_gap_between_periods:
        msg = "max_gap_from_start must be less than or equal to max_gap_between_periods"
        raise ValueError(msg)

    # Use extended algorithm for longer sequences
    if len(prices) > 28:
        result = two_phase.get_cheapest_periods_extended(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
        return _try_add_cheap_items(
            result, prices, low_price_threshold,
            min_consecutive_periods, max_gap_between_periods, max_gap_from_start,
        )

    # Use direct brute-force for shorter sequences (dispatched by two_phase module)
    result = two_phase._get_cheapest_periods(
        prices,
        low_price_threshold,
        min_selections,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
        aggressive,
    )
    return _try_add_cheap_items(
        result, prices, low_price_threshold,
        min_consecutive_periods, max_gap_between_periods, max_gap_from_start,
    )


def _try_add_cheap_items(
    result: list[int],
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    """Augment result by adding remaining below-threshold items without any bridge items.

    After the base algorithm produces a valid selection, tries all subsets of the
    remaining below-threshold items (those not in result with price <= threshold).
    For each subset the full merged set is validated against all constraints —
    no items above the threshold are added as bridges.

    Implementation uses a group-bitmask approach:
      1. Group remaining cheap items into maximal consecutive runs ("groups").
      2. Try all 2^num_groups subsets of *whole groups* — O(2^G) instead of the
         exponential O(C(n, k)) that the naive combinations approach gives.
      3. Return the subset that adds the most items; ties broken by lowest avg cost.

    A hard limit of _MAX_CHEAP_GROUPS groups is enforced.  If that limit is
    exceeded the function raises ValueError immediately rather than hanging.
    """
    n = len(prices)
    result_set = set(result)
    remaining_cheap = sorted(
        i for i in range(n)
        if i not in result_set and prices[i] <= low_price_threshold
    )

    if not remaining_cheap:
        return result

    # ------------------------------------------------------------------
    # Group remaining cheap items into maximal consecutive runs.
    # ------------------------------------------------------------------
    groups: list[list[int]] = []
    g: list[int] = [remaining_cheap[0]]
    for i in range(1, len(remaining_cheap)):
        if remaining_cheap[i] == remaining_cheap[i - 1] + 1:
            g.append(remaining_cheap[i])
        else:
            groups.append(g)
            g = [remaining_cheap[i]]
    groups.append(g)

    num_groups = len(groups)
    _LOG.debug(
        "_try_add_cheap_items: n=%d, base_result=%d items, "
        "remaining_cheap=%d items in %d groups",
        n, len(result), len(remaining_cheap), num_groups,
    )

    # -----------------------------------------------------------------------
    # Safety guard: if there are too many groups the bitmask search would take
    # too long.  Crash fast rather than hanging the process at 100 % CPU.
    # -----------------------------------------------------------------------
    if num_groups > _MAX_CHEAP_GROUPS:
        msg = (
            f"_try_add_cheap_items: {num_groups} groups of remaining cheap items "
            f"exceeds the safety limit of {_MAX_CHEAP_GROUPS} groups "
            f"(would require up to 2^{num_groups} = {2**num_groups:,} iterations). "
            f"n={n}, remaining_cheap={len(remaining_cheap)}. "
            f"This is a bug — the base algorithm should have consumed more items."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Exhaustively enumerate all 2^num_groups non-empty subsets of whole
    # groups via bitmask.  For cheap_day with 16 groups this is only
    # 65 536 iterations instead of C(40, k) for k = 40..1 which is
    # astronomically larger.
    # ------------------------------------------------------------------
    best_candidate: list[int] | None = None
    best_items_added = 0
    best_avg = Decimal("inf")

    for mask in range(1, 1 << num_groups):
        candidate = list(result)
        items_added = 0
        for gi, grp in enumerate(groups):
            if mask & (1 << gi):
                candidate.extend(grp)
                items_added += len(grp)
        candidate.sort()

        if not two_phase._validate_full_selection(
            candidate, n, min_consecutive_periods, max_gap_between_periods, max_gap_from_start
        ):
            continue

        if items_added > best_items_added:
            best_items_added = items_added
            best_avg = sum(prices[i] for i in candidate) / len(candidate)
            best_candidate = candidate
        elif items_added == best_items_added:
            avg = sum(prices[i] for i in candidate) / len(candidate)
            if avg < best_avg:
                best_avg = avg
                best_candidate = candidate

    if best_candidate is not None:
        _LOG.debug(
            "_try_add_cheap_items: added %d cheap items → total %d selected",
            best_items_added, len(best_candidate),
        )
        return best_candidate

    return result
