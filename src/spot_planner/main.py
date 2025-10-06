import itertools
from decimal import Decimal
from typing import Sequence

# Import the Rust implementation
try:
    from . import spot_planner as _rust_module

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _is_valid_combination(
    combination: tuple[tuple[int, Decimal], ...],
    min_consecutive_selections: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    full_length: int,
) -> bool:
    if not combination:
        return False

    # Items are already sorted, so indices are in order
    indices = [index for index, _ in combination]

    # Check max_gap_from_start first (fastest check)
    if indices[0] > max_gap_from_start:
        return False

    # Check start gap
    if indices[0] > max_gap_between_periods:
        return False

    # Check gaps between consecutive indices and min_consecutive_selections in single pass
    block_length = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            if block_length < min_consecutive_selections:
                return False
            block_length = 1

    # Check last block min_consecutive_selections
    if block_length < min_consecutive_selections:
        return False

    # Check end gap
    if (full_length - 1 - indices[-1]) > max_gap_between_periods:
        return False

    return True


def _get_combination_cost(combination: tuple[tuple[int, Decimal], ...]) -> Decimal:
    return sum(price for _, price in combination) or Decimal("0")


def _get_cheapest_periods_python(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_selections: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    price_items: tuple[tuple[int, Decimal], ...] = tuple(enumerate(prices))
    cheap_items: tuple[tuple[int, Decimal], ...] = tuple(
        (index, price) for index, price in price_items if price <= low_price_threshold
    )
    # Start with min_selections as minimum, increment if no valid combination found
    actual_count = max(min_selections, len(cheap_items))

    # Special case: if min_selections equals total items, return all of them
    if min_selections == len(price_items):
        return list(range(len(price_items)))

    # Special case: if all items are below threshold, return all of them
    if len(cheap_items) == len(price_items):
        return list(range(len(price_items)))

    cheapest_price_item_combination: tuple[tuple[int, Decimal], ...] = ()
    cheapest_cost: Decimal = _get_combination_cost(price_items)

    # Generate all combinations of the required size
    found = False
    current_count = actual_count

    while not found and current_count <= len(price_items):
        for price_item_combination in itertools.combinations(
            price_items, current_count
        ):
            if not _is_valid_combination(
                price_item_combination,
                min_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
                len(price_items),
            ):
                continue
            combination_cost = _get_combination_cost(price_item_combination)
            if combination_cost < cheapest_cost:
                cheapest_price_item_combination = price_item_combination
                cheapest_cost = combination_cost
                found = True
        current_count += 1

    if not found:
        msg = f"No combination found for {current_count} items"
        raise ValueError(msg)

    # Merge cheap_items with cheapest_price_item_combination, adding any items from cheap_items not already present
    merged_combination = list(cheapest_price_item_combination)
    existing_indices = {i for i, _ in cheapest_price_item_combination}
    for item in cheap_items:
        if item[0] not in existing_indices:
            merged_combination.append(item)
    # Sort by index to maintain order
    merged_combination.sort(key=lambda x: x[0])
    cheapest_price_item_combination = tuple(merged_combination)

    return [i for i, _ in cheapest_price_item_combination]


def get_cheapest_periods(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_selections: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    """
    Find optimal periods in a price sequence based on cost and timing constraints.

    This algorithm selects periods (indices) from a price sequence to minimize cost
    while satisfying various timing constraints. The primary selection criterion is
    the price threshold - all periods with prices at or below the threshold are
    automatically selected regardless of other constraints.

    Args:
        prices: Sequence of prices for each period. Each element represents the
               price for one time period (e.g., hourly, 15-minute intervals).
        low_price_threshold: Price threshold below/equal to which periods are
                           automatically selected. All periods with price <= threshold
                           will be included in the result regardless of other constraints.
        min_selections: Minimum number of individual periods (indices) that must be
                       selected. The algorithm will select at least this many periods,
                       but may select more if they are below the price threshold.
        min_consecutive_selections: Minimum number of consecutive periods that must be
                                  selected together. Any selected period must be part of
                                  a run of at least this many consecutive selections.
                                  Prevents isolated single-period selections.
        max_gap_between_periods: Maximum number of periods allowed between selected
                               periods. Controls the maximum downtime between operating
                               periods. Set to 0 to require consecutive selections only.
        max_gap_from_start: Maximum number of periods from the beginning before the
                          first selection must occur. Controls how long we can wait
                          before starting operations.

    Returns:
        List of indices representing the selected periods, sorted by price (cheapest first).
        The indices correspond to positions in the input prices sequence.

    Raises:
        ValueError: If the input parameters are invalid or no valid combination
                   can be found that satisfies all constraints.

    Examples:
        >>> prices = [Decimal('0.05'), Decimal('0.08'), Decimal('0.12'), Decimal('0.06')]
        >>> get_cheapest_periods(prices, Decimal('0.10'), 2, 1, 1, 1)
        [0, 3, 1]  # Selects periods 0, 1, 3 (all <= 0.10, sorted by price)

    Note:
        The algorithm prioritizes periods below the price threshold. If all periods
        are below the threshold, all periods will be selected regardless of other
        constraints. If the desired number of periods equals the total number of
        periods, all periods will be selected regardless of price.
    """
    # Validate input parameters before calling either implementation
    if not prices:
        raise ValueError("prices cannot be empty")

    if min_selections <= 0:
        raise ValueError("min_selections must be greater than 0")

    if min_selections > len(prices):
        raise ValueError("min_selections cannot be greater than total number of items")

    if min_consecutive_selections <= 0:
        raise ValueError("min_consecutive_selections must be greater than 0")

    if min_consecutive_selections > min_selections:
        raise ValueError("min_consecutive_selections cannot be greater than min_selections")

    if max_gap_between_periods < 0:
        raise ValueError("max_gap_between_periods must be greater than or equal to 0")

    if max_gap_from_start < 0:
        raise ValueError("max_gap_from_start must be greater than or equal to 0")

    if max_gap_from_start > max_gap_between_periods:
        raise ValueError("max_gap_from_start must be less than or equal to max_gap_between_periods")

    if _RUST_AVAILABLE:
        # Use Rust implementation - convert Decimal objects to strings
        prices_str = [str(price) for price in prices]
        low_price_threshold_str = str(low_price_threshold)
        return _rust_module.get_cheapest_periods(
            prices_str,
            low_price_threshold_str,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
    else:
        # Fallback to Python implementation
        return _get_cheapest_periods_python(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
