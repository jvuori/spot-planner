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
    min_period: int,
    max_gap: int,
    max_start_gap: int,
    full_length: int,
) -> bool:
    if not combination:
        return False

    # Items are already sorted, so indices are in order
    indices = [index for index, _ in combination]

    # Check max_start_gap first (fastest check)
    if indices[0] > max_start_gap:
        return False

    # Check start gap
    if indices[0] > max_gap:
        return False

    # Check gaps between consecutive indices and min_period in single pass
    block_length = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap:
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            if block_length < min_period:
                return False
            block_length = 1

    # Check last block min_period
    if block_length < min_period:
        return False

    # Check end gap
    if (full_length - 1 - indices[-1]) > max_gap:
        return False

    return True


def _get_combination_cost(combination: tuple[tuple[int, Decimal], ...]) -> Decimal:
    return sum(price for _, price in combination) or Decimal("0")


def _get_cheapest_periods_python(
    price_data: Sequence[Decimal],
    price_threshold: Decimal,
    desired_count: int,
    min_period: int,
    max_gap: int,
    max_start_gap: int,
) -> list[int]:
    price_items: tuple[tuple[int, Decimal], ...] = tuple(enumerate(price_data))
    cheap_items: tuple[tuple[int, Decimal], ...] = tuple(
        (index, price) for index, price in price_items if price <= price_threshold
    )
    # Start with desired_count as minimum, increment if no valid combination found
    actual_count = max(desired_count, len(cheap_items))

    # Special case: if desired_count equals total items, return all of them
    if desired_count == len(price_items):
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
                min_period,
                max_gap,
                max_start_gap,
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
    price_data: Sequence[Decimal],
    price_threshold: Decimal,
    desired_count: int,
    min_period: int,
    max_gap: int,
    max_start_gap: int,
) -> list[int]:
    """
    Find the cheapest periods in a sequence of prices.

    This function uses a Rust implementation for better performance,
    with a Python fallback if the Rust module is not available.
    """
    # Validate input parameters before calling either implementation
    if not price_data:
        raise ValueError("price_data cannot be empty")

    if desired_count <= 0:
        raise ValueError("desired_count must be greater than 0")

    if desired_count > len(price_data):
        raise ValueError("desired_count cannot be greater than total number of items")

    if min_period <= 0:
        raise ValueError("min_period must be greater than 0")

    if min_period > desired_count:
        raise ValueError("min_period cannot be greater than desired_count")

    if max_gap < 0:
        raise ValueError("max_gap must be greater than or equal to 0")

    if max_start_gap < 0:
        raise ValueError("max_start_gap must be greater than or equal to 0")

    if max_start_gap > max_gap:
        raise ValueError("max_start_gap must be less than or equal to max_gap")

    if _RUST_AVAILABLE:
        # Use Rust implementation - convert Decimal objects to strings
        price_data_str = [str(price) for price in price_data]
        price_threshold_str = str(price_threshold)
        return _rust_module.get_cheapest_periods(
            price_data_str,
            price_threshold_str,
            desired_count,
            min_period,
            max_gap,
            max_start_gap,
        )
    else:
        # Fallback to Python implementation
        return _get_cheapest_periods_python(
            price_data,
            price_threshold,
            desired_count,
            min_period,
            max_gap,
            max_start_gap,
        )
