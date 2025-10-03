import itertools
from decimal import Decimal
from typing import Sequence


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


def get_cheapest_periods(
    price_data: Sequence[Decimal],
    price_threshold: Decimal,
    desired_count: int,
    min_period: int,
    max_gap: int,
    max_start_gap: int,
) -> list[int]:
    if max_start_gap > max_gap:
        msg = "max_start_gap must be less than or equal to max_gap"
        raise ValueError(msg)

    price_items: tuple[tuple[int, Decimal], ...] = tuple(enumerate(price_data))
    cheap_items: tuple[tuple[int, Decimal], ...] = tuple(
        (index, price) for index, price in price_items if price <= price_threshold
    )
    actual_count = max(desired_count, len(cheap_items))

    cheapest_price_item_combination: tuple[tuple[int, Decimal], ...] = ()
    cheapest_cost: Decimal = _get_combination_cost(price_items)

    while not cheapest_price_item_combination:
        for price_item_combination in itertools.combinations(price_items, actual_count):
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
        actual_count += 1
        if actual_count > len(price_items):
            msg = f"No combination found for {actual_count} items"
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
