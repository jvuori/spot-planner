import itertools
from dataclasses import dataclass
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
    min_consecutive_periods: int,
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

    # Check gaps between consecutive indices and min_consecutive_periods in single pass
    block_length = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            if block_length < min_consecutive_periods:
                return False
            block_length = 1

    # Check last block min_consecutive_periods
    # If the last index is at the end of the price sequence, don't enforce min_consecutive_periods
    # because we don't know future prices that might extend this block
    last_index = indices[-1]
    is_at_end = last_index == full_length - 1
    if not is_at_end and block_length < min_consecutive_periods:
        return False

    # Check end gap
    if (full_length - 1 - indices[-1]) > max_gap_between_periods:
        return False

    return True


def _get_combination_cost(combination: tuple[tuple[int, Decimal], ...]) -> Decimal:
    return sum(price for _, price in combination) or Decimal("0")


def _group_consecutive_items(
    items: Sequence[tuple[int, Decimal]],
) -> list[list[tuple[int, Decimal]]]:
    """Group cheap items into consecutive runs."""
    if not items:
        return []

    groups = []
    current_group = [items[0]]

    for i in range(1, len(items)):
        if items[i][0] == items[i - 1][0] + 1:
            current_group.append(items[i])
        else:
            groups.append(current_group)
            current_group = [items[i]]
    groups.append(current_group)

    return groups


def _check_consecutive_runs(
    indices: list[int], min_consecutive_periods: int, full_length: int
) -> bool:
    """Check if all consecutive runs in indices meet the minimum length requirement.

    If the last index is at the end of the price sequence (full_length - 1), don't enforce
    min_consecutive_periods for the last block because we don't know future prices.

    Args:
        indices: Sorted list of indices
        min_consecutive_periods: Minimum required length for each consecutive run
        full_length: Total length of the price sequence

    Returns:
        True if all runs meet the requirement, False otherwise
    """
    if not indices:
        return False

    if len(indices) == 1:
        return min_consecutive_periods <= 1

    # Count consecutive runs
    run_length = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            run_length += 1
        else:
            # End of a run - check if it meets minimum
            if run_length < min_consecutive_periods:
                return False
            run_length = 1

    # Check the last run
    # If the last index is at the end of the price sequence, don't enforce min_consecutive_periods
    # because we don't know future prices that might extend this block
    last_index = indices[-1]
    is_at_end = last_index == full_length - 1
    if is_at_end:
        # At the end, so don't enforce min_consecutive_periods for the last block
        return True
    else:
        return run_length >= min_consecutive_periods


def _get_cheapest_periods_python(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool = True,
) -> list[int]:
    price_items: tuple[tuple[int, Decimal], ...] = tuple(enumerate(prices))
    cheap_items: tuple[tuple[int, Decimal], ...] = tuple(
        (index, price) for index, price in price_items if price <= low_price_threshold
    )

    # Special case: if min_selections equals total items, return all of them
    if min_selections == len(price_items):
        return list(range(len(price_items)))

    # Special case: if all items are below threshold, return all of them
    if len(cheap_items) == len(price_items):
        return list(range(len(price_items)))

    # Choose algorithm based on aggressive parameter
    if aggressive:
        # Aggressive mode: minimize average cost (current behavior)
        return _get_cheapest_periods_aggressive_python(
            price_items,
            cheap_items,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
        )
    else:
        # Conservative mode: maximize number of cheap items
        return _get_cheapest_periods_conservative_python(
            price_items,
            cheap_items,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
        )


def _get_cheapest_periods_aggressive_python(
    price_items: tuple[tuple[int, Decimal], ...],
    cheap_items: tuple[tuple[int, Decimal], ...],
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    # Generate all combinations and find the best one after merging with cheap items
    best_result = []
    best_cost = _get_combination_cost(price_items)
    found = False

    # Try combinations starting from min_selections
    for current_count in range(min_selections, len(price_items) + 1):
        for price_item_combination in itertools.combinations(
            price_items, current_count
        ):
            if not _is_valid_combination(
                price_item_combination,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                len(price_items),
            ):
                continue

            # Start with this combination
            result_indices = [i for i, _ in price_item_combination]
            existing_indices = set(result_indices)

            # Try every combination of cheap items that are not already included
            available_cheap_items = [
                item for item in cheap_items if item[0] not in existing_indices
            ]

            # Group cheap items into consecutive runs for efficiency
            cheap_groups = _group_consecutive_items(available_cheap_items)

            # Try every combination of consecutive groups (2^n instead of 2^20)
            best_merged_result = result_indices.copy()
            best_merged_cost = _get_combination_cost(
                [(i, price_items[i][1]) for i in result_indices]
            )

            for group_mask in range(1, 2 ** len(cheap_groups)):  # Skip empty selection
                merged_indices = result_indices.copy()

                # Add items from selected groups
                for group_idx, group in enumerate(cheap_groups):
                    if group_mask & (1 << group_idx):
                        for index, _ in group:
                            merged_indices.append(index)

                merged_indices.sort()

                # Check if merged result maintains valid consecutive runs
                if _check_consecutive_runs(
                    merged_indices, min_consecutive_periods, len(price_items)
                ):
                    # Calculate average cost of this merged result
                    merged_cost = sum(price_items[i][1] for i in merged_indices)
                    merged_avg_cost = merged_cost / len(merged_indices)

                    # Calculate average cost of current best
                    best_avg_cost = best_merged_cost / len(best_merged_result)

                    # Keep the result with lowest average cost
                    if merged_avg_cost < best_avg_cost:
                        best_merged_result = merged_indices
                        best_merged_cost = merged_cost

            # Use the best merged result
            total_cost = best_merged_cost
            avg_cost = total_cost / len(best_merged_result)

            # Compare average costs, not total costs
            best_avg_cost = (
                best_cost / len(best_result) if best_result else float("inf")
            )

            if avg_cost < best_avg_cost:
                best_result = best_merged_result
                best_cost = total_cost
                found = True

        # If we found a valid combination at this size, don't try larger sizes
        if found:
            break

    if not found:
        raise ValueError(
            f"No valid combination found that satisfies the constraints for {len(price_items)} items"
        )

    return best_result


def _get_cheapest_periods_conservative_python(
    price_items: tuple[tuple[int, Decimal], ...],
    cheap_items: tuple[tuple[int, Decimal], ...],
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    """Conservative algorithm: maximize number of cheap items while respecting constraints."""
    # First, try to use as many cheap items as possible
    best_result = []
    found = False

    # Try combinations starting from min_selections, prioritizing cheap items
    for current_count in range(min_selections, len(price_items) + 1):
        # First try combinations that include as many cheap items as possible
        cheap_indices = [i for i, _ in cheap_items]

        # Try all combinations of cheap items first (from most to least)
        for cheap_count in range(len(cheap_indices), 0, -1):
            if cheap_count < min_selections:
                continue

            for cheap_combination in itertools.combinations(cheap_indices, cheap_count):
                # Convert to price_item format for validation
                cheap_price_items = [(i, price_items[i][1]) for i in cheap_combination]

                if not _is_valid_combination(
                    cheap_price_items,
                    min_consecutive_periods,
                    max_gap_between_periods,
                    max_gap_from_start,
                    len(price_items),
                ):
                    continue

                # If we need more items, try to add non-cheap items
                if cheap_count < current_count:
                    remaining_needed = current_count - cheap_count
                    non_cheap_indices = [
                        i for i in range(len(price_items)) if i not in cheap_indices
                    ]

                    for non_cheap_combination in itertools.combinations(
                        non_cheap_indices, min(remaining_needed, len(non_cheap_indices))
                    ):
                        combined_indices = list(cheap_combination) + list(
                            non_cheap_combination
                        )
                        combined_indices.sort()

                        # Convert to price_item format for validation
                        combined_price_items = [
                            (i, price_items[i][1]) for i in combined_indices
                        ]

                        if _is_valid_combination(
                            combined_price_items,
                            min_consecutive_periods,
                            max_gap_between_periods,
                            max_gap_from_start,
                            len(price_items),
                        ):
                            best_result = combined_indices
                            found = True
                            break
                else:
                    # We have enough cheap items
                    best_result = list(cheap_combination)
                    found = True
                    break

                if found:
                    break

            if found:
                break

        if found:
            break

    if not found:
        raise ValueError(
            f"No valid combination found that satisfies the constraints for {len(price_items)} items"
        )

    return best_result


def _get_cheapest_periods(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int = 0,
    max_gap_from_start: int = 0,
    aggressive: bool = True,
) -> list[int]:
    # Validate input parameters before calling either implementation
    if not prices:
        msg = "prices cannot be empty"
        raise ValueError(msg)

    if len(prices) > 28:
        msg = "prices cannot contain more than 28 items"
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

    if _RUST_AVAILABLE:
        # Use Rust implementation - convert Decimal objects to strings
        prices_str = [str(price) for price in prices]
        low_price_threshold_str = str(low_price_threshold)
        return _rust_module.get_cheapest_periods(
            prices_str,
            low_price_threshold_str,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
    else:
        # Fallback to Python implementation
        return _get_cheapest_periods_python(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )


@dataclass
class _ChunkBoundaryState:
    """Tracks the state at the end of a processed chunk for boundary handling."""

    ended_with_selected: bool  # True if the chunk ended with a selected period
    trailing_selected_count: int  # Consecutive selected periods at the end
    trailing_unselected_count: (
        int  # Unselected periods at the end (0 if ended selected)
    )


def _validate_full_selection(
    selected_indices: list[int],
    total_length: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> bool:
    """Validate that a complete selection meets all constraints."""
    if not selected_indices:
        return False

    indices = sorted(selected_indices)

    # Check max_gap_from_start
    if indices[0] > max_gap_from_start:
        return False

    # Check gap from start (also constrained by max_gap_between_periods)
    if indices[0] > max_gap_between_periods:
        return False

    # Check gap at end
    if (total_length - 1 - indices[-1]) > max_gap_between_periods:
        return False

    # Check gaps between selections and consecutive block lengths
    block_length = 1
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1] - 1
        if gap > max_gap_between_periods:
            return False

        if indices[i] == indices[i - 1] + 1:
            block_length += 1
        else:
            # End of a block - must meet minimum
            if block_length < min_consecutive_periods:
                return False
            block_length = 1

    # Final block must also meet minimum
    if block_length < min_consecutive_periods:
        return False

    return True


def _calculate_chunk_boundary_state(
    chunk_selected: list[int], chunk_length: int
) -> _ChunkBoundaryState:
    """Calculate the boundary state after processing a chunk."""
    if not chunk_selected:
        return _ChunkBoundaryState(
            ended_with_selected=False,
            trailing_selected_count=0,
            trailing_unselected_count=chunk_length,
        )

    last_selected = max(chunk_selected)
    ended_with_selected = last_selected == chunk_length - 1

    if ended_with_selected:
        # Count trailing consecutive selected
        sorted_selected = sorted(chunk_selected)
        trailing_count = 1
        for i in range(len(sorted_selected) - 2, -1, -1):
            if sorted_selected[i] == sorted_selected[i + 1] - 1:
                trailing_count += 1
            else:
                break
        return _ChunkBoundaryState(
            ended_with_selected=True,
            trailing_selected_count=trailing_count,
            trailing_unselected_count=0,
        )
    else:
        return _ChunkBoundaryState(
            ended_with_selected=False,
            trailing_selected_count=0,
            trailing_unselected_count=chunk_length - 1 - last_selected,
        )


def _estimate_forced_prefix_cost(
    next_chunk_prices: Sequence[Decimal],
    boundary_state: _ChunkBoundaryState,
    min_consecutive_periods: int,
) -> Decimal:
    """
    Estimate the cost of forced prefix selections in the next chunk.

    When a chunk ends with an incomplete consecutive block, the next chunk
    is forced to continue it. This function calculates the cost of those
    forced selections to enable look-ahead optimization.
    """
    if not boundary_state.ended_with_selected:
        return Decimal(0)

    if boundary_state.trailing_selected_count >= min_consecutive_periods:
        return Decimal(0)

    # Calculate how many items would be forced
    forced_count = min(
        min_consecutive_periods - boundary_state.trailing_selected_count,
        len(next_chunk_prices),
    )

    # Sum the cost of forced items
    return sum(next_chunk_prices[:forced_count], Decimal(0))


def _try_chunk_selection(
    chunk_prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    target: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool,
) -> list[int] | None:
    """Try to get a valid chunk selection, return None if not possible."""
    try:
        return _get_cheapest_periods(
            chunk_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
    except ValueError:
        return None


def _find_best_chunk_selection_with_lookahead(
    chunk_prices: Sequence[Decimal],
    next_chunk_prices: Sequence[Decimal] | None,
    low_price_threshold: Decimal,
    target: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool,
) -> list[int]:
    """
    Find the best chunk selection considering the cost impact on the next chunk.

    This implements look-ahead optimization: instead of just picking the locally
    optimal selection, we consider how different ending strategies affect the
    forced selections in the next chunk.

    For each valid selection strategy, we calculate:
    - Cost of selections in current chunk
    - Cost of forced selections in next chunk (if any)
    - Total combined cost

    We pick the strategy with the lowest combined cost.
    """
    chunk_len = len(chunk_prices)

    # If there's no next chunk, just find the best selection for this chunk
    if next_chunk_prices is None:
        result = _try_chunk_selection(
            chunk_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
        if result:
            return result
        # Fallback attempts
        for fallback_target in [min_consecutive_periods, 1]:
            result = _try_chunk_selection(
                chunk_prices,
                low_price_threshold,
                min(fallback_target, chunk_len),
                min(fallback_target, min_consecutive_periods),
                max_gap_between_periods,
                max_gap_from_start,
                aggressive,
            )
            if result:
                return result
        # Last resort
        sorted_by_price = sorted(range(chunk_len), key=lambda i: chunk_prices[i])
        return sorted(sorted_by_price[:min_consecutive_periods])

    # With look-ahead: try multiple strategies and pick the best combined cost
    # Store: (selection, cost_metric) where cost_metric depends on mode
    # For aggressive: average cost
    # For conservative: (cheap_count, total_cost) tuple for comparison
    candidates: list[tuple[list[int], Decimal | tuple[int, Decimal]]] = []

    # Strategy 1: Standard selection (locally optimal)
    selection = _try_chunk_selection(
        chunk_prices,
        low_price_threshold,
        target,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
        aggressive,
    )
    if selection:
        boundary = _calculate_chunk_boundary_state(selection, chunk_len)
        chunk_cost: Decimal = sum((chunk_prices[i] for i in selection), Decimal(0))
        forced_cost = _estimate_forced_prefix_cost(
            next_chunk_prices, boundary, min_consecutive_periods
        )
        total_cost = chunk_cost + forced_cost
        
        if aggressive:
            # Aggressive mode: use average cost
            avg_cost = total_cost / Decimal(len(selection))
            candidates.append((selection, avg_cost))
        else:
            # Conservative mode: use (cheap_count, total_cost)
            cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
            candidates.append((selection, (cheap_count, total_cost)))

    # Strategy 2: Try to end with a complete block (avoid forcing next chunk)
    # This means selecting items up to the end of the chunk
    if target < chunk_len:
        # Try selecting more items to reach the end
        for extra in range(1, min(min_consecutive_periods + 1, chunk_len - target + 1)):
            extended_target = min(target + extra, chunk_len)
            selection = _try_chunk_selection(
                chunk_prices,
                low_price_threshold,
                extended_target,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                aggressive,
            )
            if selection and max(selection) == chunk_len - 1:
                # This selection ends at the chunk boundary
                boundary = _calculate_chunk_boundary_state(selection, chunk_len)
                # Check if it creates a complete block at the end
                if boundary.trailing_selected_count >= min_consecutive_periods:
                    complete_block_cost: Decimal = sum(
                        (chunk_prices[i] for i in selection), Decimal(0)
                    )
                    # No forced cost since block is complete
                    if aggressive:
                        avg_cost = complete_block_cost / Decimal(len(selection))
                        candidates.append((selection, avg_cost))
                    else:
                        cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
                        candidates.append((selection, (cheap_count, complete_block_cost)))
                    break

    # Strategy 3: Try ending with unselected items (gap at end)
    # This avoids forcing the next chunk to continue a block
    if target <= chunk_len - 1:
        # Try to not select the last item
        adjusted_prices = list(chunk_prices)
        # Make the last item very expensive to discourage selecting it
        adjusted_prices[-1] = Decimal("999999999")

        selection = _try_chunk_selection(
            adjusted_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
        if selection and chunk_len - 1 not in selection:
            # Verify the gap at end is acceptable
            last_selected = max(selection)
            gap_at_end = chunk_len - 1 - last_selected
            if gap_at_end <= max_gap_between_periods:
                boundary = _calculate_chunk_boundary_state(selection, chunk_len)
                chunk_cost_val: Decimal = sum(
                    (chunk_prices[i] for i in selection), Decimal(0)
                )
                # No forced cost since we ended with unselected
                if aggressive:
                    avg_cost = chunk_cost_val / Decimal(len(selection))
                    candidates.append((selection, avg_cost))
                else:
                    cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
                    candidates.append((selection, (cheap_count, chunk_cost_val)))

    # Strategy 4: Try to complete a min_consecutive block at the end
    # by selecting exactly min_consecutive_periods items at the end
    if min_consecutive_periods <= chunk_len:
        end_block_start = chunk_len - min_consecutive_periods
        # Check if we can make a valid selection that includes this end block
        end_block = list(range(end_block_start, chunk_len))

        # Calculate cost of end block
        end_block_cost: Decimal = sum((chunk_prices[i] for i in end_block), Decimal(0))

        # Try to find a selection that includes this end block
        if target <= min_consecutive_periods:
            # Just use the end block
            selection = end_block
            # Check if this satisfies gap constraints
            if end_block_start <= max_gap_from_start:
                boundary = _calculate_chunk_boundary_state(selection, chunk_len)
                forced_cost = _estimate_forced_prefix_cost(
                    next_chunk_prices, boundary, min_consecutive_periods
                )
                total_cost = end_block_cost + forced_cost
                if aggressive:
                    avg_cost = total_cost / Decimal(len(selection))
                    candidates.append((selection, avg_cost))
                else:
                    cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
                    candidates.append((selection, (cheap_count, total_cost)))

    # If no candidates, use fallback
    if not candidates:
        # Try with relaxed constraints
        for fallback_target in [min_consecutive_periods, 1]:
            selection = _try_chunk_selection(
                chunk_prices,
                low_price_threshold,
                min(fallback_target, chunk_len),
                min(fallback_target, chunk_len),
                max_gap_between_periods,
                chunk_len,  # Relaxed gap from start
                aggressive,
            )
            if selection:
                fallback_cost: Decimal = sum(
                    (chunk_prices[i] for i in selection), Decimal(0)
                )
                if aggressive:
                    avg_cost = fallback_cost / Decimal(len(selection))
                    candidates.append((selection, avg_cost))
                else:
                    cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
                    candidates.append((selection, (cheap_count, fallback_cost)))
                break

    if not candidates:
        # Last resort
        sorted_by_price = sorted(range(chunk_len), key=lambda i: chunk_prices[i])
        selection = sorted(sorted_by_price[:min_consecutive_periods])
        last_resort_cost: Decimal = sum(
            (chunk_prices[i] for i in selection), Decimal(0)
        )
        if aggressive:
            avg_cost = last_resort_cost / Decimal(len(selection))
            candidates.append((selection, avg_cost))
        else:
            cheap_count = sum(1 for i in selection if chunk_prices[i] <= low_price_threshold)
            candidates.append((selection, (cheap_count, last_resort_cost)))

    # Pick the best candidate based on mode
    if aggressive:
        # Aggressive mode: lowest average cost
        best_selection, _ = min(candidates, key=lambda x: x[1])
    else:
        # Conservative mode: most cheap items, then lowest total cost
        best_selection, _ = max(candidates, key=lambda x: (x[1][0], -x[1][1]))
    return best_selection


def _calculate_optimal_chunk_size(
    total_items: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
) -> int:
    """
    Calculate optimal chunk size based on sequence length and constraints.

    Smaller chunks = faster but more boundary issues
    Larger chunks = slower but fewer boundary issues

    Strategy:
    - Very small sequences (<=48): Use medium chunks (20) for good balance
    - Small sequences (49-96): Use medium chunks (18-20)
    - Medium sequences (97-192): Use smaller chunks (15-18) for performance
    - Large sequences (>192): Use smallest chunks (12-15) for speed

    Also considers constraints:
    - Large max_gap allows smaller chunks (boundaries less critical)
    - Large min_consecutive needs larger chunks (boundaries more critical)
    """
    # Base chunk size based on sequence length
    if total_items <= 48:
        base_size = 20
    elif total_items <= 96:
        base_size = 18
    elif total_items <= 192:
        base_size = 15
    else:
        base_size = 12

    # Adjust based on constraints
    # If max_gap is large, boundaries are less critical - can use smaller chunks
    if max_gap_between_periods >= 15:
        base_size = max(12, base_size - 2)
    elif max_gap_between_periods >= 10:
        base_size = max(12, base_size - 1)

    # If min_consecutive is large, boundaries are more critical - need larger chunks
    if min_consecutive_periods >= 6:
        base_size = min(24, base_size + 2)
    elif min_consecutive_periods >= 4:
        base_size = min(24, base_size + 1)

    # Ensure chunk size is reasonable (not too small, not too large)
    # Too small: < 10 items per chunk becomes inefficient
    # Too large: > 24 items per chunk becomes slow
    return max(10, min(24, base_size))


def _get_cheapest_periods_extended(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool,
) -> list[int]:
    """
    Extended algorithm for longer price sequences (> 28 items).

    Uses a two-phase approach:
    1. Rough planning: Create averages of price groups, run brute-force on averages
       to determine approximate distribution of selections.
    2. Fine-grained planning: Process actual prices in chunks, using the rough plan
       to guide target selections per chunk, with boundary-aware constraint handling
       and look-ahead optimization.

    The algorithm maintains constraints across chunk boundaries:
    - If a chunk ends with an incomplete consecutive block, the next chunk must
      continue that block to meet min_consecutive_periods.
    - max_gap_between_periods is tracked across boundaries by adjusting
      max_gap_from_start for subsequent chunks.

    Look-ahead optimization:
    - For each chunk, multiple selection strategies are evaluated
    - The cost of forced selections in the next chunk is considered
    - The strategy with lowest combined cost is chosen

    Chunk size is adaptively determined based on sequence length and constraints
    to optimize the performance/optimality trade-off.
    """
    n = len(prices)

    # Calculate optimal chunk size adaptively
    MAX_CHUNK_SIZE = _calculate_optimal_chunk_size(
        n, min_consecutive_periods, max_gap_between_periods
    )

    # 4 items per average group gives good rough planning resolution
    AVERAGE_GROUP_SIZE = 4

    # Phase 1: Rough planning with averages
    # Create averaged groups for rough selection pattern
    averages: list[Decimal] = []
    group_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) for each average

    for i in range(0, n, AVERAGE_GROUP_SIZE):
        group_end = min(i + AVERAGE_GROUP_SIZE, n)
        group = list(prices[i:group_end])
        group_sum = sum(group, Decimal(0))
        group_avg = group_sum / Decimal(len(group))
        averages.append(group_avg)
        group_ranges.append((i, group_end))

    # Scale parameters for rough planning
    # Each average represents AVERAGE_GROUP_SIZE actual items
    scale_factor = AVERAGE_GROUP_SIZE
    rough_min_selections = max(1, (min_selections + scale_factor - 1) // scale_factor)
    rough_min_consecutive = max(
        1, (min_consecutive_periods + scale_factor - 1) // scale_factor
    )
    rough_max_gap = max(0, max_gap_between_periods // scale_factor)
    rough_max_gap_start = max(0, max_gap_from_start // scale_factor)

    # Ensure constraints are valid
    rough_min_consecutive = min(rough_min_consecutive, rough_min_selections)
    rough_max_gap_start = min(rough_max_gap_start, rough_max_gap)

    # Get rough selection pattern
    try:
        rough_selected = _get_cheapest_periods(
            averages,
            low_price_threshold,
            rough_min_selections,
            rough_min_consecutive,
            rough_max_gap,
            rough_max_gap_start,
            aggressive,
        )
    except ValueError:
        # If rough planning fails with scaled constraints, try more lenient
        try:
            rough_selected = _get_cheapest_periods(
                averages,
                low_price_threshold,
                rough_min_selections,
                1,  # min_consecutive = 1
                len(averages),  # max_gap = all
                len(averages),  # max_gap_start = all
                aggressive,
            )
        except ValueError:
            # Last resort: select all averages
            rough_selected = list(range(len(averages)))

    # Phase 2: Fine-grained planning
    # Calculate target selections per chunk based on rough plan
    num_chunks = (n + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
    chunk_selection_targets: list[int] = [0] * num_chunks

    for avg_idx in rough_selected:
        start_price_idx, end_price_idx = group_ranges[avg_idx]

        # Distribute this group's selections to overlapping chunks
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * MAX_CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * MAX_CHUNK_SIZE, n)

            # Calculate overlap between this average group and chunk
            overlap_start = max(start_price_idx, chunk_start)
            overlap_end = min(end_price_idx, chunk_end)

            if overlap_start < overlap_end:
                chunk_selection_targets[chunk_idx] += overlap_end - overlap_start

    # Ensure we have enough total selections
    total_target = sum(chunk_selection_targets)
    if total_target < min_selections:
        # Distribute remaining selections proportionally
        remaining = min_selections - total_target
        for i in range(remaining):
            chunk_selection_targets[i % num_chunks] += 1

    # Process each chunk with boundary-aware constraints and look-ahead optimization
    all_selected: list[int] = []
    prev_state: _ChunkBoundaryState | None = None

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * MAX_CHUNK_SIZE
        chunk_end = min((chunk_idx + 1) * MAX_CHUNK_SIZE, n)
        chunk_prices = list(prices[chunk_start:chunk_end])
        chunk_len = len(chunk_prices)

        # Get next chunk prices for look-ahead optimization
        next_chunk_prices: list[Decimal] | None = None
        if chunk_idx + 1 < num_chunks:
            next_chunk_start = (chunk_idx + 1) * MAX_CHUNK_SIZE
            next_chunk_end = min((chunk_idx + 2) * MAX_CHUNK_SIZE, n)
            next_chunk_prices = list(prices[next_chunk_start:next_chunk_end])

        # Determine forced prefix selections and adjusted constraints
        forced_prefix_length = 0
        adjusted_max_gap_start = (
            max_gap_from_start if chunk_idx == 0 else max_gap_between_periods
        )

        if prev_state is not None:
            # Handle incomplete consecutive block from previous chunk
            if (
                prev_state.ended_with_selected
                and 0 < prev_state.trailing_selected_count < min_consecutive_periods
            ):
                # Must continue the block
                forced_prefix_length = min(
                    min_consecutive_periods - prev_state.trailing_selected_count,
                    chunk_len,
                )

            # Adjust max_gap_from_start based on trailing unselected
            if prev_state.trailing_unselected_count > 0:
                adjusted_max_gap_start = max(
                    0, max_gap_between_periods - prev_state.trailing_unselected_count
                )

        # Calculate target selections for this chunk
        target = chunk_selection_targets[chunk_idx]
        target = max(target, min_consecutive_periods)  # At least min_consecutive
        target = max(target, forced_prefix_length)  # At least forced prefix
        target = min(target, chunk_len)  # Can't exceed chunk size

        # Handle forced prefix selections
        forced_selections = list(range(forced_prefix_length))

        if forced_prefix_length >= chunk_len:
            # Entire chunk is forced to be selected
            chunk_selected = list(range(chunk_len))
        elif forced_prefix_length > 0:
            # Some items forced, process the rest with look-ahead
            remaining_start = forced_prefix_length
            remaining_prices = chunk_prices[remaining_start:]
            remaining_target = max(
                min_consecutive_periods, target - forced_prefix_length
            )
            remaining_target = min(remaining_target, len(remaining_prices))

            if len(remaining_prices) > 0 and remaining_target > 0:
                # Use look-ahead for the remaining portion
                remaining_selected = _find_best_chunk_selection_with_lookahead(
                    remaining_prices,
                    next_chunk_prices,
                    low_price_threshold,
                    remaining_target,
                    min_consecutive_periods,
                    max_gap_between_periods,
                    max_gap_between_periods,  # After forced prefix, gap from start is reset
                    aggressive,
                )
                # Offset remaining selections and combine with forced
                chunk_selected = forced_selections + [
                    i + remaining_start for i in remaining_selected
                ]
            else:
                chunk_selected = forced_selections
        else:
            # No forced prefix - use look-ahead optimization to find best selection
            chunk_selected = _find_best_chunk_selection_with_lookahead(
                chunk_prices,
                next_chunk_prices,
                low_price_threshold,
                target,
                min_consecutive_periods,
                max_gap_between_periods,
                adjusted_max_gap_start,
                aggressive,
            )

        # Convert to global indices and add to result
        for local_idx in chunk_selected:
            all_selected.append(chunk_start + local_idx)

        # Update boundary state for next chunk
        prev_state = _calculate_chunk_boundary_state(chunk_selected, chunk_len)

    # Sort and validate the final result
    all_selected = sorted(set(all_selected))

    # Validate the complete selection meets all constraints
    if not _validate_full_selection(
        all_selected,
        n,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
    ):
        # If validation fails, try a repair pass
        all_selected = _repair_selection(
            all_selected,
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
        )

    return all_selected


def _repair_selection(
    selected: list[int],
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> list[int]:
    """
    Repair a selection that doesn't meet constraints by adding necessary items.
    """
    n = len(prices)
    result = sorted(set(selected))

    # Ensure we have at least min_selections
    while len(result) < min_selections:
        # Add cheapest unselected item
        unselected = [i for i in range(n) if i not in result]
        if not unselected:
            break
        cheapest = min(unselected, key=lambda i: prices[i])
        result.append(cheapest)
        result = sorted(result)

    # Fix gap from start
    while result and result[0] > max_gap_from_start:
        # Add items before first selection
        for i in range(result[0] - 1, -1, -1):
            result.insert(0, i)
            if result[0] <= max_gap_from_start:
                break

    # Fix gaps between selections
    i = 1
    while i < len(result):
        gap = result[i] - result[i - 1] - 1
        if gap > max_gap_between_periods:
            # Fill the gap with cheapest items
            gap_items = list(range(result[i - 1] + 1, result[i]))
            gap_items.sort(key=lambda x: prices[x])
            # Add enough items to fix the gap
            items_needed = gap - max_gap_between_periods
            for item in gap_items[:items_needed]:
                result.append(item)
            result = sorted(result)
        else:
            i += 1

    # Fix gap at end
    while result and (n - 1 - result[-1]) > max_gap_between_periods:
        result.append(result[-1] + 1)

    # Fix consecutive block lengths
    result = sorted(set(result))
    i = 0
    while i < len(result):
        # Find current block
        block_start = i
        while i + 1 < len(result) and result[i + 1] == result[i] + 1:
            i += 1
        block_end = i
        block_length = block_end - block_start + 1

        # Check if this is not the last block and is too short
        is_last_block = block_end == len(result) - 1
        is_at_sequence_end = result[block_end] == n - 1

        if block_length < min_consecutive_periods and not (
            is_last_block and is_at_sequence_end
        ):
            # Extend the block
            items_needed = min_consecutive_periods - block_length
            block_start_idx = result[block_start]
            block_end_idx = result[block_end]

            # Try extending forward first
            for j in range(items_needed):
                next_idx = block_end_idx + 1 + j
                if next_idx < n and next_idx not in result:
                    result.append(next_idx)

            # If still short, extend backward
            result = sorted(set(result))
            # Recalculate block
            new_block_start = block_start
            while (
                new_block_start + 1 < len(result)
                and result[new_block_start + 1] == result[new_block_start] + 1
            ):
                new_block_start += 1
            new_block_length = new_block_start - block_start + 1

            if new_block_length < min_consecutive_periods:
                items_still_needed = min_consecutive_periods - new_block_length
                for j in range(items_still_needed):
                    prev_idx = block_start_idx - 1 - j
                    if prev_idx >= 0 and prev_idx not in result:
                        result.append(prev_idx)

            result = sorted(set(result))

        i += 1

    return sorted(set(result))


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
        return _get_cheapest_periods_extended(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )

    # Use direct brute-force for shorter sequences
    return _get_cheapest_periods(
        prices,
        low_price_threshold,
        min_selections,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
        aggressive,
    )
