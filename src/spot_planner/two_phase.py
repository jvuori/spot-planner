"""Two-phase algorithm for handling longer price sequences (> 28 items).

This module implements a two-phase approach:
1. Rough planning: Create averages of price groups, run brute-force on averages
   to determine approximate distribution of selections.
2. Fine-grained planning: Process actual prices in chunks, using the rough plan
   to guide target selections per chunk, with boundary-aware constraint handling
   and look-ahead optimization.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

# Import the Rust implementation
# Note: The Rust extension module is part of the package itself, so we use
# a relative import. This is an exception to the fully-qualified import rule
# because compiled extensions that share the package name cannot be imported
# using fully qualified syntax from within the package.
try:
    from . import spot_planner as _rust_module  # type: ignore[import-untyped]

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# Import Python brute-force implementation
from spot_planner import brute_force


def _get_cheapest_periods(
    prices: Sequence[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int = 0,
    max_gap_from_start: int = 0,
    aggressive: bool = True,
) -> list[int]:
    """
    Internal dispatcher that chooses between Rust and Python brute-force implementations.

    This function validates input parameters and dispatches to either the Rust
    implementation (if available) or the Python fallback.

    Note: This function is only for sequences <= 28 items. For longer sequences,
    use get_cheapest_periods_extended().
    """
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
        return brute_force.get_cheapest_periods_python(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )


@dataclass
class ChunkBoundaryState:
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
) -> ChunkBoundaryState:
    """Calculate the boundary state after processing a chunk."""
    if not chunk_selected:
        return ChunkBoundaryState(
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
        return ChunkBoundaryState(
            ended_with_selected=True,
            trailing_selected_count=trailing_count,
            trailing_unselected_count=0,
        )
    else:
        return ChunkBoundaryState(
            ended_with_selected=False,
            trailing_selected_count=0,
            trailing_unselected_count=chunk_length - 1 - last_selected,
        )


def _estimate_forced_prefix_cost(
    next_chunk_prices: Sequence[Decimal],
    boundary_state: ChunkBoundaryState,
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

    # Special case: if target=0, return empty selection (skip this chunk)
    if target == 0:
        return []

    # Check if we can skip this chunk entirely if it's mostly expensive
    # and we can reach the next chunk within max_gap
    cheap_items_in_chunk = sum(1 for p in chunk_prices if p <= low_price_threshold)
    chunk_is_mostly_expensive = cheap_items_in_chunk < min_consecutive_periods
    
    if chunk_is_mostly_expensive and next_chunk_prices is not None:
        # Check if next chunk has cheap items we can select instead
        next_chunk_cheap_items = sum(1 for p in next_chunk_prices if p <= low_price_threshold)
        if next_chunk_cheap_items >= min_consecutive_periods:
            # Check if we can skip this chunk (gap from previous selection to next chunk)
            # This is a heuristic - if max_gap is large enough, we can likely skip
            # The actual gap check happens at the chunk boundary level
            if max_gap_between_periods >= chunk_len:
                # Can skip this expensive chunk - return empty selection
                return []

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

    # Strategy 0: Skip expensive chunk if max_gap allows and next chunk has cheap items
    cheap_items_in_chunk = sum(1 for p in chunk_prices if p <= low_price_threshold)
    chunk_is_mostly_expensive = cheap_items_in_chunk < min_consecutive_periods
    
    if chunk_is_mostly_expensive and next_chunk_prices is not None:
        # Check if next chunk has enough cheap items
        next_chunk_cheap_items = sum(1 for p in next_chunk_prices if p <= low_price_threshold)
        if next_chunk_cheap_items >= min_consecutive_periods:
            # Check if we can skip this chunk (gap constraint allows it)
            # If max_gap is at least chunk_len, we can skip the entire chunk
            if max_gap_between_periods >= chunk_len:
                # Try selecting from next chunk instead
                # This simulates skipping current chunk and selecting from next
                skip_selection = []  # Empty selection for current chunk
                # Estimate cost of selecting from next chunk
                # Find cheapest items in next chunk
                next_chunk_cheap_indices = [
                    i for i, p in enumerate(next_chunk_prices)
                    if p <= low_price_threshold
                ]
                next_chunk_cheap_indices.sort(key=lambda i: next_chunk_prices[i])
                # Take enough to form a valid block
                estimated_next_selection = next_chunk_cheap_indices[:min_consecutive_periods]
                estimated_next_cost = sum(
                    (next_chunk_prices[i] for i in estimated_next_selection), Decimal(0)
                )
                
                # Cost of skipping: 0 for current chunk + estimated cost for next
                skip_cost = estimated_next_cost
                
                if aggressive:
                    # For aggressive, compare average cost
                    # Since we're skipping current chunk, we need to compare with
                    # what we would have selected in current chunk
                    # Use estimated cost per item from next chunk
                    if estimated_next_selection:
                        avg_skip_cost = skip_cost / Decimal(len(estimated_next_selection))
                        candidates.append((skip_selection, avg_skip_cost))
                else:
                    # For conservative, prefer skipping expensive chunks
                    # Use (cheap_count=0 for current, but next has cheap items, low cost)
                    candidates.append((skip_selection, (0, skip_cost)))

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
            cheap_count = sum(
                1 for i in selection if chunk_prices[i] <= low_price_threshold
            )
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
                        cheap_count = sum(
                            1
                            for i in selection
                            if chunk_prices[i] <= low_price_threshold
                        )
                        candidates.append(
                            (selection, (cheap_count, complete_block_cost))
                        )
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
                    cheap_count = sum(
                        1 for i in selection if chunk_prices[i] <= low_price_threshold
                    )
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
                    cheap_count = sum(
                        1 for i in selection if chunk_prices[i] <= low_price_threshold
                    )
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
                    cheap_count = sum(
                        1 for i in selection if chunk_prices[i] <= low_price_threshold
                    )
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
            cheap_count = sum(
                1 for i in selection if chunk_prices[i] <= low_price_threshold
            )
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
    max_start_fixes = n  # Safety limit
    start_fix_count = 0
    while result and result[0] > max_gap_from_start and start_fix_count < max_start_fixes:
        start_fix_count += 1
        # Add items before first selection
        for i in range(result[0] - 1, -1, -1):
            result.insert(0, i)
            if result[0] <= max_gap_from_start:
                break

    # Fix gaps between selections
    i = 1
    max_gap_fixes = n * 2  # Safety limit (allows fixing multiple gaps)
    gap_fix_count = 0
    while i < len(result) and gap_fix_count < max_gap_fixes:
        gap = result[i] - result[i - 1] - 1
        if gap > max_gap_between_periods:
            gap_fix_count += 1
            # Fill the gap with cheapest items
            gap_items = list(range(result[i - 1] + 1, result[i]))
            gap_items.sort(key=lambda x: prices[x])
            # Add enough items to fix the gap
            items_needed = gap - max_gap_between_periods
            for item in gap_items[:items_needed]:
                result.append(item)
            result = sorted(result)
            # Reset i to 1 to re-check all gaps after adding items
            i = 1
        else:
            i += 1

    # Fix gap at end
    max_end_fixes = n  # Safety limit
    end_fix_count = 0
    while result and (n - 1 - result[-1]) > max_gap_between_periods and end_fix_count < max_end_fixes:
        end_fix_count += 1
        result.append(result[-1] + 1)
        # Safety check: ensure we don't exceed array bounds
        if result[-1] >= n:
            break

    # Fix consecutive block lengths
    result = sorted(set(result))
    i = 0
    max_iterations = len(result) * 20  # Safety limit to prevent infinite loops
    iteration_count = 0
    previous_result_hash = None
    stable_iterations = 0
    while i < len(result) and iteration_count < max_iterations:
        iteration_count += 1
        # Check for cycles: if result hasn't changed in several iterations, break
        current_result_hash = hash(tuple(result))
        if current_result_hash == previous_result_hash:
            stable_iterations += 1
            if stable_iterations > 5:
                # Result is stable, no more changes needed
                break
        else:
            stable_iterations = 0
            previous_result_hash = current_result_hash
        # Find current block
        block_start = i
        while i + 1 < len(result) and result[i + 1] == result[i] + 1:
            i += 1
        block_end = i
        block_length = block_end - block_start + 1
        block_start_idx = result[block_start]
        block_end_idx = result[block_end]

        # Check if this block is entirely expensive and we can skip it
        if block_length > 0:
            block_is_expensive = all(
                prices[result[j]] > low_price_threshold
                for j in range(block_start, block_end + 1)
            )
            
            if block_is_expensive:
                # Check if we can skip this expensive block
                # Find the previous block's end (the last selected item before this block)
                prev_cheap_idx = None
                if block_start > 0:
                    prev_block_end_idx = result[block_start - 1]
                    # The previous block ends at prev_block_end_idx
                    # Check if it's cheap, and if so, use it as the previous cheap reference
                    if prices[prev_block_end_idx] <= low_price_threshold:
                        prev_cheap_idx = prev_block_end_idx
                    else:
                        # Previous block is also expensive, look further back
                        for idx in range(prev_block_end_idx - 1, max(-1, prev_block_end_idx - max_gap_between_periods - 1), -1):
                            if idx >= 0 and idx in result and prices[idx] <= low_price_threshold:
                                prev_cheap_idx = idx
                                break
                
                # Look for next cheap item after this expensive block
                next_cheap_idx = None
                for idx in range(block_end_idx + 1, min(block_end_idx + 1 + max_gap_between_periods + 1, n)):
                    if prices[idx] <= low_price_threshold:
                        next_cheap_idx = idx
                        break
                
                # Check if we can skip: gap from prev to next is within max_gap
                if prev_cheap_idx is not None and next_cheap_idx is not None:
                    gap_prev_to_next = next_cheap_idx - prev_cheap_idx - 1
                    if gap_prev_to_next <= max_gap_between_periods:
                        # Can skip this expensive block entirely
                        # Remove all items in this block (collect indices first)
                        indices_to_remove = [result[j] for j in range(block_start, block_end + 1)]
                        for idx in indices_to_remove:
                            result.remove(idx)
                        result = sorted(set(result))
                        # Restart from beginning to re-check
                        i = 0
                        continue
                elif prev_cheap_idx is None and next_cheap_idx is not None:
                    # No previous block, but can skip to next cheap
                    gap_to_next = next_cheap_idx - block_end_idx - 1
                    if gap_to_next <= max_gap_between_periods:
                        # Remove this expensive block (collect indices first)
                        indices_to_remove = [result[j] for j in range(block_start, block_end + 1)]
                        for idx in indices_to_remove:
                            result.remove(idx)
                        result = sorted(set(result))
                        i = 0
                        continue

        # Check if this is not the last block and is too short
        is_last_block = block_end == len(result) - 1
        is_at_sequence_end = result[block_end] == n - 1

        if block_length < min_consecutive_periods and not (
            is_last_block and is_at_sequence_end
        ):
            # Extend the block, preferring cheap items
            items_needed = min_consecutive_periods - block_length
            block_start_idx = result[block_start]
            block_end_idx = result[block_end]

            # Check if we can skip an expensive region entirely
            # Look ahead to find the next cheap item after this block
            next_cheap_idx = None
            for idx in range(block_end_idx + 1, min(block_end_idx + 1 + max_gap_between_periods + 1, n)):
                if idx not in result and prices[idx] <= low_price_threshold:
                    next_cheap_idx = idx
                    break
            
            # If we found a cheap item within max_gap, check if we can skip expensive items
            if next_cheap_idx is not None:
                gap_to_cheap = next_cheap_idx - block_end_idx - 1
                if gap_to_cheap <= max_gap_between_periods:
                    # Check if items between block_end and next_cheap are all expensive
                    expensive_between = all(
                        prices[idx] > low_price_threshold
                        for idx in range(block_end_idx + 1, next_cheap_idx)
                        if idx not in result
                    )
                    if expensive_between:
                        # Skip expensive region - don't extend this block
                        # The gap is valid, so we can leave it as is
                        i += 1
                        continue

            # Collect candidate items for extension (forward and backward)
            forward_candidates = []
            backward_candidates = []
            
            # Forward candidates - prefer cheap items
            for j in range(items_needed * 3):  # Look ahead more to find cheap items
                next_idx = block_end_idx + 1 + j
                if next_idx < n and next_idx not in result:
                    forward_candidates.append(next_idx)
            
            # Backward candidates - prefer cheap items
            for j in range(items_needed * 3):  # Look back more to find cheap items
                prev_idx = block_start_idx - 1 - j
                if prev_idx >= 0 and prev_idx not in result:
                    backward_candidates.append(prev_idx)
            
            # Separate cheap and expensive candidates
            cheap_forward = [idx for idx in forward_candidates if prices[idx] <= low_price_threshold]
            expensive_forward = [idx for idx in forward_candidates if prices[idx] > low_price_threshold]
            cheap_backward = [idx for idx in backward_candidates if prices[idx] <= low_price_threshold]
            expensive_backward = [idx for idx in backward_candidates if prices[idx] > low_price_threshold]
            
            # Sort by price within each category
            cheap_forward.sort(key=lambda idx: prices[idx])
            expensive_forward.sort(key=lambda idx: prices[idx])
            cheap_backward.sort(key=lambda idx: prices[idx])
            expensive_backward.sort(key=lambda idx: prices[idx])
            
            # Prefer cheap items: first try cheap forward, then cheap backward, then expensive
            added = 0
            for candidate in cheap_forward + cheap_backward + expensive_forward + expensive_backward:
                if added >= items_needed:
                    break
                if candidate not in result:
                    result.append(candidate)
                    added += 1

            result = sorted(set(result))

        i += 1
    
    # If we hit the iteration limit, log a warning but return what we have
    if iteration_count >= max_iterations:
        # This should not happen in normal operation, but if it does,
        # we return the current result to avoid infinite loops
        pass

    return sorted(set(result))


def get_cheapest_periods_extended(
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
    # If we have more than 28 averages, chunk them for brute-force processing
    if len(averages) > 28:
        # Split into chunks of max 20 averages to stay under the 28-item limit
        ROUGH_CHUNK_SIZE = 20
        rough_selected = []
        
        for chunk_start_idx in range(0, len(averages), ROUGH_CHUNK_SIZE):
            chunk_end_idx = min(chunk_start_idx + ROUGH_CHUNK_SIZE, len(averages))
            chunk_averages = averages[chunk_start_idx:chunk_end_idx]
            
            # Calculate target for this chunk (proportional)
            chunk_target = max(1, (rough_min_selections * len(chunk_averages)) // len(averages))
            
            try:
                chunk_selected = _get_cheapest_periods(
                    chunk_averages,
                    low_price_threshold,
                    chunk_target,
                    rough_min_consecutive,
                    rough_max_gap,
                    rough_max_gap_start if chunk_start_idx == 0 else rough_max_gap,
                    aggressive,
                )
                # Offset indices back to global positions
                for idx in chunk_selected:
                    rough_selected.append(chunk_start_idx + idx)
            except ValueError:
                # If this chunk fails, select cheapest items from it
                sorted_chunk = sorted(
                    range(len(chunk_averages)),
                    key=lambda i: chunk_averages[i]
                )
                for idx in sorted_chunk[:min(chunk_target, len(chunk_averages))]:
                    rough_selected.append(chunk_start_idx + idx)
        
        rough_selected = sorted(rough_selected)
    else:
        # Original logic for <=28 averages
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
                # Last resort: select only cheap average groups
                rough_selected = [
                    i for i, avg in enumerate(averages) if avg <= low_price_threshold
                ]

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
    prev_state: ChunkBoundaryState | None = None

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
        # For the first chunk, if target=0 and max_gap_from_start allows waiting
        # until the next chunk, we can allow target=0 to skip expensive periods
        if chunk_idx == 0 and target == 0:
            # Check if we can wait until next chunk without violating max_gap_from_start
            if chunk_idx + 1 < num_chunks:
                next_chunk_start = (chunk_idx + 1) * MAX_CHUNK_SIZE
                if next_chunk_start <= max_gap_from_start:
                    # Can wait until next chunk - allow target=0
                    target = 0
                else:
                    # Must select something in this chunk to satisfy max_gap_from_start
                    target = max(target, min_consecutive_periods)
            else:
                # This is the last chunk, must select something
                target = max(target, min_consecutive_periods)
        else:
            # For non-first chunks or when target > 0, enforce min_consecutive
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
                # Only forced prefix - this should not happen if target enforcement worked correctly
                # But if it does, we need to ensure we meet min_consecutive_periods
                # The forced prefix should already be sized to complete the previous chunk's block
                # So forced_prefix_length should be sufficient, but let's verify
                if forced_prefix_length < min_consecutive_periods and len(remaining_prices) > 0:
                    # This is a problem - forced prefix is too short
                    # This shouldn't happen if the algorithm is working correctly
                    # But as a safety, try to extend it minimally
                    items_needed = min_consecutive_periods - forced_prefix_length
                    sorted_remaining = sorted(
                        range(len(remaining_prices)),
                        key=lambda i: remaining_prices[i]
                    )
                    # Only add the minimum needed, not more
                    additional = []
                    for i in range(min(items_needed, len(sorted_remaining))):
                        additional.append(remaining_start + sorted_remaining[i])
                    chunk_selected = forced_selections + additional
                    chunk_selected = sorted(set(chunk_selected))
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

        # Validate that chunk selection meets minimum requirements
        # If target was enforced to min_consecutive_periods, we must have at least that many
        if target >= min_consecutive_periods and len(chunk_selected) < min_consecutive_periods:
            # Selection is too small - try to extend it
            # This should not happen if the selection logic is working correctly,
            # but as a safety measure, add cheapest items to meet the minimum
            unselected_in_chunk = [
                i for i in range(chunk_len) if i not in chunk_selected
            ]
            if unselected_in_chunk:
                unselected_in_chunk.sort(key=lambda i: chunk_prices[i])
                items_needed = min_consecutive_periods - len(chunk_selected)
                for i in range(min(items_needed, len(unselected_in_chunk))):
                    chunk_selected.append(unselected_in_chunk[i])
                chunk_selected = sorted(set(chunk_selected))

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
