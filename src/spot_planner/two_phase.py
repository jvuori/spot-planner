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
) -> list[int]:
    """
    Get a valid chunk selection.
    
    Raises ValueError if no valid selection can be found with the given constraints.
    """
    return _get_cheapest_periods(
        chunk_prices,
        low_price_threshold,
        target,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
        aggressive,
    )


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

    # If there's no next chunk, just find the best selection for this chunk
    if next_chunk_prices is None:
        return _try_chunk_selection(
            chunk_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )

    # With look-ahead: try multiple strategies and pick the best combined cost
    # Store: (selection, cost_metric) where cost_metric depends on mode
    # For aggressive: average cost
    # For conservative: (cheap_count, total_cost) tuple for comparison
    candidates: list[tuple[list[int], Decimal | tuple[int, Decimal]]] = []

    # Strategy 1: Standard selection (locally optimal)
    try:
        selection = _try_chunk_selection(
            chunk_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
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
    except ValueError:
        # Strategy 1 failed, try other strategies
        pass

    # Strategy 2: Try to end with a complete block (avoid forcing next chunk)
    # This means selecting items up to the end of the chunk
    if target < chunk_len:
        # Try selecting more items to reach the end
        for extra in range(1, min(min_consecutive_periods + 1, chunk_len - target + 1)):
            extended_target = min(target + extra, chunk_len)
            try:
                selection = _try_chunk_selection(
                    chunk_prices,
                    low_price_threshold,
                    extended_target,
                    min_consecutive_periods,
                    max_gap_between_periods,
                    max_gap_from_start,
                    aggressive,
                )
            except ValueError:
                continue
            if max(selection) == chunk_len - 1:
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

        try:
            selection = _try_chunk_selection(
                adjusted_prices,
                low_price_threshold,
                target,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                aggressive,
            )
        except ValueError:
            selection = []
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

    # If no candidates, fail fast - cannot satisfy constraints
    if not candidates:
        raise ValueError(
            f"Cannot find valid selection in chunk with look-ahead: "
            f"target={target}, min_consecutive_periods={min_consecutive_periods}, "
            f"chunk_len={chunk_len}, "
            f"cheap_items={sum(1 for p in chunk_prices if p <= low_price_threshold)}"
        )

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
        # For rough planning with many items, use a heuristic instead of exhaustive search
        # The rough plan doesn't need to be perfect - it just needs to guide the fine-grained phase
        # If we have > 20 items and the search would expand exponentially, use a simpler approach
        if len(averages) > 20:
            # Simple heuristic: distribute selections proportionally across cheap groups
            # Find the cheapest items that are approximately equally distributed
            indexed_averages = list(enumerate(averages))
            # Use a greedy gap-aware forward-selection approach
            # This maintains connectivity while preferring cheap items
            rough_selected = []
            current_pos = -1  # Start before the first group
            indexed_averages.sort(key=lambda x: x[1])  # Sort by price
            
            while current_pos < len(averages) - 1:
                # Determine the search window: next rough_max_gap positions
                window_start = current_pos + 1
                window_end = min(current_pos + 1 + rough_max_gap, len(averages) - 1)
                
                # Special case for start: respect max_gap_from_start
                if current_pos == -1:
                    window_end = min(rough_max_gap_start, len(averages) - 1)
               
                # Find cheapest group in the window
                candidates_in_window = [
                    (idx, price) for idx, price in indexed_averages
                    if window_start <= idx <= window_end
                ]
                
                if not candidates_in_window:
                    # No candidates in window - need to expand search or stop
                    # Try to jump to the cheapest available group beyond the window
                    candidates_beyond = [
                        (idx, price) for idx, price in indexed_averages
                        if idx > window_end and idx not in rough_selected
                    ]
                    if candidates_beyond:
                        best_beyond = min(candidates_beyond, key=lambda x: x[1])
                        rough_selected.append(best_beyond[0])
                        current_pos = best_beyond[0]
                    else:
                        break  # Reached the end
                else:
                    # Select the cheapest in the window
                    best_in_window = min(candidates_in_window, key=lambda x: x[1])
                    if best_in_window[0] not in rough_selected:
                        rough_selected.append(best_in_window[0])
                    current_pos = best_in_window[0]
                
                # Safety: don't select more than reasonable amount
                if len(rough_selected) >= len(averages) // 2:
                    break
            
            rough_selected = sorted(set(rough_selected))
            
            # Ensure we selected at least the minimum
            if len(rough_selected) < rough_min_selections:
                # Add cheapest remaining groups
                remaining = [
                    (idx, price) for idx, price in indexed_averages
                    if idx not in rough_selected
                ]
                remaining.sort(key=lambda x: x[1])
                for idx, _ in remaining[:rough_min_selections - len(rough_selected)]:
                    rough_selected.append(idx)
                rough_selected = sorted(rough_selected)
            
            # CRITICAL: Ensure start gap constraint is respected
            # The first selected group must not exceed rough_max_gap_start
            if rough_selected and rough_selected[0] > rough_max_gap_start:
                # Need to add a selection near the start
                candidates_at_start = [
                    (idx, price) for idx, price in indexed_averages
                    if idx <= rough_max_gap_start and idx not in rough_selected
                ]
                if candidates_at_start:
                    # Add the cheapest near-start group
                    best_at_start = min(candidates_at_start, key=lambda x: x[1])
                    rough_selected.append(best_at_start[0])
                    rough_selected = sorted(rough_selected)
            
            # CRITICAL: Ensure end gap constraint is respected
            # The last selected average group must be close enough to the end
            if rough_selected:
                last_selected_avg = rough_selected[-1]
                # How far is the last selected group from the end (in average units)?
                gap_from_end_in_avgs = len(averages) - 1 - last_selected_avg
                # If gap exceeds rough_max_gap, need to add groups near the end
                if gap_from_end_in_avgs > rough_max_gap:
                    # Find cheapest groups near the end that would satisfy constraint
                    # Need to select something within rough_max_gap of the end
                    min_required_avg_idx = len(averages) - 1 - rough_max_gap
                    # Find cheapest average in the required range that's not already selected
                    candidates_near_end = [
                        (idx, price) for idx, price in indexed_averages
                        if idx >= min_required_avg_idx and idx not in rough_selected
                    ]
                    if candidates_near_end:
                        # Add the cheapest near-end group
                        best_near_end = min(candidates_near_end, key=lambda x: x[1])
                        rough_selected.append(best_near_end[0])
                        rough_selected = sorted(rough_selected)
            
            # CRITICAL: Bridge large gaps between selected groups
            # Ensure no gaps between selected groups exceed rough_max_gap
            if rough_selected:
                bridged_selected = []
                for idx in rough_selected:
                    if bridged_selected:
                        prev_idx = bridged_selected[-1]
                        gap = idx - prev_idx - 1
                        if gap > rough_max_gap:
                            # Need to bridge the gap - find cheapest group in the gap
                            candidates_in_gap = [
                                (i, price) for i, price in indexed_averages
                                if prev_idx < i < idx and i not in rough_selected and i not in bridged_selected
                            ]
                            if candidates_in_gap:
                                best_bridge = min(candidates_in_gap, key=lambda x: x[1])
                                bridged_selected.append(best_bridge[0])
                    bridged_selected.append(idx)
                rough_selected = sorted(bridged_selected)
        else:
            # For smaller sets, try to use the exact algorithm
            # If this fails, fall back to the heuristic (which is reasonable for rough planning)
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
                # For rough planning, if exact algorithm fails, use heuristic
                # This is acceptable because rough planning is just guidance
                indexed_averages = list(enumerate(averages))
                indexed_averages.sort(key=lambda x: x[1])
                rough_selected = sorted([idx for idx, _ in indexed_averages[:max(rough_min_selections, 1)]])
                
                # CRITICAL: Bridge large gaps ONCE
                bridged_selected = []
                for idx in rough_selected:
                    if bridged_selected:
                        prev_idx = bridged_selected[-1]
                        gap = idx - prev_idx - 1
                        if gap > rough_max_gap:
                            candidates_in_gap = [
                                (i, price) for i, price in indexed_averages
                                if prev_idx < i < idx and i not in rough_selected and i not in bridged_selected
                            ]
                            if candidates_in_gap:
                                best_bridge = min(candidates_in_gap, key=lambda x: x[1])
                                bridged_selected.append(best_bridge[0])
                    bridged_selected.append(idx)
                
                rough_selected = sorted(bridged_selected)
                
                # CRITICAL: Ensure end gap constraint is respected
                if rough_selected:
                    last_selected_avg = rough_selected[-1]
                    gap_from_end_in_avgs = len(averages) - 1 - last_selected_avg
                    if gap_from_end_in_avgs > rough_max_gap:
                        min_required_avg_idx = len(averages) - 1 - rough_max_gap
                        candidates_near_end = [
                            (i, price) for i, price in indexed_averages
                            if i >= min_required_avg_idx and i not in rough_selected
                        ]
                        if candidates_near_end:
                            best_near_end = min(candidates_near_end, key=lambda x: x[1])
                            rough_selected.append(best_near_end[0])
                            rough_selected = sorted(rough_selected)
                           
                            # Bridge the gap once
                            if len(rough_selected) >= 2:
                                last_main = rough_selected[-2]
                                near_end_idx = rough_selected[-1]
                                gap = near_end_idx - last_main - 1
                                if gap > rough_max_gap:
                                    candidates_in_gap = [
                                        (i, price) for i, price in indexed_averages
                                        if last_main < i < near_end_idx and i not in rough_selected
                                    ]
                                    if candidates_in_gap:
                                        best_bridge = min(candidates_in_gap, key=lambda x: x[1])
                                        rough_selected.append(best_bridge[0])
                                        rough_selected = sorted(rough_selected)

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
    
    # Ensure chunk 0 respects max_gap_from_start constraint
    # Rough planning works in average space, but max_gap_from_start is in price index space.
    # We only need to force selections in chunk 0 if:
    # 1. The first rough selected average starts at or beyond max_gap_from_start
    # 2. chunk 0 has no target from the rough plan
    if chunk_selection_targets[0] == 0 and rough_selected:
        first_rough_start = group_ranges[rough_selected[0]][0]  # Start price index of first rough selection
        if first_rough_start >= max_gap_from_start:
            chunk_selection_targets[0] = min_consecutive_periods

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

        # Calculate target selections for this chunk based on rough planning
        target = chunk_selection_targets[chunk_idx]
        
        # CRITICAL: If target > 0, it must be at least min_consecutive_periods
        # to form a valid consecutive block
        if target > 0:
            target = max(target, min_consecutive_periods)
        
        # For chunks with target=0, allow skipping if gap constraints permit
        if target == 0:
            if chunk_idx == 0 and chunk_idx + 1 < num_chunks:
                next_chunk_start = (chunk_idx + 1) * MAX_CHUNK_SIZE
                if next_chunk_start > max_gap_from_start:
                    # Can't skip - would violate max_gap_from_start
                    target = min(min_consecutive_periods, chunk_len)
        
        # Respect forced prefix requirement
        target = max(target, forced_prefix_length)
        
        # If the chunk is too small to form a valid consecutive block, skip it
        if chunk_len < min_consecutive_periods:
            target = 0
        else:
            # Can't exceed chunk size
            target = min(target, chunk_len)

        # Handle forced prefix selections
        forced_selections = list(range(forced_prefix_length))

        if forced_prefix_length >= chunk_len:
            # Entire chunk is forced to be selected
            chunk_selected = list(range(chunk_len))
        elif forced_prefix_length > 0:
            # Some items forced, process the rest with look-ahead
            remaining_start = forced_prefix_length
            remaining_prices = chunk_prices[remaining_start:]
            remaining_target = target - forced_prefix_length
            
            # CRITICAL: If remaining_target > 0, it must be at least min_consecutive_periods
            # to form a valid consecutive block in the remaining portion
            if remaining_target > 0:
                remaining_target = max(remaining_target, min_consecutive_periods)
            
            # Can't select more than what's available
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
                # remaining_target is 0 or remaining_prices is empty
                # Use only the forced prefix
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

    # CRITICAL: Bridge any remaining large gaps between chunks
    # This ensures gaps between different chunks don't exceed constraints
    if all_selected:
        bridged_selected = []
        for idx in all_selected:
            if bridged_selected:
                prev_idx = bridged_selected[-1]
                gap = idx - prev_idx - 1
                if gap > max_gap_between_periods:
                    # Need to bridge the gap by selecting periods in between
                    # Find the cheapest periods in the gap that maintain connectivity
                    gap_start = prev_idx + 1
                    gap_end = idx - 1
                    gap_prices = prices[gap_start:gap_end + 1]
                    
                    # Select the cheapest consecutive block of min_consecutive_periods in the gap
                    if len(gap_prices) >= min_consecutive_periods:
                        # Find the cheapest consecutive block
                        min_cost = float('inf')
                        best_start = None
                        for start in range(len(gap_prices) - min_consecutive_periods + 1):
                            cost = sum(gap_prices[start:start + min_consecutive_periods])
                            if cost < min_cost:
                                min_cost = cost
                                best_start = start
                        
                        if best_start is not None:
                            # Add the consecutive block
                            for i in range(min_consecutive_periods):
                                bridged_selected.append(gap_start + best_start + i)
            bridged_selected.append(idx)
        all_selected = sorted(bridged_selected)
    
    # Fix incomplete trailing and leading blocks
    if all_selected:
        # Fix trailing block
        trailing_block_length = 1
        for i in range(len(all_selected) - 1, 0, -1):
            if all_selected[i] == all_selected[i - 1] + 1:
                trailing_block_length += 1
            else:
                break
        
        if trailing_block_length < min_consecutive_periods:
            # Try to extend the trailing block backwards
            if len(all_selected) > trailing_block_length:
                block_start_idx = all_selected[-trailing_block_length]
                can_extend = True
                extension_indices = []
                for j in range(min_consecutive_periods - trailing_block_length):
                    extend_idx = block_start_idx - 1 - j
                    if extend_idx < 0 or extend_idx in set(all_selected):
                        can_extend = False
                        break
                    extension_indices.append(extend_idx)
                
                if can_extend:
                    all_selected.extend(extension_indices)
                    all_selected = sorted(set(all_selected))
                else:
                    # Can't extend - try removing the trailing block
                    # Only safe if the end gap doesn't exceed max_gap_between_periods
                    new_last_idx = all_selected[-trailing_block_length - 1]
                    gap_from_end = n - 1 - new_last_idx
                    if gap_from_end <= max_gap_between_periods:
                        all_selected = all_selected[:-trailing_block_length]
        
        # Fix leading block
        if all_selected:
            leading_block_length = 1
            for i in range(1, len(all_selected)):
                if all_selected[i] == all_selected[i - 1] + 1:
                    leading_block_length += 1
                else:
                    break
            
            if leading_block_length < min_consecutive_periods:
                # Try to extend the leading block forward
                block_end_idx = all_selected[leading_block_length - 1]
                can_extend = True
                extension_indices = []
                for j in range(min_consecutive_periods - leading_block_length):
                    extend_idx = block_end_idx + 1 + j
                    if extend_idx >= n or extend_idx in set(all_selected):
                        can_extend = False
                        break
                    extension_indices.append(extend_idx)
                
                if can_extend:
                    all_selected.extend(extension_indices)
                    all_selected = sorted(set(all_selected))

    # Validate the complete selection meets all constraints
    validation_result = _validate_full_selection(
        all_selected,
        n,
        min_consecutive_periods,
        max_gap_between_periods,
        max_gap_from_start,
    )
    if not validation_result:
        # Fail fast: do not attempt to repair constraint violations
        # Add detailed diagnostics
        diagnostics = []
        if all_selected:
            indices = sorted(all_selected)
            if indices[0] > max_gap_from_start:
                diagnostics.append(f"first_selection={indices[0]} > max_gap_from_start={max_gap_from_start}")
            if indices[0] > max_gap_between_periods:
                diagnostics.append(f"start_gap={indices[0]} > max_gap_between_periods={max_gap_between_periods}")
            end_gap = n - 1 - indices[-1]
            if end_gap > max_gap_between_periods:
                diagnostics.append(f"end_gap={end_gap} > max_gap_between_periods={max_gap_between_periods}")
            
            # Check gaps and blocks
            block_length = 1
            for i in range(1, len(indices)):
                gap = indices[i] - indices[i - 1] - 1
                if gap > max_gap_between_periods:
                    diagnostics.append(f"gap at {indices[i-1]}->{indices[i]} = {gap} > max_gap_between_periods={max_gap_between_periods}")
                if indices[i] == indices[i - 1] + 1:
                    block_length += 1
                else:
                    if block_length < min_consecutive_periods:
                        diagnostics.append(f"block ending at {indices[i-1]} has length {block_length} < min_consecutive_periods={min_consecutive_periods}")
                    block_length = 1
            if block_length < min_consecutive_periods:
                diagnostics.append(f"final block has length {block_length} < min_consecutive_periods={min_consecutive_periods}")
        
        diag_str = "; ".join(diagnostics) if diagnostics else "unknown"
        raise ValueError(
            f"Selection does not meet constraints ({diag_str}). "
            f"Total periods: {n}, Selections: {len(all_selected)}, "
            f"min_consecutive_periods: {min_consecutive_periods}, "
            f"max_gap_between_periods: {max_gap_between_periods}, "
            f"max_gap_from_start: {max_gap_from_start}"
        )

    return all_selected
