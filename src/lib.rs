use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_decimal::Decimal;
use std::collections::HashSet;

/// Check if a combination of indices is valid according to basic constraints (ignoring consecutive requirements)

/// Group consecutive items into runs (matches Python _group_consecutive_items)
fn group_consecutive_items(items: &[(usize, Decimal)]) -> Vec<Vec<(usize, Decimal)>> {
    if items.is_empty() {
        return Vec::new();
    }

    let mut groups = Vec::new();
    let mut current_group = vec![items[0]];

    for i in 1..items.len() {
        if items[i].0 == items[i - 1].0 + 1 {
            // Consecutive item, add to current group
            current_group.push(items[i]);
        } else {
            // Non-consecutive item, start new group
            groups.push(current_group);
            current_group = vec![items[i]];
        }
    }
    groups.push(current_group);
    groups
}

/// Check if indices have valid consecutive runs (matches Python _check_consecutive_runs)
fn check_consecutive_runs(indices: &[usize], min_consecutive_periods: usize) -> bool {
    if indices.is_empty() {
        return false;
    }

    let mut current_run_length = 1;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] + 1 {
            current_run_length += 1;
        } else {
            if current_run_length < min_consecutive_periods {
                return false;
            }
            current_run_length = 1;
        }
    }

    // Check the last run
    current_run_length >= min_consecutive_periods
}

/// Check if a combination of price items is valid according to the constraints
fn is_valid_combination(
    combination: &[(usize, Decimal)],
    min_consecutive_periods: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
    full_length: usize,
) -> bool {
    if combination.is_empty() {
        return false;
    }

    // Items are already sorted, so indices are in order
    let indices: Vec<usize> = combination.iter().map(|(index, _)| *index).collect();

    // OPTIMIZED VALIDATION ORDER: Fastest + Most Selective First

    // 1. Fastest checks first (single array access)
    if indices[0] > max_gap_from_start {
        return false;
    }
    if indices[0] > max_gap_between_periods {
        return false;
    }
    if (full_length - 1 - indices[indices.len() - 1]) > max_gap_between_periods {
        return false;
    }

    // 2. Quick gap validation (fast loop, no complex logic)
    for i in 1..indices.len() {
        let gap = indices[i] - indices[i - 1] - 1;
        if gap > max_gap_between_periods {
            return false;
        }
    }

    // 3. Most expensive check last: consecutive requirements
    // Only do this if all other checks passed
    let mut block_length = 1;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] + 1 {
            block_length += 1;
        } else {
            if block_length < min_consecutive_periods {
                return false;
            }
            block_length = 1;
        }
    }

    // Check last block min_consecutive_periods
    if block_length < min_consecutive_periods {
        return false;
    }

    true
}

/// Calculate the total cost of a combination
fn get_combination_cost(combination: &[(usize, Decimal)]) -> Decimal {
    combination.iter().map(|(_, price)| *price).sum()
}

/// Find the cheapest periods in a sequence of prices
///
/// This algorithm selects periods (indices) from a price sequence to minimize cost
/// while satisfying various timing constraints. The algorithm prioritizes periods
/// with prices at or below the threshold, but still respects all constraints.
///
/// # Parameters
/// - `prices`: Sequence of prices for each period
/// - `low_price_threshold`: Price threshold below/equal to which periods are preferentially selected
/// - `min_selections`: Desired minimum number of periods to select (will be incremented if no valid solution found)
/// - `min_consecutive_periods`: Minimum consecutive periods required for each consecutive block
/// - `max_gap_between_periods`: Maximum gap allowed between selected periods
/// - `max_gap_from_start`: Maximum gap from start before first selection
/// - `aggressive`: Whether to use aggressive (cost-minimizing) or conservative (cheap-item-maximizing) strategy
///
/// # Algorithm
/// 1. Find cheapest main combination that meets constraints (with adaptive min_selections retry)
/// 2. Try adding cheap items to that combination, validating each merge
/// 3. Return the cheapest valid combination of main+cheap items
///
/// # Important Notes
/// - `min_consecutive_periods` is enforced as a minimum for each consecutive block
/// - The algorithm adaptively increases `min_selections` if no valid solution is found
///
/// # Returns
/// List of indices representing selected periods, sorted by index
#[pyfunction]
#[pyo3(signature = (prices, low_price_threshold, min_selections, min_consecutive_periods, max_gap_between_periods=0, max_gap_from_start=0, aggressive=true))]
fn get_cheapest_periods(
    _py: Python,
    prices: &Bound<'_, PyList>,
    low_price_threshold: &str,
    min_selections: usize,
    min_consecutive_periods: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
    aggressive: bool,
) -> PyResult<Vec<usize>> {
    // Convert Python list to Vec<Decimal>
    let prices: Vec<Decimal> = prices
        .iter()
        .map(|item| {
            let decimal_str = item.extract::<String>()?;
            decimal_str
                .parse::<Decimal>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid decimal"))
        })
        .collect::<PyResult<Vec<Decimal>>>()?;

    let low_price_threshold: Decimal = low_price_threshold
        .parse::<Decimal>()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid decimal"))?;

    let price_items: Vec<(usize, Decimal)> = prices.clone().into_iter().enumerate().collect();

    let cheap_items: Vec<(usize, Decimal)> = price_items
        .iter()
        .filter(|(_, price)| *price <= low_price_threshold)
        .cloned()
        .collect();

    // Validate input parameters
    if prices.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prices cannot be empty",
        ));
    }

    if prices.len() > 28 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prices cannot contain more than 28 items",
        ));
    }

    if min_selections == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_selections must be greater than 0",
        ));
    }

    if min_selections > prices.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_selections cannot be greater than total number of items",
        ));
    }

    if min_consecutive_periods == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_consecutive_periods must be greater than 0",
        ));
    }

    if max_gap_from_start > max_gap_between_periods {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_gap_from_start must be less than or equal to max_gap_between_periods",
        ));
    }

    // Start with min_selections as minimum, increment if no valid combination found
    let _actual_count = min_selections;

    // Special case: if min_selections equals total items, return all of them
    if min_selections == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    // Special case: if all items are below threshold, return all of them
    if cheap_items.len() == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    // Choose algorithm based on aggressive parameter
    if aggressive {
        // Aggressive mode: minimize average cost (current behavior)
        get_cheapest_periods_aggressive(
            &price_items,
            &cheap_items,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
        )
    } else {
        // Conservative mode: maximize number of cheap items
        get_cheapest_periods_conservative(
            &price_items,
            &cheap_items,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
        )
    }
}

/// Aggressive algorithm: minimize average cost by excluding some cheap items
fn get_cheapest_periods_aggressive(
    price_items: &[(usize, Decimal)],
    cheap_items: &[(usize, Decimal)],
    min_selections: usize,
    min_consecutive_periods: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
) -> PyResult<Vec<usize>> {
    // ALGORITHM FLOW:
    // 1. Find cheapest main combination that meets constraints
    // 2. If no valid solution found, increment min_selections by one and retry
    // 3. Add each combination of cheap items, validate if the result is ok
    // 4. Return the cheapest valid combination of main+cheap

    // Phase 1: Find cheapest main combination with adaptive min_selections
    let mut best_main_combination: Vec<usize> = Vec::new();
    let mut best_main_cost = get_combination_cost(price_items);
    let mut found_main = false;

    // Adaptive retry: try min_selections, then min_selections+1, etc. until found
    for current_count in min_selections..=price_items.len() {
        for price_item_combination in
            itertools::Itertools::combinations(price_items.iter().cloned(), current_count)
        {
            if !is_valid_combination(
                &price_item_combination,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                continue;
            }

            let combination_cost = get_combination_cost(&price_item_combination);
            if combination_cost < best_main_cost {
                best_main_combination = price_item_combination.iter().map(|(i, _)| *i).collect();
                best_main_cost = combination_cost;
                found_main = true;
            }
        }

        // Break as soon as we find any valid combination
        if found_main {
            break;
        }
    }

    if !found_main {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "No valid main combination found that satisfies the constraints for {} items",
            price_items.len()
        )));
    }

    // Phase 2: Try adding cheap items one by one to the best main combination
    let mut best_result = best_main_combination.clone();
    let mut best_cost = best_main_cost;
    let existing_indices: HashSet<usize> = best_main_combination.iter().cloned().collect();

    // Get available cheap items (not already in main combination)
    let available_cheap_items: Vec<(usize, Decimal)> = cheap_items
        .iter()
        .filter(|(i, _)| !existing_indices.contains(i))
        .cloned()
        .collect();

    // Group cheap items into consecutive runs for efficiency
    let cheap_groups = group_consecutive_items(&available_cheap_items);

    // Try every combination of consecutive groups (2^n instead of 2^20)
    for group_mask in 1..(1 << cheap_groups.len()) {
        let mut merged_indices = best_main_combination.clone();

        // Add items from selected groups
        for (group_idx, group) in cheap_groups.iter().enumerate() {
            if group_mask & (1 << group_idx) != 0 {
                for (index, _) in group {
                    merged_indices.push(*index);
                }
            }
        }

        merged_indices.sort();

        // Check if merged result maintains valid consecutive runs
        if check_consecutive_runs(&merged_indices, min_consecutive_periods) {
            // Create the merged combination for full validation
            let merged_combination: Vec<(usize, Decimal)> = merged_indices
                .iter()
                .map(|&i| (i, price_items[i].1))
                .collect();

            // Full validation including all constraints
            if is_valid_combination(
                &merged_combination,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                // Calculate total cost of this merged result
                let merged_cost: Decimal = merged_indices.iter().map(|&i| price_items[i].1).sum();

                // Keep the result with lowest total cost
                if merged_cost < best_cost {
                    best_result = merged_indices;
                    best_cost = merged_cost;
                }
            }
        }
    }

    // Sort result by index
    best_result.sort();

    Ok(best_result)
}

/// Conservative algorithm: maximize number of cheap items while respecting constraints
fn get_cheapest_periods_conservative(
    price_items: &[(usize, Decimal)],
    cheap_items: &[(usize, Decimal)],
    min_selections: usize,
    min_consecutive_periods: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
) -> PyResult<Vec<usize>> {
    // ALGORITHM FLOW (same as aggressive but with different scoring):
    // 1. Find cheapest main combination that meets constraints
    // 2. If no valid solution found, increment min_selections by one and retry
    // 3. Add each combination of cheap items, validate if the result is ok
    // 4. Return the cheapest valid combination of main+cheap

    // Phase 1: Find cheapest main combination with adaptive min_selections
    let mut best_main_combination: Vec<usize> = Vec::new();
    let mut best_main_cost = get_combination_cost(price_items);
    let mut found_main = false;

    // Adaptive retry: try min_selections, then min_selections+1, etc. until found
    for current_count in min_selections..=price_items.len() {
        for price_item_combination in
            itertools::Itertools::combinations(price_items.iter().cloned(), current_count)
        {
            if !is_valid_combination(
                &price_item_combination,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                continue;
            }

            let combination_cost = get_combination_cost(&price_item_combination);
            if combination_cost < best_main_cost {
                best_main_combination = price_item_combination.iter().map(|(i, _)| *i).collect();
                best_main_cost = combination_cost;
                found_main = true;
            }
        }

        // Break as soon as we find any valid combination
        if found_main {
            break;
        }
    }

    if !found_main {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "No valid main combination found that satisfies the constraints for {} items",
            price_items.len()
        )));
    }

    // Phase 2: Try adding cheap items one by one to the best main combination
    let mut best_result = best_main_combination.clone();
    let mut best_cost = best_main_cost;
    let existing_indices: HashSet<usize> = best_main_combination.iter().cloned().collect();

    // Get available cheap items (not already in main combination)
    let available_cheap_items: Vec<(usize, Decimal)> = cheap_items
        .iter()
        .filter(|(i, _)| !existing_indices.contains(i))
        .cloned()
        .collect();

    // Group cheap items into consecutive runs for efficiency
    let cheap_groups = group_consecutive_items(&available_cheap_items);

    // Try every combination of consecutive groups (2^n instead of 2^20)
    for group_mask in 1..(1 << cheap_groups.len()) {
        let mut merged_indices = best_main_combination.clone();

        // Add items from selected groups
        for (group_idx, group) in cheap_groups.iter().enumerate() {
            if group_mask & (1 << group_idx) != 0 {
                for (index, _) in group {
                    merged_indices.push(*index);
                }
            }
        }

        merged_indices.sort();

        // Check if merged result maintains valid consecutive runs
        if check_consecutive_runs(&merged_indices, min_consecutive_periods) {
            // Create the merged combination for full validation
            let merged_combination: Vec<(usize, Decimal)> = merged_indices
                .iter()
                .map(|&i| (i, price_items[i].1))
                .collect();

            // Full validation including all constraints
            if is_valid_combination(
                &merged_combination,
                min_consecutive_periods,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                // Calculate total cost of this merged result
                let merged_cost: Decimal = merged_indices.iter().map(|&i| price_items[i].1).sum();

                // For conservative mode, prioritize solutions with more cheap items
                let cheap_count = merged_indices
                    .iter()
                    .filter(|&&i| cheap_items.iter().any(|(j, _)| *j == i))
                    .count();

                let best_cheap_count = best_result
                    .iter()
                    .filter(|&&i| cheap_items.iter().any(|(j, _)| *j == i))
                    .count();

                // Prefer solutions with more cheap items, or if equal, lower cost
                if cheap_count > best_cheap_count
                    || (cheap_count == best_cheap_count && merged_cost < best_cost)
                {
                    best_result = merged_indices;
                    best_cost = merged_cost;
                }
            }
        }
    }

    // Sort result by index
    best_result.sort();

    Ok(best_result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn spot_planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_cheapest_periods, m)?)?;
    Ok(())
}
