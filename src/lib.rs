use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_decimal::Decimal;
use std::collections::HashSet;

/// Check if a combination of indices is valid according to basic constraints (ignoring consecutive requirements)
fn is_valid_combination_relaxed(
    indices: &[usize],
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
    full_length: usize,
) -> bool {
    if indices.is_empty() {
        return false;
    }

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

    true
}

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
fn check_consecutive_runs(indices: &[usize], min_consecutive_selections: usize) -> bool {
    if indices.is_empty() {
        return false;
    }

    let mut current_run_length = 1;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] + 1 {
            current_run_length += 1;
        } else {
            if current_run_length < min_consecutive_selections {
                return false;
            }
            current_run_length = 1;
        }
    }

    // Check the last run
    current_run_length >= min_consecutive_selections
}

/// Check if a combination of price items is valid according to the constraints
fn is_valid_combination(
    combination: &[(usize, Decimal)],
    min_consecutive_selections: usize,
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
            if block_length < min_consecutive_selections {
                return false;
            }
            block_length = 1;
        }
    }

    // Check last block min_consecutive_selections
    if block_length < min_consecutive_selections {
        return false;
    }

    true
}

/// Calculate the total cost of a combination
fn get_combination_cost(combination: &[(usize, Decimal)]) -> Decimal {
    combination.iter().map(|(_, price)| *price).sum()
}

/// Calculate dynamic consecutive_selections based on heating requirements
fn calculate_dynamic_consecutive_selections(
    min_consecutive_selections: usize,
    max_consecutive_selections: usize,
    min_selections: usize,
    total_prices: usize,
    max_gap_between_periods: usize,
) -> usize {
    // Calculate percentage of min_selections relative to total prices
    let min_selections_percentage = min_selections as f64 / total_prices as f64;

    // Base calculation based on percentage rules:
    // - < 25% of total prices: use min_consecutive_selections
    // - > 75% of total prices: use max_consecutive_selections
    // - Between 25-75%: linear interpolation
    let base_consecutive = if min_selections_percentage <= 0.25 {
        min_consecutive_selections
    } else if min_selections_percentage >= 0.75 {
        max_consecutive_selections
    } else {
        // Linear interpolation between 25% and 75%
        // Map 0.25-0.75 to 0.0-1.0 for interpolation
        let interpolation_factor = (min_selections_percentage - 0.25) / (0.75 - 0.25);
        min_consecutive_selections
            + (interpolation_factor
                * (max_consecutive_selections - min_consecutive_selections) as f64)
                as usize
    };

    // Adjust based on gap between periods
    // Larger gaps mean more consecutive heating needed
    // Scale gap adjustment: larger gaps push toward max_consecutive_selections
    let gap_factor = (max_gap_between_periods as f64 / 10.0).min(1.0); // Normalize gap to 0-1
    let gap_adjustment =
        (gap_factor * (max_consecutive_selections - min_consecutive_selections) as f64) as usize;

    // Final calculation: base + gap adjustment
    let dynamic_consecutive = base_consecutive + gap_adjustment;

    // Ensure result is within bounds
    dynamic_consecutive
        .max(min_consecutive_selections)
        .min(max_consecutive_selections)
}

/// Find the cheapest periods in a sequence of prices
#[pyfunction]
#[pyo3(signature = (prices, low_price_threshold, min_selections, min_consecutive_selections, max_consecutive_selections, max_gap_between_periods=0, max_gap_from_start=0))]
fn get_cheapest_periods(
    _py: Python,
    prices: &Bound<'_, PyList>,
    low_price_threshold: &str,
    min_selections: usize,
    min_consecutive_selections: usize,
    max_consecutive_selections: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
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

    // If we have mostly cheap items, be more permissive with consecutive requirements
    let cheap_items_count = cheap_items.len();
    let cheap_percentage = cheap_items_count as f64 / prices.len() as f64;

    // Calculate dynamic consecutive_selections based on min/max bounds
    let actual_consecutive_selections = if cheap_percentage > 0.8 {
        // If more than 80% of items are cheap, use minimum consecutive requirement
        min_consecutive_selections
    } else {
        // Otherwise, use the dynamic calculation
        calculate_dynamic_consecutive_selections(
            min_consecutive_selections,
            max_consecutive_selections,
            min_selections,
            prices.len(),
            max_gap_between_periods,
        )
    };

    // Validate input parameters
    if prices.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prices cannot be empty",
        ));
    }

    if prices.len() > 29 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prices cannot contain more than 29 items",
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

    if min_consecutive_selections == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_consecutive_selections must be greater than 0",
        ));
    }

    if max_consecutive_selections == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_consecutive_selections must be greater than 0",
        ));
    }

    if min_consecutive_selections > max_consecutive_selections {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_consecutive_selections cannot be greater than max_consecutive_selections",
        ));
    }

    // Note: actual_consecutive_selections can be greater than min_selections
    // because consecutive_selections is per-block minimum, while min_selections is total minimum

    if max_gap_from_start > max_gap_between_periods {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_gap_from_start must be less than or equal to max_gap_between_periods",
        ));
    }

    // Start with min_selections as minimum, increment if no valid combination found
    let actual_count = min_selections;

    // Special case: if min_selections equals total items, return all of them
    if min_selections == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    // Special case: if all items are below threshold, return all of them
    if cheap_items.len() == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    // Implement the sophisticated Python algorithm with merging logic
    let mut best_result: Vec<usize> = Vec::new();
    let mut best_cost = get_combination_cost(&price_items);
    let mut found = false;

    // Try combinations starting from min_selections (matching Python logic)
    for current_count in min_selections..=price_items.len() {
        for price_item_combination in
            itertools::Itertools::combinations(price_items.iter().cloned(), current_count)
        {
            if !is_valid_combination(
                &price_item_combination,
                actual_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                continue;
            }

            // Start with this combination
            let mut result_indices: Vec<usize> =
                price_item_combination.iter().map(|(i, _)| *i).collect();
            let existing_indices: HashSet<usize> = result_indices.iter().cloned().collect();

            // Try every combination of cheap items that are not already included
            let available_cheap_items: Vec<(usize, Decimal)> = cheap_items
                .iter()
                .filter(|(i, _)| !existing_indices.contains(i))
                .cloned()
                .collect();

            // Group cheap items into consecutive runs for efficiency
            let cheap_groups = group_consecutive_items(&available_cheap_items);

            // Try every combination of consecutive groups (2^n instead of 2^20)
            let mut best_merged_result = result_indices.clone();
            let mut best_merged_cost = get_combination_cost(&price_item_combination);

            for group_mask in 1..(1 << cheap_groups.len()) {
                // Skip empty selection
                let mut merged_indices = result_indices.clone();

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
                if check_consecutive_runs(&merged_indices, actual_consecutive_selections) {
                    // Calculate average cost of this merged result
                    let merged_cost: Decimal =
                        merged_indices.iter().map(|&i| price_items[i].1).sum();
                    let merged_avg_cost = merged_cost / Decimal::from(merged_indices.len());

                    // Calculate average cost of current best
                    let best_avg_cost = best_merged_cost / Decimal::from(best_merged_result.len());

                    // Keep the result with lowest average cost
                    if merged_avg_cost < best_avg_cost {
                        best_merged_result = merged_indices;
                        best_merged_cost = merged_cost;
                    }
                }
            }

            // Use the best merged result
            let total_cost = best_merged_cost;
            let avg_cost = total_cost / Decimal::from(best_merged_result.len());

            // Calculate average cost of current best result
            let best_avg_cost = best_cost
                / Decimal::from(if best_result.is_empty() {
                    1
                } else {
                    best_result.len()
                });

            // Keep the result with lowest average cost
            if avg_cost < best_avg_cost {
                best_result = best_merged_result;
                best_cost = total_cost;
                found = true;
            }
        }

        if found {
            break;
        }
    }

    if !found {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "No valid combination found that satisfies the constraints for {} items",
            price_items.len()
        )));
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
