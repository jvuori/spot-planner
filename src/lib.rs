use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_decimal::Decimal;
use std::collections::HashSet;

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

    // Check max_gap_from_start first (fastest check)
    if indices[0] > max_gap_from_start {
        return false;
    }

    // Check start gap
    if indices[0] > max_gap_between_periods {
        return false;
    }

    // Check gaps between consecutive indices and min_consecutive_selections in single pass
    let mut block_length = 1;
    for i in 1..indices.len() {
        let gap = indices[i] - indices[i - 1] - 1;
        if gap > max_gap_between_periods {
            return false;
        }

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

    // Check end gap
    if (full_length - 1 - indices[indices.len() - 1]) > max_gap_between_periods {
        return false;
    }

    true
}

/// Calculate the total cost of a combination
fn get_combination_cost(combination: &[(usize, Decimal)]) -> Decimal {
    combination.iter().map(|(_, price)| *price).sum()
}

/// Check if all consecutive runs in indices meet the minimum length requirement
fn check_consecutive_runs(indices: &[usize], min_consecutive_selections: usize) -> bool {
    if indices.is_empty() {
        return false;
    }

    if indices.len() == 1 {
        return min_consecutive_selections <= 1;
    }

    // Count consecutive runs
    let mut run_length = 1;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] + 1 {
            run_length += 1;
        } else {
            // End of a run - check if it meets minimum
            if run_length < min_consecutive_selections {
                return false;
            }
            run_length = 1;
        }
    }

    // Check the last run
    if run_length < min_consecutive_selections {
        return false;
    }

    true
}

/// Find the cheapest periods in a sequence of prices
#[pyfunction]
fn get_cheapest_periods(
    _py: Python,
    prices: &Bound<'_, PyList>,
    low_price_threshold: &str,
    min_selections: usize,
    min_consecutive_selections: usize,
    max_gap_between_periods: usize,
    max_gap_from_start: usize,
) -> PyResult<Vec<usize>> {
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

    if min_consecutive_selections > min_selections {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_consecutive_selections cannot be greater than min_selections",
        ));
    }

    if max_gap_from_start > max_gap_between_periods {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_gap_from_start must be less than or equal to max_gap_between_periods",
        ));
    }

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

    let price_items: Vec<(usize, Decimal)> = prices.into_iter().enumerate().collect();

    let cheap_items: Vec<(usize, Decimal)> = price_items
        .iter()
        .filter(|(_, price)| *price <= low_price_threshold)
        .cloned()
        .collect();

    // Start with min_selections as minimum, increment if no valid combination found
    let actual_count = std::cmp::max(min_selections, cheap_items.len());

    // Special case: if min_selections equals total items, return all of them
    if min_selections == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    // Special case: if all items are below threshold, return all of them
    if cheap_items.len() == price_items.len() {
        return Ok((0..price_items.len()).collect());
    }

    let mut cheapest_price_item_combination: Vec<(usize, Decimal)> = Vec::new();
    let mut cheapest_cost = get_combination_cost(&price_items);

    // Generate all combinations of the required size
    let mut found = false;
    let mut current_count = actual_count;

    while !found && current_count <= price_items.len() {
        for combination in
            itertools::Itertools::combinations(price_items.iter().cloned(), current_count)
        {
            if !is_valid_combination(
                &combination,
                min_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
                price_items.len(),
            ) {
                continue;
            }

            let combination_cost = get_combination_cost(&combination);
            if combination_cost < cheapest_cost {
                cheapest_price_item_combination = combination.to_vec();
                cheapest_cost = combination_cost;
                found = true;
            }
        }

        if !found {
            current_count += 1;
            if current_count > price_items.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "No combination found for {} items",
                    current_count
                )));
            }
        }
    }

    // Merge cheap_items with cheapest_price_item_combination, but only if they maintain valid consecutive runs
    // Start with the valid combination found
    let mut result_indices: Vec<usize> = cheapest_price_item_combination
        .iter()
        .map(|(i, _)| *i)
        .collect();
    let mut existing_indices: HashSet<usize> = result_indices.iter().cloned().collect();

    // Try to add each cheap item that's not already included
    for item in cheap_items {
        if !existing_indices.contains(&item.0) {
            // Try adding this item and check if it maintains valid consecutive runs
            let mut test_indices = result_indices.clone();
            test_indices.push(item.0);
            test_indices.sort();

            if check_consecutive_runs(&test_indices, min_consecutive_selections) {
                result_indices.push(item.0);
                existing_indices.insert(item.0);
            }
        }
    }

    // Sort result by index
    result_indices.sort();

    Ok(result_indices)
}

/// A Python module implemented in Rust.
#[pymodule]
fn spot_planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_cheapest_periods, m)?)?;
    Ok(())
}
