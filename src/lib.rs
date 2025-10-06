use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_decimal::Decimal;
use std::collections::HashSet;

/// Check if a combination of price items is valid according to the constraints
fn is_valid_combination(
    combination: &[(usize, Decimal)],
    min_period: usize,
    max_gap: usize,
    max_start_gap: usize,
    full_length: usize,
) -> bool {
    if combination.is_empty() {
        return false;
    }

    // Items are already sorted, so indices are in order
    let indices: Vec<usize> = combination.iter().map(|(index, _)| *index).collect();

    // Check max_start_gap first (fastest check)
    if indices[0] > max_start_gap {
        return false;
    }

    // Check start gap
    if indices[0] > max_gap {
        return false;
    }

    // Check gaps between consecutive indices and min_period in single pass
    let mut block_length = 1;
    for i in 1..indices.len() {
        let gap = indices[i] - indices[i - 1] - 1;
        if gap > max_gap {
            return false;
        }

        if indices[i] == indices[i - 1] + 1 {
            block_length += 1;
        } else {
            if block_length < min_period {
                return false;
            }
            block_length = 1;
        }
    }

    // Check last block min_period
    if block_length < min_period {
        return false;
    }

    // Check end gap
    if (full_length - 1 - indices[indices.len() - 1]) > max_gap {
        return false;
    }

    true
}

/// Calculate the total cost of a combination
fn get_combination_cost(combination: &[(usize, Decimal)]) -> Decimal {
    combination.iter().map(|(_, price)| *price).sum()
}

/// Find the cheapest periods in a sequence of prices
#[pyfunction]
fn get_cheapest_periods(
    _py: Python,
    price_data: &Bound<'_, PyList>,
    price_threshold: &str,
    desired_count: usize,
    min_period: usize,
    max_gap: usize,
    max_start_gap: usize,
) -> PyResult<Vec<usize>> {
    if max_start_gap > max_gap {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_start_gap must be less than or equal to max_gap",
        ));
    }

    // Convert Python list to Vec<Decimal>
    let price_data: Vec<Decimal> = price_data
        .iter()
        .map(|item| {
            let decimal_str = item.extract::<String>()?;
            decimal_str
                .parse::<Decimal>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid decimal"))
        })
        .collect::<PyResult<Vec<Decimal>>>()?;

    let price_threshold: Decimal = price_threshold
        .parse::<Decimal>()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid decimal"))?;

    let price_items: Vec<(usize, Decimal)> = price_data.into_iter().enumerate().collect();

    let cheap_items: Vec<(usize, Decimal)> = price_items
        .iter()
        .filter(|(_, price)| *price <= price_threshold)
        .cloned()
        .collect();

    let actual_count = std::cmp::max(desired_count, cheap_items.len());

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
                min_period,
                max_gap,
                max_start_gap,
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

    // Merge cheap_items with cheapest_price_item_combination, adding any items from cheap_items not already present
    let mut merged_combination = cheapest_price_item_combination;
    let existing_indices: HashSet<usize> = merged_combination.iter().map(|(i, _)| *i).collect();

    for item in cheap_items {
        if !existing_indices.contains(&item.0) {
            merged_combination.push(item);
        }
    }

    // Sort by index to maintain order
    merged_combination.sort_by_key(|(i, _)| *i);

    Ok(merged_combination.into_iter().map(|(i, _)| i).collect())
}

/// A Python module implemented in Rust.
#[pymodule]
fn spot_planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_cheapest_periods, m)?)?;
    Ok(())
}
