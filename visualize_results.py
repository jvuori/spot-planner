#!/usr/bin/env python3
"""
Generate realistic test scenarios with 96 price items (15-min resolution, 24 hours)
and visualize the results with bar charts showing selected vs non-selected items.
"""

from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from spot_planner import get_cheapest_periods
from spot_planner import two_phase


def validate_constraints(
    selected_indices: list[int],
    total_length: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    min_selections: int,
) -> tuple[bool, str]:
    """
    Validate that selected indices meet all constraints.
    
    Returns:
        (is_valid, error_message): True if all constraints are met, False otherwise
    """
    if not selected_indices:
        return False, "No periods selected"
    
    # Check min_selections
    if len(selected_indices) < min_selections:
        return False, f"Selected {len(selected_indices)} periods, but min_selections is {min_selections}"
    
    # Check max_gap_from_start
    if selected_indices[0] > max_gap_from_start:
        return False, f"First selection at index {selected_indices[0]}, but max_gap_from_start is {max_gap_from_start}"
    
    # Check gaps between consecutive selections and consecutive runs
    consecutive_run_length = 1
    consecutive_runs = []
    
    for i in range(1, len(selected_indices)):
        gap = selected_indices[i] - selected_indices[i - 1] - 1
        
        # Check max_gap_between_periods
        if gap > max_gap_between_periods:
            return False, f"Gap of {gap} periods between index {selected_indices[i-1]} and {selected_indices[i]}, but max_gap_between_periods is {max_gap_between_periods}"
        
        # Track consecutive runs
        if selected_indices[i] == selected_indices[i - 1] + 1:
            consecutive_run_length += 1
        else:
            # End of a run
            consecutive_runs.append((selected_indices[i - consecutive_run_length], consecutive_run_length))
            consecutive_run_length = 1
    
    # Add the last run
    consecutive_runs.append((selected_indices[len(selected_indices) - consecutive_run_length], consecutive_run_length))
    
    # Check min_consecutive_periods for all runs except possibly the last one
    for i, (start_idx, run_length) in enumerate(consecutive_runs):
        is_last_run = (i == len(consecutive_runs) - 1)
        is_at_end = (start_idx + run_length - 1) == (total_length - 1)
        
        # Only enforce min_consecutive_periods if it's not the last run at the end
        if not (is_last_run and is_at_end):
            if run_length < min_consecutive_periods:
                return False, f"Consecutive run starting at index {start_idx} has length {run_length}, but min_consecutive_periods is {min_consecutive_periods}"
    
    # Check end gap
    end_gap = total_length - 1 - selected_indices[-1]
    if end_gap > max_gap_between_periods:
        return False, f"End gap of {end_gap} periods after index {selected_indices[-1]}, but max_gap_between_periods is {max_gap_between_periods}"
    
    return True, "All constraints met"


def generate_realistic_daily_pattern() -> list[Decimal]:
    """
    Generate a realistic daily electricity price pattern (96 items = 24 hours * 4).

    Typical pattern:
    - Night (00:00-06:00): Low prices
    - Morning (06:00-09:00): Rising prices, morning peak
    - Day (09:00-17:00): Moderate to high prices
    - Evening (17:00-21:00): Peak prices
    - Night (21:00-24:00): Decreasing prices
    """
    prices = []

    # 00:00-06:00 (0-23): Night, very cheap
    for _ in range(24):
        # Add some variation: 0.02-0.04
        price = Decimal(str(np.random.uniform(0.02, 0.04)))
        prices.append(price)

    # 06:00-09:00 (24-35): Morning, rising prices
    for i in range(12):
        # Rise from 0.05 to 0.15
        price = Decimal(str(np.random.uniform(0.05 + i * 0.008, 0.15)))
        prices.append(price)

    # 09:00-12:00 (36-47): Day, moderate-high
    for _ in range(12):
        price = Decimal(str(np.random.uniform(0.10, 0.18)))
        prices.append(price)

    # 12:00-17:00 (48-67): Afternoon, moderate
    for _ in range(20):
        price = Decimal(str(np.random.uniform(0.08, 0.14)))
        prices.append(price)

    # 17:00-21:00 (68-83): Evening peak, expensive
    for i in range(16):
        # Peak around 19:00
        peak_factor = 1.0 - abs(i - 8) / 8.0
        price = Decimal(str(np.random.uniform(0.15 + peak_factor * 0.10, 0.25)))
        prices.append(price)

    # 21:00-24:00 (84-95): Night, decreasing
    for i in range(12):
        # Decrease from 0.12 to 0.03
        price = Decimal(str(np.random.uniform(0.03, 0.12 - i * 0.007)))
        prices.append(price)

    return prices


def generate_cheap_day_pattern() -> list[Decimal]:
    """Generate a day with mostly cheap prices (windy/sunny day)."""
    prices = []
    for _ in range(96):
        # Mostly cheap with occasional spikes
        if np.random.random() < 0.15:  # 15% chance of expensive
            price = Decimal(str(np.random.uniform(0.15, 0.25)))
        else:
            price = Decimal(str(np.random.uniform(0.01, 0.08)))
        prices.append(price)
    return prices


def generate_expensive_day_pattern() -> list[Decimal]:
    """Generate a day with mostly expensive prices (cold winter day, low renewables)."""
    prices = []
    for _ in range(96):
        # Mostly expensive with occasional cheap periods
        if np.random.random() < 0.2:  # 20% chance of cheap
            price = Decimal(str(np.random.uniform(0.01, 0.05)))
        else:
            price = Decimal(str(np.random.uniform(0.12, 0.30)))
        prices.append(price)
    return prices


def generate_volatile_day_pattern() -> list[Decimal]:
    """Generate a day with high price volatility."""
    prices = []
    for _ in range(96):
        # High volatility: random between very cheap and very expensive
        price = Decimal(str(np.random.uniform(0.01, 0.30)))
        prices.append(price)
    return prices


def calculate_chunk_targets(
    prices: list[Decimal],
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool,
) -> tuple[list[int], int]:
    """
    Calculate chunk selection targets from phase 1 rough planning.
    Returns (chunk_targets, total_target) where chunk_targets is a list of targets per chunk.
    """
    n = len(prices)
    
    # Only calculate for sequences > 28 items (uses extended algorithm)
    if n <= 28:
        # For short sequences, return None to indicate no chunking
        return ([], min_selections)
    
    # Calculate optimal chunk size (same logic as algorithm)
    MAX_CHUNK_SIZE = two_phase._calculate_optimal_chunk_size(
        n, min_consecutive_periods, max_gap_between_periods
    )
    
    # Phase 1: Rough planning with averages
    AVERAGE_GROUP_SIZE = 4
    averages: list[Decimal] = []
    group_ranges: list[tuple[int, int]] = []
    
    for i in range(0, n, AVERAGE_GROUP_SIZE):
        group_end = min(i + AVERAGE_GROUP_SIZE, n)
        group = list(prices[i:group_end])
        group_sum = sum(group, Decimal(0))
        group_avg = group_sum / Decimal(len(group))
        averages.append(group_avg)
        group_ranges.append((i, group_end))
    
    # Scale parameters for rough planning
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
    if len(averages) > 28:
        # Split into chunks of max 20 averages
        ROUGH_CHUNK_SIZE = 20
        rough_selected = []
        
        for chunk_start_idx in range(0, len(averages), ROUGH_CHUNK_SIZE):
            chunk_end_idx = min(chunk_start_idx + ROUGH_CHUNK_SIZE, len(averages))
            chunk_averages = averages[chunk_start_idx:chunk_end_idx]
            
            chunk_target = max(1, (rough_min_selections * len(chunk_averages)) // len(averages))
            
            try:
                chunk_selected = two_phase._get_cheapest_periods(
                    chunk_averages,
                    low_price_threshold,
                    chunk_target,
                    rough_min_consecutive,
                    rough_max_gap,
                    rough_max_gap_start if chunk_start_idx == 0 else rough_max_gap,
                    aggressive,
                )
                for idx in chunk_selected:
                    rough_selected.append(chunk_start_idx + idx)
            except ValueError:
                sorted_chunk = sorted(
                    range(len(chunk_averages)),
                    key=lambda i: chunk_averages[i]
                )
                for idx in sorted_chunk[:min(chunk_target, len(chunk_averages))]:
                    rough_selected.append(chunk_start_idx + idx)
        
        rough_selected = sorted(rough_selected)
    else:
        try:
            rough_selected = two_phase._get_cheapest_periods(
                averages,
                low_price_threshold,
                rough_min_selections,
                rough_min_consecutive,
                rough_max_gap,
                rough_max_gap_start,
                aggressive,
            )
        except ValueError:
            try:
                rough_selected = two_phase._get_cheapest_periods(
                    averages,
                    low_price_threshold,
                    rough_min_selections,
                    1,
                    len(averages),
                    len(averages),
                    aggressive,
                )
            except ValueError:
                rough_selected = [
                    i for i, avg in enumerate(averages) if avg <= low_price_threshold
                ]
    
    # Phase 2: Calculate target selections per chunk based on rough plan
    num_chunks = (n + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
    chunk_selection_targets: list[int] = [0] * num_chunks
    
    for avg_idx in rough_selected:
        start_price_idx, end_price_idx = group_ranges[avg_idx]
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * MAX_CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * MAX_CHUNK_SIZE, n)
            
            overlap_start = max(start_price_idx, chunk_start)
            overlap_end = min(end_price_idx, chunk_end)
            
            if overlap_start < overlap_end:
                chunk_selection_targets[chunk_idx] += overlap_end - overlap_start
    
    # Ensure we have enough total selections
    total_target = sum(chunk_selection_targets)
    if total_target < min_selections:
        remaining = min_selections - total_target
        for i in range(remaining):
            chunk_selection_targets[i % num_chunks] += 1
        total_target = min_selections
    
    # Apply the same enforcement logic as the algorithm (matching two_phase.py lines 1027-1048)
    # This ensures targets shown in visualization match what the algorithm actually uses
    enforced_targets = []
    for chunk_idx in range(num_chunks):
        target = chunk_selection_targets[chunk_idx]
        chunk_start = chunk_idx * MAX_CHUNK_SIZE
        chunk_end = min((chunk_idx + 1) * MAX_CHUNK_SIZE, n)
        chunk_len = chunk_end - chunk_start
        
        # Apply enforcement logic (matching two_phase.py lines 1031-1048)
        if chunk_idx == 0 and target == 0:
            # First chunk with target=0 - check if we can wait
            if chunk_idx + 1 < num_chunks:
                next_chunk_start = (chunk_idx + 1) * MAX_CHUNK_SIZE
                if next_chunk_start <= max_gap_from_start:
                    # Can wait - keep target=0
                    enforced_target = 0
                else:
                    # Must select something
                    enforced_target = max(target, min_consecutive_periods)
            else:
                # Last chunk, must select something
                enforced_target = max(target, min_consecutive_periods)
        else:
            # For non-first chunks or when target > 0, enforce min_consecutive
            enforced_target = max(target, min_consecutive_periods)
        
        # Note: We don't apply forced_prefix_length here since that depends on
        # the previous chunk's boundary state, which we can't know in advance.
        # The visualization shows the base enforced target, which is what phase 1
        # produces after enforcement, before forced prefix adjustments.
        
        enforced_target = min(enforced_target, chunk_len)  # Can't exceed chunk size
        enforced_targets.append(enforced_target)

    return (enforced_targets, sum(enforced_targets))


def generate_peak_valley_pattern() -> list[Decimal]:
    """Generate a clear peak-valley pattern."""
    prices = []
    for i in range(96):
        # Create clear valleys at night, peaks during day
        hour = i / 4.0
        if 2 <= hour <= 6 or 22 <= hour <= 24:
            # Night valleys: very cheap
            price = Decimal(str(np.random.uniform(0.01, 0.03)))
        elif 8 <= hour <= 10 or 17 <= hour <= 19:
            # Peak hours: expensive
            price = Decimal(str(np.random.uniform(0.20, 0.30)))
        else:
            # Moderate
            price = Decimal(str(np.random.uniform(0.05, 0.15)))
        prices.append(price)
    return prices


def visualize_scenario(
    prices: list[Decimal],
    selected_indices: list[int],
    title: str,
    filename: str,
    output_dir: Path,
    low_price_threshold: Decimal,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
    aggressive: bool = True,
):
    """Create a bar chart visualization of selected vs non-selected items."""
    n = len(prices)
    hours = np.arange(n) / 4.0  # Convert 15-min intervals to hours

    # Convert prices to float for plotting
    price_values = [float(p) for p in prices]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Create arrays for selected and non-selected
    selected_prices = []
    non_selected_prices = []
    selected_hours = []
    non_selected_hours = []

    for i in range(n):
        if i in selected_indices:
            selected_prices.append(price_values[i])
            selected_hours.append(hours[i])
        else:
            non_selected_prices.append(price_values[i])
            non_selected_hours.append(hours[i])

    # Plot non-selected items in light gray
    if non_selected_hours:
        ax.bar(
            non_selected_hours,
            non_selected_prices,
            width=0.25,
            color="lightgray",
            alpha=0.6,
            label="Not Selected",
            edgecolor="gray",
            linewidth=0.5,
        )

    # Plot selected items in green
    if selected_hours:
        ax.bar(
            selected_hours,
            selected_prices,
            width=0.25,
            color="green",
            alpha=0.8,
            label="Selected",
            edgecolor="darkgreen",
            linewidth=0.5,
        )

    # Add threshold line
    ax.axhline(
        y=float(low_price_threshold),
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({low_price_threshold})",
    )

    # Calculate chunk boundaries and targets (only for sequences > 28 items)
    chunk_boundaries = []
    chunk_targets = []
    total_target = min_selections
    chunk_size = 0
    
    if n > 28:
        # Calculate chunk targets from phase 1
        chunk_targets, total_target = calculate_chunk_targets(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
        
        # Calculate optimal chunk size using same logic as _calculate_optimal_chunk_size
        if n <= 48:
            base_size = 20
        elif n <= 96:
            base_size = 18
        elif n <= 192:
            base_size = 15
        else:
            base_size = 12

        # Adjust based on constraints
        if max_gap_between_periods >= 15:
            base_size = max(12, base_size - 2)
        elif max_gap_between_periods >= 10:
            base_size = max(12, base_size - 1)

        if min_consecutive_periods >= 6:
            base_size = min(24, base_size + 2)
        elif min_consecutive_periods >= 4:
            base_size = min(24, base_size + 1)

        chunk_size = max(10, min(24, base_size))

        # Calculate chunk boundaries
        # Position boundaries between bars, not at bar centers
        # Each bar is 0.25 hours wide (15 minutes), so boundary between index i-1 and i
        # should be at (i - 0.5) / 4.0 hours, or equivalently (i * 0.25 - 0.125)
        for chunk_start in range(chunk_size, n, chunk_size):
            # Position boundary between the last bar of previous chunk and first bar of this chunk
            chunk_boundary_hour = (chunk_start - 0.5) / 4.0  # Convert to hours, positioned between bars
            chunk_boundaries.append(chunk_boundary_hour)

    # Add chunk boundary lines and target numbers
    for i, boundary_hour in enumerate(chunk_boundaries):
        ax.axvline(
            x=boundary_hour,
            color="blue",
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
            label="Chunk boundary" if i == 0 else "",
        )
    
    
    # Add phase 1 target numbers inside chart area (after legend is created)
    if chunk_targets and chunk_size > 0:
        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        
        # Get legend position to position all numbers below it
        legend = ax.get_legend()
        legend_y_bottom = y_max * 0.7  # Default fallback if no legend
        
        if legend is not None:
            # Get legend bounding box in figure coordinates
            legend_bbox = legend.get_window_extent()
            # Convert to axes coordinates, then to data coordinates
            legend_bbox_axes = legend_bbox.transformed(ax.transAxes.inverted())
            # Convert y position to data coordinates
            legend_y_bottom = legend_bbox_axes.y0 * (y_max - y_min) + y_min
        
        # Position all target numbers below the legend box
        # Place them 8% below the legend bottom, but ensure they're visible
        text_y_position = legend_y_bottom - (y_max - y_min) * 0.08
        
        # Make sure it's still visible and not too low
        if text_y_position < y_min + (y_max - y_min) * 0.1:
            text_y_position = y_min + (y_max - y_min) * 0.1  # At least 10% above bottom
        
        for chunk_idx, target in enumerate(chunk_targets):
            # Calculate chunk center position
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, n)
            chunk_center_hour = ((chunk_start + chunk_end) / 2.0) / 4.0
            
            # Count actual selections in this chunk
            actual_selections = sum(1 for idx in selected_indices if chunk_start <= idx < chunk_end)
            
            # Use different color based on target vs actual
            if target == 0:
                color = "gray"  # No target
            elif actual_selections >= target:
                color = "blue"  # Target met or exceeded
            else:
                color = "orange"  # Target not met
            
            # Display target number with larger font (increased from 9 to 18)
            # All numbers positioned below legend
            ax.text(
                chunk_center_hour,
                text_y_position,
                f"T:{target}",
                ha="center",
                va="center",
                fontsize=18,
                color=color,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1.5),
                zorder=10,  # Draw on top
            )

    # Formatting
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Price (€/kWh)", fontsize=12)
    # Build title with explanation
    title_parts = [f"{title}"]
    
    # Build title with desired and selected count and parameters
    if n > 28:
        title_parts.append(
            f"Desired: {min_selections} | "
            f"Selected: {len(selected_indices)} periods | "
            f"Min consecutive: {min_consecutive_periods} | "
            f"Max gap: {max_gap_between_periods}"
        )
    else:
        title_parts.append(
            f"Desired: {min_selections} | "
            f"Selected: {len(selected_indices)} periods | "
            f"Min consecutive: {min_consecutive_periods} | "
            f"Max gap: {max_gap_between_periods}"
        )
    
    ax.set_title(
        "\n".join(title_parts),
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set x-axis to show hours - dynamically based on number of items
    max_hours = n / 4.0  # Total hours in the data
    # Round up to nearest even number for nice tick spacing
    max_hours_rounded = int(np.ceil(max_hours / 2.0)) * 2
    
    # Set x-axis ticks - show every 2 hours, up to the max
    tick_spacing = 2
    max_tick = int(np.ceil(max_hours_rounded / tick_spacing)) * tick_spacing
    ax.set_xticks(np.arange(0, max_tick + tick_spacing, tick_spacing))
    ax.set_xlim(-0.5, max_hours + 0.5)

    # Add hour markers every 6 hours
    for hour in range(0, int(max_hours_rounded) + 6, 6):
        if hour <= max_hours:
            ax.axvline(x=hour, color="black", linestyle=":", alpha=0.2, linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Generate test scenarios and visualize results."""
    # Create output directory
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)

    # Test scenarios
    scenarios = [
        {
            "name": "realistic_daily",
            "prices": [
                Decimal("0.02749080237694725"),
                Decimal("0.03901428612819832"),
                Decimal("0.0346398788362281"),
                Decimal("0.03197316968394073"),
                Decimal("0.02312037280884873"),
                Decimal("0.023119890406724054"),
                Decimal("0.021161672243363988"),
                Decimal("0.0373235229154987"),
                Decimal("0.03202230023486417"),
                Decimal("0.03416145155592091"),
                Decimal("0.02041168988591605"),
                Decimal("0.03939819704323989"),
                Decimal("0.03664885281600844"),
                Decimal("0.024246782213565524"),
                Decimal("0.023636499344142012"),
                Decimal("0.023668090197068677"),
                Decimal("0.026084844859190756"),
                Decimal("0.03049512863264476"),
                Decimal("0.028638900372842314"),
                Decimal("0.025824582803960838"),
                Decimal("0.03223705789444759"),
                Decimal("0.02278987721304084"),
                Decimal("0.025842892970704363"),
                Decimal("0.027327236865873836"),
                Decimal("0.09560699842170359"),
                Decimal("0.13023618844815726"),
                Decimal("0.08277259770130221"),
                Decimal("0.11308181731943448"),
                Decimal("0.12228419068261889"),
                Decimal("0.09278702476319986"),
                Decimal("0.1295923322988748"),
                Decimal("0.11350306144224083"),
                Decimal("0.11634185734747006"),
                Decimal("0.14856879504309334"),
                Decimal("0.1493126406614912"),
                Decimal("0.14770076817739752"),
                Decimal("0.12436910153386965"),
                Decimal("0.10781376912051072"),
                Decimal("0.15473864212097255"),
                Decimal("0.1352121994991681"),
                Decimal("0.10976305878758232"),
                Decimal("0.1396141528089016"),
                Decimal("0.10275108168921748"),
                Decimal("0.17274563216630257"),
                Decimal("0.12070239852800135"),
                Decimal("0.15300178274831855"),
                Decimal("0.12493688608715288"),
                Decimal("0.14160544169422487"),
                Decimal("0.11280261676059679"),
                Decimal("0.09109126733153163"),
                Decimal("0.13817507766587353"),
                Decimal("0.1265079694016669"),
                Decimal("0.13636993649385137"),
                Decimal("0.13368964102565895"),
                Decimal("0.11587399872866512"),
                Decimal("0.13531245410138704"),
                Decimal("0.08530955012311517"),
                Decimal("0.09175897174514872"),
                Decimal("0.08271363733463229"),
                Decimal("0.09951981984579586"),
                Decimal("0.10332063738136893"),
                Decimal("0.09628094190643376"),
                Decimal("0.12972425054911577"),
                Decimal("0.10140519960161537"),
                Decimal("0.09685607058124285"),
                Decimal("0.11256176498949491"),
                Decimal("0.08845545349848577"),
                Decimal("0.12813181884524238"),
                Decimal("0.15745506436797707"),
                Decimal("0.24885260695254527"),
                Decimal("0.2329183576972493"),
                Decimal("0.19991973009588576"),
                Decimal("0.20027610585618014"),
                Decimal("0.24307980356705627"),
                Decimal("0.24267143359619042"),
                Decimal("0.24661258960051233"),
                Decimal("0.25"),
                Decimal("0.23842555814667613"),
                Decimal("0.23396164321360682"),
                Decimal("0.21684508973219235"),
                Decimal("0.24315517129377967"),
                Decimal("0.22645613292672237"),
                Decimal("0.19981735186394867"),
                Decimal("0.16806135565002708"),
                Decimal("0.0579884089544096"),
                Decimal("0.05699021572822"),
                Decimal("0.08545006955369286"),
                Decimal("0.0739914655235097"),
                Decimal("0.08500719003973224"),
                Decimal("0.0559718208839072"),
                Decimal("0.03574052380503848"),
                Decimal("0.05924303627614279"),
                Decimal("0.05586669165297452"),
                Decimal("0.045154484334376396"),
                Decimal("0.04541934359909121"),
                Decimal("0.036419342752737074"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.10"),
                "min_selections": 24,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "custom_test_data",
            "prices": [
                Decimal("4.78"),
                Decimal("4.252"),
                Decimal("3.869"),
                Decimal("3.721"),
                Decimal("3.792"),
                Decimal("3.697"),
                Decimal("3.593"),
                Decimal("3.476"),
                Decimal("7.152"),
                Decimal("4.211"),
                Decimal("4.133"),
                Decimal("3.687"),
                Decimal("4.084"),
                Decimal("3.875"),
                Decimal("3.66"),
                Decimal("3.503"),
                Decimal("3.712"),
                Decimal("3.637"),
                Decimal("3.335"),
                Decimal("3.048"),
                Decimal("2.896"),
                Decimal("3.182"),
                Decimal("3.131"),
                Decimal("3.119"),
                Decimal("2.727"),
                Decimal("2.938"),
                Decimal("3.195"),
                Decimal("3.488"),
                Decimal("2.6"),
                Decimal("3.028"),
                Decimal("3.559"),
                Decimal("4.321"),
                Decimal("2.301"),
                Decimal("3.21"),
                Decimal("4.29"),
                Decimal("5.699"),
                Decimal("4.5"),
                Decimal("6.147"),
                Decimal("8.161"),
                Decimal("9.924"),
                Decimal("5.945"),
                Decimal("6.764"),
                Decimal("7.731"),
                Decimal("8.374"),
                Decimal("7.501"),
                Decimal("7.962"),
                Decimal("9.852"),
                Decimal("10.999"),
                Decimal("6.248"),
                Decimal("7.505"),
                Decimal("9.999"),
                Decimal("10.825"),
                Decimal("8.566"),
                Decimal("8.603"),
                Decimal("8.221"),
                Decimal("8.346"),
                Decimal("7.832"),
                Decimal("8.211"),
                Decimal("7.507"),
                Decimal("7.253"),
                Decimal("11.3"),
                Decimal("11.517"),
                Decimal("12.154"),
                Decimal("13.057"),
                Decimal("11.606"),
                Decimal("12.523"),
                Decimal("14.322"),
                Decimal("15.832"),
                Decimal("12.529"),
                Decimal("13.219"),
                Decimal("14.048"),
                Decimal("14.923"),
                Decimal("13.182"),
                Decimal("14.6"),
                Decimal("14.995"),
                Decimal("14.643"),
                Decimal("15.967"),
                Decimal("16.106"),
                Decimal("15.394"),
                Decimal("15.005"),
                Decimal("14.696"),
                Decimal("14.18"),
                Decimal("13.969"),
                Decimal("13.987"),
                Decimal("14.071"),
                Decimal("13.576"),
                Decimal("13.501"),
                Decimal("12.638"),
                Decimal("9.999"),
                Decimal("8.164"),
                Decimal("7.12"),
                Decimal("6.717"),
                Decimal("7.925"),
                Decimal("10.565"),
                Decimal("10.636"),
                Decimal("9.747"),
            ],
            "params": {
                "low_price_threshold": Decimal("4.665812749003984"),
                "min_selections": 34,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 24,
                "max_gap_from_start": 22,
                "aggressive": False,
            },
        },
        {
            "name": "realistic_daily_tight",
            "prices": [
                Decimal("0.02749080237694725"),
                Decimal("0.03901428612819832"),
                Decimal("0.0346398788362281"),
                Decimal("0.03197316968394073"),
                Decimal("0.02312037280884873"),
                Decimal("0.023119890406724054"),
                Decimal("0.021161672243363988"),
                Decimal("0.0373235229154987"),
                Decimal("0.03202230023486417"),
                Decimal("0.03416145155592091"),
                Decimal("0.02041168988591605"),
                Decimal("0.03939819704323989"),
                Decimal("0.03664885281600844"),
                Decimal("0.024246782213565524"),
                Decimal("0.023636499344142012"),
                Decimal("0.023668090197068677"),
                Decimal("0.026084844859190756"),
                Decimal("0.03049512863264476"),
                Decimal("0.028638900372842314"),
                Decimal("0.025824582803960838"),
                Decimal("0.03223705789444759"),
                Decimal("0.02278987721304084"),
                Decimal("0.025842892970704363"),
                Decimal("0.027327236865873836"),
                Decimal("0.09560699842170359"),
                Decimal("0.13023618844815726"),
                Decimal("0.08277259770130221"),
                Decimal("0.11308181731943448"),
                Decimal("0.12228419068261889"),
                Decimal("0.09278702476319986"),
                Decimal("0.1295923322988748"),
                Decimal("0.11350306144224083"),
                Decimal("0.11634185734747006"),
                Decimal("0.14856879504309334"),
                Decimal("0.1493126406614912"),
                Decimal("0.14770076817739752"),
                Decimal("0.12436910153386965"),
                Decimal("0.10781376912051072"),
                Decimal("0.15473864212097255"),
                Decimal("0.1352121994991681"),
                Decimal("0.10976305878758232"),
                Decimal("0.1396141528089016"),
                Decimal("0.10275108168921748"),
                Decimal("0.17274563216630257"),
                Decimal("0.12070239852800135"),
                Decimal("0.15300178274831855"),
                Decimal("0.12493688608715288"),
                Decimal("0.14160544169422487"),
                Decimal("0.11280261676059679"),
                Decimal("0.09109126733153163"),
                Decimal("0.13817507766587353"),
                Decimal("0.1265079694016669"),
                Decimal("0.13636993649385137"),
                Decimal("0.13368964102565895"),
                Decimal("0.11587399872866512"),
                Decimal("0.13531245410138704"),
                Decimal("0.08530955012311517"),
                Decimal("0.09175897174514872"),
                Decimal("0.08271363733463229"),
                Decimal("0.09951981984579586"),
                Decimal("0.10332063738136893"),
                Decimal("0.09628094190643376"),
                Decimal("0.12972425054911577"),
                Decimal("0.10140519960161537"),
                Decimal("0.09685607058124285"),
                Decimal("0.11256176498949491"),
                Decimal("0.08845545349848577"),
                Decimal("0.12813181884524238"),
                Decimal("0.15745506436797707"),
                Decimal("0.24885260695254527"),
                Decimal("0.2329183576972493"),
                Decimal("0.19991973009588576"),
                Decimal("0.20027610585618014"),
                Decimal("0.24307980356705627"),
                Decimal("0.24267143359619042"),
                Decimal("0.24661258960051233"),
                Decimal("0.25"),
                Decimal("0.23842555814667613"),
                Decimal("0.23396164321360682"),
                Decimal("0.21684508973219235"),
                Decimal("0.24315517129377967"),
                Decimal("0.22645613292672237"),
                Decimal("0.19981735186394867"),
                Decimal("0.16806135565002708"),
                Decimal("0.0579884089544096"),
                Decimal("0.05699021572822"),
                Decimal("0.08545006955369286"),
                Decimal("0.0739914655235097"),
                Decimal("0.08500719003973224"),
                Decimal("0.0559718208839072"),
                Decimal("0.03574052380503848"),
                Decimal("0.05924303627614279"),
                Decimal("0.05586669165297452"),
                Decimal("0.045154484334376396"),
                Decimal("0.04541934359909121"),
                Decimal("0.036419342752737074"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.10"),
                "min_selections": 32,
                "min_consecutive_periods": 6,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "cheap_day",
            "prices": [
                Decimal("0.07655000144869413"),
                Decimal("0.05190609389379257"),
                Decimal("0.02091961642353419"),
                Decimal("0.23661761457749353"),
                Decimal("0.059565080445723194"),
                Decimal("0.24699098521619944"),
                Decimal("0.024863737747479332"),
                Decimal("0.022838315689740367"),
                Decimal("0.046732950214256656"),
                Decimal("0.03038603981386294"),
                Decimal("0.019764570245642932"),
                Decimal("0.03564532903055842"),
                Decimal("0.06496231729751095"),
                Decimal("0.04599641068895282"),
                Decimal("0.013251528890399841"),
                Decimal("0.02193668865811041"),
                Decimal("0.24488855372533333"),
                Decimal("0.06658781436815228"),
                Decimal("0.01683704798044687"),
                Decimal("0.04081067456177209"),
                Decimal("0.199517691011127"),
                Decimal("0.2409320402078782"),
                Decimal("0.056376559904778745"),
                Decimal("0.04640476148244676"),
                Decimal("0.022939811886786895"),
                Decimal("0.06425929763527802"),
                Decimal("0.07263791452993541"),
                Decimal("0.07453119645161818"),
                Decimal("0.16959828624191453"),
                Decimal("0.18253303307632643"),
                Decimal("0.028994432224172716"),
                Decimal("0.03497273286855125"),
                Decimal("0.047988725821077396"),
                Decimal("0.23021969807540396"),
                Decimal("0.24868869366005172"),
                Decimal("0.02391009770739207"),
                Decimal("0.23154614284548342"),
                Decimal("0.061030501762869116"),
                Decimal("0.015183125621386327"),
                Decimal("0.01811083416675908"),
                Decimal("0.05363086887792906"),
                Decimal("0.014449084520021655"),
                Decimal("0.032762832541872296"),
                Decimal("0.054629022994864926"),
                Decimal("0.04305504476133645"),
                Decimal("0.2213244787222995"),
                Decimal("0.04928940382986474"),
                Decimal("0.044565691745507355"),
                Decimal("0.039927871285098476"),
                Decimal("0.16078914269933045"),
                Decimal("0.21364104112637805"),
                Decimal("0.0455999483815292"),
                Decimal("0.027450456040421248"),
                Decimal("0.06288857969801341"),
                Decimal("0.01538859368801551"),
                Decimal("0.02128549010778031"),
                Decimal("0.06656842656950919"),
                Decimal("0.07100224131314024"),
                Decimal("0.02305990412202251"),
                Decimal("0.04775395693409556"),
                Decimal("0.07272639099464452"),
                Decimal("0.017703634716937373"),
                Decimal("0.039897545203837946"),
                Decimal("0.07025114082794405"),
                Decimal("0.20107473025775657"),
                Decimal("0.02554754673295112"),
                Decimal("0.1837615171403628"),
                Decimal("0.03262420524145287"),
                Decimal("0.05921132712266246"),
                Decimal("0.07802474579046725"),
                Decimal("0.027624760707775496"),
                Decimal("0.031061481687173875"),
                Decimal("0.012582086314817297"),
                Decimal("0.04518753162602031"),
                Decimal("0.17786464642366115"),
                Decimal("0.026769332346688074"),
                Decimal("0.1989452760277563"),
                Decimal("0.026943869005805032"),
                Decimal("0.06331337307301023"),
                Decimal("0.060975144402830174"),
                Decimal("0.05426140814155057"),
                Decimal("0.0475042278852331"),
                Decimal("0.2335302495589238"),
                Decimal("0.023056295727989798"),
                Decimal("0.20908929431882417"),
                Decimal("0.01116114802494993"),
                Decimal("0.02585470426385566"),
                Decimal("0.022205650030349404"),
                Decimal("0.03707147424103762"),
                Decimal("0.01962646609021953"),
                Decimal("0.017943146486841234"),
                Decimal("0.07141375473666868"),
                Decimal("0.05619888322239254"),
                Decimal("0.04886405681196237"),
                Decimal("0.02692966036303162"),
                Decimal("0.23972157579533268"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.08"),
                "min_selections": 20,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "expensive_day",
            "prices": [
                Decimal("0.2911285751537849"),
                Decimal("0.22775852715546657"),
                Decimal("0.016239780813448106"),
                Decimal("0.04464704583099741"),
                Decimal("0.24745306400328818"),
                Decimal("0.048796394086479775"),
                Decimal("0.1582210399220897"),
                Decimal("0.017336180394137354"),
                Decimal("0.2144561576938028"),
                Decimal("0.17242124523564753"),
                Decimal("0.14510889491736753"),
                Decimal("0.18594513179286448"),
                Decimal("0.26133167305074245"),
                Decimal("0.030569377536544463"),
                Decimal("0.12836107428959959"),
                Decimal("0.15069434226371248"),
                Decimal("0.047955421490133335"),
                Decimal("0.265511522660963"),
                Decimal("0.1375809805211491"),
                Decimal("0.1992274488731282"),
                Decimal("0.029807076404450808"),
                Decimal("0.04637281608315129"),
                Decimal("0.23925401118371675"),
                Decimal("0.21361224381200594"),
                Decimal("0.15327380199459487"),
                Decimal("0.2595239082050006"),
                Decimal("0.2810689230769768"),
                Decimal("0.285937362304161"),
                Decimal("0.017839314496765808"),
                Decimal("0.023013213230530575"),
                Decimal("0.16884282571930126"),
                Decimal("0.18421559880484606"),
                Decimal("0.2176852949684847"),
                Decimal("0.042087879230161586"),
                Decimal("0.049475477464020694"),
                Decimal("0.15576882267615103"),
                Decimal("0.04261845713819337"),
                Decimal("0.2512212902473777"),
                Decimal("0.13332803731213627"),
                Decimal("0.14085643071452333"),
                Decimal("0.23219366282896042"),
                Decimal("0.13144050305148425"),
                Decimal("0.17853299796481448"),
                Decimal("0.23476034484393837"),
                Decimal("0.20499868652915088"),
                Decimal("0.0385297914889198"),
                Decimal("0.2210298955625093"),
                Decimal("0.20888320734559032"),
                Decimal("0.19695738330453894"),
                Decimal("0.014315657079732178"),
                Decimal("0.03545641645055122"),
                Decimal("0.2115427244096465"),
                Decimal("0.16487260124679748"),
                Decimal("0.25599920493774875"),
                Decimal("0.13385638376918274"),
                Decimal("0.14901983170572078"),
                Decimal("0.265461668321595"),
                Decimal("0.2768629062337892"),
                Decimal("0.15358261059948644"),
                Decimal("0.21708160354481712"),
                Decimal("0.2812964339862288"),
                Decimal("0.1398093464149818"),
                Decimal("0.19687940195272613"),
                Decimal("0.27493150498614183"),
                Decimal("0.03042989210310263"),
                Decimal("0.15997940588473145"),
                Decimal("0.023504606856145117"),
                Decimal("0.17817652776373594"),
                Decimal("0.246543412601132"),
                Decimal("0.2949207748897729"),
                Decimal("0.16532081324856554"),
                Decimal("0.17415809576701852"),
                Decimal("0.1266396505238159"),
                Decimal("0.21048222418119505"),
                Decimal("0.021145858569464458"),
                Decimal("0.16312114032005504"),
                Decimal("0.029578110411102518"),
                Decimal("0.16356994887207008"),
                Decimal("0.25709153075916913"),
                Decimal("0.2510789427501347"),
                Decimal("0.23381504950684429"),
                Decimal("0.21643944313345653"),
                Decimal("0.04341209982356952"),
                Decimal("0.15357333187197375"),
                Decimal("0.03363571772752967"),
                Decimal("0.1229858092070141"),
                Decimal("0.16076923953562883"),
                Decimal("0.15138595722089845"),
                Decimal("0.18961236233409673"),
                Decimal("0.14475376994627878"),
                Decimal("0.14042523382330602"),
                Decimal("0.2779210836085766"),
                Decimal("0.23879712828615224"),
                Decimal("0.2199361460879032"),
                Decimal("0.1635334123620813"),
                Decimal("0.045888630318133075"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.12"),
                "min_selections": 16,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
                "aggressive": False,
            },
        },
        {
            "name": "volatile_day",
            "prices": [
                Decimal("0.11861663446573512"),
                Decimal("0.28570714885887566"),
                Decimal("0.22227824312530747"),
                Decimal("0.18361096041714062"),
                Decimal("0.055245405728306586"),
                Decimal("0.055238410897498764"),
                Decimal("0.026844247528777843"),
                Decimal("0.2611910822747312"),
                Decimal("0.18432335340553055"),
                Decimal("0.21534104756085318"),
                Decimal("0.01596950334578271"),
                Decimal("0.29127385712697834"),
                Decimal("0.2514083658321223"),
                Decimal("0.07157834209670008"),
                Decimal("0.06272924049005918"),
                Decimal("0.06318730785749581"),
                Decimal("0.09823025045826593"),
                Decimal("0.16217936517334897"),
                Decimal("0.13526405540621358"),
                Decimal("0.09445645065743215"),
                Decimal("0.18743733946949004"),
                Decimal("0.05045321958909213"),
                Decimal("0.09472194807521325"),
                Decimal("0.11624493455517058"),
                Decimal("0.1422602954229404"),
                Decimal("0.23770102880397392"),
                Decimal("0.06790539682592432"),
                Decimal("0.15912798713994736"),
                Decimal("0.18180022496999232"),
                Decimal("0.02347061968879934"),
                Decimal("0.1861880070514171"),
                Decimal("0.059451995869314544"),
                Decimal("0.02886496196573106"),
                Decimal("0.2851768058034666"),
                Decimal("0.2900332895916222"),
                Decimal("0.24443523095377373"),
                Decimal("0.09833799306027749"),
                Decimal("0.03832491306185132"),
                Decimal("0.2084275776885255"),
                Decimal("0.13764422318448438"),
                Decimal("0.045391088104985856"),
                Decimal("0.15360130393226834"),
                Decimal("0.019972671123413333"),
                Decimal("0.2737029166028468"),
                Decimal("0.0850461946640049"),
                Decimal("0.20213146246265476"),
                Decimal("0.10039621206592916"),
                Decimal("0.16081972614156514"),
                Decimal("0.1685459810095511"),
                Decimal("0.06360779210240283"),
                Decimal("0.291179542051722"),
                Decimal("0.2347885187747232"),
                Decimal("0.2824546930536148"),
                Decimal("0.26949993162401814"),
                Decimal("0.18339099385521468"),
                Decimal("0.2773435281567039"),
                Decimal("0.03566282559505665"),
                Decimal("0.0668350301015521"),
                Decimal("0.023115913784056037"),
                Decimal("0.10434579592134664"),
                Decimal("0.12271641400994977"),
                Decimal("0.08869121921442981"),
                Decimal("0.2503338776540595"),
                Decimal("0.11345846474114088"),
                Decimal("0.09147100780934041"),
                Decimal("0.16738186411589207"),
                Decimal("0.050868025242681164"),
                Decimal("0.2426371244186715"),
                Decimal("0.03161968666713354"),
                Decimal("0.29619721161415"),
                Decimal("0.23395098309603066"),
                Decimal("0.06762754764491"),
                Decimal("0.011601413965844696"),
                Decimal("0.2464838142519019"),
                Decimal("0.21498862971580895"),
                Decimal("0.2214120787318863"),
                Decimal("0.23366840053892426"),
                Decimal("0.031472949002886205"),
                Decimal("0.11395506127783904"),
                Decimal("0.04360202726228762"),
                Decimal("0.26029999350392213"),
                Decimal("0.19075645677999178"),
                Decimal("0.10596042720726825"),
                Decimal("0.028431921582946856"),
                Decimal("0.10018487329754203"),
                Decimal("0.10430316338775664"),
                Decimal("0.22158579171803858"),
                Decimal("0.1948916666930118"),
                Decimal("0.2672916953471347"),
                Decimal("0.1469423282969653"),
                Decimal("0.04468233132210749"),
                Decimal("0.21684098829466855"),
                Decimal("0.23062766409890026"),
                Decimal("0.1727703872951539"),
                Decimal("0.23358048218682267"),
                Decimal("0.1532007229456733"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.10"),
                "min_selections": 24,
                "min_consecutive_periods": 3,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "peak_valley",
            "prices": [
                Decimal("0.08745401188473625"),
                Decimal("0.1450714306409916"),
                Decimal("0.12319939418114051"),
                Decimal("0.10986584841970365"),
                Decimal("0.06560186404424365"),
                Decimal("0.06559945203362026"),
                Decimal("0.05580836121681995"),
                Decimal("0.1366176145774935"),
                Decimal("0.022022300234864175"),
                Decimal("0.024161451555920907"),
                Decimal("0.010411689885916049"),
                Decimal("0.029398197043239885"),
                Decimal("0.026648852816008435"),
                Decimal("0.014246782213565522"),
                Decimal("0.013636499344142012"),
                Decimal("0.013668090197068676"),
                Decimal("0.016084844859190754"),
                Decimal("0.020495128632644757"),
                Decimal("0.018638900372842312"),
                Decimal("0.01582458280396084"),
                Decimal("0.02223705789444759"),
                Decimal("0.012789877213040837"),
                Decimal("0.01584289297070436"),
                Decimal("0.017327236865873834"),
                Decimal("0.019121399684340717"),
                Decimal("0.12851759613930136"),
                Decimal("0.06996737821583597"),
                Decimal("0.10142344384136115"),
                Decimal("0.10924145688620424"),
                Decimal("0.054645041271999775"),
                Decimal("0.11075448519014383"),
                Decimal("0.06705241236872915"),
                Decimal("0.20650515929852797"),
                Decimal("0.2948885537253333"),
                Decimal("0.2965632033074559"),
                Decimal("0.2808397348116461"),
                Decimal("0.23046137691733706"),
                Decimal("0.2097672114006384"),
                Decimal("0.26842330265121567"),
                Decimal("0.24401524937396013"),
                Decimal("0.2122038234844779"),
                Decimal("0.09951769101112701"),
                Decimal("0.05343885211152184"),
                Decimal("0.1409320402078782"),
                Decimal("0.0758779981600017"),
                Decimal("0.11625222843539819"),
                Decimal("0.0811711076089411"),
                Decimal("0.10200680211778107"),
                Decimal("0.10467102793432796"),
                Decimal("0.0684854455525527"),
                Decimal("0.14695846277645586"),
                Decimal("0.12751328233611145"),
                Decimal("0.1439498941564189"),
                Decimal("0.13948273504276487"),
                Decimal("0.10978999788110852"),
                Decimal("0.14218742350231167"),
                Decimal("0.05884925020519195"),
                Decimal("0.06959828624191453"),
                Decimal("0.05452272889105381"),
                Decimal("0.08253303307632644"),
                Decimal("0.0888677289689482"),
                Decimal("0.0771349031773896"),
                Decimal("0.13287375091519293"),
                Decimal("0.08567533266935892"),
                Decimal("0.07809345096873807"),
                Decimal("0.10426960831582485"),
                Decimal("0.06409242249747626"),
                Decimal("0.13021969807540396"),
                Decimal("0.2074550643679771"),
                Decimal("0.2986886936600517"),
                Decimal("0.27722447692966573"),
                Decimal("0.21987156815341724"),
                Decimal("0.20055221171236026"),
                Decimal("0.2815461428454834"),
                Decimal("0.27068573438476173"),
                Decimal("0.2729007168040987"),
                Decimal("0.27712703466859456"),
                Decimal("0.05740446517340904"),
                Decimal("0.08584657285442726"),
                Decimal("0.061586905952512976"),
                Decimal("0.13631034258755936"),
                Decimal("0.1123298126827558"),
                Decimal("0.08308980248526492"),
                Decimal("0.056355835028602363"),
                Decimal("0.08109823217156623"),
                Decimal("0.08251833220267471"),
                Decimal("0.12296061783380641"),
                Decimal("0.1137557471355213"),
                Decimal("0.027744254851526526"),
                Decimal("0.019444298503238984"),
                Decimal("0.012391884918766034"),
                Decimal("0.024264895744459898"),
                Decimal("0.025215700972337947"),
                Decimal("0.021225543951389925"),
                Decimal("0.025419343599091218"),
                Decimal("0.019875911927287812"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.10"),
                "min_selections": 28,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "realistic_conservative",
            "prices": [
                Decimal("0.02749080237694725"),
                Decimal("0.03901428612819832"),
                Decimal("0.0346398788362281"),
                Decimal("0.03197316968394073"),
                Decimal("0.02312037280884873"),
                Decimal("0.023119890406724054"),
                Decimal("0.021161672243363988"),
                Decimal("0.0373235229154987"),
                Decimal("0.03202230023486417"),
                Decimal("0.03416145155592091"),
                Decimal("0.02041168988591605"),
                Decimal("0.03939819704323989"),
                Decimal("0.03664885281600844"),
                Decimal("0.024246782213565524"),
                Decimal("0.023636499344142012"),
                Decimal("0.023668090197068677"),
                Decimal("0.026084844859190756"),
                Decimal("0.03049512863264476"),
                Decimal("0.028638900372842314"),
                Decimal("0.025824582803960838"),
                Decimal("0.03223705789444759"),
                Decimal("0.02278987721304084"),
                Decimal("0.025842892970704363"),
                Decimal("0.027327236865873836"),
                Decimal("0.09560699842170359"),
                Decimal("0.13023618844815726"),
                Decimal("0.08277259770130221"),
                Decimal("0.11308181731943448"),
                Decimal("0.12228419068261889"),
                Decimal("0.09278702476319986"),
                Decimal("0.1295923322988748"),
                Decimal("0.11350306144224083"),
                Decimal("0.11634185734747006"),
                Decimal("0.14856879504309334"),
                Decimal("0.1493126406614912"),
                Decimal("0.14770076817739752"),
                Decimal("0.12436910153386965"),
                Decimal("0.10781376912051072"),
                Decimal("0.15473864212097255"),
                Decimal("0.1352121994991681"),
                Decimal("0.10976305878758232"),
                Decimal("0.1396141528089016"),
                Decimal("0.10275108168921748"),
                Decimal("0.17274563216630257"),
                Decimal("0.12070239852800135"),
                Decimal("0.15300178274831855"),
                Decimal("0.12493688608715288"),
                Decimal("0.14160544169422487"),
                Decimal("0.11280261676059679"),
                Decimal("0.09109126733153163"),
                Decimal("0.13817507766587353"),
                Decimal("0.1265079694016669"),
                Decimal("0.13636993649385137"),
                Decimal("0.13368964102565895"),
                Decimal("0.11587399872866512"),
                Decimal("0.13531245410138704"),
                Decimal("0.08530955012311517"),
                Decimal("0.09175897174514872"),
                Decimal("0.08271363733463229"),
                Decimal("0.09951981984579586"),
                Decimal("0.10332063738136893"),
                Decimal("0.09628094190643376"),
                Decimal("0.12972425054911577"),
                Decimal("0.10140519960161537"),
                Decimal("0.09685607058124285"),
                Decimal("0.11256176498949491"),
                Decimal("0.08845545349848577"),
                Decimal("0.12813181884524238"),
                Decimal("0.15745506436797707"),
                Decimal("0.24885260695254527"),
                Decimal("0.2329183576972493"),
                Decimal("0.19991973009588576"),
                Decimal("0.20027610585618014"),
                Decimal("0.24307980356705627"),
                Decimal("0.24267143359619042"),
                Decimal("0.24661258960051233"),
                Decimal("0.25"),
                Decimal("0.23842555814667613"),
                Decimal("0.23396164321360682"),
                Decimal("0.21684508973219235"),
                Decimal("0.24315517129377967"),
                Decimal("0.22645613292672237"),
                Decimal("0.19981735186394867"),
                Decimal("0.16806135565002708"),
                Decimal("0.0579884089544096"),
                Decimal("0.05699021572822"),
                Decimal("0.08545006955369286"),
                Decimal("0.0739914655235097"),
                Decimal("0.08500719003973224"),
                Decimal("0.0559718208839072"),
                Decimal("0.03574052380503848"),
                Decimal("0.05924303627614279"),
                Decimal("0.05586669165297452"),
                Decimal("0.045154484334376396"),
                Decimal("0.04541934359909121"),
                Decimal("0.036419342752737074"),
            ],
            "params": {
                "low_price_threshold": Decimal("0.10"),
                "min_selections": 24,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
                "aggressive": False,  # Conservative mode
            },
        },
        {
            "name": "custom_conservative",
            "prices": [
                Decimal("11.606"),
                Decimal("12.523"),
                Decimal("14.322"),
                Decimal("15.832"),
                Decimal("12.529"),
                Decimal("13.219"),
                Decimal("14.048"),
                Decimal("14.923"),
                Decimal("13.182"),
                Decimal("14.6"),
                Decimal("14.995"),
                Decimal("14.643"),
                Decimal("15.967"),
                Decimal("16.106"),
                Decimal("15.394"),
                Decimal("15.005"),
                Decimal("14.696"),
                Decimal("14.18"),
                Decimal("13.969"),
                Decimal("13.987"),
                Decimal("14.071"),
                Decimal("13.576"),
                Decimal("13.501"),
                Decimal("12.638"),
                Decimal("9.999"),
                Decimal("8.164"),
                Decimal("7.12"),
                Decimal("6.717"),
                Decimal("7.925"),
                Decimal("10.565"),
                Decimal("10.636"),
                Decimal("9.747"),
                Decimal("11.489"),
                Decimal("7.968"),
                Decimal("6.598"),
                Decimal("5.945"),
                Decimal("10.0"),
                Decimal("7.929"),
                Decimal("7.108"),
                Decimal("6.126"),
                Decimal("8.994"),
                Decimal("7.941"),
                Decimal("7.52"),
                Decimal("7.119"),
                Decimal("8.994"),
                Decimal("7.324"),
                Decimal("6.743"),
                Decimal("6.141"),
                Decimal("6.191"),
                Decimal("6.062"),
                Decimal("5.209"),
                Decimal("5.0"),
                Decimal("5.263"),
                Decimal("6.14"),
                Decimal("5.001"),
                Decimal("5.001"),
                Decimal("4.393"),
                Decimal("4.999"),
                Decimal("4.829"),
                Decimal("5.0"),
                Decimal("6.658"),
                Decimal("7.0"),
                Decimal("7.0"),
                Decimal("7.119"),
                Decimal("7.119"),
                Decimal("6.999"),
                Decimal("6.999"),
                Decimal("6.442"),
                Decimal("8.105"),
                Decimal("8.066"),
                Decimal("7.509"),
                Decimal("7.12"),
                Decimal("9.873"),
                Decimal("9.991"),
                Decimal("9.999"),
                Decimal("10.035"),
                Decimal("11.0"),
                Decimal("10.378"),
                Decimal("9.872"),
                Decimal("7.996"),
                Decimal("10.762"),
                Decimal("7.713"),
                Decimal("7.689"),
                Decimal("6.812"),
                Decimal("7.889"),
                Decimal("7.738"),
                Decimal("7.712"),
                Decimal("7.632"),
                Decimal("10.348"),
                Decimal("10.247"),
                Decimal("10.249"),
                Decimal("10.349"),
                Decimal("10.347"),
                Decimal("10.591"),
                Decimal("10.72"),
                Decimal("9.358"),
            ],
            "params": {
                "low_price_threshold": Decimal("7.915812749003984"),
                "min_selections": 40,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 24,
                "max_gap_from_start": 20,
                "aggressive": False,
            },
        },
        {
            "name": "custom_all",
            "prices": [
                Decimal("4.494"),
                Decimal("4.286"),
                Decimal("3.957"),
                Decimal("3.94"),
                Decimal("4.884"),
                Decimal("4.239"),
                Decimal("3.658"),
                Decimal("3.264"),
                Decimal("4.9"),
                Decimal("4.325"),
                Decimal("3.542"),
                Decimal("3.195"),
                Decimal("3.889"),
                Decimal("4.142"),
                Decimal("3.598"),
                Decimal("3.371"),
                Decimal("4.0"),
                Decimal("3.661"),
                Decimal("3.28"),
                Decimal("2.833"),
                Decimal("3.376"),
                Decimal("3.192"),
                Decimal("3.013"),
                Decimal("3.0"),
                Decimal("3.101"),
                Decimal("3.045"),
                Decimal("2.991"),
                Decimal("2.999"),
                Decimal("3.154"),
                Decimal("3.067"),
                Decimal("2.991"),
                Decimal("2.93"),
                Decimal("2.966"),
                Decimal("2.906"),
                Decimal("2.814"),
                Decimal("2.599"),
                Decimal("2.819"),
                Decimal("2.711"),
                Decimal("2.599"),
                Decimal("2.26"),
                Decimal("2.573"),
                Decimal("2.465"),
                Decimal("2.348"),
                Decimal("2.244"),
                Decimal("2.386"),
                Decimal("2.299"),
                Decimal("2.199"),
                Decimal("2.084"),
                Decimal("2.446"),
                Decimal("2.448"),
                Decimal("2.178"),
                Decimal("2.226"),
                Decimal("2.377"),
                Decimal("2.401"),
                Decimal("2.414"),
                Decimal("2.447"),
                Decimal("2.491"),
                Decimal("2.495"),
                Decimal("2.6"),
                Decimal("2.603"),
                Decimal("2.599"),
                Decimal("2.661"),
                Decimal("2.672"),
                Decimal("2.709"),
                Decimal("2.663"),
                Decimal("2.662"),
                Decimal("2.647"),
                Decimal("2.756"),
                Decimal("2.61"),
                Decimal("2.655"),
                Decimal("2.708"),
                Decimal("2.708"),
                Decimal("2.751"),
                Decimal("2.682"),
                Decimal("2.745"),
                Decimal("2.827"),
                Decimal("2.582"),
                Decimal("2.774"),
                Decimal("2.827"),
                Decimal("2.907"),
                Decimal("2.789"),
                Decimal("2.878"),
                Decimal("2.951"),
                Decimal("2.985"),
                Decimal("2.885"),
                Decimal("2.967"),
                Decimal("3.008"),
                Decimal("3.036"),
                Decimal("2.94"),
                Decimal("2.974"),
                Decimal("3.005"),
                Decimal("3.162"),
                Decimal("2.848"),
                Decimal("3.056"),
                Decimal("3.174"),
                Decimal("3.179"),
            ],
            "params": {
                "low_price_threshold": Decimal("3.507812749003984"),
                "min_selections": 81,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 16,
                "max_gap_from_start": 15,
                "aggressive": False,
            },
        },
        {
            "name": "custom_daily_peak",
            "prices": [
                Decimal("1.138"),
                Decimal("1.144"),
                Decimal("1.173"),
                Decimal("1.234"),
                Decimal("1.14"),
                Decimal("1.227"),
                Decimal("1.362"),
                Decimal("1.546"),
                Decimal("1.247"),
                Decimal("1.46"),
                Decimal("1.647"),
                Decimal("1.807"),
                Decimal("1.399"),
                Decimal("1.576"),
                Decimal("1.813"),
                Decimal("2.01"),
                Decimal("1.74"),
                Decimal("1.935"),
                Decimal("2.098"),
                Decimal("2.216"),
                Decimal("2.032"),
                Decimal("2.118"),
                Decimal("2.271"),
                Decimal("2.328"),
                Decimal("2.118"),
                Decimal("2.184"),
                Decimal("2.234"),
                Decimal("2.476"),
                Decimal("2.253"),
                Decimal("2.469"),
                Decimal("2.599"),
                Decimal("2.636"),
                Decimal("2.49"),
                Decimal("2.498"),
                Decimal("2.443"),
                Decimal("2.383"),
                Decimal("2.383"),
                Decimal("2.408"),
                Decimal("2.391"),
                Decimal("2.448"),
                Decimal("2.553"),
                Decimal("2.574"),
                Decimal("2.599"),
                Decimal("2.6"),
                Decimal("2.6"),
                Decimal("2.599"),
                Decimal("2.618"),
                Decimal("2.737"),
                Decimal("2.62"),
                Decimal("2.654"),
                Decimal("2.672"),
                Decimal("2.665"),
                Decimal("2.602"),
                Decimal("2.6"),
                Decimal("2.599"),
                Decimal("2.599"),
                Decimal("2.444"),
                Decimal("2.412"),
                Decimal("2.414"),
                Decimal("2.439"),
                Decimal("2.411"),
                Decimal("2.935"),
                Decimal("4.283"),
                Decimal("5.97"),
                Decimal("5.872"),
                Decimal("7.001"),
                Decimal("7.007"),
                Decimal("7.324"),
                Decimal("9.705"),
                Decimal("10.761"),
                Decimal("10.009"),
                Decimal("8.594"),
                Decimal("9.999"),
                Decimal("7.008"),
                Decimal("5.872"),
                Decimal("4.999"),
                Decimal("5.0"),
                Decimal("4.999"),
                Decimal("4.78"),
                Decimal("4.061"),
                Decimal("4.999"),
                Decimal("3.722"),
                Decimal("3.488"),
                Decimal("2.6"),
                Decimal("3.5"),
                Decimal("3.489"),
                Decimal("2.734"),
                Decimal("2.582"),
                Decimal("3.256"),
                Decimal("2.898"),
                Decimal("2.67"),
                Decimal("2.655"),
                Decimal("2.81"),
                Decimal("2.803"),
                Decimal("2.709"),
                Decimal("2.663"),
                Decimal("2.7"),
                Decimal("2.667"),
                Decimal("2.748"),
                Decimal("2.781"),
                Decimal("2.52"),
                Decimal("2.591"),
                Decimal("2.599"),
                Decimal("2.59"),
                Decimal("2.564"),
                Decimal("2.555"),
                Decimal("2.423"),
                Decimal("2.328"),
                Decimal("2.6"),
                Decimal("2.432"),
                Decimal("2.32"),
                Decimal("2.186"),
                Decimal("2.417"),
                Decimal("2.31"),
                Decimal("2.169"),
                Decimal("2.133"),
                Decimal("2.197"),
                Decimal("2.135"),
                Decimal("2.094"),
                Decimal("1.931"),
                Decimal("2.098"),
                Decimal("1.987"),
                Decimal("1.834"),
                Decimal("1.728"),
                Decimal("1.968"),
                Decimal("1.953"),
                Decimal("1.862"),
                Decimal("1.773"),
                Decimal("1.901"),
                Decimal("1.771"),
                Decimal("1.628"),
                Decimal("1.523"),
                Decimal("1.64"),
                Decimal("1.521"),
                Decimal("1.44"),
                Decimal("1.358"),
            ],
            "params": {
                "low_price_threshold": Decimal("3.186"),
                "min_selections": 44,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 12,
                "max_gap_from_start": 8,
                "aggressive": False,
            },
        },
        {
            "name": "custom_new_scenario",
            "prices": [
                Decimal("2.383"),
                Decimal("2.408"),
                Decimal("2.391"),
                Decimal("2.448"),
                Decimal("2.553"),
                Decimal("2.574"),
                Decimal("2.599"),
                Decimal("2.6"),
                Decimal("2.6"),
                Decimal("2.599"),
                Decimal("2.618"),
                Decimal("2.737"),
                Decimal("2.62"),
                Decimal("2.654"),
                Decimal("2.672"),
                Decimal("2.665"),
                Decimal("2.602"),
                Decimal("2.6"),
                Decimal("2.599"),
                Decimal("2.599"),
                Decimal("2.444"),
                Decimal("2.412"),
                Decimal("2.414"),
                Decimal("2.439"),
                Decimal("2.411"),
                Decimal("2.935"),
                Decimal("4.283"),
                Decimal("5.97"),
                Decimal("5.872"),
                Decimal("7.001"),
                Decimal("7.007"),
                Decimal("7.324"),
                Decimal("9.705"),
                Decimal("10.761"),
                Decimal("10.009"),
                Decimal("8.594"),
                Decimal("9.999"),
                Decimal("7.008"),
                Decimal("5.872"),
                Decimal("4.999"),
                Decimal("5.0"),
                Decimal("4.999"),
                Decimal("4.78"),
                Decimal("4.061"),
                Decimal("4.999"),
                Decimal("3.722"),
                Decimal("3.488"),
                Decimal("2.6"),
                Decimal("3.5"),
                Decimal("3.489"),
                Decimal("2.734"),
                Decimal("2.582"),
                Decimal("3.256"),
                Decimal("2.898"),
                Decimal("2.67"),
                Decimal("2.655"),
                Decimal("2.81"),
                Decimal("2.803"),
                Decimal("2.709"),
                Decimal("2.663"),
                Decimal("2.7"),
                Decimal("2.667"),
                Decimal("2.748"),
                Decimal("2.781"),
                Decimal("2.52"),
                Decimal("2.591"),
                Decimal("2.599"),
                Decimal("2.59"),
                Decimal("2.564"),
                Decimal("2.555"),
                Decimal("2.423"),
                Decimal("2.328"),
                Decimal("2.6"),
                Decimal("2.432"),
                Decimal("2.32"),
                Decimal("2.186"),
                Decimal("2.417"),
                Decimal("2.31"),
                Decimal("2.169"),
                Decimal("2.133"),
                Decimal("2.197"),
                Decimal("2.135"),
                Decimal("2.094"),
                Decimal("1.931"),
                Decimal("2.098"),
                Decimal("1.987"),
                Decimal("1.834"),
                Decimal("1.728"),
                Decimal("1.968"),
                Decimal("1.953"),
                Decimal("1.862"),
                Decimal("1.773"),
                Decimal("1.901"),
                Decimal("1.771"),
                Decimal("1.628"),
                Decimal("1.523"),
                Decimal("1.64"),
                Decimal("1.521"),
                Decimal("1.44"),
                Decimal("1.358"),
            ],
            "params": {
                "low_price_threshold": Decimal("3.32"),
                "min_selections": 24,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 24,
                "max_gap_from_start": 18,
                "aggressive": False,
            },
        },
        {
            "name": "custom_minimal_constraints",
            "prices": [
                Decimal("9.824"),
                Decimal("8.534"),
                Decimal("7.981"),
                Decimal("7.296"),
                Decimal("8.586"),
                Decimal("7.75"),
                Decimal("7.523"),
                Decimal("7.027"),
                Decimal("6.923"),
                Decimal("6.718"),
                Decimal("6.447"),
                Decimal("6.061"),
                Decimal("6.182"),
                Decimal("5.652"),
                Decimal("5.188"),
                Decimal("4.957"),
                Decimal("5.665"),
                Decimal("5.2"),
                Decimal("5.159"),
                Decimal("4.991"),
                Decimal("5.262"),
                Decimal("5.2"),
                Decimal("5.228"),
                Decimal("5.2"),
                Decimal("4.88"),
                Decimal("4.965"),
                Decimal("5.366"),
                Decimal("5.79"),
                Decimal("6.009"),
                Decimal("6.695"),
                Decimal("6.78"),
                Decimal("6.974"),
                Decimal("6.824"),
                Decimal("7.085"),
                Decimal("7.173"),
                Decimal("7.226"),
                Decimal("7.125"),
                Decimal("7.254"),
                Decimal("7.83"),
                Decimal("9.228"),
                Decimal("9.318"),
                Decimal("9.838"),
                Decimal("8.972"),
                Decimal("8.961"),
                Decimal("9.421"),
                Decimal("9.189"),
                Decimal("9.01"),
                Decimal("8.494"),
                Decimal("9.777"),
                Decimal("8.982"),
                Decimal("8.723"),
                Decimal("8.274"),
                Decimal("8.569"),
                Decimal("8.393"),
                Decimal("8.205"),
                Decimal("8.0"),
                Decimal("8.33"),
                Decimal("8.2"),
                Decimal("8.2"),
                Decimal("8.489"),
                Decimal("8.31"),
                Decimal("9.928"),
                Decimal("11.565"),
                Decimal("14.995"),
                Decimal("8.423"),
                Decimal("10.567"),
                Decimal("15.13"),
                Decimal("18.988"),
                Decimal("11.047"),
                Decimal("13.994"),
                Decimal("14.999"),
                Decimal("15.004"),
                Decimal("14.999"),
                Decimal("16.852"),
                Decimal("15.695"),
                Decimal("17.173"),
                Decimal("13.695"),
                Decimal("13.788"),
                Decimal("14.999"),
                Decimal("17.025"),
                Decimal("14.653"),
                Decimal("15.69"),
                Decimal("15.453"),
                Decimal("15.004"),
                Decimal("16.784"),
                Decimal("15.297"),
                Decimal("13.152"),
                Decimal("10.767"),
                Decimal("15.595"),
                Decimal("13.405"),
                Decimal("12.563"),
                Decimal("10.999"),
                Decimal("14.628"),
                Decimal("14.997"),
                Decimal("12.029"),
                Decimal("10.311"),
                Decimal("14.69"),
                Decimal("11.0"),
                Decimal("10.682"),
                Decimal("9.922"),
                Decimal("10.872"),
                Decimal("8.515"),
                Decimal("8.154"),
                Decimal("7.119"),
            ],
            "params": {
                "low_price_threshold": Decimal("5.88"),
                "min_selections": 1,
                "min_consecutive_periods": 1,
                "max_gap_between_periods": 24,
                "max_gap_from_start": 23,
                "aggressive": False,
            },
        },
        {
            "name": "customer_issue_2026_02_16",
            "prices": [
                Decimal("10.477"),
                Decimal("9.962"),
                Decimal("7.271"),
                Decimal("9.37"),
                Decimal("10.168"),
                Decimal("10.3"),
                Decimal("9.103"),
                Decimal("9.29"),
                Decimal("9.259"),
                Decimal("9.587"),
                Decimal("9.053"),
                Decimal("9.251"),
                Decimal("9.444"),
                Decimal("9.255"),
                Decimal("10.216"),
                Decimal("9.161"),
                Decimal("8.773"),
                Decimal("14.999"),
                Decimal("8.762"),
                Decimal("11.004"),
                Decimal("11.829"),
                Decimal("11.999"),
                Decimal("14.999"),
                Decimal("13.193"),
                Decimal("8.898"),
                Decimal("6.911"),
                Decimal("12.999"),
                Decimal("8.852"),
                Decimal("8.135"),
                Decimal("7.413"),
                Decimal("8.239"),
                Decimal("7.322"),
                Decimal("6.999"),
                Decimal("7.616"),
                Decimal("6.787"),
                Decimal("7.276"),
                Decimal("7.608"),
                Decimal("8.327"),
                Decimal("8.181"),
                Decimal("8.212"),
                Decimal("8.355"),
                Decimal("8.425"),
                Decimal("9.527"),
                Decimal("10.501"),
                Decimal("9.899"),
                Decimal("9.334"),
                Decimal("9.008"),
                Decimal("8.361"),
                Decimal("7.712"),
                Decimal("7.225"),
                Decimal("7.552"),
                Decimal("6.295"),
                Decimal("5.033"),
                Decimal("4.3"),
                Decimal("3.5"),
                Decimal("3.219"),
                Decimal("3.164"),
                Decimal("2.999"),
                Decimal("3.219"),
                Decimal("3.187"),
                Decimal("3.029"),
                Decimal("2.977"),
                Decimal("2.947"),
                Decimal("2.796"),
                Decimal("2.644"),
                Decimal("2.471"),
                Decimal("2.688"),
                Decimal("2.524"),
                Decimal("2.392"),
                Decimal("2.313"),
            ],
            "params": {
                "low_price_threshold": Decimal("7.608"),
                "min_selections": 29,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 36,
                "max_gap_from_start": 33,
            },
        },
        {
            "name": "mlp_daemon_2026_02_25",
            "prices": [
                Decimal("9.779"),
                Decimal("12.416"),
                Decimal("14.999"),
                Decimal("17.878"),
                Decimal("10.21"),
                Decimal("14.997"),
                Decimal("14.999"),
                Decimal("14.999"),
                Decimal("13.486"),
                Decimal("14.994"),
                Decimal("14.997"),
                Decimal("14.057"),
                Decimal("16.448"),
                Decimal("14.51"),
                Decimal("13.709"),
                Decimal("12.919"),
                Decimal("15.145"),
                Decimal("14.999"),
                Decimal("13.775"),
                Decimal("11.488"),
                Decimal("13.482"),
                Decimal("11.418"),
                Decimal("9.939"),
                Decimal("8.865"),
                Decimal("9.94"),
                Decimal("9.939"),
                Decimal("9.2"),
                Decimal("8.13"),
                Decimal("9.997"),
                Decimal("9.94"),
                Decimal("9.94"),
                Decimal("9.939"),
                Decimal("9.939"),
                Decimal("9.939"),
                Decimal("10.116"),
                Decimal("10.006"),
                Decimal("14.992"),
                Decimal("11.218"),
                Decimal("9.015"),
                Decimal("8.558"),
                Decimal("6.492"),
                Decimal("5.0"),
                Decimal("3.58"),
                Decimal("2.945"),
                Decimal("3.521"),
                Decimal("2.767"),
                Decimal("2.351"),
                Decimal("1.528"),
                Decimal("3.103"),
                Decimal("3.195"),
                Decimal("2.884"),
                Decimal("1.859"),
                Decimal("2.999"),
                Decimal("2.999"),
                Decimal("2.845"),
                Decimal("2.518"),
                Decimal("2.809"),
                Decimal("2.649"),
                Decimal("2.587"),
                Decimal("2.558"),
                Decimal("2.401"),
                Decimal("2.487"),
                Decimal("2.369"),
                Decimal("2.486"),
                Decimal("2.604"),
                Decimal("2.795"),
                Decimal("2.914"),
                Decimal("2.999"),
                Decimal("2.655"),
                Decimal("2.772"),
                Decimal("2.934"),
                Decimal("2.85"),
                Decimal("2.679"),
                Decimal("2.679"),
                Decimal("2.739"),
                Decimal("2.862"),
            ],
            "params": {
                "low_price_threshold": Decimal("8.13"),
                "min_selections": 37,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 0,
            },
        },
        {
            "name": "mlp_daemon_2026_02_25_01_23_35",
            "prices": [
                Decimal("14.997"),
                Decimal("14.999"),
                Decimal("14.999"),
                Decimal("13.486"),
                Decimal("14.994"),
                Decimal("14.997"),
                Decimal("14.057"),
                Decimal("16.448"),
                Decimal("14.51"),
                Decimal("13.709"),
                Decimal("12.919"),
                Decimal("15.145"),
                Decimal("14.999"),
                Decimal("13.775"),
                Decimal("11.488"),
                Decimal("13.482"),
                Decimal("11.418"),
                Decimal("9.939"),
                Decimal("8.865"),
                Decimal("9.94"),
                Decimal("9.939"),
                Decimal("9.2"),
                Decimal("8.13"),
                Decimal("9.997"),
                Decimal("9.94"),
                Decimal("9.94"),
                Decimal("9.939"),
                Decimal("9.939"),
                Decimal("9.939"),
                Decimal("10.116"),
                Decimal("10.006"),
                Decimal("14.992"),
                Decimal("11.218"),
                Decimal("9.015"),
                Decimal("8.558"),
                Decimal("6.492"),
                Decimal("5.0"),
                Decimal("3.58"),
                Decimal("2.945"),
                Decimal("3.521"),
                Decimal("2.767"),
                Decimal("2.351"),
                Decimal("1.528"),
                Decimal("3.103"),
                Decimal("3.195"),
                Decimal("2.884"),
                Decimal("1.859"),
                Decimal("2.999"),
                Decimal("2.999"),
                Decimal("2.845"),
                Decimal("2.518"),
                Decimal("2.809"),
                Decimal("2.649"),
                Decimal("2.587"),
                Decimal("2.558"),
                Decimal("2.401"),
                Decimal("2.487"),
                Decimal("2.369"),
                Decimal("2.486"),
                Decimal("2.604"),
                Decimal("2.795"),
                Decimal("2.914"),
                Decimal("2.999"),
                Decimal("2.655"),
                Decimal("2.772"),
                Decimal("2.934"),
                Decimal("2.85"),
                Decimal("2.679"),
                Decimal("2.679"),
                Decimal("2.739"),
                Decimal("2.862"),
            ],
            "params": {
                "low_price_threshold": Decimal("5.0"),
                "min_selections": 35,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 17,
            },
        },
    ]

    print(f"Generating {len(scenarios)} test scenarios...")
    print(f"Output directory: {output_dir.absolute()}\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}/{len(scenarios)}: {scenario['name']}")

        # Get prices - either from generator or provided directly
        if "prices" in scenario:
            prices = scenario["prices"]
        else:
            if "prices" in scenario:
                prices = scenario["prices"]
            elif "generator" in scenario:
                prices = scenario["generator"]()
            else:
                raise ValueError(f"Scenario {scenario['name']} must have either 'prices' or 'generator'")

        # Get parameters
        params = scenario["params"].copy()
        aggressive = params.pop("aggressive", True)

        # Run algorithm
        selected = get_cheapest_periods(
            prices=prices,
            aggressive=aggressive,
            **params,
        )

        # Validate constraints
        is_valid, error_message = validate_constraints(
            selected_indices=selected,
            total_length=len(prices),
            min_consecutive_periods=params["min_consecutive_periods"],
            max_gap_between_periods=params["max_gap_between_periods"],
            max_gap_from_start=params["max_gap_from_start"],
            min_selections=params["min_selections"],
        )

        if not is_valid:
            # Print detailed diagnostic information
            print(f"  ❌ CONSTRAINT VIOLATION DETECTED!")
            print(f"     Error: {error_message}")
            print(f"     Selected indices: {selected}")
            
            # Analyze consecutive runs
            consecutive_runs = []
            if selected:
                run_start = selected[0]
                run_length = 1
                
                for i in range(1, len(selected)):
                    if selected[i] == selected[i-1] + 1:
                        run_length += 1
                    else:
                        consecutive_runs.append((run_start, run_length))
                        run_start = selected[i]
                        run_length = 1
                
                consecutive_runs.append((run_start, run_length))
                
                print(f"     Consecutive runs found: {len(consecutive_runs)}")
                for start, length in consecutive_runs:
                    status = "✓" if length >= params["min_consecutive_periods"] or (start + length - 1 == len(prices) - 1) else "✗"
                    at_end = "(at end)" if (start + length - 1 == len(prices) - 1) else ""
                    print(f"       {status} Start: {start}, Length: {length} {at_end}")
            
            raise ValueError(f"Constraint validation failed: {error_message}")

        # Create visualization
        title = scenario["name"].replace("_", " ").title()
        filename = f"{scenario['name']}.png"

        visualize_scenario(
            prices=prices,
            selected_indices=selected,
            title=title,
            filename=filename,
            output_dir=output_dir,
            low_price_threshold=params["low_price_threshold"],
            min_selections=params["min_selections"],
            min_consecutive_periods=params["min_consecutive_periods"],
            max_gap_between_periods=params["max_gap_between_periods"],
            max_gap_from_start=params["max_gap_from_start"],
            aggressive=aggressive,
        )

        # Print summary
        total_cost = sum(prices[i] for i in selected)
        avg_cost = total_cost / len(selected) if selected else Decimal(0)
        print(
            f"  Selected {len(selected)} periods, "
            f"Total cost: {total_cost:.2f}, "
            f"Avg cost: {avg_cost:.4f}\n"
        )

    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
