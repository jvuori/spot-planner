"""Debug script for realistic_daily scenario."""
from decimal import Decimal
import visualize_results
from spot_planner import main

# Monkey-patch validation to print detailed violations
from spot_planner import two_phase

original_validate = two_phase._validate_full_selection

def debug_validate(selected, total_periods, min_consecutive, max_gap_between, max_gap_from_start):
    """Debug version that prints details before validation."""
    print("\nGenerated selection before validation:")
    print(f"  Total selected: {len(selected)}")
    print(f"  Indices: {selected[:50]}...")  # First 50
    
    # Check consecutive runs
    runs = []
    if selected:
        current_run_start = selected[0]
        current_run_end = selected[0]
        
        for i in range(1, len(selected)):
            if selected[i] == current_run_end + 1:
                current_run_end = selected[i]
            else:
                runs.append((current_run_start, current_run_end, current_run_end - current_run_start + 1))
                current_run_start = selected[i]
                current_run_end = selected[i]
        
        # Add last run
        runs.append((current_run_start, current_run_end, current_run_end - current_run_start + 1))
    
    print(f"  Consecutive runs: {len(runs)}")
    for start, end, length in runs:
        status = "✓" if length >= min_consecutive else "✗"
        print(f"    [{start}-{end}] length={length} {status}")
    
    # Check gaps
    if selected:
        print(f"\n  Gap from start: {selected[0]} {'✓' if selected[0] <= max_gap_from_start else '✗'}")
        
        for i in range(len(selected) - 1):
            gap = selected[i + 1] - selected[i] - 1
            if gap > 0:
                status = "✓" if gap <= max_gap_between else "✗"
                print(f"  Gap at {selected[i] + 1}-{selected[i + 1] - 1}: {gap} {status}")
        
        # Check end gap
        end_gap = total_periods - 1 - selected[-1]
        if end_gap > 0:
            status = "✓" if end_gap <= max_gap_between else "✗"
            print(f"  End gap {end_gap} {status}")
    
    return original_validate(selected, total_periods, min_consecutive, max_gap_between, max_gap_from_start)

two_phase._validate_full_selection = debug_validate

# Test realistic_daily
print("Testing realistic_daily with 96 prices")
prices = visualize_results.generate_realistic_daily_pattern()
print(f"Parameters: min_selections=24, min_consecutive=4, max_gap_between=20, max_gap_from_start=10")

try:
    result = main.get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("0.10"),
        min_selections=24,
        min_consecutive_periods=4,
        max_gap_between_periods=20,
        max_gap_from_start=10,
        aggressive=False,
    )
    print(f"\nSuccess! Selected {len(result)} periods")
except ValueError as e:
    print(f"\n❌ Error: {e}")
