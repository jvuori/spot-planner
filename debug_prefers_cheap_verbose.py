from decimal import Decimal
from spot_planner import main, two_phase

# Monkey-patch validation
original_validate = two_phase._validate_full_selection

def debug_validate(selected, total_periods, min_consecutive, max_gap_between, max_gap_from_start):
    print(f"\n=== Validation ===")
    print(f"Selected: {sorted(selected)}")
    print(f"Total: {len(selected)}, min_consecutive: {min_consecutive}, max_gap_between: {max_gap_between}, max_gap_from_start: {max_gap_from_start}")
    
    # Check gaps
    if selected:
        print(f"Gap from start: {selected[0]}")
        for i in range(len(selected) - 1):
            gap = selected[i + 1] - selected[i] - 1
            if gap > 0:
                print(f"Gap {selected[i]+1}-{selected[i+1]-1}: {gap}")
        end_gap = total_periods - 1 - selected[-1]
        print(f"End gap: {end_gap}")
    
    return original_validate(selected, total_periods, min_consecutive, max_gap_between, max_gap_from_start)

two_phase._validate_full_selection = debug_validate

# Test
prices = []
for i in range(96):
    if i % 4 == 0:
        prices.append(Decimal("1"))
    else:
        prices.append(Decimal("10"))

try:
    result = main.get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal("2"),
        min_selections=20,
        min_consecutive_periods=1,
        max_gap_between_periods=10,
        max_gap_from_start=10,
    )
    print(f"\n✓ Success! Selected {len(result)} periods")
except ValueError as e:
    print(f"\n✗ Failed: {e}")
