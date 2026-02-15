from decimal import Decimal
from spot_planner import main

# Create test data
prices = []
for i in range(96):
    if i % 4 == 0:  # Every 4th item is cheap
        prices.append(Decimal("1"))
    else:
        prices.append(Decimal("10"))

print(f"Total prices: {len(prices)}")
cheap_indices = [i for i in range(len(prices)) if prices[i] == Decimal("1")]
print(f"Cheap indices (price=1): {cheap_indices[:10]}... (total: {len(cheap_indices)})")

result = main.get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("2"),
    min_selections=20,
    min_consecutive_periods=1,
    max_gap_between_periods=10,
    max_gap_from_start=10,
)

cheap_selected = sum(1 for idx in result if prices[idx] == Decimal("1"))
print(f"\nResult:")
print(f"  Total selected: {len(result)}")
print(f"  Cheap selected: {cheap_selected}")
print(f"  Ratio: {cheap_selected}/{len(result)} = {cheap_selected/len(result):.2%}")
print(f"  Expected: > {len(result) // 2}")
print(f"  Test {'PASS' if cheap_selected > len(result) // 2 else 'FAIL'}")

# Show first 30 selections
print(f"\nFirst 30 selections:")
for idx in result[:30]:
    print(f"  {idx}: ${prices[idx]}")
