#!/usr/bin/env python3

from decimal import Decimal


def group_consecutive_items(items):
    """Group cheap items into consecutive runs."""
    if not items:
        return []

    groups = []
    current_group = [items[0]]

    for i in range(1, len(items)):
        if items[i][0] == items[i - 1][0] + 1:
            current_group.append(items[i])
        else:
            groups.append(current_group)
            current_group = [items[i]]
    groups.append(current_group)

    return groups


def analyze_cheap_groups():
    """Analyze the cheap groups for the problematic input"""
    prices = [
        Decimal("3.17"),
        Decimal("2.98"),
        Decimal("3.702"),
        Decimal("5.067"),
        Decimal("4.19"),
        Decimal("4.692"),
        Decimal("4.493"),
        Decimal("3.813"),
        Decimal("4.902"),
        Decimal("3.559"),
        Decimal("2.396"),
        Decimal("1.758"),
        Decimal("2.599"),
        Decimal("1.531"),
        Decimal("1.25"),
        Decimal("1.134"),
        Decimal("1.167"),
        Decimal("1.077"),
        Decimal("0.583"),
        Decimal("0.404"),
        Decimal("0.508"),
        Decimal("0.473"),
        Decimal("0.4"),
        Decimal("0.341"),
    ]

    low_price_threshold = Decimal("2.554812749003984063745019920")

    price_items = list(enumerate(prices))
    cheap_items = [(i, p) for i, p in price_items if p <= low_price_threshold]

    print("Cheap items (index, price):")
    for idx, price in cheap_items:
        print(f"  {idx}: {price}")

    print(f"\nTotal cheap items: {len(cheap_items)}")

    # Group consecutive items
    cheap_groups = group_consecutive_items(cheap_items)

    print(f"\nCheap groups: {len(cheap_groups)}")
    for i, group in enumerate(cheap_groups):
        indices = [idx for idx, _ in group]
        print(f"  Group {i}: indices {indices} (length {len(group)})")

    # Calculate total combinations of groups
    total_group_combinations = 2 ** len(cheap_groups)
    print(
        f"\nTotal group combinations: 2^{len(cheap_groups)} = {total_group_combinations:,}"
    )

    # Show some example combinations
    print("\nFirst 10 group combinations:")
    for group_mask in range(min(10, total_group_combinations)):
        selected_groups = []
        for group_idx, group in enumerate(cheap_groups):
            if group_mask & (1 << group_idx):
                selected_groups.append(group_idx)
        print(f"  Mask {group_mask}: groups {selected_groups}")


if __name__ == "__main__":
    analyze_cheap_groups()
