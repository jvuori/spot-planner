# Client Update Guide: Removing max_consecutive_periods

## Overview

The `get_cheapest_periods` function has been simplified to remove the `max_consecutive_periods` parameter and the internal dynamic calculation logic. The function now uses `min_consecutive_periods` directly as provided by the client.

## What Changed

### Before (Old API)
```python
from spot_planner.main import get_cheapest_periods
from decimal import Decimal

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=12,
    min_consecutive_periods=2,
    max_consecutive_periods=5,  # ❌ This parameter is removed
    max_gap_between_periods=3,
    max_gap_from_start=2,
)
```

### After (New API)
```python
from spot_planner.main import get_cheapest_periods
from decimal import Decimal

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=12,
    min_consecutive_periods=2,  # ✅ Use this directly
    max_gap_between_periods=3,
    max_gap_from_start=2,
)
```

## Migration Steps

1. **Remove `max_consecutive_periods` parameter** from all `get_cheapest_periods` calls
2. **Decide on `min_consecutive_periods` value** based on your requirements
3. **Optionally implement dynamic logic** on the client side if needed (see examples below)

## Implementing Dynamic Logic on the Client Side

If you previously relied on the automatic dynamic calculation, you can now implement similar logic in your application. Here are several approaches:

### Approach 1: Simple Percentage-Based Calculation

Calculate `min_consecutive_periods` based on the percentage of `min_selections` relative to total prices:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods

def calculate_min_consecutive_periods(
    min_selections: int,
    total_prices: int,
    min_consecutive: int = 2,
    max_consecutive: int = 5,
) -> int:
    """
    Calculate min_consecutive_periods based on percentage of selections.
    
    Args:
        min_selections: Minimum number of periods to select
        total_prices: Total number of price periods available
        min_consecutive: Minimum allowed consecutive periods (default: 2)
        max_consecutive: Maximum allowed consecutive periods (default: 5)
    
    Returns:
        Calculated min_consecutive_periods value
    """
    if total_prices == 0:
        return min_consecutive
    
    percentage = min_selections / total_prices
    
    # < 25% of total prices: use min_consecutive
    if percentage <= 0.25:
        return min_consecutive
    # > 75% of total prices: use max_consecutive
    elif percentage >= 0.75:
        return max_consecutive
    # Between 25-75%: linear interpolation
    else:
        interpolation_factor = (percentage - 0.25) / (0.75 - 0.25)
        return int(min_consecutive + interpolation_factor * (max_consecutive - min_consecutive))


# Usage example
prices = [Decimal(str(i)) for i in range(24)]  # 24 hours
min_selections = 12  # 50% of 24

min_consecutive_periods = calculate_min_consecutive_periods(
    min_selections=min_selections,
    total_prices=len(prices),
    min_consecutive=2,
    max_consecutive=5,
)

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=min_selections,
    min_consecutive_periods=min_consecutive_periods,
    max_gap_between_periods=3,
    max_gap_from_start=2,
)
```

### Approach 2: Percentage-Based with Gap Adjustment

Include gap adjustment logic similar to the old implementation:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods

def calculate_min_consecutive_periods_with_gaps(
    min_selections: int,
    total_prices: int,
    max_gap_between_periods: int,
    min_consecutive: int = 2,
    max_consecutive: int = 5,
) -> int:
    """
    Calculate min_consecutive_periods with gap adjustment.
    
    Args:
        min_selections: Minimum number of periods to select
        total_prices: Total number of price periods available
        max_gap_between_periods: Maximum gap between periods
        min_consecutive: Minimum allowed consecutive periods (default: 2)
        max_consecutive: Maximum allowed consecutive periods (default: 5)
    
    Returns:
        Calculated min_consecutive_periods value
    """
    if total_prices == 0:
        return min_consecutive
    
    percentage = min_selections / total_prices
    
    # Base calculation based on percentage
    if percentage <= 0.25:
        base_consecutive = min_consecutive
    elif percentage >= 0.75:
        base_consecutive = max_consecutive
    else:
        interpolation_factor = (percentage - 0.25) / (0.75 - 0.25)
        base_consecutive = int(
            min_consecutive + interpolation_factor * (max_consecutive - min_consecutive)
        )
    
    # Gap adjustment: larger gaps push toward max_consecutive
    gap_factor = min(max_gap_between_periods / 10.0, 1.0)  # Normalize to 0-1
    gap_adjustment = int(gap_factor * (max_consecutive - min_consecutive))
    
    # Final calculation: base + gap adjustment
    dynamic_consecutive = base_consecutive + gap_adjustment
    
    # Ensure result is within bounds
    return max(min_consecutive, min(dynamic_consecutive, max_consecutive))


# Usage example
prices = [Decimal(str(i)) for i in range(24)]
min_selections = 9  # 37.5% of 24
max_gap_between_periods = 8  # Large gap

min_consecutive_periods = calculate_min_consecutive_periods_with_gaps(
    min_selections=min_selections,
    total_prices=len(prices),
    max_gap_between_periods=max_gap_between_periods,
    min_consecutive=2,
    max_consecutive=5,
)

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=min_selections,
    min_consecutive_periods=min_consecutive_periods,
    max_gap_between_periods=max_gap_between_periods,
    max_gap_from_start=2,
)
```

### Approach 3: Context-Aware Calculation

Use external context (e.g., season, weather, historical data) to determine the appropriate value:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods
from datetime import datetime
from enum import Enum

class Season(Enum):
    SUMMER = "summer"
    WINTER = "winter"
    SPRING = "spring"
    FALL = "fall"


def get_season() -> Season:
    """Determine current season based on date."""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return Season.WINTER
    elif month in [3, 4, 5]:
        return Season.SPRING
    elif month in [6, 7, 8]:
        return Season.SUMMER
    else:
        return Season.FALL


def calculate_min_consecutive_periods_by_context(
    min_selections: int,
    total_prices: int,
    season: Season = None,
    historical_heating_hours: float = None,
) -> int:
    """
    Calculate min_consecutive_periods based on context.
    
    Args:
        min_selections: Minimum number of periods to select
        total_prices: Total number of price periods available
        season: Current season (optional)
        historical_heating_hours: Average heating hours per day (optional)
    
    Returns:
        Calculated min_consecutive_periods value
    """
    if season is None:
        season = get_season()
    
    # Base values by season
    season_bases = {
        Season.SUMMER: 1,
        Season.SPRING: 2,
        Season.FALL: 2,
        Season.WINTER: 3,
    }
    
    base_consecutive = season_bases.get(season, 2)
    
    # Adjust based on historical heating hours if available
    if historical_heating_hours is not None:
        percentage = min_selections / total_prices
        if historical_heating_hours > 18:  # High heating demand
            base_consecutive = max(base_consecutive, 3)
        elif historical_heating_hours < 6:  # Low heating demand
            base_consecutive = min(base_consecutive, 1)
    
    # Adjust based on percentage of selections
    percentage = min_selections / total_prices
    if percentage > 0.75:
        base_consecutive = max(base_consecutive, 4)
    elif percentage < 0.25:
        base_consecutive = min(base_consecutive, 1)
    
    return base_consecutive


# Usage example
prices = [Decimal(str(i)) for i in range(24)]
min_selections = 12

min_consecutive_periods = calculate_min_consecutive_periods_by_context(
    min_selections=min_selections,
    total_prices=len(prices),
    season=Season.WINTER,
    historical_heating_hours=16.5,
)

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=min_selections,
    min_consecutive_periods=min_consecutive_periods,
    max_gap_between_periods=3,
    max_gap_from_start=2,
)
```

### Approach 4: Fixed Value Based on Requirements

If you have specific requirements, use a fixed value:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods

# Simple fixed value based on your heating system requirements
MIN_CONSECUTIVE_PERIODS = 2  # Your heating system needs at least 2 consecutive hours

prices = [Decimal(str(i)) for i in range(24)]
min_selections = 12

result = get_cheapest_periods(
    prices=prices,
    low_price_threshold=Decimal("20.0"),
    min_selections=min_selections,
    min_consecutive_periods=MIN_CONSECUTIVE_PERIODS,
    max_gap_between_periods=3,
    max_gap_from_start=2,
)
```

## Complete Migration Example

Here's a complete example showing how to migrate existing code:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods

# OLD CODE (before migration)
def get_heating_periods_old(prices, threshold, min_selections):
    return get_cheapest_periods(
        prices=prices,
        low_price_threshold=threshold,
        min_selections=min_selections,
        min_consecutive_periods=2,
        max_consecutive_periods=5,  # ❌ Remove this
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )


# NEW CODE (after migration) - Option 1: Fixed value
def get_heating_periods_new_fixed(prices, threshold, min_selections):
    return get_cheapest_periods(
        prices=prices,
        low_price_threshold=threshold,
        min_selections=min_selections,
        min_consecutive_periods=2,  # ✅ Fixed value
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )


# NEW CODE (after migration) - Option 2: Dynamic calculation
def calculate_dynamic_min_consecutive(min_selections, total_prices, min_val=2, max_val=5):
    """Calculate min_consecutive_periods dynamically."""
    if total_prices == 0:
        return min_val
    percentage = min_selections / total_prices
    if percentage <= 0.25:
        return min_val
    elif percentage >= 0.75:
        return max_val
    else:
        factor = (percentage - 0.25) / (0.75 - 0.25)
        return int(min_val + factor * (max_val - min_val))


def get_heating_periods_new_dynamic(prices, threshold, min_selections):
    min_consecutive = calculate_dynamic_min_consecutive(
        min_selections=min_selections,
        total_prices=len(prices),
        min_val=2,
        max_val=5,
    )
    return get_cheapest_periods(
        prices=prices,
        low_price_threshold=threshold,
        min_selections=min_selections,
        min_consecutive_periods=min_consecutive,  # ✅ Calculated dynamically
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )


# Usage
prices = [Decimal(str(i)) for i in range(24)]
threshold = Decimal("20.0")
min_selections = 12

# Choose one approach:
result = get_heating_periods_new_fixed(prices, threshold, min_selections)
# OR
result = get_heating_periods_new_dynamic(prices, threshold, min_selections)
```

## Key Points

1. **`max_consecutive_periods` is removed** - Remove it from all function calls
2. **`min_consecutive_periods` is now direct** - Use the value you want directly
3. **Dynamic logic is optional** - Implement it on the client side if needed
4. **Flexibility** - You can now implement any custom logic that fits your use case

## Testing Your Migration

After updating your code, test it to ensure it works correctly:

```python
from decimal import Decimal
from spot_planner.main import get_cheapest_periods

# Test with various scenarios
test_cases = [
    {
        "prices": [Decimal("50"), Decimal("40"), Decimal("30"), Decimal("20"), Decimal("10")],
        "threshold": Decimal("25"),
        "min_selections": 2,
        "min_consecutive_periods": 1,
    },
    {
        "prices": [Decimal("50"), Decimal("40"), Decimal("30"), Decimal("20"), Decimal("10")],
        "threshold": Decimal("25"),
        "min_selections": 3,
        "min_consecutive_periods": 2,
    },
]

for case in test_cases:
    result = get_cheapest_periods(
        prices=case["prices"],
        low_price_threshold=case["threshold"],
        min_selections=case["min_selections"],
        min_consecutive_periods=case["min_consecutive_periods"],
        max_gap_between_periods=3,
        max_gap_from_start=2,
    )
    print(f"Result: {result}")
    assert len(result) >= case["min_selections"]
```

## Questions?

If you need help with your specific use case, consider:
- What factors should influence `min_consecutive_periods` in your application?
- Do you have historical data or context that can help determine the value?
- Is a fixed value sufficient, or do you need dynamic calculation?

