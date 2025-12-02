# Visualization Results

This directory contains bar chart visualizations of the spot planner algorithm results for 96-item price sequences (15-minute resolution, 24 hours).

## Generated Visualizations

### 1. `realistic_daily.png`

- **Pattern**: Realistic daily electricity price pattern
- **Parameters**:
  - Threshold: 0.10 €/kWh
  - Min selections: 24 periods (6 hours)
  - Min consecutive: 4 periods (1 hour)
  - Max gap: 8 periods (2 hours)
- **Result**: 69 periods selected, avg cost: 0.0872 €/kWh

### 2. `realistic_daily_tight.png`

- **Pattern**: Realistic daily pattern with tighter constraints
- **Parameters**:
  - Threshold: 0.10 €/kWh
  - Min selections: 32 periods (8 hours)
  - Min consecutive: 6 periods (1.5 hours)
  - Max gap: 4 periods (1 hour)
- **Result**: 84 periods selected, avg cost: 0.0960 €/kWh

### 3. `cheap_day.png`

- **Pattern**: Day with mostly cheap prices (windy/sunny day)
- **Parameters**:
  - Threshold: 0.08 €/kWh
  - Min selections: 20 periods (5 hours)
  - Min consecutive: 4 periods (1 hour)
  - Max gap: 10 periods (2.5 hours)
- **Result**: 49 periods selected, avg cost: 0.0422 €/kWh

### 4. `expensive_day.png`

- **Pattern**: Day with mostly expensive prices (cold winter, low renewables)
- **Parameters**:
  - Threshold: 0.12 €/kWh
  - Min selections: 16 periods (4 hours)
  - Min consecutive: 4 periods (1 hour)
  - Max gap: 12 periods (3 hours)
- **Result**: 29 periods selected, avg cost: 0.1173 €/kWh

### 5. `volatile_day.png`

- **Pattern**: High price volatility throughout the day
- **Parameters**:
  - Threshold: 0.10 €/kWh
  - Min selections: 24 periods (6 hours)
  - Min consecutive: 3 periods (45 minutes)
  - Max gap: 10 periods (2.5 hours)
- **Result**: 38 periods selected, avg cost: 0.1024 €/kWh

### 6. `peak_valley.png`

- **Pattern**: Clear peak-valley pattern (cheap nights, expensive peaks)
- **Parameters**:
  - Threshold: 0.10 €/kWh
  - Min selections: 28 periods (7 hours)
  - Min consecutive: 4 periods (1 hour)
  - Max gap: 6 periods (1.5 hours)
- **Result**: 76 periods selected, avg cost: 0.0863 €/kWh

### 7. `realistic_conservative.png`

- **Pattern**: Realistic daily pattern using conservative mode
- **Parameters**:
  - Threshold: 0.10 €/kWh
  - Min selections: 24 periods (6 hours)
  - Min consecutive: 4 periods (1 hour)
  - Max gap: 8 periods (2 hours)
  - **Mode**: Conservative (maximizes cheap items)
- **Result**: 57 periods selected, avg cost: 0.0839 €/kWh

## Chart Interpretation

- **Green bars**: Selected periods (meet all constraints and are cost-optimized)
- **Gray bars**: Non-selected periods
- **Red dashed line**: Price threshold (low_price_threshold)
- **X-axis**: Time in hours (0-24)
- **Y-axis**: Price in €/kWh

## Key Observations

1. **Constraint adherence**: Selected periods form consecutive blocks meeting `min_consecutive_periods`
2. **Gap constraints**: Gaps between selected periods respect `max_gap_between_periods`
3. **Cost optimization**: Algorithm prefers cheaper periods while respecting constraints
4. **Boundary handling**: Constraints are maintained across chunk boundaries (for 96 items, uses extended algorithm)

## Regenerating Visualizations

To regenerate with different parameters or patterns, edit `visualize_results.py` and run:

```bash
uv run python visualize_results.py
```
