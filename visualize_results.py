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

    # Formatting
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Price (€/kWh)", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"Selected: {len(selected_indices)}/{n} periods | "
        f"Min consecutive: {min_consecutive_periods} | "
        f"Max gap: {max_gap_between_periods}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set x-axis to show hours
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(-0.5, 24.5)

    # Add hour markers
    for hour in range(0, 25, 6):
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
            "generator": generate_realistic_daily_pattern,
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
            "generator": generate_realistic_daily_pattern,
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
            "generator": generate_cheap_day_pattern,
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
            "generator": generate_expensive_day_pattern,
            "params": {
                "low_price_threshold": Decimal("0.12"),
                "min_selections": 16,
                "min_consecutive_periods": 4,
                "max_gap_between_periods": 20,
                "max_gap_from_start": 10,
            },
        },
        {
            "name": "volatile_day",
            "generator": generate_volatile_day_pattern,
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
            "generator": generate_peak_valley_pattern,
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
            "generator": generate_realistic_daily_pattern,
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
    ]

    print(f"Generating {len(scenarios)} test scenarios...")
    print(f"Output directory: {output_dir.absolute()}\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}/{len(scenarios)}: {scenario['name']}")

        # Get prices - either from generator or provided directly
        if "prices" in scenario:
            prices = scenario["prices"]
        else:
            prices = scenario["generator"]()

        # Get parameters
        params = scenario["params"].copy()
        aggressive = params.pop("aggressive", True)

        try:
            # Run algorithm
            selected = get_cheapest_periods(
                prices=prices,
                aggressive=aggressive,
                **params,
            )

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
            )

            # Print summary
            total_cost = sum(prices[i] for i in selected)
            avg_cost = total_cost / len(selected) if selected else Decimal(0)
            print(
                f"  Selected {len(selected)} periods, "
                f"Total cost: {total_cost:.2f}, "
                f"Avg cost: {avg_cost:.4f}\n"
            )

        except Exception as e:
            print(f"  ERROR: {e}\n")

    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
