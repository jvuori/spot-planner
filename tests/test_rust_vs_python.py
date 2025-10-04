"""
Comprehensive data-driven test suite comparing Rust and Python implementations.

This test suite ensures that the Rust implementation behaves identically
to the original Python implementation across various scenarios.
"""

from decimal import Decimal

import pytest

from spot_planner.main import _get_cheapest_periods_python, get_cheapest_periods


class TestRustVsPython:
    """Test suite comparing Rust and Python implementations."""

    @pytest.mark.parametrize(
        "scenario_name,price_data,price_threshold,desired_count,min_period,max_gap,max_start_gap",
        [
            pytest.param(*scenario, id=scenario[0])
            for scenario in [
                # Almost flat prices (small variations)
                (
                    "flat_prices",
                    [Decimal("10.0") + Decimal(str(i * 0.1)) for i in range(20)],
                    Decimal("10.5"),
                    5,
                    2,
                    3,
                    2,
                ),
                # Clear peaks and valleys
                (
                    "peak_valley",
                    [
                        Decimal("50"),
                        Decimal("20"),
                        Decimal("45"),
                        Decimal("15"),
                        Decimal("40"),
                        Decimal("10"),
                        Decimal("35"),
                        Decimal("25"),
                        Decimal("30"),
                        Decimal("20"),
                        Decimal("25"),
                        Decimal("15"),
                        Decimal("20"),
                        Decimal("30"),
                        Decimal("25"),
                        Decimal("35"),
                        Decimal("20"),
                        Decimal("40"),
                        Decimal("15"),
                        Decimal("45"),
                    ],
                    Decimal("25"),
                    8,
                    2,
                    2,
                    1,
                ),
                # Very cheap prices (all below threshold)
                (
                    "cheap_prices",
                    [Decimal("5") + Decimal(str(i)) for i in range(20)],
                    Decimal("15"),
                    10,
                    3,
                    1,
                    1,
                ),
                # Very expensive prices (all above threshold)
                (
                    "expensive_prices",
                    [Decimal("100") + Decimal(str(i * 2)) for i in range(20)],
                    Decimal("50"),
                    5,
                    2,
                    2,
                    1,
                ),
                # Alternating high/low pattern
                (
                    "alternating",
                    [Decimal("50") if i % 2 == 0 else Decimal("10") for i in range(20)],
                    Decimal("30"),
                    6,
                    1,
                    1,
                    1,
                ),
                # Gradual increase
                (
                    "increasing",
                    [Decimal("10") + Decimal(str(i)) for i in range(20)],
                    Decimal("20"),
                    4,
                    2,
                    2,
                    1,
                ),
                # Gradual decrease
                (
                    "decreasing",
                    [Decimal("30") - Decimal(str(i * 0.5)) for i in range(20)],
                    Decimal("20"),
                    6,
                    2,
                    1,
                    1,
                ),
                # Single very cheap price among expensive ones
                (
                    "single_cheap",
                    [Decimal("100")] * 19 + [Decimal("5")],
                    Decimal("10"),
                    1,
                    1,
                    1,
                    1,
                ),
                # Two cheap prices far apart
                (
                    "two_cheap_far",
                    [Decimal("100")] * 5
                    + [Decimal("5")]
                    + [Decimal("100")] * 8
                    + [Decimal("5")]
                    + [Decimal("100")] * 5,
                    Decimal("10"),
                    2,
                    1,
                    10,
                    5,
                ),
                # All same price
                ("same_prices", [Decimal("25")] * 20, Decimal("25"), 8, 2, 1, 1),
                # Extreme values
                (
                    "extreme_values",
                    [Decimal("0.01"), Decimal("999.99")] * 10,
                    Decimal("500"),
                    5,
                    1,
                    1,
                    1,
                ),
                # Very small differences
                (
                    "small_differences",
                    [Decimal("10.0001") + Decimal(str(i * 0.0001)) for i in range(20)],
                    Decimal("10.001"),
                    5,
                    1,
                    1,
                    1,
                ),
            ]
        ],
    )
    def test_rust_vs_python_scenarios(
        self,
        scenario_name,
        price_data,
        price_threshold,
        desired_count,
        min_period,
        max_gap,
        max_start_gap,
    ):
        """Test that Rust and Python implementations produce identical results for all scenarios."""
        try:
            # Get results from both implementations
            rust_result = get_cheapest_periods(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )
            python_result = _get_cheapest_periods_python(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )

            # Results should be identical
            assert rust_result == python_result, (
                f"Scenario '{scenario_name}' failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Price data: {price_data}\n"
                f"Threshold: {price_threshold}\n"
                f"Desired count: {desired_count}\n"
                f"Min period: {min_period}\n"
                f"Max gap: {max_gap}\n"
                f"Max start gap: {max_start_gap}"
            )

        except ValueError as e:
            # Both implementations should raise the same error
            with pytest.raises(ValueError, match=str(e)):
                _get_cheapest_periods_python(
                    price_data,
                    price_threshold,
                    desired_count,
                    min_period,
                    max_gap,
                    max_start_gap,
                )

    @pytest.mark.parametrize(
        "desired_count,min_period,max_gap,max_start_gap",
        [
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (3, 1, 1, 1),
            (5, 1, 1, 1),
            (10, 1, 1, 1),
            (15, 1, 1, 1),
            (20, 1, 1, 1),
            (5, 2, 2, 1),
            (5, 3, 2, 1),
            (5, 5, 2, 1),
            (5, 1, 0, 1),
            (5, 1, 2, 1),
            (5, 1, 3, 1),
            (5, 1, 5, 1),
            (5, 1, 10, 1),
            (5, 1, 3, 0),
            (5, 1, 3, 2),
            (5, 1, 3, 3),
            (5, 1, 3, 5),
            (0, 20, 1, 1),
            (1, 1, 0, 0),
            (10, 1, 20, 20),
        ],
    )
    def test_rust_vs_python_parameter_combinations(
        self, desired_count, min_period, max_gap, max_start_gap
    ):
        """Test various parameter combinations to ensure consistency."""
        # Use a fixed price dataset for parameter testing
        price_data = [
            Decimal("50"),
            Decimal("40"),
            Decimal("30"),
            Decimal("20"),
            Decimal("10"),
            Decimal("15"),
            Decimal("25"),
            Decimal("35"),
            Decimal("45"),
            Decimal("55"),
            Decimal("12"),
            Decimal("22"),
            Decimal("32"),
            Decimal("42"),
            Decimal("52"),
            Decimal("18"),
            Decimal("28"),
            Decimal("38"),
            Decimal("48"),
            Decimal("58"),
        ]
        price_threshold = Decimal("30")

        try:
            # Get results from both implementations
            rust_result = get_cheapest_periods(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )
            python_result = _get_cheapest_periods_python(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )

            # Results should be identical
            assert rust_result == python_result, (
                f"Parameter combination failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Desired count: {desired_count}\n"
                f"Min period: {min_period}\n"
                f"Max gap: {max_gap}\n"
                f"Max start gap: {max_start_gap}"
            )

        except ValueError as e:
            # Both implementations should raise the same error
            with pytest.raises(ValueError, match=str(e)):
                _get_cheapest_periods_python(
                    price_data,
                    price_threshold,
                    desired_count,
                    min_period,
                    max_gap,
                    max_start_gap,
                )

    @pytest.mark.parametrize(
        "price_data,price_threshold,desired_count,min_period,max_gap,max_start_gap",
        [
            # Empty price data
            ([], Decimal("10"), 1, 1, 1, 1),
            # Single price
            ([Decimal("25")], Decimal("30"), 1, 1, 1, 1),
            # Two prices
            ([Decimal("20"), Decimal("30")], Decimal("25"), 1, 1, 1, 1),
            # All prices below threshold
            ([Decimal("5"), Decimal("10"), Decimal("15")], Decimal("20"), 2, 1, 1, 1),
            # All prices above threshold
            ([Decimal("50"), Decimal("60"), Decimal("70")], Decimal("40"), 1, 1, 1, 1),
            # Very strict constraints (impossible to satisfy)
            ([Decimal("10")] * 5, Decimal("5"), 3, 10, 0, 0),
        ],
    )
    def test_rust_vs_python_edge_cases(
        self,
        price_data,
        price_threshold,
        desired_count,
        min_period,
        max_gap,
        max_start_gap,
    ):
        """Test specific edge cases that might cause issues."""
        try:
            rust_result = get_cheapest_periods(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )
            python_result = _get_cheapest_periods_python(
                price_data,
                price_threshold,
                desired_count,
                min_period,
                max_gap,
                max_start_gap,
            )

            assert rust_result == python_result, (
                f"Edge case failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Price data: {price_data}\n"
                f"Threshold: {price_threshold}\n"
                f"Desired count: {desired_count}\n"
                f"Min period: {min_period}\n"
                f"Max gap: {max_gap}\n"
                f"Max start gap: {max_start_gap}"
            )

        except ValueError as e:
            # Both implementations should raise the same error
            with pytest.raises(ValueError, match=str(e)):
                _get_cheapest_periods_python(
                    price_data,
                    price_threshold,
                    desired_count,
                    min_period,
                    max_gap,
                    max_start_gap,
                )

    def test_rust_vs_python_random_scenarios(self):
        """Test with randomly generated scenarios to catch edge cases."""
        import random

        # Set seed for reproducible tests
        random.seed(42)

        for i in range(10):  # Test 10 random scenarios (reduced for speed)
            # Generate random price data (20 prices)
            price_data = [Decimal(str(random.uniform(1, 100))) for _ in range(20)]

            # Generate random parameters
            price_threshold = Decimal(str(random.uniform(10, 80)))
            desired_count = random.randint(0, 10)
            min_period = random.randint(1, 5)
            max_gap = random.randint(1, 5)
            max_start_gap = random.randint(1, max_gap)

            try:
                rust_result = get_cheapest_periods(
                    price_data,
                    price_threshold,
                    desired_count,
                    min_period,
                    max_gap,
                    max_start_gap,
                )
                python_result = _get_cheapest_periods_python(
                    price_data,
                    price_threshold,
                    desired_count,
                    min_period,
                    max_gap,
                    max_start_gap,
                )

                assert rust_result == python_result, (
                    f"Random scenario {i} failed:\n"
                    f"Rust result: {rust_result}\n"
                    f"Python result: {python_result}\n"
                    f"Price data: {price_data}\n"
                    f"Threshold: {price_threshold}\n"
                    f"Desired count: {desired_count}\n"
                    f"Min period: {min_period}\n"
                    f"Max gap: {max_gap}\n"
                    f"Max start gap: {max_start_gap}"
                )

            except ValueError as e:
                # Both implementations should raise the same error
                with pytest.raises(ValueError, match=str(e)):
                    _get_cheapest_periods_python(
                        price_data,
                        price_threshold,
                        desired_count,
                        min_period,
                        max_gap,
                        max_start_gap,
                    )

    @pytest.mark.parametrize(
        "price_threshold,desired_count,min_period,max_gap,max_start_gap",
        [
            (Decimal("10"), 12, 1, 1, 1),
            (Decimal("5"), 15, 2, 1, 1),
            (Decimal("15"), 8, 1, 2, 1),
            (Decimal("8"), 10, 1, 1, 1),
        ],
    )
    def test_rust_vs_python_performance_consistency(
        self, price_threshold, desired_count, min_period, max_gap, max_start_gap
    ):
        """Test that both implementations handle the same performance scenarios consistently."""
        # Use the same data as the original performance test but with more variations
        base_prices = [Decimal(f"{i}") for i in range(20)]

        rust_result = get_cheapest_periods(
            base_prices,
            price_threshold,
            desired_count,
            min_period,
            max_gap,
            max_start_gap,
        )
        python_result = _get_cheapest_periods_python(
            base_prices,
            price_threshold,
            desired_count,
            min_period,
            max_gap,
            max_start_gap,
        )

        assert rust_result == python_result, (
            f"Performance test failed:\n"
            f"Rust result: {rust_result}\n"
            f"Python result: {python_result}\n"
            f"Parameters: threshold={price_threshold}, count={desired_count}, "
            f"min_period={min_period}, max_gap={max_gap}, max_start_gap={max_start_gap}"
        )

    def test_rust_vs_python_decimal_precision(self):
        """Test that both implementations handle decimal precision identically."""
        # Test with high precision decimals
        high_precision_prices = [
            Decimal("10.123456789"),
            Decimal("20.987654321"),
            Decimal("15.555555555"),
            Decimal("25.111111111"),
            Decimal("30.999999999"),
        ]

        price_threshold = Decimal("20.5")

        rust_result = get_cheapest_periods(
            high_precision_prices, price_threshold, 2, 1, 1, 1
        )
        python_result = _get_cheapest_periods_python(
            high_precision_prices, price_threshold, 2, 1, 1, 1
        )

        assert rust_result == python_result, (
            f"Decimal precision test failed:\n"
            f"Rust result: {rust_result}\n"
            f"Python result: {python_result}\n"
            f"High precision prices: {high_precision_prices}"
        )
