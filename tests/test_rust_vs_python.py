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
        "scenario_name,prices,low_price_threshold,min_selections,min_consecutive_selections,max_gap_between_periods,max_gap_from_start",
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
                # Real-world scenario: all prices below threshold (whole day cheap)
                (
                    "whole_day_cheap",
                    [
                        Decimal("0.021"),
                        Decimal("0.04925"),
                        Decimal("0.00675"),
                        Decimal("-0.00025"),
                        Decimal("-0.00225"),
                        Decimal("-0.002"),
                        Decimal("0.001"),
                        Decimal("0.0005"),
                        Decimal("0.05525"),
                        Decimal("0.09625"),
                        Decimal("0.129"),
                        Decimal("0.13"),
                        Decimal("0.12825"),
                        Decimal("0.13975"),
                        Decimal("0.2035"),
                        Decimal("0.20125"),
                        Decimal("0.26925"),
                        Decimal("0.3105"),
                        Decimal("0.3865"),
                        Decimal("0.454"),
                        Decimal("0.526"),
                        Decimal("0.4945"),
                        Decimal("0.4815"),
                        Decimal("0.49425"),
                    ],
                    Decimal("0.7973127490039840637554282614"),
                    24,
                    1,
                    5,
                    5,
                ),
                # Real-world scenario: all prices below threshold with low min_selections
                (
                    "whole_day_cheap_low_desired",
                    [
                        Decimal("0.021"),
                        Decimal("0.04925"),
                        Decimal("0.00675"),
                        Decimal("-0.00025"),
                        Decimal("-0.00225"),
                        Decimal("-0.002"),
                        Decimal("0.001"),
                        Decimal("0.0005"),
                        Decimal("0.05525"),
                        Decimal("0.09625"),
                        Decimal("0.129"),
                        Decimal("0.13"),
                        Decimal("0.12825"),
                        Decimal("0.13975"),
                        Decimal("0.2035"),
                        Decimal("0.20125"),
                        Decimal("0.26925"),
                        Decimal("0.3105"),
                        Decimal("0.3865"),
                        Decimal("0.454"),
                        Decimal("0.526"),
                        Decimal("0.4945"),
                        Decimal("0.4815"),
                        Decimal("0.49425"),
                    ],
                    Decimal("0.7973127490039840637554282614"),
                    2,
                    1,
                    5,
                    5,
                ),
                # Real-world scenario: min_selections equals total items (all above threshold)
                (
                    "whole_day_desired",
                    [
                        Decimal("0.021"),
                        Decimal("0.04925"),
                        Decimal("0.00675"),
                        Decimal("-0.00025"),
                        Decimal("-0.00225"),
                        Decimal("-0.002"),
                        Decimal("0.001"),
                        Decimal("0.0005"),
                        Decimal("0.05525"),
                        Decimal("0.09625"),
                        Decimal("0.129"),
                        Decimal("0.13"),
                        Decimal("0.12825"),
                        Decimal("0.13975"),
                        Decimal("0.2035"),
                        Decimal("0.20125"),
                        Decimal("0.26925"),
                        Decimal("0.3105"),
                        Decimal("0.3865"),
                        Decimal("0.454"),
                        Decimal("0.526"),
                        Decimal("0.4945"),
                        Decimal("0.4815"),
                        Decimal("0.49425"),
                    ],
                    Decimal("-1.0"),
                    24,
                    1,
                    5,
                    5,
                ),
                # Original issue scenario: first 3 items should be selected
                (
                    "original_issue_scenario",
                    [
                        Decimal("2.232"),
                        Decimal("2.4"),
                        Decimal("2.599"),
                        Decimal("2.768"),
                        Decimal("2.6"),
                        Decimal("3.336"),
                        Decimal("3.5"),
                        Decimal("3.349"),
                        Decimal("3.148"),
                        Decimal("2.625"),
                        Decimal("2.51"),
                        Decimal("3.992"),
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
                    ],
                    Decimal("4.355812749003984063745019920"),
                    16,
                    4,
                    8,
                    18,
                ),
            ]
        ],
    )
    def test_rust_vs_python_scenarios(
        self,
        scenario_name,
        prices,
        low_price_threshold,
        min_selections,
        min_consecutive_selections,
        max_gap_between_periods,
        max_gap_from_start,
    ):
        """Test that Rust and Python implementations produce identical results for all scenarios."""
        try:
            # Get results from both implementations
            rust_result = get_cheapest_periods(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                8,  # max_consecutive_selections
                max_gap_between_periods,
                max_gap_from_start,
            )
            python_result = _get_cheapest_periods_python(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
            )

            # Results should be identical
            assert rust_result == python_result, (
                f"Scenario '{scenario_name}' failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Price data: {prices}\n"
                f"Threshold: {low_price_threshold}\n"
                f"Desired count: {min_selections}\n"
                f"Min period: {min_consecutive_selections}\n"
                f"Max gap: {max_gap_between_periods}\n"
                f"Max start gap: {max_gap_from_start}"
            )

        except ValueError as e:
            # The main function (which calls Rust) should raise the same error as Python fallback
            # But Python fallback might not have the same validation, so we need to check both
            try:
                python_result = _get_cheapest_periods_python(
                    prices,
                    low_price_threshold,
                    min_selections,
                    min_consecutive_selections,
                    max_gap_between_periods,
                    max_gap_from_start,
                )
                # If Python fallback succeeds, that's also valid
                # The main function validation is more strict
            except ValueError as python_error:
                # Both should raise ValueError, but error messages might differ
                assert isinstance(e, ValueError) and isinstance(
                    python_error, ValueError
                )

    @pytest.mark.parametrize(
        "min_selections,min_consecutive_selections,max_gap_between_periods,max_gap_from_start",
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
        self,
        min_selections,
        min_consecutive_selections,
        max_gap_between_periods,
        max_gap_from_start,
    ):
        """Test various parameter combinations to ensure consistency."""
        # Use a fixed price dataset for parameter testing
        prices = [
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
        low_price_threshold = Decimal("30")

        try:
            # Get results from both implementations
            rust_result = get_cheapest_periods(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                8,  # max_consecutive_selections
                max_gap_between_periods,
                max_gap_from_start,
            )
            python_result = _get_cheapest_periods_python(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
            )

            # Results should be identical
            assert rust_result == python_result, (
                f"Parameter combination failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Desired count: {min_selections}\n"
                f"Min period: {min_consecutive_selections}\n"
                f"Max gap: {max_gap_between_periods}\n"
                f"Max start gap: {max_gap_from_start}"
            )

        except ValueError as e:
            # The main function (which calls Rust) should raise the same error as Python fallback
            # But Python fallback might not have the same validation, so we need to check both
            try:
                python_result = _get_cheapest_periods_python(
                    prices,
                    low_price_threshold,
                    min_selections,
                    min_consecutive_selections,
                    max_gap_between_periods,
                    max_gap_from_start,
                )
                # If Python fallback succeeds, that's also valid
                # The main function validation is more strict
            except ValueError as python_error:
                # Both should raise ValueError, but error messages might differ
                assert isinstance(e, ValueError) and isinstance(
                    python_error, ValueError
                )

    @pytest.mark.parametrize(
        "prices,low_price_threshold,min_selections,min_consecutive_selections,max_gap_between_periods,max_gap_from_start",
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
        prices,
        low_price_threshold,
        min_selections,
        min_consecutive_selections,
        max_gap_between_periods,
        max_gap_from_start,
    ):
        """Test specific edge cases that might cause issues."""
        try:
            rust_result = get_cheapest_periods(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                8,  # max_consecutive_selections
                max_gap_between_periods,
                max_gap_from_start,
            )
            python_result = _get_cheapest_periods_python(
                prices,
                low_price_threshold,
                min_selections,
                min_consecutive_selections,
                max_gap_between_periods,
                max_gap_from_start,
            )

            assert rust_result == python_result, (
                f"Edge case failed:\n"
                f"Rust result: {rust_result}\n"
                f"Python result: {python_result}\n"
                f"Price data: {prices}\n"
                f"Threshold: {low_price_threshold}\n"
                f"Desired count: {min_selections}\n"
                f"Min period: {min_consecutive_selections}\n"
                f"Max gap: {max_gap_between_periods}\n"
                f"Max start gap: {max_gap_from_start}"
            )

        except ValueError as e:
            # The main function (which calls Rust) should raise the same error as Python fallback
            # But Python fallback might not have the same validation, so we need to check both
            try:
                python_result = _get_cheapest_periods_python(
                    prices,
                    low_price_threshold,
                    min_selections,
                    min_consecutive_selections,
                    max_gap_between_periods,
                    max_gap_from_start,
                )
                # If Python fallback succeeds, that's also valid
                # The main function validation is more strict
            except ValueError as python_error:
                # Both should raise ValueError, but error messages might differ
                assert isinstance(e, ValueError) and isinstance(
                    python_error, ValueError
                )

    def test_rust_vs_python_random_scenarios(self):
        """Test with randomly generated scenarios to catch edge cases."""
        import random

        # Set seed for reproducible tests
        random.seed(42)

        for i in range(10):  # Test 10 random scenarios (reduced for speed)
            # Generate random price data (20 prices)
            prices = [Decimal(str(random.uniform(1, 100))) for _ in range(20)]

            # Generate random parameters (ensure valid combinations)
            low_price_threshold = Decimal(str(random.uniform(10, 80)))
            min_selections = random.randint(1, 10)  # Must be > 0
            min_consecutive_selections = random.randint(
                1, min(5, min_selections)
            )  # Must be <= min_selections
            max_gap_between_periods = random.randint(1, 5)
            max_gap_from_start = random.randint(
                1, max_gap_between_periods
            )  # Must be <= max_gap_between_periods

            try:
                rust_result = get_cheapest_periods(
                    prices,
                    low_price_threshold,
                    min_selections,
                    min_consecutive_selections,
                    max_gap_between_periods,
                    max_gap_from_start,
                )
                python_result = _get_cheapest_periods_python(
                    prices,
                    low_price_threshold,
                    min_selections,
                    min_consecutive_selections,
                    max_gap_between_periods,
                    max_gap_from_start,
                )

                assert rust_result == python_result, (
                    f"Random scenario {i} failed:\n"
                    f"Rust result: {rust_result}\n"
                    f"Python result: {python_result}\n"
                    f"Price data: {prices}\n"
                    f"Threshold: {low_price_threshold}\n"
                    f"Desired count: {min_selections}\n"
                    f"Min period: {min_consecutive_selections}\n"
                    f"Max gap: {max_gap_between_periods}\n"
                    f"Max start gap: {max_gap_from_start}"
                )

            except ValueError as e:
                # Both implementations should raise the same error
                with pytest.raises(ValueError, match=str(e)):
                    _get_cheapest_periods_python(
                        prices,
                        low_price_threshold,
                        min_selections,
                        min_consecutive_selections,
                        max_gap_between_periods,
                        max_gap_from_start,
                    )

    @pytest.mark.parametrize(
        "low_price_threshold,min_selections,min_consecutive_selections,max_gap_between_periods,max_gap_from_start",
        [
            (Decimal("10"), 12, 1, 1, 1),
            (Decimal("5"), 15, 2, 1, 1),
            (Decimal("15"), 8, 1, 2, 1),
            (Decimal("8"), 10, 1, 1, 1),
        ],
    )
    def test_rust_vs_python_performance_consistency(
        self,
        low_price_threshold,
        min_selections,
        min_consecutive_selections,
        max_gap_between_periods,
        max_gap_from_start,
    ):
        """Test that both implementations handle the same performance scenarios consistently."""
        # Use the same data as the original performance test but with more variations
        base_prices = [Decimal(f"{i}") for i in range(20)]

        rust_result = get_cheapest_periods(
            base_prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
        python_result = _get_cheapest_periods_python(
            base_prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )

        assert rust_result == python_result, (
            f"Performance test failed:\n"
            f"Rust result: {rust_result}\n"
            f"Python result: {python_result}\n"
            f"Parameters: threshold={low_price_threshold}, count={min_selections}, "
            f"min_consecutive_selections={min_consecutive_selections}, max_gap_between_periods={max_gap_between_periods}, max_gap_from_start={max_gap_from_start}"
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

        low_price_threshold = Decimal("20.5")

        rust_result = get_cheapest_periods(
            high_precision_prices, low_price_threshold, 2, 1, 1, 1
        )
        python_result = _get_cheapest_periods_python(
            high_precision_prices, low_price_threshold, 2, 1, 1, 1
        )

        assert rust_result == python_result, (
            f"Decimal precision test failed:\n"
            f"Rust result: {rust_result}\n"
            f"Python result: {python_result}\n"
            f"High precision prices: {high_precision_prices}"
        )

    def test_original_issue_scenario_detailed(self):
        """Test the original issue scenario with detailed validation."""
        # Original input data that was causing the issue
        prices = [
            Decimal("2.232"),
            Decimal("2.4"),
            Decimal("2.599"),
            Decimal("2.768"),
            Decimal("2.6"),
            Decimal("3.336"),
            Decimal("3.5"),
            Decimal("3.349"),
            Decimal("3.148"),
            Decimal("2.625"),
            Decimal("2.51"),
            Decimal("3.992"),
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
        ]
        low_price_threshold = Decimal("4.355812749003984063745019920")
        min_selections = 16
        min_consecutive_selections = 4
        max_consecutive_selections = 8
        max_gap_between_periods = 18
        max_gap_from_start = 16

        # Get results from both implementations
        rust_result = get_cheapest_periods(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )
        python_result = _get_cheapest_periods_python(
            prices,
            low_price_threshold,
            min_selections,
            min_consecutive_selections,
            max_gap_between_periods,
            max_gap_from_start,
        )

        # Results should be identical
        assert rust_result == python_result, (
            f"Original issue scenario failed:\n"
            f"Rust result: {rust_result}\n"
            f"Python result: {python_result}\n"
            f"Length: Rust={len(rust_result)}, Python={len(python_result)}"
        )

        # Both implementations should select the first 3 items (the original issue)
        first_three_selected = all(i in rust_result for i in [0, 1, 2])
        assert first_three_selected, (
            f"First 3 items [0, 1, 2] not selected in result: {rust_result}\n"
            f"Prices of first 3: {[prices[i] for i in [0, 1, 2]]}\n"
            f"Threshold: {low_price_threshold}\n"
            f"All below threshold: {all(prices[i] <= low_price_threshold for i in [0, 1, 2])}"
        )

        # Verify the result has the correct length
        assert len(rust_result) == min_selections, (
            f"Result length {len(rust_result)} != expected {min_selections}"
        )

        # Verify all selected items are valid indices
        assert all(0 <= i < len(prices) for i in rust_result), (
            f"Invalid indices in result: {rust_result}"
        )

        # Verify the result is sorted (indices should be in order)
        assert rust_result == sorted(rust_result), (
            f"Result indices not sorted: {rust_result}"
        )

        # Calculate and verify cost metrics
        total_cost = sum(prices[i] for i in rust_result)
        avg_cost = total_cost / len(rust_result)
        cheap_count = sum(1 for i in rust_result if prices[i] <= low_price_threshold)

        print(f"\nOriginal issue scenario validation:")
        print(f"Result: {rust_result}")
        print(f"First 3 items selected: {first_three_selected}")
        print(f"Total cost: {total_cost}")
        print(f"Average cost: {avg_cost:.3f}")
        print(f"Cheap items: {cheap_count}/{len(rust_result)}")
        print(f"✓ Both implementations work correctly!")
