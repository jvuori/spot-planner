"""Regression test for the 2026-09-29 crash found by sweeping mlp inputs.

Bug: the extended algorithm measured the remaining gap budget for a chunk from
the previous chunk's trailing unselected count only. When two or more
consecutive chunks were skipped (target=0), the earlier skipped chunks were
forgotten, so the budget was overestimated. The 2026-03-01 fix handled a single
skipped chunk 0; this is the multi-chunk generalization.

Example (150 prices, chunk size 14, max_gap_from_start=32): chunks 0 and 1 are
skipped, then chunk 2 (starting at index 28) was given
  adjusted_max_gap_start = 32 - 14 = 18    # WRONG, ignores chunk 0
  -> first selection at index 36 > 32 -> ValueError
The correct budget is 32 - 28 = 4.

Fix: compute the budget globally, from the last selection so far (or from the
sequence start), and force a selection in a chunk that cannot be skipped
without exceeding it.

Prices are real Finnish day-ahead prices (c/kWh excl. VAT) and the parameters
are exactly what the mlp heating planner passes at 0 degC and +5 degC.
"""

from decimal import Decimal

import pytest

from spot_planner import get_cheapest_periods, two_phase

# 150 quarters starting 2026-01-09T12:00:00+00:00: expensive start, cheap ~9-12 h later.
PRICES_2026_01_09 = [
    Decimal("16.592"), Decimal("16.24"), Decimal("15.0"), Decimal("15.0"), Decimal("14.999"), Decimal("14.314"), Decimal("12.327"), Decimal("14.999"),
    Decimal("14.999"), Decimal("15.0"), Decimal("15.106"), Decimal("13.906"), Decimal("15.182"), Decimal("15.59"), Decimal("15.927"), Decimal("15.0"),
    Decimal("13.997"), Decimal("14.632"), Decimal("12.326"), Decimal("12.278"), Decimal("11.618"), Decimal("12.802"), Decimal("12.584"), Decimal("11.0"),
    Decimal("12.655"), Decimal("11.237"), Decimal("11.236"), Decimal("10.905"), Decimal("11.331"), Decimal("10.999"), Decimal("10.49"), Decimal("8.263"),
    Decimal("11.696"), Decimal("10.676"), Decimal("10.448"), Decimal("8.806"), Decimal("10.441"), Decimal("9.655"), Decimal("7.991"), Decimal("7.565"),
    Decimal("7.753"), Decimal("7.295"), Decimal("7.193"), Decimal("6.991"), Decimal("5.881"), Decimal("5.741"), Decimal("6.486"), Decimal("5.748"),
    Decimal("6.737"), Decimal("6.503"), Decimal("6.106"), Decimal("5.642"), Decimal("6.332"), Decimal("6.15"), Decimal("6.095"), Decimal("5.986"),
    Decimal("6.392"), Decimal("6.209"), Decimal("6.274"), Decimal("6.126"), Decimal("6.231"), Decimal("6.301"), Decimal("6.143"), Decimal("6.249"),
    Decimal("6.109"), Decimal("6.282"), Decimal("6.543"), Decimal("6.768"), Decimal("6.327"), Decimal("6.694"), Decimal("7.027"), Decimal("7.262"),
    Decimal("6.812"), Decimal("7.493"), Decimal("7.897"), Decimal("8.176"), Decimal("7.988"), Decimal("8.134"), Decimal("8.539"), Decimal("8.869"),
    Decimal("8.784"), Decimal("9.068"), Decimal("9.098"), Decimal("8.987"), Decimal("8.894"), Decimal("9.177"), Decimal("9.762"), Decimal("9.98"),
    Decimal("9.181"), Decimal("9.856"), Decimal("10.032"), Decimal("10.269"), Decimal("9.727"), Decimal("10.014"), Decimal("9.701"), Decimal("9.892"),
    Decimal("9.819"), Decimal("9.979"), Decimal("10.108"), Decimal("10.756"), Decimal("10.181"), Decimal("10.43"), Decimal("10.903"), Decimal("11.196"),
    Decimal("10.414"), Decimal("11.164"), Decimal("11.326"), Decimal("12.489"), Decimal("11.7"), Decimal("12.108"), Decimal("11.976"), Decimal("13.832"),
    Decimal("12.984"), Decimal("13.476"), Decimal("13.767"), Decimal("13.853"), Decimal("13.912"), Decimal("13.371"), Decimal("13.233"), Decimal("12.55"),
    Decimal("13.069"), Decimal("12.55"), Decimal("12.147"), Decimal("11.989"), Decimal("12.48"), Decimal("11.617"), Decimal("11.372"), Decimal("10.87"),
    Decimal("11.236"), Decimal("10.869"), Decimal("10.84"), Decimal("10.113"), Decimal("10.859"), Decimal("9.822"), Decimal("9.536"), Decimal("8.984"),
    Decimal("9.103"), Decimal("8.794"), Decimal("8.715"), Decimal("8.584"), Decimal("8.369"), Decimal("7.6"), Decimal("7.171"), Decimal("6.36"),
    Decimal("7.483"), Decimal("7.245"), Decimal("6.721"), Decimal("6.552"), Decimal("6.964"), Decimal("7.012"),
]

# 150 quarters starting 2026-01-23T08:00:00+00:00: expensive start, cheap ~9-12 h later.
PRICES_2026_01_23 = [
    Decimal("20.918"), Decimal("21.859"), Decimal("19.999"), Decimal("17.224"), Decimal("22.101"), Decimal("19.814"), Decimal("17.224"), Decimal("15.0"),
    Decimal("19.999"), Decimal("19.898"), Decimal("18.996"), Decimal("19.018"), Decimal("20.0"), Decimal("19.999"), Decimal("19.999"), Decimal("18.991"),
    Decimal("18.937"), Decimal("18.914"), Decimal("18.911"), Decimal("18.943"), Decimal("18.007"), Decimal("18.906"), Decimal("18.969"), Decimal("18.998"),
    Decimal("18.963"), Decimal("20.517"), Decimal("20.0"), Decimal("19.999"), Decimal("15.72"), Decimal("16.395"), Decimal("19.331"), Decimal("20.634"),
    Decimal("17.586"), Decimal("20.985"), Decimal("18.199"), Decimal("17.417"), Decimal("20.29"), Decimal("17.659"), Decimal("16.834"), Decimal("16.601"),
    Decimal("17.84"), Decimal("16.052"), Decimal("15.1"), Decimal("14.429"), Decimal("14.607"), Decimal("13.867"), Decimal("13.806"), Decimal("11.501"),
    Decimal("13.687"), Decimal("14.999"), Decimal("13.088"), Decimal("11.059"), Decimal("12.785"), Decimal("11.799"), Decimal("11.765"), Decimal("11.562"),
    Decimal("11.521"), Decimal("10.009"), Decimal("9.494"), Decimal("9.398"), Decimal("14.999"), Decimal("11.924"), Decimal("10.755"), Decimal("10.322"),
    Decimal("11.554"), Decimal("10.705"), Decimal("10.552"), Decimal("10.375"), Decimal("10.523"), Decimal("10.612"), Decimal("10.501"), Decimal("10.454"),
    Decimal("10.367"), Decimal("10.486"), Decimal("10.384"), Decimal("10.465"), Decimal("10.334"), Decimal("10.181"), Decimal("10.164"), Decimal("10.22"),
    Decimal("10.226"), Decimal("10.447"), Decimal("10.144"), Decimal("10.573"), Decimal("10.253"), Decimal("10.688"), Decimal("10.991"), Decimal("11.18"),
    Decimal("10.716"), Decimal("11.026"), Decimal("11.304"), Decimal("11.669"), Decimal("11.69"), Decimal("11.84"), Decimal("13.184"), Decimal("13.142"),
    Decimal("13.163"), Decimal("13.974"), Decimal("14.586"), Decimal("14.64"), Decimal("14.718"), Decimal("15.0"), Decimal("15.0"), Decimal("15.0"),
    Decimal("14.987"), Decimal("14.999"), Decimal("14.731"), Decimal("14.453"), Decimal("14.59"), Decimal("14.4"), Decimal("14.319"), Decimal("14.49"),
    Decimal("14.589"), Decimal("14.647"), Decimal("14.375"), Decimal("13.982"), Decimal("13.126"), Decimal("13.746"), Decimal("14.589"), Decimal("14.619"),
    Decimal("13.601"), Decimal("14.589"), Decimal("14.306"), Decimal("14.348"), Decimal("13.717"), Decimal("14.039"), Decimal("14.648"), Decimal("14.601"),
    Decimal("14.387"), Decimal("14.999"), Decimal("15.0"), Decimal("14.999"), Decimal("14.886"), Decimal("14.84"), Decimal("13.99"), Decimal("12.982"),
    Decimal("14.108"), Decimal("13.614"), Decimal("12.735"), Decimal("11.543"), Decimal("13.55"), Decimal("12.681"), Decimal("11.126"), Decimal("10.283"),
    Decimal("12.311"), Decimal("11.836"), Decimal("11.562"), Decimal("11.132"), Decimal("11.901"), Decimal("11.088"),
]


@pytest.mark.parametrize(
    ("prices", "low_price_threshold", "min_selections"),
    [
        pytest.param(PRICES_2026_01_09, Decimal("8.584"), 53, id="2026-01-09-0C"),
        pytest.param(PRICES_2026_01_23, Decimal("12.735"), 53, id="2026-01-23-0C"),
        pytest.param(PRICES_2026_01_09, Decimal("7.897"), 45, id="2026-01-09-5C"),
        pytest.param(PRICES_2026_01_23, Decimal("11.765"), 45, id="2026-01-23-5C"),
    ],
)
def test_crash_2026_09_29(prices, low_price_threshold, min_selections):
    """Must not raise, and must satisfy every constraint."""
    selected = get_cheapest_periods(
        prices=prices,
        low_price_threshold=low_price_threshold,
        min_selections=min_selections,
        min_consecutive_periods=4,
        max_gap_between_periods=32,
        max_gap_from_start=32,
    )

    assert selected[0] <= 32
    assert len(selected) >= min_selections
    assert two_phase._validate_full_selection(selected, len(prices), 4, 32, 32)
