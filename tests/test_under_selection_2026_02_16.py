"""
Test case for the under-selection issue reported on 2026-02-16.

Customer complaint: spot_planner returned 28 selected periods but 29 requested.
Input: 70 periods with specific price data and constraints.
"""

from decimal import Decimal

from spot_planner import get_cheapest_periods


def test_under_selection_issue_2026_02_16():
    """Reproduce the under-selection issue."""
    # Exact data from the customer's log
    prices = [
        Decimal('10.477'), Decimal('9.962'), Decimal('7.271'), Decimal('9.37'),
        Decimal('10.168'), Decimal('10.3'), Decimal('9.103'), Decimal('9.29'),
        Decimal('9.259'), Decimal('9.587'), Decimal('9.053'), Decimal('9.251'),
        Decimal('9.444'), Decimal('9.255'), Decimal('10.216'), Decimal('9.161'),
        Decimal('8.773'), Decimal('14.999'), Decimal('8.762'), Decimal('11.004'),
        Decimal('11.829'), Decimal('11.999'), Decimal('14.999'), Decimal('13.193'),
        Decimal('8.898'), Decimal('6.911'), Decimal('12.999'), Decimal('8.852'),
        Decimal('8.135'), Decimal('7.413'), Decimal('8.239'), Decimal('7.322'),
        Decimal('6.999'), Decimal('7.616'), Decimal('6.787'), Decimal('7.276'),
        Decimal('7.608'), Decimal('8.327'), Decimal('8.181'), Decimal('8.212'),
        Decimal('8.355'), Decimal('8.425'), Decimal('9.527'), Decimal('10.501'),
        Decimal('9.899'), Decimal('9.334'), Decimal('9.008'), Decimal('8.361'),
        Decimal('7.712'), Decimal('7.225'), Decimal('7.552'), Decimal('6.295'),
        Decimal('5.033'), Decimal('4.3'), Decimal('3.5'), Decimal('3.219'),
        Decimal('3.164'), Decimal('2.999'), Decimal('3.219'), Decimal('3.187'),
        Decimal('3.029'), Decimal('2.977'), Decimal('2.947'), Decimal('2.796'),
        Decimal('2.644'), Decimal('2.471'), Decimal('2.688'), Decimal('2.524'),
        Decimal('2.392'), Decimal('2.313'),
    ]
    
    result = get_cheapest_periods(
        prices=prices,
        low_price_threshold=Decimal('7.608'),
        min_selections=29,
        min_consecutive_periods=4,
        max_gap_between_periods=36,
        max_gap_from_start=33,
    )
    
    print(f"Requested: 29 selections")
    print(f"Received: {len(result)} selections")
    print(f"Selected indices: {result}")
    
    # Verify the result meets constraints
    assert len(result) == 29, f"Under-selected: got {len(result)} but wanted 29"
    
    # Verify min_consecutive_periods constraint
    # Check all consecutive blocks
    if result:
        current_block = [result[0]]
        for i in range(1, len(result)):
            if result[i] - result[i-1] == 1:
                # Consecutive
                current_block.append(result[i])
            else:
                # End of block
                assert len(current_block) >= 4, f"Block {current_block} has less than 4 consecutive periods"
                current_block = [result[i]]
        # Check last block
        assert len(current_block) >= 4, f"Block {current_block} has less than 4 consecutive periods"
    
    # Verify max_gap_from_start constraint
    assert result[0] <= 33, f"First selection at {result[0]} violates max_gap_from_start=33"
    
    # Verify max_gap_between_periods constraint
    for i in range(1, len(result)):
        gap = result[i] - result[i-1] - 1
        assert gap <= 36, f"Gap between {result[i-1]} and {result[i]} is {gap}, exceeds max_gap_between_periods=36"


if __name__ == "__main__":
    test_under_selection_issue_2026_02_16()
    print("Test passed!")
