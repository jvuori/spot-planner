#!/usr/bin/env python3

from decimal import Decimal
import time
import sys
from src.spot_planner.main import get_cheapest_periods

def test_hanging_input():
    """Test the specific input that causes hanging"""
    prices = [
        Decimal('3.17'), Decimal('2.98'), Decimal('3.702'), Decimal('5.067'), 
        Decimal('4.19'), Decimal('4.692'), Decimal('4.493'), Decimal('3.813'), 
        Decimal('4.902'), Decimal('3.559'), Decimal('2.396'), Decimal('1.758'), 
        Decimal('2.599'), Decimal('1.531'), Decimal('1.25'), Decimal('1.134'), 
        Decimal('1.167'), Decimal('1.077'), Decimal('0.583'), Decimal('0.404'), 
        Decimal('0.508'), Decimal('0.473'), Decimal('0.4'), Decimal('0.341')
    ]
    
    low_price_threshold = Decimal('2.554812749003984063745019920')
    min_selections = 12
    min_consecutive_selections = 1
    max_consecutive_selections = 1
    max_gap_between_periods = 18
    max_gap_from_start = 17
    
    print(f"Testing with {len(prices)} prices")
    print(f"Low price threshold: {low_price_threshold}")
    print(f"Min selections: {min_selections}")
    print(f"Min consecutive: {min_consecutive_selections}")
    print(f"Max consecutive: {max_consecutive_selections}")
    print(f"Max gap between periods: {max_gap_between_periods}")
    print(f"Max gap from start: {max_gap_from_start}")
    
    # Count cheap items
    cheap_items = [p for p in prices if p <= low_price_threshold]
    print(f"Cheap items (<= threshold): {len(cheap_items)} out of {len(prices)}")
    print(f"Cheap percentage: {len(cheap_items)/len(prices)*100:.1f}%")
    
    print("\nStarting calculation...")
    start_time = time.time()
    
    try:
        result = get_cheapest_periods(
            prices=prices,
            low_price_threshold=low_price_threshold,
            min_selections=min_selections,
            min_consecutive_selections=min_consecutive_selections,
            max_consecutive_selections=max_consecutive_selections,
            max_gap_between_periods=max_gap_between_periods,
            max_gap_from_start=max_gap_from_start
        )
        
        end_time = time.time()
        print(f"Success! Result: {result}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        end_time = time.time()
        print(f"Error after {end_time - start_time:.2f} seconds: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Set a timeout to prevent infinite hanging
    import signal
    
    def timeout_handler(signum, frame):
        print("\nTimeout! The algorithm is taking too long.")
        sys.exit(1)
    
    # Set 30 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        test_hanging_input()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        signal.alarm(0)  # Cancel the alarm

