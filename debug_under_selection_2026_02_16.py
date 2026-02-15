"""
Visualization for the under-selection issue.
Shows selected vs non-selected periods and price thresholds.
"""

import matplotlib.pyplot as plt
from decimal import Decimal
from spot_planner import get_cheapest_periods


def visualize_under_selection_issue():
    """Visualize the under-selection issue."""
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
    
    threshold = Decimal('7.608')
    result = get_cheapest_periods(
        prices=prices,
        low_price_threshold=threshold,
        min_selections=29,
        min_consecutive_periods=4,
        max_gap_between_periods=36,
        max_gap_from_start=33,
    )
    
    # Convert to floats for plotting
    prices_float = [float(p) for p in prices]
    threshold_float = float(threshold)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Prices with selected/non-selected highlighting
    ax = axes[0]
    indices = list(range(len(prices)))
    selected = set(result)
    colors = ['green' if i in selected else 'red' for i in indices]
    
    ax.bar(indices, prices_float, color=colors, alpha=0.6)
    ax.axhline(y=threshold_float, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({threshold_float})')
    ax.set_xlabel('Period Index')
    ax.set_ylabel('Price')
    ax.set_title(f'Price Selection - Green=Selected ({len(result)}), Red=Not Selected - UNDER-SELECTED (want 29, got 28)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Consecutive blocks analysis
    ax = axes[1]
    if result:
        blocks = []
        current_block = [result[0]]
        for i in range(1, len(result)):
            if result[i] - result[i-1] == 1:
                current_block.append(result[i])
            else:
                blocks.append(current_block)
                current_block = [result[i]]
        blocks.append(current_block)
        
        # Show blocks
        for block in blocks:
            block_start = block[0]
            block_end = block[-1]
            block_prices = [float(prices[i]) for i in block]
            block_indices = list(range(len(block)))
            ax.bar([i + block_start for i in block_indices], block_prices, alpha=0.7, width=0.8)
        
        ax.set_xlabel('Period Index')
        ax.set_ylabel('Price')
        ax.set_title(f'Selected Blocks Analysis - Total {len(blocks)} blocks')
        ax.grid(True, alpha=0.3)
        
        # Print block analysis
        print("\nBlock Analysis:")
        for i, block in enumerate(blocks):
            block_prices = [prices[idx] for idx in block]
            total_price = sum(block_prices, Decimal(0))
            avg_price = total_price / len(block)
            print(f"  Block {i+1}: indices {block[0]}-{block[-1]} ({len(block)} periods), avg price = {avg_price:.3f}")
    
    plt.tight_layout()
    plt.savefig('/home/jaakko/prj/spot-planner/visualization_results/under_selection_2026_02_16.png', dpi=100)
    print(f"\nVisualization saved to visualization_results/under_selection_2026_02_16.png")
    print(f"Selected {len(result)} periods (requested 29)")
    print(f"Selected indices: {result}")


if __name__ == "__main__":
    visualize_under_selection_issue()
