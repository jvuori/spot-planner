"""Debug script to understand why realistic_daily fails."""
from decimal import Decimal
from spot_planner import two_phase

prices = [
    Decimal("0.02749080237694725"),
    Decimal("0.03901428612819832"),
    Decimal("0.0346398788362281"),
    Decimal("0.03197316968394073"),
    Decimal("0.02312037280884873"),
    Decimal("0.023119890406724054"),
    Decimal("0.021161672243363988"),
    Decimal("0.0373235229154987"),
    Decimal("0.03202230023486417"),
    Decimal("0.03416145155592091"),
    Decimal("0.02041168988591605"),
    Decimal("0.03939819704323989"),
    Decimal("0.03664885281600844"),
    Decimal("0.024246782213565524"),
    Decimal("0.023636499344142012"),
    Decimal("0.023668090197068677"),
    Decimal("0.026084844859190756"),
    Decimal("0.03049512863264476"),
    Decimal("0.028638900372842314"),
    Decimal("0.025824582803960838"),
    Decimal("0.03223705789444759"),
    Decimal("0.02278987721304084"),
    Decimal("0.025842892970704363"),
    Decimal("0.027327236865873836"),
    Decimal("0.09560699842170359"),
    Decimal("0.13023618844815726"),
    Decimal("0.08277259770130221"),
    Decimal("0.11308181731943448"),
    Decimal("0.12228419068261889"),
    Decimal("0.09278702476319986"),
    Decimal("0.1295923322988748"),
    Decimal("0.11350306144224083"),
    Decimal("0.11634185734747006"),
    Decimal("0.14856879504309334"),
    Decimal("0.1493126406614912"),
    Decimal("0.14770076817739752"),
    Decimal("0.12436910153386965"),
    Decimal("0.10781376912051072"),
    Decimal("0.15473864212097255"),
    Decimal("0.1352121994991681"),
    Decimal("0.10976305878758232"),
    Decimal("0.1396141528089016"),
    Decimal("0.10275108168921748"),
    Decimal("0.17274563216630257"),
    Decimal("0.12070239852800135"),
    Decimal("0.15300178274831855"),
    Decimal("0.12493688608715288"),
    Decimal("0.14160544169422487"),
    Decimal("0.11280261676059679"),
    Decimal("0.09109126733153163"),
    Decimal("0.13817507766587353"),
    Decimal("0.1265079694016669"),
    Decimal("0.13636993649385137"),
    Decimal("0.13368964102565895"),
    Decimal("0.11587399872866512"),
    Decimal("0.13531245410138704"),
    Decimal("0.08530955012311517"),
    Decimal("0.09175897174514872"),
    Decimal("0.08271363733463229"),
    Decimal("0.09951981984579586"),
    Decimal("0.10332063738136893"),
    Decimal("0.09628094190643376"),
    Decimal("0.12972425054911577"),
    Decimal("0.10140519960161537"),
    Decimal("0.09685607058124285"),
    Decimal("0.11256176498949491"),
    Decimal("0.08845545349848577"),
    Decimal("0.12813181884524238"),
    Decimal("0.15745506436797707"),
    Decimal("0.24885260695254527"),
    Decimal("0.2329183576972493"),
    Decimal("0.19991973009588576"),
    Decimal("0.20027610585618014"),
    Decimal("0.24307980356705627"),
    Decimal("0.24267143359619042"),
    Decimal("0.24661258960051233"),
    Decimal("0.25"),
    Decimal("0.25"),
    Decimal("0.23842555814667613"),
    Decimal("0.23396164321360682"),
    Decimal("0.21684508973219235"),
    Decimal("0.24315517129377967"),
    Decimal("0.22645613292672237"),
    Decimal("0.19981735186394867"),
    Decimal("0.16806135565002708"),
    Decimal("0.0579884089544096"),
    Decimal("0.05699021572822"),
    Decimal("0.08545006955369286"),
    Decimal("0.0739914655235097"),
    Decimal("0.08500719003973224"),
    Decimal("0.0559718208839072"),
    Decimal("0.03574052380503848"),
    Decimal("0.05924303627614279"),
    Decimal("0.05586669165297452"),
    Decimal("0.045154484334376396"),
    Decimal("0.04541934359909121"),
    Decimal("0.036419342752737074"),
]

params = {
    "low_price_threshold": Decimal("0.10"),
    "min_selections": 24,
    "min_consecutive_periods": 4,
    "max_gap_between_periods": 20,
    "max_gap_from_start": 10,
}

print(f"Running with {len(prices)} prices")
print(f"Parameters: {params}")
print()

try:
    result = two_phase.get_cheapest_periods_extended(
        prices,
        aggressive=False,
        **params
    )
    print(f"Success! Selected {len(result)} periods")
    print(f"Indices: {result}")
except ValueError as e:
    print(f"Validation failed: {e}")
    print()
    print("Adding debug output to see what was generated before validation...")
    
    # Now monkey-patch the validation to see what was generated
    import spot_planner.two_phase as tp
    original_validate = tp._validate_full_selection
    
    def debug_validate(selected, n, min_consecutive, max_gap_between, max_gap_from_start):
        print(f"\nGenerated selection before validation:")
        print(f"  Total selected: {len(selected)}")
        print(f"  Indices: {selected}")
        print()
        
        # Analyze consecutive runs
        if selected:
            runs = []
            run_start = selected[0]
            for i in range(1, len(selected)):
                if selected[i] != selected[i-1] + 1:
                    runs.append((run_start, selected[i-1]))
                    run_start = selected[i]
            runs.append((run_start, selected[-1]))
            
            print(f"  Consecutive runs: {len(runs)}")
            for start, end in runs:
                length = end - start + 1
                marker = "✓" if length >= min_consecutive else "✗"
                print(f"    [{start}-{end}] length={length} {marker}")
            print()
            
            # Check gaps
            print(f"  Gap from start: {selected[0]}" + 
                  (" ✓" if selected[0] <= max_gap_from_start else " ✗"))
            for i in range(1, len(selected)):

                gap = selected[i] - selected[i-1] - 1
                if gap > 0:
                    marker = "✓" if gap <= max_gap_between else "✗"
                    print(f"  Gap at {selected[i-1]+1}-{selected[i]-1}: {gap} {marker}")
        
        # Call original
        return original_validate(selected, n, min_consecutive, max_gap_between, max_gap_from_start)
    
    tp._validate_full_selection = debug_validate
    
    try:
        result = two_phase.get_cheapest_periods_extended(
            prices,
            aggressive=False,
            **params
        )
    except ValueError as e2:
        print(f"\nFinal error: {e2}")
