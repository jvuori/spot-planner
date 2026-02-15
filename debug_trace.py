#!/usr/bin/env python3
"""Debug script to trace algorithm calls."""

import sys
from decimal import Decimal
from spot_planner import get_cheapest_periods
from spot_planner import two_phase

# Patch _find_best_chunk_selection_with_lookahead to log calls
original_find_best = two_phase._find_best_chunk_selection_with_lookahead

call_count = 0

def patched_find_best(
    chunk_prices,
    next_chunk_prices,
    low_price_threshold,
    target,
    min_consecutive_periods,
    max_gap_between_periods,
    max_gap_from_start,
    aggressive,
):
    global call_count
    call_count += 1
    print(f"[CALL #{call_count}] _find_best_chunk_selection_with_lookahead:", file=sys.stderr)
    print(f"  chunk_len={len(chunk_prices)}, target={target}, min_consecutive={min_consecutive_periods}", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        result = original_find_best(
            chunk_prices,
            next_chunk_prices,
            low_price_threshold,
            target,
            min_consecutive_periods,
            max_gap_between_periods,
            max_gap_from_start,
            aggressive,
        )
        print(f"  -> returned {len(result)} selections: {result}", file=sys.stderr)
        sys.stderr.flush()
        return result
    except Exception as e:
        print(f"  -> raised exception: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise

two_phase._find_best_chunk_selection_with_lookahead = patched_find_best

# realistic_daily_tight scenario (first chunk only)
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
    "min_selections": 32,
    "min_consecutive_periods": 6,
    "max_gap_between_periods": 20,
    "max_gap_from_start": 10,
}

print(f"Running with {len(prices)} prices")
print(f"Parameters: {params}")
print()

selected = get_cheapest_periods(prices=prices, **params)

print()
print(f"Selected {len(selected)} periods")
print(f"Selected indices: {selected}")
