#!/usr/bin/env python3
"""Debug script to investigate constraint violation in realistic_daily_tight."""

from decimal import Decimal
from spot_planner import get_cheapest_periods

# realistic_daily_tight scenario
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
    Decimal("0.026886084618607965"),
    Decimal("0.03397577172267782"),
    Decimal("0.03611424775858451"),
    Decimal("0.03742765959063124"),
    Decimal("0.03928127206176044"),
    Decimal("0.05096968833825107"),
    Decimal("0.062094056838838786"),
    Decimal("0.09341669871092024"),
    Decimal("0.13734074032831196"),
    Decimal("0.11629075959851388"),
    Decimal("0.1447072607857175"),
    Decimal("0.11999851837869898"),
    Decimal("0.14998852838867754"),
    Decimal("0.12019206092881928"),
    Decimal("0.13926062949690643"),
    Decimal("0.1268687838732067"),
    Decimal("0.12036806329802803"),
    Decimal("0.14997506996447706"),
    Decimal("0.13953829453024776"),
    Decimal("0.10002195943636964"),
    Decimal("0.14869829943798197"),
    Decimal("0.1799954858988963"),
    Decimal("0.17996963082568208"),
    Decimal("0.13047453291636447"),
    Decimal("0.1315421051127758"),
    Decimal("0.10041838084654038"),
    Decimal("0.13993754654886775"),
    Decimal("0.16951997893926046"),
    Decimal("0.11981074584009913"),
    Decimal("0.11998994776803695"),
    Decimal("0.16991769063906804"),
    Decimal("0.13962799551003022"),
    Decimal("0.13972234032009334"),
    Decimal("0.099986683866582"),
    Decimal("0.11023726929988998"),
    Decimal("0.11974536906887964"),
    Decimal("0.10984468193850078"),
    Decimal("0.11993677925329823"),
    Decimal("0.10022697926726598"),
    Decimal("0.09953876091006836"),
    Decimal("0.08008933073267815"),
    Decimal("0.11992895099031857"),
    Decimal("0.08995886082850831"),
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

print(f"Selected {len(selected)} periods: {selected}")
print()

# Analyze consecutive runs
consecutive_runs = []
if selected:
    run_start = selected[0]
    run_length = 1
    
    for i in range(1, len(selected)):
        if selected[i] == selected[i-1] + 1:
            run_length += 1
        else:
            consecutive_runs.append((run_start, run_length))
            run_start = selected[i]
            run_length = 1
    
    consecutive_runs.append((run_start, run_length))
    
    print(f"Found {len(consecutive_runs)} consecutive runs:")
    for start, length in consecutive_runs:
        print(f"  Start: {start}, Length: {length}")
        if length < params["min_consecutive_periods"]:
            is_at_end = (start + length - 1) == (len(prices) - 1)
            print(f"    ⚠️ VIOLATION: Length {length} < min_consecutive_periods {params['min_consecutive_periods']} (at_end={is_at_end})")
