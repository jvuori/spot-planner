"""Debug expensive_day scenario to see exact constraint violations."""
from decimal import Decimal
from spot_planner import two_phase

prices = [
    Decimal("0.2911285751537849"),
    Decimal("0.22775852715546657"),
    Decimal("0.016239780813448106"),
    Decimal("0.04464704583099741"),
    Decimal("0.24745306400328818"),
    Decimal("0.048796394086479775"),
    Decimal("0.1582210399220897"),
    Decimal("0.017336180394137354"),
    Decimal("0.2144561576938028"),
    Decimal("0.17242124523564753"),
    Decimal("0.14510889491736753"),
    Decimal("0.18594513179286448"),
    Decimal("0.26133167305074245"),
    Decimal("0.030569377536544463"),
    Decimal("0.12836107428959959"),
    Decimal("0.15069434226371248"),
    Decimal("0.047955421490133335"),
    Decimal("0.265511522660963"),
    Decimal("0.1375809805211491"),
    Decimal("0.1992274488731282"),
    Decimal("0.029807076404450808"),
    Decimal("0.04637281608315129"),
    Decimal("0.23925401118371675"),
    Decimal("0.21361224381200594"),
    Decimal("0.15327380199459487"),
    Decimal("0.2595239082050006"),
    Decimal("0.2810689230769768"),
    Decimal("0.285937362304161"),
    Decimal("0.017839314496765808"),
    Decimal("0.023013213230530575"),
    Decimal("0.16884282571930126"),
    Decimal("0.18421559880484606"),
    Decimal("0.2176852949684847"),
    Decimal("0.042087879230161586"),
    Decimal("0.049475477464020694"),
    Decimal("0.15576882267615103"),
    Decimal("0.04261845713819337"),
    Decimal("0.2512212902473777"),
    Decimal("0.13332803731213627"),
    Decimal("0.14085643071452333"),
    Decimal("0.23219366282896042"),
    Decimal("0.13144050305148425"),
    Decimal("0.17853299796481448"),
    Decimal("0.23476034484393837"),
    Decimal("0.20499868652915088"),
    Decimal("0.0385297914889198"),
    Decimal("0.2210298955625093"),
    Decimal("0.20888320734559032"),
    Decimal("0.19695738330453894"),
    Decimal("0.014315657079732178"),
    Decimal("0.03545641645055122"),
    Decimal("0.2115427244096465"),
    Decimal("0.16487260124679748"),
    Decimal("0.25599920493774875"),
    Decimal("0.13385638376918274"),
    Decimal("0.14901983170572078"),
    Decimal("0.265461668321595"),
    Decimal("0.2768629062337892"),
    Decimal("0.15358261059948644"),
    Decimal("0.21708160354481712"),
    Decimal("0.2812964339862288"),
    Decimal("0.1398093464149818"),
    Decimal("0.19687940195272613"),
    Decimal("0.27493150498614183"),
    Decimal("0.03042989210310263"),
    Decimal("0.15997940588473145"),
    Decimal("0.023504606856145117"),
    Decimal("0.17817652776373594"),
    Decimal("0.246543412601132"),
    Decimal("0.2949207748897729"),
    Decimal("0.16532081324856554"),
    Decimal("0.17415809576701852"),
    Decimal("0.1266396505238159"),
    Decimal("0.21048222418119505"),
    Decimal("0.021145858569464458"),
    Decimal("0.16312114032005504"),
    Decimal("0.029578110411102518"),
    Decimal("0.16356994887207008"),
    Decimal("0.25709153075916913"),
    Decimal("0.2510789427501347"),
    Decimal("0.23381504950684429"),
    Decimal("0.21643944313345653"),
    Decimal("0.04341209982356952"),
    Decimal("0.15357333187197375"),
    Decimal("0.03363571772752967"),
    Decimal("0.1229858092070141"),
    Decimal("0.16076923953562883"),
    Decimal("0.15138595722089845"),
    Decimal("0.18961236233409673"),
    Decimal("0.14475376994627878"),
    Decimal("0.14042523382330602"),
    Decimal("0.2779210836085766"),
    Decimal("0.23879712828615224"),
    Decimal("0.2199361460879032"),
    Decimal("0.1635334123620813"),
    Decimal("0.045888630318133075"),
]

params = {
    "low_price_threshold": Decimal("0.12"),
    "min_selections": 16,
    "min_consecutive_periods": 4,
    "max_gap_between_periods": 20,
    "max_gap_from_start": 10,
}

print(f"Testing expensive_day with {len(prices)} prices")
print(f"Parameters: min_selections={params['min_selections']}, "
      f"min_consecutive={params['min_consecutive_periods']}")
print()

# Monkey-patch validation to see what was generated
original_validate = two_phase._validate_full_selection

def debug_validate(selected, n, min_consecutive, max_gap_between, max_gap_from_start):
    print(f"Generated selection before validation:")
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
        violations = []
        for start, end in runs:
            length = end - start + 1
            marker = "✓" if length >= min_consecutive else "✗"
            print(f"    [{start}-{end}] length={length} {marker}")
            if length < min_consecutive:
                violations.append(f"Run [{start}-{end}] length={length}")
        print()
        
        # Check gaps
        print(f"  Gap from start: {selected[0]}" + 
              (" ✓" if selected[0] <= max_gap_from_start else " ✗"))
        gap_violations = []
        for i in range(1, len(selected)):
            gap = selected[i] - selected[i-1] - 1
            if gap > 0:
                marker = "✓" if gap <= max_gap_between else "✗"
                if gap > max_gap_between:
                    gap_violations.append(f"Gap {gap} > {max_gap_between} between {selected[i-1]} and {selected[i]}")
                print(f"  Gap at {selected[i-1]+1}-{selected[i]-1}: {gap} {marker}")
        
        end_gap = n - 1 - selected[-1]
        if end_gap > 0:
            marker = "✓" if end_gap <= max_gap_between else "✗"
            print(f"  Gap at end: {end_gap} {marker}")
            if end_gap > max_gap_between:
                gap_violations.append(f"End gap {end_gap} > {max_gap_between}")
        
        if violations:
            print(f"\n  CONSTRAINT VIOLATIONS:")
            print(f"    Short consecutive runs: {len(violations)}")
            for v in violations:
                print(f"      - {v}")
        
        if gap_violations:
            if not violations:
                print(f"\n  CONSTRAINT VIOLATIONS:")
            print(f"    Gap violations: {len(gap_violations)}")
            for v in gap_violations:
                print(f"      - {v}")
    
    # Call original
    return original_validate(selected, n, min_consecutive, max_gap_between, max_gap_from_start)

two_phase._validate_full_selection = debug_validate

try:
    result = two_phase.get_cheapest_periods_extended(
        prices,
        aggressive=False,
        **params
    )
    print(f"\nSuccess! Selected {len(result)} periods")
except ValueError as e:
    print(f"\nFailed with error: {e}")
