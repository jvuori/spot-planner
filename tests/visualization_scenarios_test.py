"""Tests for all visualization scenarios to ensure they meet constraints."""
from decimal import Decimal
import pytest
from spot_planner import get_cheapest_periods


def validate_constraints(
    selected: list[int],
    total_length: int,
    min_selections: int,
    min_consecutive_periods: int,
    max_gap_between_periods: int,
    max_gap_from_start: int,
) -> tuple[bool, list[str]]:
    """
    Validate that a selection meets all constraints.
    
    Returns:
        (is_valid, violations) - tuple of bool and list of violation messages
    """
    violations = []
    
    if not selected:
        violations.append("Selection is empty")
        return False, violations
    
    indices = sorted(selected)
    
    # Check minimum selections
    if len(indices) < min_selections:
        violations.append(
            f"Not enough selections: {len(indices)} < {min_selections}"
        )
    
    # Check max_gap_from_start
    if indices[0] > max_gap_from_start:
        violations.append(
            f"Gap from start too large: {indices[0]} > {max_gap_from_start}"
        )
    
    # Check gap at end
    end_gap = total_length - 1 - indices[-1]
    if end_gap > max_gap_between_periods:
        violations.append(
            f"Gap at end too large: {end_gap} > {max_gap_between_periods}"
        )
    
    # Check consecutive runs and gaps
    runs = []
    run_start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            # End of run
            runs.append((run_start, indices[i - 1]))
            
            # Check gap
            gap = indices[i] - indices[i - 1] - 1
            if gap > max_gap_between_periods:
                violations.append(
                    f"Gap too large between indices {indices[i-1]} and {indices[i]}: "
                    f"{gap} > {max_gap_between_periods}"
                )
            
            run_start = indices[i]
    runs.append((run_start, indices[-1]))
    
    # Check consecutive run lengths
    for start, end in runs:
        length = end - start + 1
        if length < min_consecutive_periods:
            violations.append(
                f"Consecutive run [{start}-{end}] too short: "
                f"length={length} < {min_consecutive_periods}"
            )
    
    return len(violations) == 0, violations


class TestVisualizationScenarios:
    """Test all visualization scenarios to ensure they meet constraints."""
    
    def test_expensive_day(self):
        """Test expensive_day scenario with constraint validation."""
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
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        # Validate constraints
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"expensive_day constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_realistic_daily(self):
        """Test realistic_daily scenario with constraint validation."""
        # Using generate function from visualize_results.py
        import visualize_results
        prices = visualize_results.generate_realistic_daily_pattern()
        
        params = {
            "low_price_threshold": Decimal("0.10"),
            "min_selections": 24,
            "min_consecutive_periods": 4,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 10,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"realistic_daily constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_cheap_day(self):
        """Test cheap_day scenario with constraint validation."""
        import visualize_results
        prices = visualize_results.generate_cheap_day_pattern()
        
        params = {
            "low_price_threshold": Decimal("0.10"),
            "min_selections": 24,
            "min_consecutive_periods": 4,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 10,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"cheap_day constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_realistic_daily_tight(self):
        """Test realistic_daily_tight scenario with constraint validation."""
        import visualize_results
        prices = visualize_results.generate_realistic_daily_pattern()
        
        params = {
            "low_price_threshold": Decimal("0.10"),
            "min_selections": 32,
            "min_consecutive_periods": 6,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 10,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"realistic_daily_tight constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_volatile_day(self):
        """Test volatile_day scenario with constraint validation."""
        import visualize_results
        prices = visualize_results.generate_volatile_day_pattern()
        
        params = {
            "low_price_threshold": Decimal("0.15"),
            "min_selections": 24,
            "min_consecutive_periods": 4,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 10,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"volatile_day constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_peak_valley(self):
        """Test peak_valley scenario with constraint validation."""
        import visualize_results
        prices = visualize_results.generate_peak_valley_pattern()
        
        params = {
            "low_price_threshold": Decimal("0.15"),
            "min_selections": 24,
            "min_consecutive_periods": 4,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 10,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"peak_valley constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
    
    def test_mlp_daemon_2026_02_25(self):
        """Test mlp.daemon.planner scenario from 2026-02-25 with constraint validation."""
        prices = [
            Decimal("9.779"),
            Decimal("12.416"),
            Decimal("14.999"),
            Decimal("17.878"),
            Decimal("10.21"),
            Decimal("14.997"),
            Decimal("14.999"),
            Decimal("14.999"),
            Decimal("13.486"),
            Decimal("14.994"),
            Decimal("14.997"),
            Decimal("14.057"),
            Decimal("16.448"),
            Decimal("14.51"),
            Decimal("13.709"),
            Decimal("12.919"),
            Decimal("15.145"),
            Decimal("14.999"),
            Decimal("13.775"),
            Decimal("11.488"),
            Decimal("13.482"),
            Decimal("11.418"),
            Decimal("9.939"),
            Decimal("8.865"),
            Decimal("9.94"),
            Decimal("9.939"),
            Decimal("9.2"),
            Decimal("8.13"),
            Decimal("9.997"),
            Decimal("9.94"),
            Decimal("9.94"),
            Decimal("9.939"),
            Decimal("9.939"),
            Decimal("9.939"),
            Decimal("10.116"),
            Decimal("10.006"),
            Decimal("14.992"),
            Decimal("11.218"),
            Decimal("9.015"),
            Decimal("8.558"),
            Decimal("6.492"),
            Decimal("5.0"),
            Decimal("3.58"),
            Decimal("2.945"),
            Decimal("3.521"),
            Decimal("2.767"),
            Decimal("2.351"),
            Decimal("1.528"),
            Decimal("3.103"),
            Decimal("3.195"),
            Decimal("2.884"),
            Decimal("1.859"),
            Decimal("2.999"),
            Decimal("2.999"),
            Decimal("2.845"),
            Decimal("2.518"),
            Decimal("2.809"),
            Decimal("2.649"),
            Decimal("2.587"),
            Decimal("2.558"),
            Decimal("2.401"),
            Decimal("2.487"),
            Decimal("2.369"),
            Decimal("2.486"),
            Decimal("2.604"),
            Decimal("2.795"),
            Decimal("2.914"),
            Decimal("2.999"),
            Decimal("2.655"),
            Decimal("2.772"),
            Decimal("2.934"),
            Decimal("2.85"),
            Decimal("2.679"),
            Decimal("2.679"),
            Decimal("2.739"),
            Decimal("2.862"),
        ]
        
        params = {
            "low_price_threshold": Decimal("8.13"),
            "min_selections": 37,
            "min_consecutive_periods": 4,
            "max_gap_between_periods": 20,
            "max_gap_from_start": 0,
        }
        
        result = get_cheapest_periods(
            prices=prices,
            aggressive=False,
            **params
        )
        
        is_valid, violations = validate_constraints(
            result,
            len(prices),
            params["min_selections"],
            params["min_consecutive_periods"],
            params["max_gap_between_periods"],
            params["max_gap_from_start"],
        )
        
        if not is_valid:
            pytest.fail(
                f"mlp_daemon_2026_02_25 constraint violations:\\n" + 
                "\\n".join(f"  - {v}" for v in violations)
            )
