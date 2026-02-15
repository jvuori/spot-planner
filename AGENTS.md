# Agent Guidelines for spot-planner

This document contains project-specific rules and guidelines for AI agents working on this codebase.

## Fail Fast Principle

**CRITICAL**: The algorithm must always know which path to choose. Never use try/except fallbacks to progressively relax constraints when the primary approach fails.

### What This Means

1. **No fallback attempts with relaxed constraints** - If the algorithm can't find a valid solution with the given constraints, it should fail cleanly with a clear error message, not try progressively more lenient constraints.

2. **No "last resort" selections** - Never select items arbitrarily (e.g., "select cheapest N items") just to return something when the algorithm can't find a valid solution.

3. **Deterministic path selection** - The algorithm should know upfront which approach to use based on the input constraints, not try multiple approaches and use what works.

### Why This Matters

- **Constraint violations**: Fallbacks that relax constraints (e.g., `min_consecutive_periods = 1` when it should be `6`) lead to invalid results that violate user requirements.
- **Debugging difficulty**: When an algorithm tries multiple approaches, it's hard to understand which path was taken and why the result looks a certain way.
- **Correctness guarantees**: Users depend on the algorithm respecting all constraints. Returning a result that violates constraints is worse than returning an error.

### Bad Pattern Examples

```python
# ❌ DON'T DO THIS
try:
    result = algorithm_with_strict_constraints()
except ValueError:
    # Try with relaxed constraints
    result = algorithm_with_relaxed_constraints()
    if not result:
        # Last resort: return something invalid
        result = select_cheapest_items()
```

```python
# ❌ DON'T DO THIS
for fallback_target in [min_consecutive_periods, 1]:
    result = try_selection(min_consecutive=min(fallback_target, min_consecutive_periods))
    if result:
        return result
```

### Good Pattern Examples

```python
# ✅ DO THIS
if cannot_satisfy_constraints(params):
    raise ValueError(f"Cannot satisfy constraints: {explanation}")
return algorithm_with_strict_constraints()
```

```python
# ✅ DO THIS
if scenario_type_a(params):
    return algorithm_a(params)
elif scenario_type_b(params):
    return algorithm_b(params)
else:
    raise ValueError("Unsupported parameter combination")
```

## Python Import Rules

**IMPORTANT**: When importing from Python modules, you can only import:

- **Modules** (e.g., `from spot_planner import two_phase`)
- **Types** (e.g., classes, type aliases: `from spot_planner.two_phase import ChunkBoundaryState`)
- **Constants** (e.g., module-level constants: `from spot_planner import MAX_ITEMS`)

**NEVER import functions directly** (e.g., `from spot_planner.two_phase import get_cheapest_periods_extended`).

Instead, import the module and access functions through the module namespace:

```python
# ✅ Correct
from spot_planner import two_phase
result = two_phase.get_cheapest_periods_extended(...)

# ❌ Incorrect
from spot_planner.two_phase import get_cheapest_periods_extended
result = get_cheapest_periods_extended(...)
```

### Fully Qualified Package Names

**ALWAYS use fully qualified package names instead of relative imports** (`.` notation).

```python
# ✅ Correct
from spot_planner import two_phase
from spot_planner import brute_force
from spot_planner import main

# ❌ Incorrect
from . import two_phase
from . import brute_force
from . import main
```

This rule ensures:

- **Clarity**: It's immediately clear which package a module belongs to
- **Consistency**: All imports follow the same pattern regardless of file location
- **Easier refactoring**: Moving files doesn't require updating relative import paths
- **Better IDE support**: IDEs can better resolve and navigate fully qualified imports

**Exception**: Compiled extension modules (e.g., Rust extensions) that share the package name cannot use fully qualified imports from within the package itself. In such cases, use a relative import with a `type: ignore` comment and document why:

```python
# Exception: Rust extension module shares package name
from . import spot_planner as _rust_module  # type: ignore[import-untyped]
```

### Rationale

This rule helps maintain:

- **Better encapsulation**: Functions are accessed through their module namespace
- **Clearer dependencies**: It's immediately clear which module a function belongs to
- **Easier refactoring**: Moving functions between modules doesn't break imports
- **Consistency**: All code follows the same import pattern
