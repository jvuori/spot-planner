# Agent Guidelines for spot-planner

This document contains project-specific rules and guidelines for AI agents working on this codebase.

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
