```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns, coding conventions, and workflow processes used in the FinceptTerminal codebase. FinceptTerminal is a Python-based financial terminal for integrating data providers, analytics engines, and trading strategies. The repository emphasizes modularity, clear documentation, and conventional commit practices to ensure maintainability and extensibility.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files and modules.
  ```
  # Good
  yfinance_provider.py
  strategies_alpha.py

  # Bad
  YFinanceProvider.py
  StrategiesAlpha2.py
  ```

- **Import Style:**  
  Use relative imports within the package.
  ```python
  # In finscope/core/providers/yfinance_provider.py
  from .base_provider import BaseProvider
  ```

- **Export Style:**  
  Use named exports (explicitly define what is exported).
  ```python
  # In yfinance_provider.py
  class YFinanceProvider(BaseProvider):
      ...
  __all__ = ["YFinanceProvider"]
  ```

- **Commit Messages:**  
  Follow [Conventional Commits](https://www.conventionalcommits.org/):
  ```
  feat: add stooq data provider integration
  docs: update REQUEST_LEDGER.md with new research item
  ```

## Workflows

### Add New Data Provider
**Trigger:** When you want to integrate a new market data source or API provider.  
**Command:** `/add-provider`

1. Create a new provider implementation in `finscope/core/providers/` (e.g., `yfinance_provider.py`).
2. Register the provider with the DataHub or provider registry (e.g., update `datahub.py` or `public_apis_registry.py`).
3. Optionally, add demo data or update `public_apis.json` if relevant.
4. Add or update tests in `finscope/tests/test_providers.py`.
5. Update UI command handlers in `finscope/ui/commands.py` if new commands are exposed.

**Example:**
```python
# finscope/core/providers/yfinance_provider.py
from .base_provider import BaseProvider

class YFinanceProvider(BaseProvider):
    def fetch(self, symbol):
        # Implementation here
        pass

__all__ = ["YFinanceProvider"]
```
```python
# finscope/core/datahub.py
from .providers.yfinance_provider import YFinanceProvider

PROVIDERS = {
    "yfinance": YFinanceProvider(),
    # other providers...
}
```

### Add New Analytics or Strategy Engine
**Trigger:** When you want to implement a new analytics calculation or trading strategy.  
**Command:** `/add-strategy`

1. Create a new strategy or analytics module in `finscope/engines/` or `finscope/analytics/` (e.g., `strategies_alpha.py`).
2. Register the strategy/engine in the engine registry or main engine file (e.g., `engines/_discover.py`).
3. Update or add tests in `finscope/tests/test_analytics.py` to verify correctness.
4. Expose the new strategy via agent/dashboard surface if relevant (e.g., `agent/dashboard.py`).
5. Document the new strategy in `finscope/research/program.md` or related docs.

**Example:**
```python
# finscope/engines/strategies_alpha.py
def alpha_strategy(data):
    # Strategy logic here
    return ...

__all__ = ["alpha_strategy"]
```
```python
# finscope/engines/_discover.py
from .strategies_alpha import alpha_strategy

STRATEGIES = {
    "alpha": alpha_strategy,
    # other strategies...
}
```

### Add or Update Documentation Ledger
**Trigger:** When you want to document new features, research specs, or track user requests.  
**Command:** `/add-doc`

1. Create or update markdown documentation in `finscope/docs/` or `finscope/research/` (e.g., `REQUEST_LEDGER.md`, `program.md`).
2. Force-add `*.md` files if ignored by `.gitignore`.
3. Reference new docs in commit messages or PRs.

**Example:**
```
docs: add new research spec for volatility clustering
```
```markdown
# finscope/research/REQUEST_LEDGER.md

## 2024-06-01
- [x] Add yfinance provider
- [ ] Implement alpha2 strategy
```

## Testing Patterns

- **Test Framework:** Unknown (no standard framework detected).
- **Test File Pattern:** Test files are named `*.test.ts` (suggests some TypeScript-based tests, possibly for UI or integration).
- **Location:** Tests for providers and analytics are found in `finscope/tests/`, e.g., `test_providers.py`, `test_analytics.py`.
- **Best Practice:** Add or update tests when adding new providers or strategies.

**Example:**
```python
# finscope/tests/test_providers.py
from finscope.core.providers.yfinance_provider import YFinanceProvider

def test_yfinance_provider_fetch():
    provider = YFinanceProvider()
    data = provider.fetch("AAPL")
    assert data is not None
```

## Commands

| Command        | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| /add-provider  | Start workflow to add a new data provider                    |
| /add-strategy  | Start workflow to add a new analytics or strategy engine     |
| /add-doc       | Start workflow to add or update documentation ledger         |
```
