```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you how to contribute to the FinceptTerminal Python codebase, a modular terminal for financial data analytics, backtesting, and dashboarding. You'll learn the project's coding conventions, how to add new data providers, analytics engines, strategies, and UI commands, and how to follow the repository's workflow and testing patterns.

## Coding Conventions

- **File Naming:** Use `snake_case` for Python files.
  - Example: `yfinance_provider.py`, `test_providers.py`
- **Import Style:** Use relative imports within the package.
  - Example:
    ```python
    from .base_provider import BaseProvider
    from ..analytics import risk_metrics
    ```
- **Export Style:** Use named exports (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["YFinanceProvider", "StooqProvider"]
    ```
- **Commit Messages:** Use [Conventional Commits](https://www.conventionalcommits.org/) with the `feat` prefix for new features.
  - Example: `feat: add cryptocom provider for crypto data integration`

## Workflows

### Add New Data Provider
**Trigger:** When integrating a new external data source (e.g., exchange, API, market) into the terminal.  
**Command:** `/add-provider`

1. **Implement the Provider:**  
   Create a new provider file in `finscope/core/providers/` (e.g., `myapi_provider.py`).
   ```python
   # finscope/core/providers/myapi_provider.py
   from .base_provider import BaseProvider

   class MyAPIProvider(BaseProvider):
       def fetch_data(self, symbol):
           # Implementation here
           pass
   ```
2. **Register the Provider:**  
   Add your provider to the registry so it's discoverable.
   ```python
   # finscope/core/providers/public_apis_registry.py
   from .myapi_provider import MyAPIProvider
   PROVIDERS.append(MyAPIProvider)
   ```
3. **Update Demo Data (Optional):**  
   If needed, add example data to `finscope/data/demo_seed.json`.
4. **Update Tests:**  
   Add or update tests in `finscope/tests/test_providers.py`.
   ```python
   def test_myapi_provider_fetch():
       provider = MyAPIProvider()
       data = provider.fetch_data("AAPL")
       assert data is not None
   ```
5. **Update UI (If Needed):**  
   If your provider exposes new commands, update `finscope/ui/functions_ext.py` or `finscope/ui/functions_ext_market.py`.

### Add New Analytics or Strategy Engine
**Trigger:** When implementing a new analytics calculation, risk metric, or trading strategy.  
**Command:** `/add-strategy`

1. **Implement Analytics or Strategy:**  
   Create or update files in `finscope/analytics/` (for analytics) or `finscope/engines/` (for strategies).
   ```python
   # finscope/analytics/my_metric.py
   def calculate_my_metric(data):
       # Calculation logic
       return result
   ```
2. **Register the Engine/Strategy:**  
   Add your new module to the appropriate registry or `__init__.py`.
   ```python
   # finscope/engines/__init__.py
   from .my_strategy import MyStrategy
   ```
3. **Update Tests:**  
   Add or update tests in `finscope/tests/test_analytics.py`.
   ```python
   def test_calculate_my_metric():
       result = calculate_my_metric(sample_data)
       assert result == expected
   ```
4. **Expose in UI (If Relevant):**  
   Update `finscope/ui/commands.py` or agent interfaces to make the new functionality available to users.

### Add New Dashboard or UI Command
**Trigger:** When exposing new functionality or data to users via the TUI dashboard or command interface.  
**Command:** `/add-ui-command`

1. **Implement Command or Panel:**  
   Add logic in `finscope/ui/commands.py` or `finscope/ui/panels.py`.
   ```python
   # finscope/ui/commands.py
   def do_show_my_metric(args):
       # Command logic
       pass
   ```
2. **Add/Update Function Extensions (If Needed):**  
   Update `finscope/ui/functions_ext.py` or related files.
3. **Wire into UI:**  
   Register the new command or panel in the main UI or command registry.
   ```python
   COMMANDS["show-my-metric"] = do_show_my_metric
   ```
4. **Update Documentation (Optional):**  
   Update help mnemonics or documentation if necessary.

## Testing Patterns

- **Test File Naming:** Test files use the pattern `test_*.py` (e.g., `test_providers.py`, `test_analytics.py`).
- **Testing Framework:** Not explicitly specified; standard Python `unittest` or `pytest` patterns are likely.
- **Test Example:**
  ```python
  def test_provider_returns_data():
      provider = SomeProvider()
      assert provider.fetch_data("AAPL") is not None
  ```

## Commands

| Command         | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| /add-provider   | Add a new data provider (API or data source)                   |
| /add-strategy   | Add a new analytics engine or trading strategy                 |
| /add-ui-command | Add a new dashboard panel, UI command, or function extension   |
```
