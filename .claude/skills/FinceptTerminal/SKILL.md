```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the FinceptTerminal Python codebase. You will learn about file organization, code style, commit practices, and how to write and run tests. This guide is designed to help contributors maintain consistency and quality across the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all file and module names.
  - Example: `data_loader.py`, `user_profile_manager.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_config
    from ..models import User
    ```

### Export Style
- Use **named exports**; explicitly define what is exported from each module.
  - Example:
    ```python
    __all__ = ["DataLoader", "UserProfileManager"]
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use the `feat` prefix for new features.
  - Example:
    ```
    feat: add support for CSV data import in portfolio module
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module  
**Command:** `/add-feature`

1. Create a new Python file using snake_case naming.
2. Use relative imports for internal modules.
3. Define `__all__` for named exports.
4. Write clear, conventional commit messages prefixed with `feat`.
5. Add or update tests in a corresponding `*.test.*` file.

### Writing and Running Tests
**Trigger:** When adding or modifying functionality  
**Command:** `/run-tests`

1. Create test files matching the pattern `*.test.*` (e.g., `portfolio.test.py`).
2. Write test functions for each feature or module.
3. Use the project's preferred test runner (framework not specified; check project docs or use `pytest` as default).
4. Run tests and ensure all pass before committing changes.

## Testing Patterns

- Test files are named using the pattern `*.test.*` (e.g., `module.test.py`).
- Each test file should focus on a single module or feature.
- The testing framework is not explicitly specified; use standard Python testing practices.
- Example test file:
  ```python
  # portfolio.test.py
  from .portfolio import Portfolio

  def test_portfolio_returns():
      p = Portfolio([100, 200, 300])
      assert p.total() == 600
  ```

## Commands
| Command        | Purpose                                      |
|----------------|----------------------------------------------|
| /add-feature   | Start the workflow for adding a new feature  |
| /run-tests     | Run the test suite for the project           |
```
