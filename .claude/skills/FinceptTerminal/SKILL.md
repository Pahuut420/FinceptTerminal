```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the FinceptTerminal Python codebase. You'll learn how to structure files, write imports and exports, follow commit message conventions, and understand the project's testing approach. This guide also provides command suggestions for common workflows.

## Coding Conventions

### File Naming
- Use **snake_case** for all filenames.
  - Example: `data_loader.py`, `user_profile_manager.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_config
    from ..models import User
    ```

### Export Style
- Use **named exports** (explicitly define what is exported from a module).
  - Example:
    ```python
    __all__ = ['DataLoader', 'parse_config']
    ```

### Commit Messages
- Follow the **conventional commits** standard.
- Use the `feat` prefix for new features.
  - Example:
    ```
    feat: add support for multiple user profiles in dashboard module
    ```

## Workflows

### Adding a New Feature
**Trigger:** When you need to implement a new feature.
**Command:** `/add-feature`

1. Create a new Python file using snake_case if needed.
2. Write your code, using relative imports for intra-package dependencies.
3. Define `__all__` in modules to explicitly export public objects.
4. Write or update tests in a corresponding `*.test.*` file.
5. Commit your changes with a message starting with `feat:`.
   - Example: `feat: implement transaction export functionality`

### Writing Tests
**Trigger:** When you add or modify code that requires testing.
**Command:** `/write-test`

1. Create a test file matching the pattern `*.test.*` (e.g., `data_loader.test.py`).
2. Write tests for your new or updated functionality.
3. Use the project's preferred (undetected) testing framework.
4. Run tests to ensure correctness.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `module.test.py`).
- The specific testing framework is not specified; follow existing patterns in the repository.
- Place tests alongside or near the modules they test.

## Commands

| Command        | Purpose                                         |
|----------------|------------------------------------------------|
| /add-feature   | Start the workflow for adding a new feature    |
| /write-test    | Begin writing tests for new or updated code    |
```