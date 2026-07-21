```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the FinceptTerminal Python codebase. You'll learn about file organization, import/export styles, commit message conventions, and how to structure and run tests. While no specific frameworks or automated workflows are detected, this guide will help you contribute code that fits seamlessly with the project's established practices.

## Coding Conventions

### File Naming
- **Style:** snake_case
- **Example:**  
  ```plaintext
  data_loader.py
  utils/helpers.py
  ```

### Import Style
- **Style:** Relative imports are preferred.
- **Example:**  
  ```python
  from .utils import calculate_metrics
  from ..models import Portfolio
  ```

### Export Style
- **Style:** Named exports are used (explicitly listing what is exported).
- **Example:**  
  ```python
  __all__ = ["calculate_metrics", "Portfolio"]
  ```

### Commit Messages
- **Style:** Conventional commits with the `feat` prefix.
- **Format Example:**  
  ```
  feat: add support for new financial instrument parsing
  ```
- **Average Length:** 79 characters

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module  
**Command:** `/add-feature`

1. Create a new Python file using snake_case for the filename.
2. Use relative imports to include any utilities or models.
3. List all exported functions/classes in `__all__`.
4. Write a descriptive commit message using the `feat` prefix.
5. If applicable, add or update test files.

### Refactoring Code
**Trigger:** When improving existing code without changing its functionality  
**Command:** `/refactor`

1. Rename files or functions to follow snake_case if needed.
2. Update imports to use relative paths.
3. Ensure all exports are explicitly listed.
4. Write a commit message describing the refactor.

### Writing Tests
**Trigger:** When adding or updating tests  
**Command:** `/add-test`

1. Create test files with the `.test.ts` extension (note: TypeScript test pattern detected, but main code is Python).
2. Place test files alongside the modules they test or in a dedicated tests directory.
3. Follow the same import/export conventions as production code.

## Testing Patterns

- **Framework:** Not explicitly detected.
- **File Pattern:** Test files use the `*.test.ts` pattern (suggests possible use of a TypeScript-based test runner or legacy tests).
- **Example Test File Name:**  
  ```plaintext
  portfolio.test.ts
  ```
- **Note:** If writing new Python tests, consider using `pytest` and naming files `test_*.py` for consistency with Python standards.

## Commands
| Command       | Purpose                                             |
|---------------|-----------------------------------------------------|
| /add-feature  | Start the workflow for adding a new feature         |
| /refactor     | Begin the code refactoring workflow                 |
| /add-test     | Add or update tests for modules                     |
```
