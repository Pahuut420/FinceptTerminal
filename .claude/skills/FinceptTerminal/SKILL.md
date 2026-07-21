```markdown
# FinceptTerminal Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill introduces you to the core development patterns and conventions used in the FinceptTerminal Python codebase. You'll learn about file naming, import/export styles, commit message conventions, and how to structure and run tests. This guide is designed to help maintain consistency and productivity when contributing to the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files.
  - Example: `data_loader.py`, `user_profile.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import calculate_totals
    from ..models import User
    ```

### Export Style
- Use **named exports** (i.e., explicitly define what is exported from a module).
  - Example:
    ```python
    __all__ = ['calculate_totals', 'User']
    ```

### Commit Messages
- Use **conventional commits** with the `feat` prefix for new features.
  - Example:
    ```
    feat: add user authentication to terminal session
    ```

## Workflows

### Feature Development
**Trigger:** When adding a new feature to the codebase  
**Command:** `/feature-dev`

1. Create a new branch for your feature.
2. Implement the feature using snake_case file naming and relative imports.
3. Add or update named exports in your modules.
4. Write a commit message starting with `feat:`.
5. Open a pull request for review.

### Code Review
**Trigger:** When reviewing a pull request  
**Command:** `/code-review`

1. Check that file names use snake_case.
2. Ensure all imports are relative.
3. Verify that named exports are used.
4. Confirm commit messages follow the `feat:` convention.
5. Approve or request changes as needed.

## Testing Patterns

- **Testing framework:** Unknown (no framework detected).
- **Test file pattern:** Files end with `.test.ts` (TypeScript test files).
  - Example: `user_profile.test.ts`
- **Note:** While the main codebase is Python, tests may be written in TypeScript. Ensure tests are placed in appropriately named files.

## Commands

| Command         | Purpose                                   |
|-----------------|-------------------------------------------|
| /feature-dev    | Start a new feature development workflow   |
| /code-review    | Begin the code review workflow            |
```