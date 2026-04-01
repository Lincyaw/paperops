---
name: verify
description: Run linting and format checks on the paperops codebase to verify code quality before committing.
---

# Verify

Run all code quality checks for the paperops project.

## Steps

1. **Format check** (black):
   ```bash
   black --check src/
   ```

2. **Import sort check** (isort):
   ```bash
   isort --check src/
   ```

3. **Lint** (flake8):
   ```bash
   flake8 src/
   ```

4. **Tests** (if any exist):
   ```bash
   pytest
   ```

If any check fails, fix the issues and re-run. Report all results to the user.
