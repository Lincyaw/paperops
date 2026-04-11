---
name: verify
description: "Run repo-aware verification for paperops changes, including code checks, skill-doc consistency checks, and slide/plot smoke tests when relevant."
---

# Verify

Run the smallest complete verification pass that matches the surface area of the change.

## Use This Skill When

- the user asks for verification, checks, or a pre-commit sanity pass
- a change touched `paperops.plotting`, `paperops.slides`, or the skill system itself
- you need confidence that docs, skill routing, and code paths still agree

## Do Not Use This Skill When

- the user is still deciding the design or workflow
- no code or skill artifact has changed and there is nothing to validate
- the task is specifically to design a figure, talk, or deck rather than to check it

## Workflow

1. Identify the changed surface
   - plotting, slides, skills/docs, or a mix
2. Run baseline checks
   - formatting/lint/test commands appropriate to the repo state
3. Add targeted smoke checks
   - plotting changes: import and minimal render path when dependencies exist
   - slides changes: build/review/preview-oriented smoke checks when dependencies exist
   - skill changes: validate mirrored file presence, references, and cross-skill links
4. Report gaps explicitly
   - if optional dependencies are missing, say which checks were skipped and why

## Default Checks

- `black --check src/ tests/` when those paths exist
- `isort --check src/ tests/` when those paths exist
- `flake8 src/ tests/` when configured
- `pytest`

Adjust the exact command set to match the modified area and installed extras.

## Handoff

- return findings to the skill or subsystem that needs fixes
- if failures point to deck rendering issues, route through `slide-review`
- if failures point to broken skill links or missing mirrored docs, fix the skill layer before code-level changes
