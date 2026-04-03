# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

paperops is a Python SDK designed for AI agents to programmatically generate academic presentations (PowerPoint) and publication-quality plots. The primary consumers are LLMs — API design, docs, and skills should be optimized for AI ergonomics.

## Build & Development

- **Package manager**: uv (not pip)
- **Python**: 3.13+
- **Install with extras**: `uv pip install -e ".[all]"` or specific extras: `[plotting]`, `[slides]`, `[dev]`
- **Build**: `uv build`
- **Publish**: `make upload` (requires local `token` file with PyPI token)
- **Tag release**: `make tag TAG=v1.0.0` (bumps version in pyproject.toml, commits, tags, pushes)
- **Lint**: `flake8 src/`
- **Format**: `black src/ && isort src/`
- **Test**: `pytest` (no tests exist yet)

## Code Style

- Black formatter, line-length 88, target Python 3.13
- isort with "black" profile
- Standard flake8

## Architecture

- `src/paperops/plotting/` — Publication-quality plotting presets and utilities (themes, figure sizing, save helpers) on top of matplotlib/seaborn
- `src/paperops/slides/` — SlideCraft: declarative PPT generation with component trees, layout engine, themes, and animations (python-pptx)
- All dependencies are optional extras — the base package has zero dependencies
- The two modules are decoupled: plotting saves files, slides references them via `Image(path=...)`

## Skills & Agents

Skills in `.claude/skills/` define design guidelines. Agents in `.claude/agents/` handle delegated execution.

When a task matches one of these skills, open the relevant `SKILL.md` and follow it before making code changes or generating output. If the user explicitly names a skill, treat that as a direct instruction to use it.

### Skills
- `/slidecraft` — PPT design philosophy + component API + 4-phase workflow
- `/slide-review` — Review existing decks via integrated preview/check artifacts and diagnose what to fix next
- `/plotting` — Academic figure guidelines: information hierarchy, chart type selection, visual rules
- `/verify` — Project verification workflow: black, isort, flake8, and pytest

### How To Trigger Skills
- Say `use .claude/skills/slidecraft for this` when working on PowerPoint or `paperops.slides` tasks
- Say `use .claude/skills/slide-review for this` when reviewing a generated deck or diagnosing layout issues from previews
- Say `use .claude/skills/plotting for this` when working on figures, charts, or `paperops.plotting`
- Say `use .claude/skills/verify` when you want the full repo verification pass
- If a task clearly falls into one of these areas, apply the matching skill even if the user does not mention it explicitly

### Agents
- `chart-maker` — Creates publication-quality matplotlib figures; returns file paths
- `slide-designer` — Builds individual slides from a brief (claim + layout spec)
- `slide-reviewer` — Reviews generated slides via preview PNGs; produces structured QA report

## Gotchas

- `make tag` uses `sed -i ''` (macOS syntax) — will fail on Linux
- `make upload` reads a local `token` file (gitignored) for PyPI auth; CI uses trusted publishing instead
- Always run code via `uv run python`, not bare `python` — ensures extras are available
