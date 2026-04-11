# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

paperops is a Python SDK designed for AI agents to programmatically generate academic presentations (PowerPoint) and publication-quality plots. The primary consumers are LLMs, so API design, docs, and skills should be optimized for AI ergonomics.

## Build & Development

- **Package manager**: uv
- **Python**: 3.13+
- **Install with extras**: `uv pip install -e ".[all]"` or specific extras: `[plotting]`, `[slides]`, `[dev]`
- **Build**: `uv build`
- **Publish**: `make upload` (requires local `token` file with PyPI token)
- **Tag release**: `make tag TAG=v1.0.0`
- **Run code**: prefer `uv run python`

## Architecture

- `src/paperops/plotting/` -> publication-quality plotting presets and helpers
- `src/paperops/slides/` -> SlideCraft: declarative PPT generation with layout, themes, and review utilities
- optional dependencies are split by extras; base package stays dependency-light

## Skills & Agents

This repo intentionally mirrors skill design across hosts:
- Claude-facing skills live under `.claude/skills/`
- Codex-facing skills live under `.codex/skills/`
- the two trees should stay structurally and semantically aligned

When a task matches one of these skills, open the corresponding `SKILL.md` and follow it before generating output or making code changes.

### Skill Set

- `/talk-architect` -> talk planning, pacing, and slide-by-slide narrative
- `/visual-language` -> PPT visual system, symbol vocabulary, image policy, and style brief
- `/slidecraft` -> deck implementation with `paperops.slides`
- `/slide-review` -> rendered-deck diagnosis and iteration loop
- `/plotting` -> publication-quality figures with `paperops.plotting`
- `/verify` -> repo-aware verification for code and skill changes

### Routing Hints

- use `.claude/skills/talk-architect` before deck generation when the story is unstable
- use `.claude/skills/visual-language` when the deck needs a coherent visual system
- use `.claude/skills/slidecraft` for PPT generation and layout implementation
- use `.claude/skills/slide-review` when diagnosing rendered deck problems
- use `.claude/skills/plotting` for paper/report figures
- use `.claude/skills/verify` for repo checks

## Agents

- `chart-maker` -> creates publication-quality matplotlib figures and returns file paths
- `slide-designer` -> builds individual slides from a brief
- `slide-reviewer` -> reviews generated slides via previews and reports issues

## Maintenance Rule

When editing a repo skill, make the matching change in both `.claude/skills/` and `.codex/skills/` unless a host-specific note explicitly says otherwise.
