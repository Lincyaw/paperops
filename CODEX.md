# CODEX.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

paperops is a Python SDK for AI agents to generate academic presentations and publication-quality plots. The repository keeps its workflow skills mirrored across hosts so Codex and Claude can follow the same task boundaries.

## Build & Development

- **Package manager**: uv
- **Python**: 3.13+
- **Install with extras**: `uv pip install -e ".[all]"` or the specific extras you need
- **Run code**: prefer `uv run python`
- **Build**: `uv build`
- **Tests**: `pytest`

## Skill Layout

- Codex-facing skills live under `.codex/skills/`
- Claude-facing mirrors live under `.claude/skills/`
- the two trees should stay structurally and semantically aligned

### Skill Set

- `/talk-architect` -> talk planning and audience-aware narrative
- `/visual-language` -> presentation visual system: hierarchy, symbol language, image policy, and style brief
- `/slidecraft` -> PPT implementation with `paperops.slides`
- `/slide-review` -> deck diagnosis from review/preview artifacts
- `/plotting` -> paper/report figure creation
- `/verify` -> repo-aware verification

### Routing Hints

- use `.codex/skills/talk-architect` before slide generation if the story is not stable
- use `.codex/skills/visual-language` before implementation if the deck style is not explicit
- use `.codex/skills/slidecraft` for slide code and layout work
- use `.codex/skills/slide-review` when the task is diagnostic rather than generative
- use `.codex/skills/plotting` for quantitative figures
- use `.codex/skills/verify` for checks after changes

## Maintenance Rule

When editing a repo skill, update the matching file under both `.codex/skills/` and `.claude/skills/` unless a host-specific note explicitly says otherwise.
