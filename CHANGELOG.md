# Changelog

## 0.3.0 - 2026-04-25

### Breaking

- breaking: slides API refactor
- adopt the IR-first `paperops.slides` authoring model across README, examples, and mirrored skills
- switch public examples to sheet-driven JSON/MDX/Python authoring instead of the legacy imperative slide builder
- promote the new builder classes (`Deck`, `Slide`, `Title`, `Subtitle`, `Heading`, `Text`, `Grid`, `KPI`, ...) as the primary public entry points
