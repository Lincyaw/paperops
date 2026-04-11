# Handoff Rules

Use this file to decide which skill owns the next step.

## `talk-architect` -> `slidecraft`

Use when the talk plan already defines:
- slide sequence or ordered story arc
- audience and duration assumptions
- primary message per slide
- what belongs in main deck vs backup

## `visual-language` -> `slidecraft`

Use when the style brief already defines:
- palette roles, type scale, and spacing rhythm
- layout families for recurring slide types
- shape, icon, and connector vocabulary
- image policy and emphasis rules
- drift checks that later review can enforce

Before implementing, translate that brief with `references/brief-consumption.md`.

## `slidecraft` -> `slide-review`

Switch when:
- the deck has been rendered and the question becomes diagnostic
- there are clipping, overflow, balance, or preview mismatch issues
- you need to know the smallest fix surface before editing code

## `slide-review` fallback routing

- narrative failure -> `talk-architect`
- visual inconsistency -> `visual-language`
- implementation/layout change -> `slidecraft`
