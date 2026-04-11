# Style Brief Template

Use this template when handing a deck design language to `slidecraft`.

## Required Fields

- `style_name`: short label for the system
- `style_keywords`: 3-5 descriptive keywords
- `audience_fit`: why this visual system matches the room and talk type
- `palette_roles`: primary actor, secondary actor, neutral structure, evidence, warning, success, low-emphasis support
- `type_scale`: title, section, claim, body, caption, numeric highlight
- `spacing_scale`: small / medium / large spacing rhythm and when each is used
- `layout_families`: 2-4 recurring slide skeletons the deck should reuse
- `symbol_vocab`: actor, process, artifact, evidence, warning, transition, comparison
- `icon_style_rules`: line/fill treatment, stroke weight, corner logic, level of detail, reuse rules
- `connector_rules`: arrows, association lines, grouping enclosures, comparison separators
- `image_policy`: when to use abstract diagrams, plots, screenshots, photos, or no image
- `emphasis_rules`: how to create a focal point and what combinations to avoid
- `accessibility_baseline`: font-size floor, contrast rule, non-color encoding rule, chart-label rule
- `drift_checks`: what counts as palette drift, symbol drift, layout drift, and emphasis drift
- `dos`: 3-5 positive rules
- `donts`: 3-5 prohibitions

## Output Style

- write it as an implementation brief, not as art criticism
- keep terms concrete enough that another agent can map them to `RoundedBox`, `Badge`, `Flowchart`, `SvgImage`, and theme overrides
- prefer rules that can be checked in review, not vague taste statements
