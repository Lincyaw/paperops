# SlideCraft Layout Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade SlideCraft so agents can rely more on intrinsic layout behavior and can review a deck through one integrated pipeline that exposes the current PPT state clearly.

**Architecture:** Extend layout nodes with lightweight fit/fill/grow/shrink semantics, teach the layout engine to negotiate space using those semantics, and add a unified deck review API that combines in-memory validation, saved-file checks, preview images, and per-slide summaries. Update local `.claude/skills` so PPT work defaults to a review-driven iteration loop.

**Tech Stack:** Python 3.13, `python-pptx`, PIL-based preview rendering, local `.claude/skills` markdown docs.

---

## File Structure

### New files

- Create: `docs/superpowers/specs/2026-04-03-slidecraft-layout-review-design.md`
- Create: `docs/superpowers/plans/2026-04-03-slidecraft-layout-review.md`
- Create: `.claude/skills/slide-review/SKILL.md`

### Modified files

- Modify: `.claude/skills/slidecraft/SKILL.md`
- Modify: `src/paperops/slides/layout/containers.py`
- Modify: `src/paperops/slides/layout/engine.py`
- Modify: `src/paperops/slides/layout/auto_size.py`
- Modify: `src/paperops/slides/components/text.py`
- Modify: `src/paperops/slides/components/shapes.py`
- Modify: `src/paperops/slides/components/composite.py`
- Modify: `src/paperops/slides/preview.py`
- Modify: `src/paperops/slides/build.py`
- Modify as needed for exercising workflow: `examples/self-intro/build.py`

### Responsibility split

- `containers.py`: declarative sizing fields and layout-node semantics
- `engine.py`: actual space negotiation rules
- `auto_size.py`: shared text measurement helpers
- `text.py`, `shapes.py`, `composite.py`: intrinsic sizing behavior for common components
- `preview.py`: saved-file checks, slide summaries, preview-aware review helpers
- `build.py`: high-level orchestration API for integrated deck review
- `.claude/skills/*`: reusable agent workflow guidance

---

### Task 1: Extend LayoutNode Semantics

**Files:**
- Modify: `src/paperops/slides/layout/containers.py`
- Modify: `src/paperops/slides/layout/engine.py`

- [ ] **Step 1: Add declarative sizing fields to `LayoutNode`**

Add new keyword-only fields to `LayoutNode` in `src/paperops/slides/layout/containers.py`:

```python
size_mode_x: str = field(default="auto", kw_only=True)
size_mode_y: str = field(default="auto", kw_only=True)
grow: float = field(default=0.0, kw_only=True)
shrink: float = field(default=1.0, kw_only=True)
basis: float | None = field(default=None, kw_only=True)
wrap: bool = field(default=False, kw_only=True)
```

Keep `width`, `height`, `min_width`, `min_height` unchanged for backward compatibility.

- [ ] **Step 2: Add helper functions in `engine.py` to classify children**

Implement local helpers in `src/paperops/slides/layout/engine.py`:

```python
def _size_mode_x(node) -> str:
    if getattr(node, "width", None) is not None:
        return "fixed"
    mode = getattr(node, "size_mode_x", "auto")
    return "fit" if mode == "auto" else mode


def _size_mode_y(node) -> str:
    if getattr(node, "height", None) is not None:
        return "fixed"
    mode = getattr(node, "size_mode_y", "auto")
    return "fit" if mode == "auto" else mode
```

- [ ] **Step 3: Teach `HStack` width allocation to use fixed / fit / fill**

Refactor `_layout_hstack()` in `src/paperops/slides/layout/engine.py` so it:

```python
# fixed width children -> reserve exact width
# fit children -> ask preferred_size()
# fill children -> divide remaining width by grow
# shrink if over-constrained, respecting min_width
```

Use `basis` when set before falling back to `preferred_size()`.

- [ ] **Step 4: Teach `VStack` height allocation to use fixed / fit / fill**

Refactor `_layout_vstack()` similarly:

```python
# fixed height children -> reserve exact height
# fit children -> preferred_size(...).height
# fill children -> divide remaining height by grow
# shrink if over-constrained, respecting min_height
```

- [ ] **Step 5: Preserve current behavior for legacy callers**

Ensure existing decks that only use `width` / `height` / `min_width` still render with no call-site changes. Do not require examples to adopt the new fields immediately.

---

### Task 2: Improve Intrinsic Component Sizing

**Files:**
- Modify: `src/paperops/slides/layout/auto_size.py`
- Modify: `src/paperops/slides/components/text.py`
- Modify: `src/paperops/slides/components/shapes.py`
- Modify: `src/paperops/slides/components/composite.py`

- [ ] **Step 1: Centralize shared width/height measurement helpers**

Add helper functions in `src/paperops/slides/layout/auto_size.py`:

```python
def measure_wrapped_text_height(text, font_family, font_size_pt, usable_width_inches) -> float:
    _w, h = measure_text(text, font_family, font_size_pt, max_width_inches=usable_width_inches)
    return h


def estimate_min_text_width(text, font_family, font_size_pt) -> float:
    tokens = [token for token in text.replace("\n", " ").split(" ") if token]
    sample = max(tokens, key=len) if tokens else text
    w, _h = measure_text(sample, font_family, font_size_pt)
    return w
```

- [ ] **Step 2: Upgrade `TextBlock.preferred_size()`**

Update `src/paperops/slides/components/text.py` so `TextBlock`:

```python
- derives usable width from textbox margins
- computes wrapped height from shared helpers
- exposes a meaningful minimum width from longest token width
- defaults to fit-content behavior unless explicitly fixed
```

- [ ] **Step 3: Upgrade `BulletList` sizing**

Replace the current line-count heuristic with shared wrapped measurement so multi-line bullets grow more realistically:

```python
joined = "\n".join(
    item if isinstance(item, str) else item[0]
    for item in self.items
)
```

Then measure against available width with a small bullet-indent allowance.

- [ ] **Step 4: Strengthen shape and composite intrinsic sizing**

Update `Badge`, `RoundedBox`, `Callout`, and `Flow` to better reflect content width and padding. Keep arrow widths narrow and content nodes fit-driven. Avoid hard-coding widths where content-fit behavior is more appropriate.

- [ ] **Step 5: Make self-intro flow and callout usage rely less on manual sizing**

Update `examples/self-intro/build.py` only where needed so representative slides benefit from the new sizing behavior without changing deck structure.

---

### Task 3: Add Integrated Deck Review API

**Files:**
- Modify: `src/paperops/slides/preview.py`
- Modify: `src/paperops/slides/build.py`

- [ ] **Step 1: Add slide-summary helpers in `preview.py`**

Create helper functions in `src/paperops/slides/preview.py`:

```python
def summarize_slide_shapes(slide) -> dict:
    ...


def summarize_presentation_issues(issues: list[dict]) -> dict[int, list[dict]]:
    ...
```

The summary should include shape counts, text-shape counts, narrow text boxes, and a simple crowding signal.

- [ ] **Step 2: Add a deck review function that merges checks**

Implement a function in `src/paperops/slides/preview.py` along these lines:

```python
def review_deck_artifacts(pptx_path: str, preview_paths: list[str] | None = None, slide_titles: list[str] | None = None) -> dict:
    saved_issues = check_presentation(pptx_path)
    # group by slide, attach preview paths, attach heuristics
    return {
        "saved_issue_count": len(saved_issues),
        "slides": [...],
    }
```

- [ ] **Step 3: Add `Presentation.review_deck()`**

Implement a high-level API in `src/paperops/slides/build.py`:

```python
def review_deck(self, output_path: str, render_preview: bool = True, output_dir: str | None = None) -> dict:
    self.save(output_path)
    preview_paths = self.preview(output_dir=output_dir) if render_preview else []
    return review_deck_artifacts(output_path, preview_paths=preview_paths, slide_titles=[...])
```

Do not remove `review()` or `preview()`; keep them as lower-level building blocks.

- [ ] **Step 4: Make review output agent-friendly**

Ensure the integrated report includes:

```python
{
    "total_slides": ...,
    "layout_issue_count": ...,
    "saved_issue_count": ...,
    "top_problem_slides": [...],
    "preview_paths": [...],
    "slides": [
        {
            "slide_number": 1,
            "title": "...",
            "layout_issues": [...],
            "saved_file_issues": [...],
            "preview_path": "...",
            "summary": {...},
        }
    ],
}
```

- [ ] **Step 5: Keep stale preview handling stable**

Preserve the current behavior where rerendered previews replace stale `slide_*.png` assets rather than accumulating misleading old files.

---

### Task 4: Improve Review Heuristics

**Files:**
- Modify: `src/paperops/slides/preview.py`

- [ ] **Step 1: Add narrow-label heuristics**

Add a deterministic heuristic that flags text-heavy shapes with poor width budgets:

```python
if text and width_inches / max(len(longest_token), 1) < threshold:
    issues.append({... "type": "crowding_risk"})
```

- [ ] **Step 2: Add dense-slide heuristics**

Flag slides likely to need visual simplification:

```python
if text_shape_count >= 8 or small_text_shape_count >= 4:
    slide_summary["density"] = "high"
```

- [ ] **Step 3: Rank problematic slides**

Create a stable ranking based on saved-file issues + crowding heuristics + density so the integrated report can surface likely problem slides first.

---

### Task 5: Update Skills

**Files:**
- Modify: `.claude/skills/slidecraft/SKILL.md`
- Create: `.claude/skills/slide-review/SKILL.md`
- Optionally modify: `CLAUDE.md`

- [ ] **Step 1: Update `slidecraft` guidance**

Edit `.claude/skills/slidecraft/SKILL.md` so it explicitly documents:

```markdown
1. Build the deck.
2. Run integrated review.
3. Inspect preview PNGs and per-slide issues.
4. Fix targeted layout/content problems.
```

Also add short guidance on intrinsic sizing and when manual dimensions are still appropriate.

- [ ] **Step 2: Create `slide-review` skill**

Create `.claude/skills/slide-review/SKILL.md` with:

```markdown
- Trigger conditions for deck review / PPT QA / slide iteration
- Review workflow using `review_deck()`
- How to distinguish content-density issues from layout negotiation issues
- What artifacts to inspect before editing code
```

- [ ] **Step 3: Ensure repo instructions mention the new skill if helpful**

If useful, add a brief line to `CLAUDE.md` so future agents can discover the dedicated review skill more easily. Keep the change minimal.

---

### Task 6: Verify On A Real Deck

**Files:**
- Modify as needed: `examples/self-intro/build.py`

- [ ] **Step 1: Run the self-intro deck through the new pipeline**

Run:

```bash
uv run python examples/self-intro/build.py
```

Then run the integrated review entry point against the generated deck and inspect its returned structure.

- [ ] **Step 2: Fix any regressions introduced by the new layout behavior**

If the new semantics change slide geometry unexpectedly, apply targeted adjustments to the example deck rather than weakening the new layout model immediately.

- [ ] **Step 3: Confirm the pipeline is useful for iteration**

Manually inspect the review result for:

- preview paths are present
- slide summaries are populated
- problematic slides are surfaced sensibly
- no mismatch between saved-file checks and preview outputs is obvious

---

### Task 7: Final Verification And Cleanup

**Files:**
- Modify any touched files from prior tasks

- [ ] **Step 1: Run focused verification**

Run:

```bash
uv run pytest tests/examples/test_self_intro_build.py -q
```

Expected: passing tests without needing new test coverage.

- [ ] **Step 2: Rebuild the example deck**

Run:

```bash
uv run python examples/self-intro/build.py
```

Expected: `.pptx` regenerated successfully with no runtime errors.

- [ ] **Step 3: Sanity-check integrated review output**

Run a short inline script:

```bash
uv run python - <<'PY'
from pathlib import Path
from examples.self_intro.build import build_presentation

out = Path("examples/self-intro/diagnostic_world_model.pptx")
prs = build_presentation(output_path=out, render_preview=False)
report = prs.review_deck(output_path=str(out), render_preview=True, output_dir="examples/self-intro/preview")
print(report["total_slides"], report["layout_issue_count"], report["saved_issue_count"])
print(report["top_problem_slides"][:3])
PY
```

Expected: structured output with preview paths and slide summaries.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/slidecraft/SKILL.md .claude/skills/slide-review/SKILL.md CLAUDE.md \
  src/paperops/slides/layout/containers.py src/paperops/slides/layout/engine.py \
  src/paperops/slides/layout/auto_size.py src/paperops/slides/components/text.py \
  src/paperops/slides/components/shapes.py src/paperops/slides/components/composite.py \
  src/paperops/slides/preview.py src/paperops/slides/build.py \
  examples/self-intro/build.py \
  docs/superpowers/specs/2026-04-03-slidecraft-layout-review-design.md \
  docs/superpowers/plans/2026-04-03-slidecraft-layout-review.md
git commit -m "Improve SlideCraft layout negotiation and deck review"
```

---

## Self-Review

### Spec coverage

- Layout semantics upgrade -> Task 1
- Intrinsic sizing upgrade -> Task 2
- Integrated review API -> Task 3
- Perceptual heuristics -> Task 4
- Skill updates -> Task 5
- Real deck validation -> Task 6

No uncovered spec sections remain.

### Placeholder scan

No `TODO`, `TBD`, or deferred implementation markers are intentionally left in this plan.

### Type consistency

Planned public names are consistent across tasks:

- `size_mode_x`, `size_mode_y`, `grow`, `shrink`, `basis`, `wrap`
- `review_deck()`
- `review_deck_artifacts()`

