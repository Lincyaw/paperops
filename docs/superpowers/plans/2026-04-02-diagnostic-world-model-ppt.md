# Diagnostic World Model PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete 20-slide English SlideCraft presentation from the local self-intro references, generate a `.pptx`, and verify the deck with smoke tests and preview/review checks.

**Architecture:** Create one focused example build script under `examples/self-intro/` that owns the theme, helper layout primitives, slide construction, and output path. Add one smoke test that imports the builder, generates the deck in a temporary directory, and verifies slide count plus file creation so the presentation is reproducible instead of being a one-off artifact.

**Tech Stack:** Python 3.13, `uv`, `paperops.slides`, `pytest`

---

## File Structure

- Create: `examples/self-intro/build.py`
  Responsibility: build the full deck, including theme, slide helpers, slide notes, and save/review/preview entrypoints.
- Create: `examples/self-intro/diagnostic_world_model.pptx`
  Responsibility: generated presentation artifact from the build script.
- Create: `tests/examples/test_self_intro_build.py`
  Responsibility: smoke-test deck generation and validate slide count/output file creation.
- Reference: `examples/self-intro/references/MinerU_markdown_A_Goal_Driven_Survey_on_Root_Cause_Analysis_2039681713986326528.md`
  Responsibility: RCA framing, seven-goal taxonomy, ideal RCA formalization.
- Reference: `examples/self-intro/references/MinerU_markdown_FSE_26_RCA_dataset_study_2039681663084261376.md`
  Responsibility: benchmark crisis, dataset defects, benchmark scale, Top@1 performance collapse.
- Reference: `examples/self-intro/references/MinerU_markdown_ICML_Bench_2039681728423124992.md`
  Responsibility: process-supervision and benchmark wording where relevant.
- Reference: `examples/self-intro/references/MinerU_markdown_TRIANGLE_ASE25_(2)_2039681685943218176.md`
  Responsibility: TRIANGLE slide wording and key metrics.

### Task 1: Scaffold The Deck Builder And Smoke Test

**Files:**
- Create: `examples/self-intro/build.py`
- Create: `tests/examples/test_self_intro_build.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path

from examples.self_intro.build import OUTPUT_FILE, build_presentation


def test_build_self_intro_deck(tmp_path: Path):
    out_path = tmp_path / OUTPUT_FILE.name
    prs = build_presentation(output_path=out_path, render_preview=False)
    assert out_path.exists()
    assert out_path.suffix == ".pptx"
    assert len(prs.slides) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_build_self_intro_deck -v`
Expected: FAIL with `ModuleNotFoundError` or import failure because `examples/self-intro/build.py` does not exist yet.

- [ ] **Step 3: Write the minimal builder scaffold**

```python
from pathlib import Path

from paperops.slides import Presentation, themes

OUTPUT_FILE = Path(__file__).with_name("diagnostic_world_model.pptx")


def build_presentation(output_path: Path | None = None, render_preview: bool = False):
    prs = Presentation(theme=themes.professional)
    for i in range(20):
        prs.content(f"Slide {i + 1}", bullets=[f"Placeholder {i + 1}"])
    save_path = output_path or OUTPUT_FILE
    prs.save(str(save_path))
    return prs


if __name__ == "__main__":
    build_presentation()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_build_self_intro_deck -v`
Expected: PASS with one generated temporary `.pptx`.

- [ ] **Step 5: Commit**

```bash
git add examples/self-intro/build.py tests/examples/test_self_intro_build.py
git commit -m "feat: scaffold self-intro deck builder"
```

### Task 2: Implement Theme, Shared Layout Helpers, And Reference-Grounded Slide Copy

**Files:**
- Modify: `examples/self-intro/build.py`

- [ ] **Step 1: Add a focused copy/outline test as a failing assertion**

```python
from examples.self_intro.build import SLIDE_TITLES


def test_self_intro_slide_titles():
    assert SLIDE_TITLES[:5] == [
        "Building World Models for Diagnostic Intelligence",
        "The Vision",
        "Premise 1: Atomic Faults",
        "Premise 2: Simulatable",
        "Premise 3: Verifiable",
    ]
    assert SLIDE_TITLES[-2:] == [
        "My Research Philosophy",
        "Three Takeaways",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_self_intro_slide_titles -v`
Expected: FAIL because `SLIDE_TITLES` is not defined yet.

- [ ] **Step 3: Replace placeholder content with reference-grounded slide data and helper primitives**

```python
SLIDE_TITLES = [
    "Building World Models for Diagnostic Intelligence",
    "The Vision",
    "Premise 1: Atomic Faults",
    "Premise 2: Simulatable",
    "Premise 3: Verifiable",
    "The Problem with RCA",
    "Effectiveness-Data-Cost Triangle",
    "Ideal RCA Formalization",
    "The Benchmark Crisis",
    "Why Existing Benchmarks Fail",
    "Our Solution",
    "The Process Gap",
    "FORGE: Forward Verification",
    "OpenRCA 2.0",
    "Triangle: Multi-Agent Triage",
    "Stage 1: Train the Agent",
    "Stage 2: Distill World Model",
    "Stage 3: Universal Expert",
    "My Research Philosophy",
    "Three Takeaways",
]


def make_theme():
    return themes.professional.override(
        font_family="Aptos",
        colors={
            "primary": "#163B65",
            "secondary": "#5B84B1",
            "accent": "#E67E22",
            "positive": "#1F8A70",
            "negative": "#C44900",
            "highlight": "#2F6B9A",
            "warning": "#D97706",
            "text": "#1F2937",
            "text_mid": "#6B7280",
            "text_light": "#9CA3AF",
            "bg": "#F8FAFC",
            "bg_alt": "#EEF2F7",
            "bg_accent": "#E8F1F8",
            "border": "#D7DFE8",
        },
    )


def title_block(title: str, subtitle: str | None = None):
    ...


def stat_card(label: str, value: str, tone: str = "primary"):
    ...
```

- [ ] **Step 4: Run both smoke tests**

Run: `uv run pytest tests/examples/test_self_intro_build.py -v`
Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add examples/self-intro/build.py tests/examples/test_self_intro_build.py
git commit -m "feat: add self-intro theme and deck content model"
```

### Task 3: Build Slides 1-11 With Structured Visuals And Notes

**Files:**
- Modify: `examples/self-intro/build.py`
- Test: `tests/examples/test_self_intro_build.py`

- [ ] **Step 1: Add a failing structure test for the first half**

```python
from pathlib import Path

from examples.self_intro.build import build_presentation


def test_first_half_builds_without_placeholder_titles(tmp_path: Path):
    prs = build_presentation(output_path=tmp_path / "deck.pptx", render_preview=False)
    titles = [slide.shapes.title.text for slide in prs.slides[:11] if slide.shapes.title]
    assert "Slide 1" not in titles
    assert "Slide 11" not in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_first_half_builds_without_placeholder_titles -v`
Expected: FAIL because placeholder slides still use generic titles/content.

- [ ] **Step 3: Implement slides 1-11**

```python
def add_slides_1_11(prs):
    add_title_slide(prs)
    add_vision_slide(prs)
    add_atomic_faults_slide(prs)
    add_simulatable_slide(prs)
    add_verifiable_slide(prs)
    add_rca_problem_slide(prs)
    add_triangle_tradeoff_slide(prs)
    add_formalization_slide(prs)
    add_benchmark_crisis_slide(prs)
    add_dataset_failure_slide(prs)
    add_solution_slide(prs)
```

Implementation requirements:
- Use the survey reference for slides 6-8 wording and the ideal `F: O -> G` formalization.
- Use the dataset-study reference for slides 9-11 wording and metrics: `9,152`, `1,430`, `25`, `0.21`, `0.37`.
- Attach speaker notes summarizing the spoken transitions for each slide.
- Add 2-step or 3-step `sb.animate(...)` groups to every content slide in this range.

- [ ] **Step 4: Run the targeted tests and build**

Run: `uv run pytest tests/examples/test_self_intro_build.py -v`
Expected: PASS

Run: `uv run python examples/self-intro/build.py`
Expected: `examples/self-intro/diagnostic_world_model.pptx` is created successfully.

- [ ] **Step 5: Commit**

```bash
git add examples/self-intro/build.py tests/examples/test_self_intro_build.py examples/self-intro/diagnostic_world_model.pptx
git commit -m "feat: add vision and benchmark sections to self-intro deck"
```

### Task 4: Build Slides 12-20 And Close The Narrative Arc

**Files:**
- Modify: `examples/self-intro/build.py`
- Test: `tests/examples/test_self_intro_build.py`

- [ ] **Step 1: Add a failing structure test for the second half**

```python
from pathlib import Path

from examples.self_intro.build import build_presentation


def test_second_half_contains_world_model_roadmap_titles(tmp_path: Path):
    prs = build_presentation(output_path=tmp_path / "deck.pptx", render_preview=False)
    titles = [slide.shapes.title.text for slide in prs.slides]
    assert "FORGE: Forward Verification" in titles
    assert "Stage 2: Distill World Model" in titles
    assert "Three Takeaways" in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_second_half_contains_world_model_roadmap_titles -v`
Expected: FAIL until slides 12-20 are fully implemented.

- [ ] **Step 3: Implement slides 12-20**

```python
def add_slides_12_20(prs):
    add_process_gap_slide(prs)
    add_forge_slide(prs)
    add_openrca20_slide(prs)
    add_triangle_slide(prs)
    add_train_agent_slide(prs)
    add_distill_world_model_slide(prs)
    add_universal_expert_slide(prs)
    add_research_philosophy_slide(prs)
    add_takeaways_slide(prs)
```

Implementation requirements:
- Use the process-benchmark reference for slides 12-14 and the TRIANGLE reference for slide 15.
- Slides 16-18 should shift from evidence-heavy layouts to roadmap/process diagrams.
- Slide 20 should be the strongest closing slide typographically and should not read like a dense summary.
- Add speaker notes and animation groups throughout.

- [ ] **Step 4: Run full verification**

Run: `uv run pytest tests/examples/test_self_intro_build.py -v`
Expected: PASS

Run: `uv run python examples/self-intro/build.py`
Expected: PPT generated at `examples/self-intro/diagnostic_world_model.pptx`

Run: `uv run python -c "from examples.self_intro.build import build_presentation; prs = build_presentation(render_preview=False); print(prs.review())"`
Expected: review dictionary printed without fatal layout errors

- [ ] **Step 5: Commit**

```bash
git add examples/self-intro/build.py tests/examples/test_self_intro_build.py examples/self-intro/diagnostic_world_model.pptx
git commit -m "feat: complete diagnostic world model self-intro deck"
```

### Task 5: Preview Key Slides And Polish Visual Consistency

**Files:**
- Modify: `examples/self-intro/build.py`
- Update: `examples/self-intro/diagnostic_world_model.pptx`

- [ ] **Step 1: Add a preview smoke test**

```python
from pathlib import Path

from examples.self_intro.build import build_presentation


def test_preview_generation_for_key_slides(tmp_path: Path):
    prs = build_presentation(output_path=tmp_path / "deck.pptx", render_preview=False)
    paths = prs.preview(slides=[0, 1, 10, 14, 19], output_dir=tmp_path / "preview")
    assert len(paths) == 5
```

- [ ] **Step 2: Run test to verify current preview behavior**

Run: `uv run pytest tests/examples/test_self_intro_build.py::test_preview_generation_for_key_slides -v`
Expected: PASS if local preview rendering works, otherwise FAIL with a concrete rendering/import error to fix.

- [ ] **Step 3: Polish visuals based on preview output**

```python
KEY_PREVIEW_SLIDES = [0, 1, 10, 14, 19]


def main():
    prs = build_presentation()
    report = prs.review()
    print(report)
    preview_dir = Path(__file__).with_name("preview")
    paths = prs.preview(slides=KEY_PREVIEW_SLIDES, output_dir=preview_dir)
    print(paths)
```

Polish checklist:
- Remove any repeated box-heavy layouts across adjacent slides.
- Ensure quantitative slides use stat contrast rather than paragraph explanation.
- Ensure closing slides feel cleaner and more spacious than middle evidence slides.
- Keep all visible text short enough for live speaking.

- [ ] **Step 4: Run final build and preview**

Run: `uv run pytest tests/examples/test_self_intro_build.py -v`
Expected: PASS

Run: `uv run python examples/self-intro/build.py`
Expected: review dict plus preview paths printed; `.pptx` updated.

- [ ] **Step 5: Commit**

```bash
git add examples/self-intro/build.py tests/examples/test_self_intro_build.py examples/self-intro/diagnostic_world_model.pptx examples/self-intro/preview
git commit -m "feat: polish and verify self-intro deck previews"
```

## Self-Review

### Spec Coverage

- Vision, feasibility, achieved path, roadmap, and closing all map to Tasks 3 and 4.
- Reference grounding maps to Tasks 2-4.
- Native SlideCraft implementation maps to Tasks 1-4.
- Verification and previews map to Task 5.

### Placeholder Scan

- No `TODO`, `TBD`, or deferred implementation markers remain.
- All file paths and commands are explicit.
- Each test step includes concrete code.

### Type Consistency

- The plan consistently uses `build_presentation(output_path=..., render_preview=False)` as the build entrypoint.
- `SLIDE_TITLES` and `OUTPUT_FILE` names are used consistently across tasks.
