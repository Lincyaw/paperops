"""Layout engine — compute absolute positions for a tree of LayoutNodes."""

from __future__ import annotations

from paperops.slides.core.constants import Region, SLIDE_WIDTH, SLIDE_HEIGHT, CONTENT_REGION
from paperops.slides.layout.containers import LayoutNode, HStack, VStack, Grid, Padding


def compute_layout(root: LayoutNode, region: Region, theme) -> list[dict]:
    """Compute absolute positions for all leaf nodes.

    After this function every leaf node has its ``_region`` set.

    Returns a list of validation issues (dicts with keys: type, detail, region).
    """
    issues: list[dict] = []

    # Phase 1: size negotiation (bottom-up) — happens lazily inside layout
    # Phase 2+3: space allocation + positioning (top-down)
    _layout(root, region, theme)

    # Phase 4: validation
    _validate(root, issues)

    return issues


# ---------------------------------------------------------------------------
# Internal recursive layout
# ---------------------------------------------------------------------------

def _layout(node: LayoutNode, region: Region, theme) -> None:
    """Recursively assign _region to *node* (and all descendants)."""

    if isinstance(node, HStack):
        _layout_hstack(node, region, theme)
    elif isinstance(node, VStack):
        _layout_vstack(node, region, theme)
    elif isinstance(node, Grid):
        _layout_grid(node, region, theme)
    elif isinstance(node, Padding):
        _layout_padding(node, region, theme)
    else:
        # Leaf node — just assign the region
        node._region = region


def _layout_hstack(node: HStack, region: Region, theme) -> None:
    node._region = region

    if not node.children:
        return

    total_gap = node.gap * (len(node.children) - 1)
    available = region.width - total_gap

    # Determine fixed vs flexible children
    fixed_total = 0.0
    flex_indices: list[int] = []
    for i, child in enumerate(node.children):
        if hasattr(child, "width") and child.width is not None:
            fixed_total += child.width
        else:
            flex_indices.append(i)

    if flex_indices:
        per_flex = max((available - fixed_total) / len(flex_indices), 0.0)
    else:
        per_flex = 0.0

    x = region.left
    for i, child in enumerate(node.children):
        if hasattr(child, "width") and child.width is not None:
            cw = child.width
        else:
            cw = per_flex
        # Respect min_width
        if hasattr(child, "min_width") and child.min_width is not None:
            cw = max(cw, child.min_width)
        # Respect explicit height: use child's height and vertically center
        ch = region.height
        ct = region.top
        if hasattr(child, "height") and child.height is not None:
            ch = min(child.height, region.height)
            ct = region.top + (region.height - ch) / 2
        child_region = Region(left=x, top=ct, width=cw, height=ch)
        _layout(child, child_region, theme)
        x += cw + node.gap


def _layout_vstack(node: VStack, region: Region, theme) -> None:
    node._region = region

    if not node.children:
        return

    total_gap = node.gap * (len(node.children) - 1)
    available = region.height - total_gap

    # Determine fixed vs flexible children
    fixed_total = 0.0
    flex_indices: list[int] = []
    for i, child in enumerate(node.children):
        if hasattr(child, "height") and child.height is not None:
            fixed_total += child.height
        else:
            flex_indices.append(i)

    if flex_indices:
        per_flex = max((available - fixed_total) / len(flex_indices), 0.0)
    else:
        per_flex = 0.0

    y = region.top
    for i, child in enumerate(node.children):
        if hasattr(child, "height") and child.height is not None:
            ch = child.height
        else:
            ch = per_flex
        if hasattr(child, "min_height") and child.min_height is not None:
            ch = max(ch, child.min_height)
        # Respect explicit width: use child's width and horizontally center
        cw = region.width
        cl = region.left
        if hasattr(child, "width") and child.width is not None:
            cw = min(child.width, region.width)
            cl = region.left + (region.width - cw) / 2
        child_region = Region(left=cl, top=y, width=cw, height=ch)
        _layout(child, child_region, theme)
        y += ch + node.gap


def _layout_grid(node: Grid, region: Region, theme) -> None:
    node._region = region

    if not node.children:
        return

    n = len(node.children)
    rows = (n + node.cols - 1) // node.cols

    col_gap_total = node.gap * (node.cols - 1)
    row_gap_total = node.gap * (rows - 1)

    cell_w = (region.width - col_gap_total) / node.cols
    cell_h = (region.height - row_gap_total) / rows

    # Determine if last row needs centering
    last_row_count = n % node.cols
    last_row_offset = 0.0
    if node.center_last_row and last_row_count != 0:
        # Width occupied by last-row items
        used_w = last_row_count * cell_w + node.gap * (last_row_count - 1)
        last_row_offset = (region.width - used_w) / 2

    for idx, child in enumerate(node.children):
        r = idx // node.cols
        c = idx % node.cols
        x = region.left + c * (cell_w + node.gap)
        # Apply centering offset for the last (incomplete) row
        if node.center_last_row and last_row_count != 0 and r == rows - 1:
            x += last_row_offset
        y = region.top + r * (cell_h + node.gap)
        child_region = Region(left=x, top=y, width=cell_w, height=cell_h)
        _layout(child, child_region, theme)


def _layout_padding(node: Padding, region: Region, theme) -> None:
    node._region = region

    if node.child is None:
        return

    inner = Region(
        left=region.left + node._left,
        top=region.top + node._top,
        width=max(region.width - node._left - node._right, 0.0),
        height=max(region.height - node._top - node._bottom, 0.0),
    )
    _layout(node.child, inner, theme)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(node: LayoutNode, issues: list[dict]) -> None:
    """Walk the tree and check for overflow / out-of-bounds."""
    region = node._region
    if region is None:
        return

    # Check slide bounds
    if region.left < -0.01 or region.top < -0.01:
        issues.append({
            "type": "out_of_bounds",
            "detail": f"Region starts at ({region.left:.2f}, {region.top:.2f}) — negative coordinates",
            "region": region,
        })
    if region.right > SLIDE_WIDTH + 0.01:
        issues.append({
            "type": "overflow",
            "detail": f"Region extends to x={region.right:.2f}, slide width is {SLIDE_WIDTH}",
            "region": region,
        })
    if region.bottom > SLIDE_HEIGHT + 0.01:
        issues.append({
            "type": "overflow",
            "detail": f"Region extends to y={region.bottom:.2f}, slide height is {SLIDE_HEIGHT}",
            "region": region,
        })

    # Check children overflow for container nodes
    if isinstance(node, HStack) and node.children:
        total_gap = node.gap * (len(node.children) - 1)
        fixed_total = sum(
            child.width for child in node.children
            if hasattr(child, "width") and child.width is not None
        )
        if fixed_total + total_gap > region.width + 0.01:
            issues.append({
                "type": "children_overflow",
                "detail": (
                    f"HStack children fixed width ({fixed_total:.2f}) + gaps ({total_gap:.2f}) "
                    f"= {fixed_total + total_gap:.2f} exceeds available width {region.width:.2f}"
                ),
                "region": region,
            })
    elif isinstance(node, VStack) and node.children:
        total_gap = node.gap * (len(node.children) - 1)
        fixed_total = sum(
            child.height for child in node.children
            if hasattr(child, "height") and child.height is not None
        )
        if fixed_total + total_gap > region.height + 0.01:
            issues.append({
                "type": "children_overflow",
                "detail": (
                    f"VStack children fixed height ({fixed_total:.2f}) + gaps ({total_gap:.2f}) "
                    f"= {fixed_total + total_gap:.2f} exceeds available height {region.height:.2f}"
                ),
                "region": region,
            })
    elif isinstance(node, Grid) and node.children:
        n = len(node.children)
        rows = (n + node.cols - 1) // node.cols
        col_gap_total = node.gap * (node.cols - 1)
        row_gap_total = node.gap * (rows - 1)
        # Check per-row width: max fixed widths in any single row
        for row_idx in range(rows):
            row_start = row_idx * node.cols
            row_end = min(row_start + node.cols, n)
            row_children = node.children[row_start:row_end]
            row_fixed_w = sum(
                child.width for child in row_children
                if hasattr(child, "width") and child.width is not None
            )
            if row_fixed_w + col_gap_total > region.width + 0.01:
                issues.append({
                    "type": "children_overflow",
                    "detail": (
                        f"Grid row {row_idx} fixed width ({row_fixed_w:.2f}) + gaps ({col_gap_total:.2f}) "
                        f"= {row_fixed_w + col_gap_total:.2f} exceeds available width {region.width:.2f}"
                    ),
                    "region": region,
                })
                break  # one warning is enough

    # Recurse into children
    if isinstance(node, (HStack, VStack, Grid)):
        for child in node.children:
            _validate(child, issues)
    elif isinstance(node, Padding) and node.child is not None:
        _validate(node.child, issues)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from paperops.slides.core.theme import themes

    theme = themes.professional

    # Build a simple tree: HStack with 3 leaf nodes
    a = LayoutNode()
    b = LayoutNode()
    c = LayoutNode()

    stack = HStack(gap=0.2, children=[a, b, c])

    issues = compute_layout(stack, CONTENT_REGION, theme)

    print("=== Layout results ===")
    print(f"Container region: {stack._region}")
    for i, child in enumerate(stack.children):
        print(f"  Child {i}: {child._region}")
    if issues:
        print(f"\nValidation issues ({len(issues)}):")
        for iss in issues:
            print(f"  [{iss['type']}] {iss['detail']}")
    else:
        print("\nNo validation issues.")

    # Also test VStack
    x = LayoutNode()
    y = LayoutNode(height=1.0)
    z = LayoutNode()
    vstack = VStack(gap=0.2, children=[x, y, z])
    compute_layout(vstack, CONTENT_REGION, theme)
    print("\n=== VStack results ===")
    print(f"Container region: {vstack._region}")
    for i, child in enumerate(vstack.children):
        print(f"  Child {i}: {child._region}")

    # Test Grid
    items = [LayoutNode() for _ in range(5)]
    grid = Grid(cols=3, gap=0.2, children=items)
    compute_layout(grid, CONTENT_REGION, theme)
    print("\n=== Grid (3 cols, 5 items) results ===")
    print(f"Container region: {grid._region}")
    for i, child in enumerate(grid.children):
        print(f"  Cell {i}: {child._region}")

    # Test Padding
    inner = LayoutNode()
    padded = Padding(child=inner, all=0.5)
    compute_layout(padded, CONTENT_REGION, theme)
    print("\n=== Padding results ===")
    print(f"Padding region: {padded._region}")
    print(f"Inner region:   {inner._region}")
