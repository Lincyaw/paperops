"""Style-driven animation collection and OOXML timing generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree

NSMAP = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

_P = NSMAP["p"]


def _pn(tag: str) -> str:
    return f"{{{_P}}}{tag}"


@dataclass(frozen=True)
class AnimationPreset:
    preset_id: str
    preset_class: str
    preset_subtype: str = "0"
    transition: str = "in"
    filter: str | None = None


@dataclass(frozen=True)
class AnimationSpec:
    node_id: int
    animation: str
    trigger: str
    group: str
    delay_ms: int
    duration_ms: int
    order: int


_PRESETS: dict[str, AnimationPreset] = {
    "fade-in": AnimationPreset("10", "entr", filter="fade"),
    "fade-up": AnimationPreset("10", "entr", filter="fade"),
    "fade-down": AnimationPreset("10", "entr", filter="fade"),
    "slide-in-left": AnimationPreset("1", "entr", "8"),
    "slide-in-right": AnimationPreset("1", "entr", "4"),
    "slide-in-up": AnimationPreset("1", "entr", "2"),
    "slide-in-down": AnimationPreset("1", "entr", "1"),
    "zoom-in": AnimationPreset("23", "entr"),
    "zoom-out": AnimationPreset("24", "entr"),
    "scale-in": AnimationPreset("23", "entr"),
    "fly-in": AnimationPreset("2", "entr"),
    "emphasis-pulse": AnimationPreset("34", "emph", transition="out"),
    "emphasis-color": AnimationPreset("27", "emph", transition="out"),
}


def _style_snapshot(node) -> dict[str, Any]:
    computed = getattr(node, "computed_style", None)
    snapshot = computed.snapshot() if computed is not None else {}
    for key, value in (getattr(node, "style", None) or {}).items():
        snapshot.setdefault(key, value)
    return snapshot


def _duration_ms(theme, value: Any, default_token: str = "base") -> int:
    raw = value if value is not None else default_token
    if isinstance(raw, (int, float)):
        return int(float(raw) * 1000)
    resolved = theme.resolve_token("duration", raw)
    return int(float(resolved) * 1000)


def collect_animation_specs(slide_node, *, theme) -> list[AnimationSpec]:
    specs: list[AnimationSpec] = []
    group_offsets: dict[str, int] = {}

    def walk(node, order_seed: list[int]) -> None:
        style = _style_snapshot(node)
        animation = style.get("animate")
        if isinstance(animation, str) and animation in _PRESETS:
            group = str(style.get("animate-group") or f"node:{order_seed[0]}")
            stagger_ms = (
                _duration_ms(theme, style.get("stagger"), default_token="instant")
                if style.get("stagger") is not None
                else 0
            )
            delay_ms = (
                _duration_ms(theme, style.get("delay"), default_token="instant")
                if style.get("delay") is not None
                else 0
            )
            delay_ms += group_offsets.get(group, 0)
            duration_ms = _duration_ms(
                theme, style.get("duration"), default_token="base"
            )
            specs.append(
                AnimationSpec(
                    node_id=id(node),
                    animation=animation,
                    trigger=str(style.get("animate-trigger") or "on-load"),
                    group=group,
                    delay_ms=delay_ms,
                    duration_ms=duration_ms,
                    order=order_seed[0],
                )
            )
            group_offsets[group] = group_offsets.get(group, 0) + stagger_ms
        order_seed[0] += 1
        for child in getattr(node, "children", None) or []:
            if hasattr(child, "type"):
                walk(child, order_seed)

    walk(slide_node, [0])
    return specs


def _inject_initial_hide(
    parent_child_tnlst, shape_ids: list[int], id_start: int
) -> int:
    if not shape_ids:
        return id_start

    hide_par = etree.SubElement(parent_child_tnlst, _pn("par"))
    hide_cTn = etree.SubElement(hide_par, _pn("cTn"))
    hide_cTn.set("id", str(id_start))
    hide_cTn.set("fill", "hold")
    st_cond = etree.SubElement(hide_cTn, _pn("stCondLst"))
    cond = etree.SubElement(st_cond, _pn("cond"))
    cond.set("delay", "0")
    hide_child = etree.SubElement(hide_cTn, _pn("childTnLst"))

    next_id = id_start + 1
    for shape_id in shape_ids:
        shape_par = etree.SubElement(hide_child, _pn("par"))
        shape_ctn = etree.SubElement(shape_par, _pn("cTn"))
        shape_ctn.set("id", str(next_id))
        next_id += 1
        shape_ctn.set("fill", "hold")
        shape_st = etree.SubElement(shape_ctn, _pn("stCondLst"))
        shape_cond = etree.SubElement(shape_st, _pn("cond"))
        shape_cond.set("delay", "0")
        shape_child = etree.SubElement(shape_ctn, _pn("childTnLst"))

        set_el = etree.SubElement(shape_child, _pn("set"))
        set_bhvr = etree.SubElement(set_el, _pn("cBhvr"))
        set_ctn = etree.SubElement(set_bhvr, _pn("cTn"))
        set_ctn.set("id", str(next_id))
        next_id += 1
        set_ctn.set("dur", "1")
        set_ctn.set("fill", "hold")
        set_st = etree.SubElement(set_ctn, _pn("stCondLst"))
        set_cond = etree.SubElement(set_st, _pn("cond"))
        set_cond.set("delay", "0")
        target = etree.SubElement(
            etree.SubElement(set_bhvr, _pn("tgtEl")), _pn("spTgt")
        )
        target.set("spid", str(shape_id))
        attr_list = etree.SubElement(set_bhvr, _pn("attrNameLst"))
        attr = etree.SubElement(attr_list, _pn("attrName"))
        attr.text = "style.visibility"
        to = etree.SubElement(set_el, _pn("to"))
        value = etree.SubElement(to, _pn("strVal"))
        value.set("val", "hidden")

    return next_id


def inject_appear_animations(
    slide_element,
    click_groups: list[list[int]],
    *,
    initially_hidden: list[int] | None = None,
):
    specs: list[AnimationSpec] = []
    order = 0
    for group_index, group in enumerate(click_groups):
        for shape_id in group:
            specs.append(
                AnimationSpec(
                    node_id=shape_id,
                    animation="fade-in",
                    trigger="on-click",
                    group=f"legacy-{group_index}",
                    delay_ms=0,
                    duration_ms=500,
                    order=order,
                )
            )
            order += 1
    inject_timing(
        slide_element,
        specs,
        node_to_shape={
            shape_id: shape_id for group in click_groups for shape_id in group
        },
        initially_hidden=initially_hidden,
    )


def inject_timing(
    slide_element,
    specs: list[AnimationSpec],
    *,
    node_to_shape: dict[int, int],
    initially_hidden: list[int] | None = None,
) -> None:
    bound_specs = [spec for spec in specs if spec.node_id in node_to_shape]
    if not bound_specs:
        return

    timing = etree.SubElement(slide_element, _pn("timing"))
    tn_lst = etree.SubElement(timing, _pn("tnLst"))
    root_par = etree.SubElement(tn_lst, _pn("par"))
    root_ctn = etree.SubElement(root_par, _pn("cTn"))
    root_ctn.set("id", "1")
    root_ctn.set("dur", "indefinite")
    root_ctn.set("restart", "never")
    root_ctn.set("nodeType", "tmRoot")
    root_child = etree.SubElement(root_ctn, _pn("childTnLst"))

    all_shape_ids = [node_to_shape[spec.node_id] for spec in bound_specs]
    next_id = _inject_initial_hide(root_child, initially_hidden or all_shape_ids, 3)

    seq = etree.SubElement(root_child, _pn("seq"))
    seq.set("concurrent", "1")
    seq.set("nextAc", "seek")
    seq_ctn = etree.SubElement(seq, _pn("cTn"))
    seq_ctn.set("id", str(next_id))
    next_id += 1
    seq_ctn.set("dur", "indefinite")
    seq_ctn.set("nodeType", "mainSeq")
    seq_child = etree.SubElement(seq_ctn, _pn("childTnLst"))

    grouped: dict[str, list[AnimationSpec]] = {}
    for spec in sorted(bound_specs, key=lambda item: (item.order, item.delay_ms)):
        grouped.setdefault(spec.group, []).append(spec)

    previous_trigger = "on-load"
    for group_specs in grouped.values():
        trigger = group_specs[0].trigger
        group_par = etree.SubElement(seq_child, _pn("par"))
        group_ctn = etree.SubElement(group_par, _pn("cTn"))
        group_ctn.set("id", str(next_id))
        next_id += 1
        group_ctn.set("fill", "hold")
        st_cond = etree.SubElement(group_ctn, _pn("stCondLst"))
        cond = etree.SubElement(st_cond, _pn("cond"))
        if trigger == "on-click":
            cond.set("delay", "indefinite")
            group_ctn.set("nodeType", "clickEffect")
        elif trigger == "with-previous":
            cond.set("delay", "0")
            group_ctn.set("nodeType", "withEffect")
        elif trigger == "after-previous":
            cond.set("delay", "0")
            group_ctn.set("nodeType", "afterEffect")
        else:
            cond.set("delay", "0")
            group_ctn.set(
                "nodeType",
                "afterEffect" if previous_trigger == "on-click" else "withEffect",
            )
        group_child = etree.SubElement(group_ctn, _pn("childTnLst"))

        for index, spec in enumerate(group_specs):
            preset = _PRESETS.get(spec.animation, _PRESETS["fade-in"])
            shape_par = etree.SubElement(group_child, _pn("par"))
            shape_ctn = etree.SubElement(shape_par, _pn("cTn"))
            shape_ctn.set("id", str(next_id))
            next_id += 1
            shape_ctn.set("fill", "hold")
            shape_ctn.set("presetID", preset.preset_id)
            shape_ctn.set("presetClass", preset.preset_class)
            shape_ctn.set("presetSubtype", preset.preset_subtype)
            shape_ctn.set("grpId", "0")
            shape_ctn.set(
                "nodeType",
                "withEffect" if index else group_ctn.get("nodeType", "clickEffect"),
            )
            shape_st = etree.SubElement(shape_ctn, _pn("stCondLst"))
            shape_cond = etree.SubElement(shape_st, _pn("cond"))
            shape_cond.set("delay", str(spec.delay_ms))
            shape_child = etree.SubElement(shape_ctn, _pn("childTnLst"))

            anim_effect = etree.SubElement(shape_child, _pn("animEffect"))
            anim_effect.set("transition", preset.transition)
            if preset.filter is not None:
                anim_effect.set("filter", preset.filter)
            eff_bhvr = etree.SubElement(anim_effect, _pn("cBhvr"))
            eff_ctn = etree.SubElement(eff_bhvr, _pn("cTn"))
            eff_ctn.set("id", str(next_id))
            next_id += 1
            eff_ctn.set("dur", str(spec.duration_ms))
            target = etree.SubElement(
                etree.SubElement(eff_bhvr, _pn("tgtEl")), _pn("spTgt")
            )
            target.set("spid", str(node_to_shape[spec.node_id]))

            set_el = etree.SubElement(shape_child, _pn("set"))
            set_bhvr = etree.SubElement(set_el, _pn("cBhvr"))
            set_ctn = etree.SubElement(set_bhvr, _pn("cTn"))
            set_ctn.set("id", str(next_id))
            next_id += 1
            set_ctn.set("dur", "1")
            set_ctn.set("fill", "hold")
            set_st = etree.SubElement(set_ctn, _pn("stCondLst"))
            set_cond = etree.SubElement(set_st, _pn("cond"))
            set_cond.set("delay", str(spec.delay_ms))
            set_target = etree.SubElement(
                etree.SubElement(set_bhvr, _pn("tgtEl")), _pn("spTgt")
            )
            set_target.set("spid", str(node_to_shape[spec.node_id]))
            attr_list = etree.SubElement(set_bhvr, _pn("attrNameLst"))
            attr = etree.SubElement(attr_list, _pn("attrName"))
            attr.text = "style.visibility"
            to = etree.SubElement(set_el, _pn("to"))
            value = etree.SubElement(to, _pn("strVal"))
            value.set("val", "visible")
        previous_trigger = trigger

    prev_cond = etree.SubElement(seq, _pn("prevCondLst"))
    pc = etree.SubElement(prev_cond, _pn("cond"))
    pc.set("evt", "onPrev")
    pc.set("delay", "0")
    etree.SubElement(etree.SubElement(pc, _pn("tgtEl")), _pn("sldTgt"))
    next_cond = etree.SubElement(seq, _pn("nextCondLst"))
    nc = etree.SubElement(next_cond, _pn("cond"))
    nc.set("evt", "onNext")
    nc.set("delay", "0")
    etree.SubElement(etree.SubElement(nc, _pn("tgtEl")), _pn("sldTgt"))


__all__ = [
    "AnimationSpec",
    "collect_animation_specs",
    "inject_appear_animations",
    "inject_timing",
]
