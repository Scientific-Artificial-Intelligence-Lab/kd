
from __future__ import annotations

import ast
from dataclasses import dataclass

from kd.core.equation.signature import law_term_entry, law_term_key
from kd.core.equation.sketch import Sketch, TermConstraint, constraint_admits




from kd.core.expr.executor import _DIFF_PATTERN
from kd.core.expr.naming import parse_compound_derivative
from kd.core.expr.term_features import TermFeatures, analyze_term
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.search.sga.config import OperatorPool, SGAConfig
from kd.search.sga.convert import tree_to_kd_expr
from kd.search.sga.tree import Node, Tree

_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)

_NATIVE_OPERATOR_FEATURES = {
    "+": {"add"},
    "-": {"sub"},
    "*": {"mul"},
    "/": {"mul", "recip"},
    "^2": {"n2"},
    "^3": {"n3"},
}
_IR_OPERATOR_MAP = {
    "add": ("+", 2),
    "sub": ("-", 2),
    "mul": ("*", 2),
    "div": ("/", 2),
    "n2": ("^2", 1),
    "n3": ("^3", 1),
}
_DIFF_HEAD = _DIFF_PATTERN

_OUTSIDE_VOCABULARY = "outside-vocabulary"
_CONSTRAINT_FILTERED = "constraint-filtered"
_PINNED = "pinned"


PinFingerprint = tuple[
    frozenset[str],
    frozenset[str],
    frozenset[tuple[str, tuple[tuple[str, int], ...]]],
    frozenset[str],
]


@dataclass(frozen=True, kw_only=True)
class SGACompiled:

    vars: tuple[str, ...]
    den: OperatorPool
    ops: OperatorPool
    root: OperatorPool
    op1: OperatorPool
    op2: OperatorPool
    pinned: tuple[tuple[str, float, Tree], ...]
    pinned_keys: frozenset[str]
    pinned_fingerprints: frozenset[PinFingerprint]
    anchored_keys: frozenset[str]
    default_kept: bool
    default_law_key: str | None
    default_hole_id: str | None
    report: CompileReport
    dropped: tuple[tuple[str, str], ...]


def _compose_derivative(node: Node, axis: str, order: int) -> Node:
    for _ in range(order // 2):
        node = Node("d^2", 2, [node, Node(axis, 0)])
    if order % 2:
        node = Node("d", 2, [node, Node(axis, 0)])
    return node


def _compound_node(name: str, variables: frozenset[str], sketch: Sketch) -> Node:
    parsed = parse_compound_derivative(
        name,
        known_fields=set(sketch.vocabulary.fields),
        known_axes=set(sketch.vocabulary.coordinates),
    )
    if parsed is None:
        raise ValueError(f"unsupported terminal {name!r}")
    field, segments = parsed
    if field not in variables or sum(order for _axis, order in segments) < 2:
        raise ValueError(f"terminal {name!r} is absent from SGA's variable pool")
    orders: dict[str, int] = {}
    for axis, order in segments:
        orders[axis] = orders.get(axis, 0) + order
    node = Node(field, 0)
    for axis, order in sorted(orders.items()):
        node = _compose_derivative(node, axis, order)
    return node


def _parse_node(node: ast.expr, variables: frozenset[str], sketch: Sketch) -> Node:
    if isinstance(node, ast.Name):
        if node.id in variables:
            return Node(node.id, 0)
        return _compound_node(node.id, variables, sketch)
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"unsupported syntax {type(node).__name__}")
    if node.keywords:
        raise ValueError("keyword arguments are outside SGA's tree language")
    head = node.func.id
    diff_match = _DIFF_HEAD.fullmatch(head)
    if diff_match is not None:
        if len(node.args) != 1:
            raise ValueError(f"{head!r} requires exactly one argument")
        order = int(diff_match.group(1) or "1")
        axis = diff_match.group(2)
        if order < 1 or axis not in sketch.vocabulary.coordinates:
            raise ValueError(f"unsupported derivative head {head!r}")
        return _compose_derivative(
            _parse_node(node.args[0], variables, sketch), axis, order
        )
    native = _IR_OPERATOR_MAP.get(head)
    if native is None:
        raise ValueError(f"operator {head!r} is outside SGA's tree language")
    native_name, arity = native
    if len(node.args) != arity:
        raise ValueError(f"{head!r} requires exactly {arity} arguments")
    return Node(
        native_name,
        arity,
        [_parse_node(argument, variables, sketch) for argument in node.args],
    )


def _parse_term_tree(term_ir: str, variables: tuple[str, ...], sketch: Sketch) -> Tree:
    try:
        parsed = ast.parse(term_ir.strip(), mode="eval").body
        while (
            isinstance(parsed, ast.Call)
            and isinstance(parsed.func, ast.Name)
            and parsed.func.id == "neg"
            and len(parsed.args) == 1
            and not parsed.keywords
        ):
            parsed = parsed.args[0]
        return Tree(_parse_node(parsed, frozenset(variables), sketch))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"SGA cannot parse sketch term {term_ir!r}: {exc}; use native "
            "spellings such as div(u,x) instead of mul(u,recip(x))"
        ) from exc


def _term_axes(features: TermFeatures) -> frozenset[str]:
    axes = set(features.coordinate_dependencies)
    for _field, multiindex in features.derivative_multiindices:
        axes.update(axis for axis, _order in multiindex)
    return frozenset(axes)


def _dimensionally_admitted(constraint: TermConstraint, features: TermFeatures) -> bool:
    if (
        constraint.max_deriv_order is not None
        and features.max_total_derivative_order > constraint.max_deriv_order
    ):
        return False
    if (
        constraint.fields is not None
        and features.base_fields
        and not features.base_fields <= constraint.fields
    ):
        return False
    return constraint.axes is None or _term_axes(features) <= constraint.axes


def _anchor_needs(candidate: TermFeatures, anchor: TermFeatures) -> bool:
    return (
        candidate.base_fields <= anchor.base_fields
        and _term_axes(candidate) <= _term_axes(anchor)
        and candidate.max_total_derivative_order <= anchor.max_total_derivative_order
    )


def _parse_pins(
    sketch: Sketch, variables: tuple[str, ...]
) -> tuple[tuple[tuple[str, float, Tree], ...], frozenset[str]]:
    pinned: list[tuple[str, float, Tree]] = []
    for pin in sketch.pinned:
        key, signed_value = law_term_entry(pin.term_ir, pin.value)
        pinned.append(
            (key, signed_value, _parse_term_tree(pin.term_ir, variables, sketch))
        )
    return tuple(pinned), frozenset(key for key, _value, _tree in pinned)


def _fingerprint(features: TermFeatures) -> PinFingerprint:
    return (
        features.base_fields,
        features.coordinate_dependencies,
        features.derivative_multiindices,
        features.operators,
    )


def _pinned_fingerprints(sketch: Sketch) -> frozenset[PinFingerprint]:
    seen: dict[PinFingerprint, str] = {}
    for pin in sketch.pinned:
        fp = _fingerprint(analyze_term(pin.term_ir, sketch.vocabulary))
        if fp in seen:
            raise ValueError(
                f"pinned terms {seen[fp]!r} and {pin.term_ir!r} are alias "
                "spellings of one physical column; pin it once"
            )
        seen[fp] = pin.term_ir
    for anchor in sketch.anchored:
        fp = _fingerprint(analyze_term(anchor.term_ir, sketch.vocabulary))
        if fp in seen:
            raise ValueError(
                f"anchored term {anchor.term_ir!r} is an alias spelling of the "
                f"pinned column {seen[fp]!r}; a pinned column is never "
                "refit"
            )
    return frozenset(seen)


def _operator_rooted_hint(term_ir: str, tree: Tree, sketch: Sketch) -> str:
    parsed = parse_compound_derivative(
        term_ir,
        known_fields=set(sketch.vocabulary.fields),
        known_axes=set(sketch.vocabulary.coordinates),
    )
    if parsed is None:
        return tree_to_kd_expr(tree)
    field, segments = parsed
    orders: dict[str, int] = {}
    for axis, order in segments:
        orders[axis] = orders.get(axis, 0) + order
    node = Node(field, 0)
    for axis, order in sorted(orders.items()):
        node = _compose_derivative(node, axis, order)
    return tree_to_kd_expr(Tree(node))


def _parse_anchors(
    sketch: Sketch, variables: tuple[str, ...], default_key: str | None
) -> tuple[frozenset[str], tuple[TermFeatures, ...]]:
    keys: list[str] = []
    features: list[TermFeatures] = []
    for anchor in sketch.anchored:
        key = law_term_key(anchor.term_ir)
        tree = _parse_term_tree(anchor.term_ir, variables, sketch)
        round_trip = law_term_key(tree_to_kd_expr(tree))
        if key != default_key and (tree.root.is_leaf or round_trip != key):
            raise ValueError(
                f"SGA anchor {anchor.term_ir!r} is not reachable; use the "
                f"operator-rooted spelling "
                f"{_operator_rooted_hint(anchor.term_ir, tree, sketch)!r}"
            )
        keys.append(key)
        features.append(analyze_term(anchor.term_ir, sketch.vocabulary))
    return frozenset(keys), tuple(features)


def _default_decision(
    sketch: Sketch,
    default_term_name: str | None,
    pinned_keys: frozenset[str],
) -> tuple[bool, str | None, str | None, tuple[str, str] | None]:
    if default_term_name is None:
        return False, None, None, None
    try:
        key = law_term_key(default_term_name)
        features = analyze_term(default_term_name, sketch.vocabulary)
    except ValueError:
        return False, None, None, (default_term_name, _OUTSIDE_VOCABULARY)
    if key in pinned_keys:
        return False, key, None, (default_term_name, _PINNED)
    if any(key == law_term_key(anchor.term_ir) for anchor in sketch.anchored):
        return True, key, None, None
    for hole in sketch.holes:
        if constraint_admits(hole.constraint, features):
            return True, key, hole.id, None
    return False, key, None, (default_term_name, _CONSTRAINT_FILTERED)


def _assert_feasible(sketch: Sketch, config: SGAConfig, default_kept: bool) -> None:
    required = sum(hole.min_count for hole in sketch.holes) + len(sketch.anchored)
    capacity = config.width + int(default_kept)
    if required > capacity:
        raise ValueError(
            f"SGA sketch requires {required} terms but config.width={config.width} "
            f"plus retained-default capacity {int(default_kept)} provides only "
            f"{capacity}"
        )


def _narrow_vars(
    sketch: Sketch,
    variables: tuple[str, ...],
    anchor_features: tuple[TermFeatures, ...],
    dropped: list[tuple[str, str]],
) -> tuple[str, ...]:
    kept: list[str] = []
    for name in variables:
        try:
            features = analyze_term(name, sketch.vocabulary)
        except ValueError:
            dropped.append((name, _OUTSIDE_VOCABULARY))
            continue
        hole_needed = any(
            _dimensionally_admitted(hole.constraint, features) for hole in sketch.holes
        )
        if hole_needed or any(
            _anchor_needs(features, item) for item in anchor_features
        ):
            kept.append(name)
        else:
            dropped.append((name, _CONSTRAINT_FILTERED))
    return tuple(kept)


def _narrow_den(
    sketch: Sketch,
    den: OperatorPool,
    anchor_features: tuple[TermFeatures, ...],
    dropped: list[tuple[str, str]],
) -> OperatorPool:
    anchor_axes = set().union(*(_term_axes(item) for item in anchor_features))
    kept: list[tuple[str, int]] = []
    for axis_entry in den:
        axis = axis_entry[0]
        hole_needed = any(
            (
                hole.constraint.max_deriv_order is None
                or hole.constraint.max_deriv_order >= 1
            )
            and (hole.constraint.axes is None or axis in hole.constraint.axes)
            for hole in sketch.holes
        )
        if hole_needed or axis in anchor_axes:
            kept.append(axis_entry)
        else:
            dropped.append((axis, _CONSTRAINT_FILTERED))
    return tuple(kept)


def _max_required_order(
    sketch: Sketch, anchor_features: tuple[TermFeatures, ...]
) -> int | None:
    caps = [hole.constraint.max_deriv_order for hole in sketch.holes]
    if any(cap is None for cap in caps):
        return None
    finite = [cap for cap in caps if cap is not None]
    finite.extend(item.max_total_derivative_order for item in anchor_features)
    return max(finite, default=0)


def _allowed_operator_names(
    sketch: Sketch, anchor_features: tuple[TermFeatures, ...]
) -> frozenset[str]:
    allowed: set[str] = set()
    for name, features in _NATIVE_OPERATOR_FEATURES.items():
        hole_admitted = any(
            hole.constraint.operators is None or features <= hole.constraint.operators
            for hole in sketch.holes
        )





        anchor_needed = any(features <= item.operators for item in anchor_features)
        if hole_admitted or anchor_needed:
            allowed.add(name)
    max_order = _max_required_order(sketch, anchor_features)
    if max_order is None or max_order >= 1:
        allowed.add("d")
    if max_order is None or max_order >= 2:
        allowed.add("d^2")
    return frozenset(allowed)


def _assert_live_pools(
    variables: tuple[str, ...],
    den: OperatorPool,
    ops: OperatorPool,
    root: OperatorPool,
    op1: OperatorPool,
    op2: OperatorPool,
) -> None:
    if not variables:
        raise ValueError("SGA narrowed vars pool is empty")
    if not ops or not root:
        raise ValueError("SGA narrowed ops/root pool is empty")
    derivative_live = any(name in {"d", "d^2"} for name, _arity in (*ops, *root, *op2))
    if derivative_live and not den:
        raise ValueError("SGA narrowed den pool is empty while derivatives survive")
    ops_set = set(ops)
    if not set(root) <= ops_set or not set(op1) <= ops_set or not set(op2) <= ops_set:
        raise ValueError("SGA narrowed operator pools are inconsistent")


def _notes(
    *,
    default_kept: bool,
    default_hole_id: str | None,
    dropped: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    seat = default_hole_id or "none"
    notes = [
        "hole min_count is exit-checked; support filtering and cancellation "
        "cannot be predicted",
        "generation enforces hole membership and max_count on distinct law keys",
        f"default column kept={default_kept}; hole seat={seat}",
        "pinned columns use SGA execute_tree footing; full verification uses "
        "the platform footing",
        "pin reservation is syntactic (law key + feature fingerprint); "
        "algebraically equal neutral wrappers are outside it (charter S-B-2)",
    ]
    notes.extend(f"dropped {element!r}: {reason}" for element, reason in dropped)
    return tuple(notes)


def _closed_compilation(
    sketch: Sketch,
    *,
    variables: tuple[str, ...],
    den: OperatorPool,
    ops: OperatorPool,
    root: OperatorPool,
    op1: OperatorPool,
    op2: OperatorPool,
    default_term_name: str | None,
) -> SGACompiled:
    default_key = None if default_term_name is None else law_term_key(default_term_name)
    return SGACompiled(
        vars=variables,
        den=den,
        ops=ops,
        root=root,
        op1=op1,
        op2=op2,
        pinned=(),
        pinned_keys=frozenset(law_term_key(pin.term_ir) for pin in sketch.pinned),
        pinned_fingerprints=frozenset(),
        anchored_keys=frozenset(),
        default_kept=default_term_name is not None,
        default_law_key=default_key,
        default_hole_id=None,
        report=CompileReport(
            levels=_LEVELS,
            notes=(
                "closed sketch: pools and target remain unchanged because the "
                "search result is discarded; lift from pins alone",
            ),
        ),
        dropped=(),
    )


def compile_for_sga(
    sketch: Sketch,
    *,
    vars: list[str],
    den: OperatorPool,
    ops: OperatorPool,
    root: OperatorPool,
    op1: OperatorPool,
    op2: OperatorPool,
    default_term_name: str | None,
    config: SGAConfig,
) -> SGACompiled:
    variables = tuple(vars)
    if not sketch.anchored and not sketch.holes:
        return _closed_compilation(
            sketch,
            variables=variables,
            den=den,
            ops=ops,
            root=root,
            op1=op1,
            op2=op2,
            default_term_name=default_term_name,
        )

    pinned, pinned_keys = _parse_pins(sketch, variables)
    pinned_fingerprints = _pinned_fingerprints(sketch)
    default_kept, default_key, default_hole_id, default_drop = _default_decision(
        sketch, default_term_name, pinned_keys
    )
    anchored_keys, anchor_features = _parse_anchors(sketch, variables, default_key)
    _assert_feasible(sketch, config, default_kept)

    dropped: list[tuple[str, str]] = []
    if default_drop is not None:
        dropped.append(default_drop)
    narrowed_vars = _narrow_vars(sketch, variables, anchor_features, dropped)
    narrowed_den = _narrow_den(sketch, den, anchor_features, dropped)
    allowed_names = _allowed_operator_names(sketch, anchor_features)
    for name, _arity in ops:
        if name not in allowed_names:
            dropped.append((name, _CONSTRAINT_FILTERED))
    narrowed_ops = tuple(entry for entry in ops if entry[0] in allowed_names)
    narrowed_root = tuple(entry for entry in root if entry[0] in allowed_names)
    narrowed_op1 = tuple(entry for entry in op1 if entry[0] in allowed_names)
    narrowed_op2 = tuple(entry for entry in op2 if entry[0] in allowed_names)
    _assert_live_pools(
        narrowed_vars,
        narrowed_den,
        narrowed_ops,
        narrowed_root,
        narrowed_op1,
        narrowed_op2,
    )

    frozen_dropped = tuple(dropped)
    return SGACompiled(
        vars=narrowed_vars,
        den=narrowed_den,
        ops=narrowed_ops,
        root=narrowed_root,
        op1=narrowed_op1,
        op2=narrowed_op2,
        pinned=pinned,
        pinned_keys=pinned_keys,
        pinned_fingerprints=pinned_fingerprints,
        anchored_keys=anchored_keys,
        default_kept=default_kept,
        default_law_key=default_key,
        default_hole_id=default_hole_id,
        report=CompileReport(
            levels=_LEVELS,
            notes=_notes(
                default_kept=default_kept,
                default_hole_id=default_hole_id,
                dropped=frozen_dropped,
            ),
        ),
        dropped=frozen_dropped,
    )


__all__ = ["SGACompiled", "compile_for_sga"]
