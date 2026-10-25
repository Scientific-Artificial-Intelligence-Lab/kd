
from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import torch

from kd.core.expr.executor import (
    UNITY_TOKEN,
    _is_diff_operator,
    _parse_diff_name,
    _stencil_override,
)
from kd.core.expr.naming import parse_compound_derivative
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data import interface_mask
from kd.data.schema import compute_dataset_fingerprint

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.executor.context import ExecutionContext
    from kd.data.schema import AxisInfo, PDEDataset
    from kd.models import StoredSurrogate
    from kd.search.protocol import PlatformComponents

__all__ = [
    "METHODS",
    "RefitFrame",
    "Trim",
    "build_components",
    "check_frame_arguments",
    "make_frame",
    "require_source_binding",
    "resolve_rows",
    "unmodeled_field_reasons",
]

Method = Literal["finite_diff", "autograd", "weak"]
Trim = int | Mapping[str, int] | Literal["by_order"]

METHODS: Final[tuple[str, ...]] = ("finite_diff", "autograd", "weak")






FD_IMPLEMENTATION: Final[str] = "kd-fd-v1"
AUTOGRAD_IMPLEMENTATION: Final[str] = "kd-autograd-v1"
IMPLEMENTATIONS: Final[dict[str, str]] = {
    "finite_diff": FD_IMPLEMENTATION,
    "autograd": AUTOGRAD_IMPLEMENTATION,
    "weak": "kd-weak-v1",
}




_STENCIL_REACH: Final[int] = 2




_SOURCE_KEYS: Final[tuple[str, ...]] = ("dataset", "target_sha256")


@dataclass(frozen=True)
class RefitFrame:

    input_identity: str
    lhs_field: str
    lhs_axis: str
    lhs_order: int
    field_source: str
    derivative_method: str
    implementation: str
    surrogate_identity: Mapping[str, Any] | None
    trim: Mapping[str, int]
    rows_sha256: str | None
    n_samples: int
    condition_number_basis: str


def check_frame_arguments(
    method: str,
    *,
    trim: Trim,
    rows: Tensor | None,
    surrogate: StoredSurrogate | None,
    weak: object | None,
) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}; got {method!r}")
    _check_trim_value(trim)
    trimmed = not _trim_is_zero(trim)
    if method == "weak":
        if rows is not None:
            raise ValueError(
                "method='weak' fits test-function windows; rows is refused"
            )
        if trimmed:
            raise ValueError(
                "method='weak' fits test-function windows; trim is refused"
            )
        if surrogate is not None:
            raise ValueError(
                "method='weak' integrates the dataset it is given; surrogate is "
                "refused (pass a sampled field as its own dataset)"
            )
        return
    if weak is not None:
        raise ValueError(f"weak options require method='weak'; got method={method!r}")
    if method == "finite_diff" and surrogate is not None:
        raise ValueError("a surrogate is used only by method='autograd'")
    if method == "autograd" and surrogate is None:
        raise ValueError("method='autograd' needs surrogate=<a stored surrogate>")
    if trimmed and rows is not None:
        raise ValueError("trim and rows select rows two ways; set only one of them")
    if method == "autograd" and trim == "by_order":
        raise ValueError(
            "trim='by_order' follows finite-difference stencil footprints; "
            "method='autograd' has none, so pass an integer or per-axis trim"
        )


def _check_trim_value(trim: Trim) -> None:
    if isinstance(trim, str):
        if trim != "by_order":
            raise ValueError(
                f"trim must be an int, a mapping or 'by_order'; got {trim!r}"
            )
        return
    counts = trim.values() if isinstance(trim, Mapping) else (trim,)
    if any(count < 0 for count in counts):
        raise ValueError(f"trim counts must be nonnegative; got {trim!r}")


def _trim_is_zero(trim: Trim) -> bool:
    if isinstance(trim, str):
        return False
    if isinstance(trim, Mapping):
        return all(count == 0 for count in trim.values())
    return trim == 0


def build_components(
    dataset: PDEDataset,
    *,
    max_order: int,
    lhs_order: int,
    surrogate: StoredSurrogate | None = None,
) -> PlatformComponents:
    if surrogate is None:
        reqs = DerivativeReqs(max_atomic_order=max_order, lhs_order=lhs_order)
    else:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            max_atomic_order=max_order,
            lhs_order=lhs_order,
            needs_surrogate=True,
            surrogate_model=surrogate.module,
        )
    return PlatformBuilder(dataset, reqs, compute_condition_number=True).build()


def require_source_binding(dataset: PDEDataset, surrogate: StoredSurrogate) -> None:
    if surrogate.identity is None:
        raise ValueError(
            f"surrogate {surrogate.path} carries no identity, so the dataset it "
            "was trained on cannot be verified; save it with an identity "
            "(Model.train_surrogate(save_to=...))"
        )


    from kd.search.dlga.config import DLGAConfig
    from kd.search.dlga.surrogate_store import dlga_surrogate_identity

    expected = dlga_surrogate_identity(dataset, DLGAConfig())
    differing = [
        key for key in _SOURCE_KEYS if surrogate.identity.get(key) != expected[key]
    ]
    if differing:
        raise ValueError(
            f"surrogate {surrogate.path} was trained on another dataset; "
            f"differing identity keys: {differing}"
        )


def unmodeled_field_reasons(
    terms: Sequence[str], dataset: PDEDataset
) -> dict[str, str]:

    from kd.search.dlga.plugin import _unmodeled_field_terms

    lhs = dataset.lhs_field
    return {
        term: (
            f"reads {sorted(fields)}; method='autograd' takes values and "
            f"derivatives from the model of the LHS field {lhs!r} only"
        )
        for term, fields in _unmodeled_field_terms(tuple(terms), dataset).items()
    }


def resolve_rows(
    components: PlatformComponents,
    *,
    trim: Trim,
    rows: Tensor | None,
    terms: Sequence[str],
) -> tuple[dict[str, int], Tensor | None, str | None]:
    dataset = components.dataset
    axes = cast(list[str], dataset.axis_order)
    if rows is not None:
        return dict.fromkeys(axes, 0), rows, _rows_sha256(rows)
    if trim == "by_order":
        margins = by_order_margins(components, terms)
    elif isinstance(trim, Mapping):
        unknown = sorted(set(trim) - set(axes))
        if unknown:
            raise ValueError(f"trim names axes the dataset lacks: {unknown}")
        margins = {axis: int(trim.get(axis, 0)) for axis in axes}
    else:
        margins = dict.fromkeys(axes, int(trim))
    if not any(margins.values()):
        return margins, None, None
    values = dataset.get_field(dataset.lhs_field)
    mask = interface_mask(
        values,
        axis=0,
        width=0,
        trim={axes.index(axis): count for axis, count in margins.items()},
    )
    return margins, mask, None


def _rows_sha256(rows: Tensor) -> str:
    indices = rows.flatten().nonzero().flatten() if rows.dtype == torch.bool else rows
    data = indices.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def by_order_margins(
    components: PlatformComponents, terms: Sequence[str]
) -> dict[str, int]:
    dataset = components.dataset
    context = cast("ExecutionContext", components.context)
    axes = _axes(dataset)
    margins = dict.fromkeys(cast(list[str], dataset.axis_order), 0)
    margins.update(
        _apply_pass({}, dataset.lhs_axis, axes[dataset.lhs_axis].is_periodic)
    )
    for term in terms:
        node = ast.parse(term.strip(), mode="eval").body
        for axis, margin in _node_margins(node, context).items():
            margins[axis] = max(margins[axis], margin)
    return margins


def _axes(dataset: PDEDataset) -> dict[str, AxisInfo]:
    assert dataset.axes is not None
    return dataset.axes


def _apply_pass(inner: dict[str, int], axis: str, periodic: bool) -> dict[str, int]:
    margin = inner.get(axis, 0)
    if periodic and margin == 0:
        return inner
    return {**inner, axis: margin + _STENCIL_REACH}


def _merge(parts: Sequence[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for part in parts:
        for axis, margin in part.items():
            merged[axis] = max(merged.get(axis, 0), margin)
    return merged


def _diff_pass(
    argument: ast.expr, axis: str, context: ExecutionContext
) -> dict[str, int]:
    override = _stencil_override(argument, axis, context)
    periodic = (
        _axes(context.dataset)[axis].is_periodic if override is None else override
    )
    return _apply_pass(_node_margins(argument, context), axis, periodic)


def _node_margins(node: ast.expr, context: ExecutionContext) -> dict[str, int]:
    if isinstance(node, ast.Name):
        return _name_margins(node.id, context.dataset)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        head = node.func.id
        if _is_diff_operator(head):
            return _diff_pass(node.args[0], _parse_diff_name(head)[0], context)
        if head == "lap":
            spatial = context.dataset.spatial_axes
            return _merge([_diff_pass(node.args[0], axis, context) for axis in spatial])
        return _merge([_node_margins(argument, context) for argument in node.args])
    children = [
        child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)
    ]
    return _merge([_node_margins(child, context) for child in children])


def _name_margins(name: str, dataset: PDEDataset) -> dict[str, int]:
    fields = set(dataset.fields or {})
    axes = _axes(dataset)
    if name == UNITY_TOKEN or name in fields or name in axes:
        return {}
    parsed = parse_compound_derivative(name, known_fields=fields, known_axes=set(axes))
    if parsed is None:
        return {}
    margins: dict[str, int] = {}
    for axis, _order in parsed[1]:
        margins = _apply_pass(margins, axis, axes[axis].is_periodic)
    return margins


def make_frame(
    components_dataset: PDEDataset,
    *,
    method: str,
    lhs_order: int,
    trim: Mapping[str, int],
    n_samples: int,
    rows_sha256: str | None = None,
    surrogate: StoredSurrogate | None = None,
) -> RefitFrame:
    return RefitFrame(
        input_identity=compute_dataset_fingerprint(components_dataset),
        lhs_field=components_dataset.lhs_field,
        lhs_axis=components_dataset.lhs_axis,
        lhs_order=lhs_order,
        field_source="surrogate" if method == "autograd" else "observed",
        derivative_method=method,
        implementation=IMPLEMENTATIONS[method],
        surrogate_identity=None if surrogate is None else surrogate.identity,
        trim=dict(trim),
        rows_sha256=rows_sha256,
        n_samples=n_samples,
        condition_number_basis="unit_l2_columns" if method == "weak" else "raw_columns",
    )
