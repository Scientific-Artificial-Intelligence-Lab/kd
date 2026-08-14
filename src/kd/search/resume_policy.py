
from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from kd.search.run_spec import (
    CONFIG_CANON_SCHEME,
    canonicalize_config,
)

if TYPE_CHECKING:
    from kd.search.descriptor import ResumeTier
    from kd.search.protocol import FacadeWiringContract

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_ARTIFACT_KEYS",
    "SCIENCE_AXIS_DENYLISTS",
    "check_resume_config",
    "config_artifact_overlay",
    "resolve_field_tier",
]









SCIENCE_AXIS_DENYLISTS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "sga": frozenset({"use_autograd", "field_model"}),





    "dlga": frozenset(
        {"library", "lhs_auto_select", "target_lhs_order", "surrogate_model"}
    ),
    "discover": frozenset({"library", "max_diff_order", "pinn"}),
    "pysr": frozenset({"terms", "binary_operators", "unary_operators"}),
    "eqgpt": frozenset(
        {"variables", "start_words", "masked_tokens", "steady_constant_column"}
    ),
    "pysindy": frozenset({"terms"}),
    "llm4ed": frozenset(),
})













CONFIG_ARTIFACT_KEYS: Final[dict[str, frozenset[str]]] = {
    "sga": frozenset({"field_model"}),
    "dlga": frozenset({"surrogate_model"}),
}





_PHANTOM_KEYS: Final[frozenset[str]] = frozenset({"algorithm"})








_FINGERPRINT_PHANTOM_KEY: Final[str] = "library_fingerprint"
_FINGERPRINT_PHANTOM_ALGORITHMS: Final[frozenset[str]] = frozenset(
    {"pysr", "pysindy"}
)



_MAX_FIELD_VALUE_CHARS: Final[int] = 120


class _Missing:

    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING: Final[_Missing] = _Missing()


def config_artifact_overlay(
    canon: dict[str, Any], algorithm: str, artifacts: object
) -> dict[str, Any]:
    keys = CONFIG_ARTIFACT_KEYS.get(algorithm)
    if not keys or not isinstance(artifacts, Mapping):
        return canon
    for key in keys:
        if key in artifacts:
            canon[key] = artifacts[key]
    return canon


def resolve_field_tier(
    plugin_cls: type[FacadeWiringContract], algorithm: str, field: str
) -> ResumeTier:
    from kd.core.platform.sketch_compile import SKETCH_CONFIG_KEY

    if field == SKETCH_CONFIG_KEY:
        return "identity_breaking"
    for knob in plugin_cls.descriptor.knobs:
        if knob.name == field:
            return knob.resume_tier
    if field in SCIENCE_AXIS_DENYLISTS.get(algorithm, frozenset()):
        return "identity_breaking"
    return "init_only"


def _short(value: object) -> str:
    text = repr(value)
    if len(text) > _MAX_FIELD_VALUE_CHARS:
        return text[: _MAX_FIELD_VALUE_CHARS - 3] + "..."
    return text


def _change_detail(
    field: str, tier: ResumeTier, stored: object, live: object
) -> dict[str, object]:
    stored_missing = stored is _MISSING
    live_missing = live is _MISSING
    return {
        "field": field,
        "tier": tier,
        "stored": None if stored_missing else _short(stored),
        "live": None if live_missing else _short(live),
        "stored_missing": stored_missing,
        "live_missing": live_missing,
    }


def _render_change(field: str, stored: object, live: object) -> str:
    if stored is _MISSING:
        return (
            f"{field} (present only in the live config, absent from the checkpoint)"
        )
    if live is _MISSING:
        return (
            f"{field} (present only in the checkpoint, absent from the live config)"
        )
    return f"{field} (stored {_short(stored)} -> live {_short(live)})"


def _build_message(
    algorithm: str,
    identity_changes: list[str],
    init_changes: list[str],
    init_has_missing: bool,
) -> str:
    segments: list[str] = []
    if identity_changes:
        segments.append(
            "identity_breaking field(s) changed vs the checkpoint: "
            + ", ".join(identity_changes)
            + ". A science-identity change starts a new lineage: run a fresh fit "
            "in a new checkpoint directory; this checkpoint cannot be resumed "
            "under the new config."
        )
    if init_changes:
        if init_has_missing:


            remedy = (
                "This checkpoint predates or postdates the current kd config "
                "schema; start a fresh fit / new lineage."
            )
        else:
            remedy = (
                "These fields can only be set on a fresh fit; drop resume_from "
                "or revert them."
            )
        segments.append(
            "init_only field(s) changed vs the checkpoint: "
            + ", ".join(init_changes)
            + ". "
            + remedy
        )
    return f"resume config mismatch for algorithm {algorithm!r}: " + " ".join(segments)


def check_resume_config(
    stored_config: object,
    stored_scheme: object,
    *,
    algorithm: str,
    plugin_cls: type[FacadeWiringContract],
    live_config: dict[str, Any],
    live_artifacts: Mapping[str, Any] | None = None,
) -> None:



    if stored_config is None:
        return



    if stored_scheme != CONFIG_CANON_SCHEME:
        raise ValueError(
            "resume config mismatch: checkpoint config_canon_scheme "
            f"{stored_scheme!r} is not readable by this kd build (expected "
            f"{CONFIG_CANON_SCHEME!r}); cannot verify resume safety."
        )
    if not isinstance(stored_config, dict):
        raise ValueError(
            "not a kd checkpoint payload: stored 'config' must be a dict, "
            f"got {type(stored_config).__name__}"
        )






    live_canon = config_artifact_overlay(
        canonicalize_config(dict(live_config)), algorithm, live_artifacts
    )

    identity_changes: list[str] = []
    init_changes: list[str] = []
    resume_safe_changes: list[str] = []


    identity_detail: list[dict[str, object]] = []
    init_detail: list[dict[str, object]] = []
    init_has_missing = False

    phantom_keys = _PHANTOM_KEYS
    if algorithm in _FINGERPRINT_PHANTOM_ALGORITHMS:
        phantom_keys = phantom_keys | {_FINGERPRINT_PHANTOM_KEY}
    fields = (set(stored_config) | set(live_canon)) - phantom_keys
    for field in sorted(fields):
        stored_val = stored_config.get(field, _MISSING)
        live_val = live_canon.get(field, _MISSING)
        if stored_val == live_val:
            continue
        tier = resolve_field_tier(plugin_cls, algorithm, field)
        rendered = _render_change(field, stored_val, live_val)
        if tier == "resume_safe":
            resume_safe_changes.append(rendered)
            continue
        detail = _change_detail(field, tier, stored_val, live_val)
        if tier == "identity_breaking":
            identity_changes.append(rendered)
            identity_detail.append(detail)
        else:
            init_changes.append(rendered)
            init_detail.append(detail)
            if stored_val is _MISSING or live_val is _MISSING:
                init_has_missing = True

    if not identity_changes and not init_changes:



        if resume_safe_changes:
            logger.info(
                "resume config accepted for algorithm %r: resume_safe field(s) "
                "changed: %s",
                algorithm,
                ", ".join(resume_safe_changes),
            )
        return

    err = ValueError(
        _build_message(algorithm, identity_changes, init_changes, init_has_missing)
    )






    setattr(err, "changes", tuple(identity_detail + init_detail))
    raise err
