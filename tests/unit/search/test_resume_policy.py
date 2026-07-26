
from __future__ import annotations

import json

import pytest

from kd.api import _PLUGIN_CLASS_BY_ALGORITHM
from kd.search import resume_policy
from kd.search.run_spec import CONFIG_CANON_SCHEME

pytestmark = pytest.mark.unit

_SGA = _PLUGIN_CLASS_BY_ALGORITHM["sga"]
_DLGA = _PLUGIN_CLASS_BY_ALGORITHM["dlga"]
_DISCOVER = _PLUGIN_CLASS_BY_ALGORITHM["discover"]
_PYSR = _PLUGIN_CLASS_BY_ALGORITHM["pysr"]







def test_c1_every_declared_knob_resolves_to_its_declared_tier() -> None:



    for algorithm, plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.items():
        for knob in plugin_cls.descriptor.knobs:
            assert (
                resume_policy.resolve_field_tier(plugin_cls, algorithm, knob.name)
                == knob.resume_tier
            )


def test_c1_denylist_members_resolve_to_identity_breaking() -> None:
    for algorithm, plugin_cls in _PLUGIN_CLASS_BY_ALGORITHM.items():
        knob_names = {knob.name for knob in plugin_cls.descriptor.knobs}
        for member in resume_policy.SCIENCE_AXIS_DENYLISTS[algorithm]:
            if member in knob_names:
                continue
            assert (
                resume_policy.resolve_field_tier(plugin_cls, algorithm, member)
                == "identity_breaking"
            )


def test_c1_unlisted_field_defaults_to_init_only() -> None:



    assert (
        resume_policy.resolve_field_tier(_SGA, "sga", "totally_new_field_xyz")
        == "init_only"
    )

    assert resume_policy.resolve_field_tier(_SGA, "sga", "p_mute") == "init_only"
    assert resume_policy.resolve_field_tier(_SGA, "sga", "seed") == "init_only"







def _live(**fields: object) -> dict[str, object]:
    return dict(fields)


def test_c2_nested_dict_change_is_detected() -> None:


    stored = {"algorithm": "discover", "library": {"operators": ["u"]}}
    live = _live(algorithm="discover", library={"operators": ["u", "u_x"]})
    with pytest.raises(ValueError, match="library"):
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="discover",
            plugin_cls=_DISCOVER,
            live_config=live,
        )


def test_c2_single_sided_key_is_changed_and_fail_closed_init_only() -> None:


    stored = {"algorithm": "discover"}
    with pytest.raises(ValueError, match="init_only"):
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="discover",
            plugin_cls=_DISCOVER,
            live_config=_live(algorithm="discover", brand_new_field=1),
        )


def test_c2_tuple_vs_list_normalized_equal_does_not_false_positive() -> None:


    stored = {"algorithm": "pysr", "binary_operators": ["+", "-"]}
    assert (
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="pysr",
            plugin_cls=_PYSR,
            live_config=_live(algorithm="pysr", binary_operators=("+", "-")),
        )
        is None
    )







def test_c3_init_only_message_names_field_and_fresh_fit() -> None:
    stored = {"algorithm": "sga", "aic_ratio": 1.0}
    with pytest.raises(ValueError, match="init_only") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", aic_ratio=2.0),
        )
    message = str(exc.value)
    assert "aic_ratio" in message
    assert "fresh fit" in message


def test_c3_identity_breaking_message_names_field_and_new_lineage() -> None:
    stored = {"algorithm": "sga", "use_autograd": False}
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", use_autograd=True),
        )
    message = str(exc.value)
    assert "use_autograd" in message
    assert "new lineage" in message


def test_c3_both_buckets_one_error_identity_headline_first() -> None:
    stored = {"algorithm": "sga", "use_autograd": False, "aic_ratio": 1.0}
    with pytest.raises(ValueError) as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", use_autograd=True, aic_ratio=2.0),
        )
    message = str(exc.value)
    assert "identity_breaking" in message
    assert "init_only" in message

    assert message.index("identity_breaking") < message.index("init_only")







def test_c4_legacy_none_snapshot_passes_silently() -> None:
    assert (
        resume_policy.check_resume_config(
            None,
            None,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", aic_ratio=99.0),
        )
        is None
    )


def test_c5_foreign_scheme_fails_closed() -> None:


    stored = {"algorithm": "sga", "aic_ratio": 1.0}
    with pytest.raises(ValueError, match="config_canon_scheme"):
        resume_policy.check_resume_config(
            stored,
            "kd-config-v2",
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", aic_ratio=1.0),
        )


def test_c6_malformed_stored_config_fails_closed() -> None:




    with pytest.raises(ValueError, match="not a kd checkpoint payload"):
        resume_policy.check_resume_config(
            ["not", "a", "dict"],
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga"),
        )







def test_c7_resume_safe_only_change_passes_silently() -> None:
    stored = {"algorithm": "sga", "num": 8}
    assert (
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", num=4),
        )
        is None
    )







def test_phantom_library_fingerprint_is_exempt_from_diff() -> None:


    stored = {"algorithm": "pysr", "library_fingerprint": "OLD"}
    assert (
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="pysr",
            plugin_cls=_PYSR,
            live_config=_live(algorithm="pysr", library_fingerprint="NEW"),
        )
        is None
    )


def test_phantom_library_fingerprint_is_not_exempt_for_other_algorithm() -> None:




    stored = {"algorithm": "sga", "library_fingerprint": "OLD"}
    with pytest.raises(ValueError, match="init_only") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", library_fingerprint="NEW"),
        )
    assert "library_fingerprint" in str(exc.value)


def test_missing_variant_present_only_in_live_renders_schema_evolution() -> None:


    stored = {"algorithm": "discover"}
    with pytest.raises(ValueError, match="init_only") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="discover",
            plugin_cls=_DISCOVER,
            live_config=_live(algorithm="discover", added_field=7),
        )
    message = str(exc.value)
    assert "added_field" in message
    assert "present only in the live config" in message
    assert "fresh fit" in message
    assert "revert" not in message


def test_missing_variant_present_only_in_checkpoint_renders_single_sided() -> None:
    stored = {"algorithm": "discover", "removed_field": 3}
    with pytest.raises(ValueError, match="init_only") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="discover",
            plugin_cls=_DISCOVER,
            live_config=_live(algorithm="discover"),
        )
    message = str(exc.value)
    assert "removed_field" in message
    assert "present only in the checkpoint" in message


def test_mixed_init_only_bucket_uses_schema_evolution_remedy() -> None:









    stored = {"algorithm": "sga", "aic_ratio": 1.0}
    with pytest.raises(ValueError, match="init_only") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", aic_ratio=2.0, added_field=7),
        )
    message = str(exc.value)
    assert "aic_ratio" in message
    assert "added_field" in message

    assert "predates" in message
    assert "fresh fit" in message
    assert "revert" not in message


def test_f13a_live_config_canonicalization_failure_raises_at_guard() -> None:


    stored = {"algorithm": "sga", "aic_ratio": 1.0}
    with pytest.raises(ValueError):
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config={"algorithm": "sga", "bad": {1, 2, 3}},
        )







def test_present_snapshot_with_none_scheme_fails_closed() -> None:




    stored = {"algorithm": "sga", "aic_ratio": 1.0}
    with pytest.raises(ValueError, match="config_canon_scheme"):
        resume_policy.check_resume_config(
            stored,
            None,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", aic_ratio=1.0),
        )







def _identity(sha: str, size: int = 1) -> dict[str, object]:
    return {"format": "kd-torch-module-v1", "sha256": sha, "size": size}




_PLACEHOLDER: dict[str, object] = {"artifact": "field_model", "format": "placeholder"}


def test_sga_field_model_identity_change_is_identity_breaking() -> None:


    stored = {"algorithm": "sga", "field_model": _identity("AAA")}
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", field_model=dict(_PLACEHOLDER)),
            live_artifacts={"field_model": _identity("BBB")},
        )
    assert "field_model" in str(exc.value)


def test_sga_field_model_identical_identity_accepts() -> None:



    stored = {"algorithm": "sga", "field_model": _identity("AAA")}
    assert (
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", field_model=dict(_PLACEHOLDER)),
            live_artifacts={"field_model": _identity("AAA")},
        )
        is None
    )


def test_sga_field_model_added_on_resume_is_identity_breaking() -> None:


    stored = {"algorithm": "sga", "field_model": None}
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="sga",
            plugin_cls=_SGA,
            live_config=_live(algorithm="sga", field_model=dict(_PLACEHOLDER)),
            live_artifacts={"field_model": _identity("BBB")},
        )
    assert "field_model" in str(exc.value)


def test_dlga_surrogate_model_identity_change_is_identity_breaking() -> None:


    stored = {"algorithm": "dlga", "surrogate_model": _identity("AAA")}
    with pytest.raises(ValueError, match="identity_breaking") as exc:
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="dlga",
            plugin_cls=_DLGA,
            live_config=_live(algorithm="dlga", surrogate_model=dict(_PLACEHOLDER)),
            live_artifacts={"surrogate_model": _identity("BBB")},
        )
    assert "surrogate_model" in str(exc.value)









def test_config_artifact_keys_excludes_run_provenance_plugins() -> None:



    assert set(resume_policy.CONFIG_ARTIFACT_KEYS) == {"sga", "dlga"}
    assert resume_policy.CONFIG_ARTIFACT_KEYS["sga"] == frozenset({"field_model"})
    assert resume_policy.CONFIG_ARTIFACT_KEYS["dlga"] == frozenset({"surrogate_model"})


def test_overlay_copies_sga_and_dlga_config_artifact_keys() -> None:
    sga_canon = resume_policy.config_artifact_overlay(
        {"field_model": dict(_PLACEHOLDER)}, "sga", {"field_model": _identity("AAA")}
    )
    assert sga_canon["field_model"] == _identity("AAA")
    dlga_canon = resume_policy.config_artifact_overlay(
        {"surrogate_model": dict(_PLACEHOLDER)},
        "dlga",
        {"surrogate_model": _identity("BBB")},
    )
    assert dlga_canon["surrogate_model"] == _identity("BBB")


def test_overlay_copies_only_listed_keys_not_incidental_artifacts() -> None:


    canon = resume_policy.config_artifact_overlay(
        {"field_model": dict(_PLACEHOLDER)},
        "sga",
        {"field_model": _identity("AAA"), "stray_provenance": "leak"},
    )
    assert canon["field_model"] == _identity("AAA")
    assert "stray_provenance" not in canon


def test_overlay_ignores_eqgpt_run_provenance_artifacts() -> None:



    canon = {"algorithm": "eqgpt", "variables": ["t", "x"]}
    out = resume_policy.config_artifact_overlay(
        dict(canon),
        "eqgpt",
        {"variables": ["x", "t"], "weights": _identity("W"), "vocab": _identity("V")},
    )
    assert out == canon


def test_overlay_ignores_llm4ed_tape_artifacts() -> None:
    canon = {"algorithm": "llm4ed", "temperature": 0.8}
    out = resume_policy.config_artifact_overlay(
        dict(canon),
        "llm4ed",
        {"llm_tape_path": "/x.jsonl", "llm_tape_sha256": "s", "llm_tape_entries": 3},
    )
    assert out == canon


def test_overlay_non_mapping_artifacts_is_noop() -> None:


    canon = {"algorithm": "sga", "field_model": dict(_PLACEHOLDER)}
    assert resume_policy.config_artifact_overlay(dict(canon), "sga", None) == canon
    assert resume_policy.config_artifact_overlay(dict(canon), "sga", object()) == canon






_G3_KEYS = frozenset(
    {"field", "tier", "stored", "live", "stored_missing", "live_missing"}
)


class TestStructuredChanges:

    def test_init_only_single_change_carries_one_structured_entry(self) -> None:
        stored = {"algorithm": "sga", "aic_ratio": 1.0}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                CONFIG_CANON_SCHEME,
                algorithm="sga",
                plugin_cls=_SGA,
                live_config=_live(algorithm="sga", aic_ratio=2.0),
            )
        changes = excinfo.value.changes
        assert isinstance(changes, tuple)
        assert len(changes) == 1
        entry = changes[0]
        assert set(entry) == _G3_KEYS
        assert entry["field"] == "aic_ratio"
        assert entry["tier"] == "init_only"
        assert entry["stored_missing"] is False
        assert entry["live_missing"] is False
        assert isinstance(entry["stored"], str)
        assert isinstance(entry["live"], str)

    def test_mixed_bucket_identity_entries_precede_init_entries(self) -> None:
        stored = {"algorithm": "sga", "use_autograd": False, "aic_ratio": 1.0}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                CONFIG_CANON_SCHEME,
                algorithm="sga",
                plugin_cls=_SGA,
                live_config=_live(algorithm="sga", use_autograd=True, aic_ratio=2.0),
            )
        changes = excinfo.value.changes
        tiers = [c["tier"] for c in changes]
        assert tiers == ["identity_breaking", "init_only"]
        assert changes[0]["field"] == "use_autograd"
        assert changes[1]["field"] == "aic_ratio"

    def test_live_side_missing_field_marks_missing_and_hides_sentinel(self) -> None:



        stored = {"algorithm": "discover", "removed_field": 3}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                CONFIG_CANON_SCHEME,
                algorithm="discover",
                plugin_cls=_DISCOVER,
                live_config=_live(algorithm="discover"),
            )
        (entry,) = excinfo.value.changes
        assert entry["field"] == "removed_field"
        assert entry["live"] is None
        assert entry["live_missing"] is True
        assert entry["stored_missing"] is False
        assert isinstance(entry["stored"], str)
        for slot in ("stored", "live"):
            assert entry[slot] != "<MISSING>"

    def test_stored_side_missing_field_marks_missing_and_hides_sentinel(self) -> None:



        stored = {"algorithm": "discover"}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                CONFIG_CANON_SCHEME,
                algorithm="discover",
                plugin_cls=_DISCOVER,
                live_config=_live(algorithm="discover", added_field=7),
            )
        (entry,) = excinfo.value.changes
        assert entry["field"] == "added_field"
        assert entry["stored"] is None
        assert entry["stored_missing"] is True
        assert entry["live_missing"] is False
        assert isinstance(entry["live"], str)
        for slot in ("stored", "live"):
            assert entry[slot] != "<MISSING>"

    def test_changes_is_json_safe(self) -> None:
        stored = {"algorithm": "sga", "use_autograd": False, "aic_ratio": 1.0}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                CONFIG_CANON_SCHEME,
                algorithm="sga",
                plugin_cls=_SGA,
                live_config=_live(algorithm="sga", use_autograd=True, aic_ratio=2.0),
            )

        json.dumps(excinfo.value.changes, allow_nan=False)

    def test_structural_scheme_mismatch_carries_no_changes(self) -> None:


        stored = {"algorithm": "sga", "aic_ratio": 1.0}
        with pytest.raises(ValueError) as excinfo:
            resume_policy.check_resume_config(
                stored,
                "kd-config-v2",
                algorithm="sga",
                plugin_cls=_SGA,
                live_config=_live(algorithm="sga", aic_ratio=1.0),
            )
        assert getattr(excinfo.value, "changes", None) is None


def test_check_resume_config_ignores_eqgpt_provenance_artifacts() -> None:




    stored = {"algorithm": "eqgpt", "variables": ["t", "x"]}
    assert (
        resume_policy.check_resume_config(
            stored,
            CONFIG_CANON_SCHEME,
            algorithm="eqgpt",
            plugin_cls=_PLUGIN_CLASS_BY_ALGORITHM["eqgpt"],
            live_config=_live(algorithm="eqgpt", variables=("t", "x")),
            live_artifacts={"variables": ["x", "t"], "weights": _identity("W")},
        )
        is None
    )
