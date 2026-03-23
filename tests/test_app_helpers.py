"""Tests for Gradio app helper functions (Gradio quick-wins).

Functions under test live in ``kd.app_helpers`` (no gradio dependency),
so they can be tested without installing gradio.

Covers:
- Step 1: DSCV_BINARY_DEFAULT includes diff2
- Step 2: _parse_sym_true_operators (prefix expression -> operator set)
- Step 2: _format_dataset_info (Markdown dataset summary)
- Step 3: validate_operators (operator name validation)
- Step 4: get_compatible_models EqGPT annotation

TDD RED phase: new functions are stubs (raise NotImplementedError).
"""

import pytest


# ===================================================================
# Step 1: Default operator constant
# ===================================================================


class TestDscvBinaryDefault:
    """DSCV_BINARY_DEFAULT must include diff2 alongside add, mul, diff."""

    def test_contains_diff2(self):
        """diff2 is required by most PDE datasets (burgers, compound, etc.)."""
        from kd.app_helpers import DSCV_BINARY_DEFAULT, _parse_ops

        ops = _parse_ops(DSCV_BINARY_DEFAULT)
        assert "diff2" in ops

    def test_contains_core_operators(self):
        """add, mul, diff must still be present."""
        from kd.app_helpers import DSCV_BINARY_DEFAULT, _parse_ops

        ops = _parse_ops(DSCV_BINARY_DEFAULT)
        for op in ("add", "mul", "diff"):
            assert op in ops, f"Core operator '{op}' missing from default"

    def test_no_duplicates(self):
        """Default string should not list any operator twice."""
        from kd.app_helpers import DSCV_BINARY_DEFAULT, _parse_ops

        ops = _parse_ops(DSCV_BINARY_DEFAULT)
        assert len(ops) == len(set(ops)), f"Duplicates in {ops}"


# ===================================================================
# Step 2: _parse_sym_true_operators
# ===================================================================


class TestParseSymTrueOperators:
    """Extract operator names from prefix-notation sym_true strings."""

    # -- Happy path: known registry entries ---

    def test_burgers(self):
        """burgers: add,mul,u1,diff,u1,x1,diff2,u1,x1 -> {add, mul, diff, diff2}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,mul,u1,diff,u1,x1,diff2,u1,x1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "mul", "diff", "diff2"}

    def test_pde_compound(self):
        """PDE_compound: add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1
        -> {add, mul, diff, diff2}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "mul", "diff", "diff2"}

    def test_kdv(self):
        """kdv: add,mul,u1,diff,u1,x1,diff3,u1,x1 -> {add, mul, diff, diff3}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,mul,u1,diff,u1,x1,diff3,u1,x1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "mul", "diff", "diff3"}

    def test_chafee_infante(self):
        """chafee-infante: add,add,u1,n3,u1,diff2,u1,x1 -> {add, n3, diff2}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,add,u1,n3,u1,diff2,u1,x1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "n3", "diff2"}

    def test_pde_divide(self):
        """PDE_divide: add,div,diff,u1,x1,x1,diff2,u1,x1
        -> {add, div, diff, diff2}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,div,diff,u1,x1,x1,diff2,u1,x1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "div", "diff", "diff2"}

    def test_fisher(self):
        """fisher: add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1
        -> {add, mul, diff2, n2, diff}"""
        from kd.app_helpers import _parse_sym_true_operators

        sym = "add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1"
        ops = _parse_sym_true_operators(sym)
        assert ops == {"add", "mul", "diff2", "n2", "diff"}

    # -- Edge cases ---

    def test_empty_string(self):
        """Empty sym_true should return an empty set."""
        from kd.app_helpers import _parse_sym_true_operators

        ops = _parse_sym_true_operators("")
        assert ops == set()

    def test_none_input(self):
        """None sym_true should return an empty set (not raise)."""
        from kd.app_helpers import _parse_sym_true_operators

        ops = _parse_sym_true_operators(None)
        assert ops == set()

    def test_single_operator(self):
        """Single token that is an operator."""
        from kd.app_helpers import _parse_sym_true_operators

        ops = _parse_sym_true_operators("n2,u1")
        assert ops == {"n2"}

    def test_only_operands(self):
        """A string with only operands (u1, x1) should return empty set."""
        from kd.app_helpers import _parse_sym_true_operators

        ops = _parse_sym_true_operators("u1,x1")
        assert ops == set()

    def test_return_type_is_set(self):
        """Return type should be set, not list or tuple."""
        from kd.app_helpers import _parse_sym_true_operators

        result = _parse_sym_true_operators("add,u1,u1")
        assert isinstance(result, set)

    def test_whitespace_tolerance(self):
        """Whitespace around tokens should be tolerated."""
        from kd.app_helpers import _parse_sym_true_operators

        ops = _parse_sym_true_operators("add , mul , u1 , diff , u1 , x1")
        assert ops == {"add", "mul", "diff"}


# ===================================================================
# Step 2: _format_dataset_info
# ===================================================================


class TestFormatDatasetInfo:
    """Dataset info formatter produces Markdown with key fields."""

    def test_returns_string(self):
        """Must return a str (Markdown)."""
        from kd.app_helpers import _format_dataset_info

        result = _format_dataset_info("burgers")
        assert isinstance(result, str)

    def test_contains_sym_true_equation(self):
        """Output should show the actual ground-truth sym_true string."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("burgers")
        # The actual sym_true string from registry must appear
        assert "add,mul,u1,diff,u1,x1,diff2,u1,x1" in md

    def test_contains_recommended_operators(self):
        """Output for PDE_compound should mention recommended operators."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("PDE_compound")
        for op in ("add", "mul", "diff", "diff2"):
            assert op in md, f"Recommended operator '{op}' missing from info"

    def test_contains_shape_when_available(self):
        """PDE_compound has shape (100, 251); info should show it."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("PDE_compound")
        assert "100" in md and "251" in md

    def test_contains_domain_when_available(self):
        """PDE_compound has domain x=(1,2), t=(0,0.5); info should show values."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("PDE_compound")
        # Must contain the actual domain boundary values, not just "x"
        assert "1" in md and "2" in md, "Domain x=(1,2) values missing"
        assert "0.5" in md, "Domain t=(0,0.5) upper bound missing"

    def test_dataset_without_shape(self):
        """burgers has no explicit shape key; should not crash."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("burgers")
        assert isinstance(md, str)
        assert len(md) > 0

    def test_dataset_with_none_sym_true(self):
        """advection_diffusion has sym_true=None; should handle gracefully."""
        from kd.app_helpers import _format_dataset_info

        md = _format_dataset_info("advection_diffusion")
        assert isinstance(md, str)

    def test_unknown_dataset_raises(self):
        """Requesting info for a non-existent dataset should raise."""
        from kd.app_helpers import _format_dataset_info

        with pytest.raises((KeyError, ValueError)):
            _format_dataset_info("__nonexistent_dataset__")


# ===================================================================
# Step 3: validate_operators
# ===================================================================


SAMPLE_VALID_SET = {
    "add", "sub", "mul", "div",
    "diff", "diff2", "diff3", "diff4",
    "n2", "n3", "n4", "n5",
    "sin", "cos", "exp", "log", "sqrt",
    "neg", "abs", "lap",
}


class TestValidateOperators:
    """Operator name validation against a known-valid set."""

    # -- Happy path: all valid ---

    def test_all_valid(self):
        """All operators recognized -> (True, [])."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(
            ["add", "mul", "diff"], SAMPLE_VALID_SET
        )
        assert is_valid is True
        assert bad == []

    def test_single_valid(self):
        """Single valid operator."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(["diff2"], SAMPLE_VALID_SET)
        assert is_valid is True
        assert bad == []

    # -- Negative: invalid operators ---

    def test_single_invalid(self):
        """One unknown operator -> (False, ['mull'])."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(["mull"], SAMPLE_VALID_SET)
        assert is_valid is False
        assert bad == ["mull"]

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid -> False, list of invalid only."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(
            ["add", "mull", "diff", "sinn"], SAMPLE_VALID_SET
        )
        assert is_valid is False
        assert set(bad) == {"mull", "sinn"}

    def test_multiple_invalid(self):
        """All operators invalid."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(
            ["foo", "bar", "baz"], SAMPLE_VALID_SET
        )
        assert is_valid is False
        assert len(bad) == 3

    def test_typo_detection(self):
        """Common typo 'dif' (missing f) should be caught."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(
            ["add", "dif"], SAMPLE_VALID_SET
        )
        assert is_valid is False
        assert "dif" in bad

    # -- Edge cases ---

    def test_empty_ops_list(self):
        """Empty operator list should be valid (nothing to check)."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators([], SAMPLE_VALID_SET)
        assert is_valid is True
        assert bad == []

    def test_empty_valid_set(self):
        """If valid_set is empty, any operator is invalid."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(["add"], set())
        assert is_valid is False
        assert bad == ["add"]

    def test_return_types(self):
        """Return is (bool, list[str])."""
        from kd.app_helpers import validate_operators

        result = validate_operators(["add"], SAMPLE_VALID_SET)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_preserves_order_of_invalid(self):
        """Invalid names should appear in input order."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(
            ["zzz", "aaa", "mmm"], SAMPLE_VALID_SET
        )
        assert bad == ["zzz", "aaa", "mmm"]

    def test_case_sensitive(self):
        """Operator names are case-sensitive: 'Add' is not 'add'."""
        from kd.app_helpers import validate_operators

        is_valid, bad = validate_operators(["Add"], SAMPLE_VALID_SET)
        assert is_valid is False
        assert bad == ["Add"]


# ===================================================================
# Step 4: get_compatible_models + EqGPT annotation
# ===================================================================


class TestGetCompatibleModelsEqGPT:
    """EqGPT should be annotated as using its own data, not the selected dataset."""

    def test_eqgpt_always_in_result(self):
        """EqGPT should appear in compatible models for any active dataset."""
        from kd.app_helpers import get_compatible_models

        models = get_compatible_models("burgers")
        eqgpt_entries = [m for m in models if "EqGPT" in str(m)]
        assert len(eqgpt_entries) >= 1

    def test_eqgpt_annotated_with_data_note(self):
        """EqGPT entry should carry an annotation about using its own data.

        The annotation can be:
        - A suffix/marker in the display name (e.g. "KD_EqGPT (built-in data)")
        - A separate field in a structured return
        - A companion function that provides the note

        We check that there is SOME indication distinguishing EqGPT from
        models that use the selected dataset.
        """
        from kd.app_helpers import get_compatible_models

        models = get_compatible_models("PDE_compound")
        eqgpt_entries = [m for m in models if "EqGPT" in str(m)]
        assert len(eqgpt_entries) >= 1

        eqgpt_name = eqgpt_entries[0]
        # The display name should hint that EqGPT uses built-in data,
        # OR the function should return structured data with a note.
        # We check for either approach:
        has_annotation = (
            eqgpt_name != "KD_EqGPT"  # name was modified to include note
            or isinstance(eqgpt_name, tuple)  # structured: (name, note)
        )
        assert has_annotation, (
            f"EqGPT entry '{eqgpt_name}' has no annotation about "
            "using built-in data instead of the selected dataset"
        )

    def test_non_eqgpt_models_not_annotated(self):
        """Only EqGPT should have the data-source annotation, not others."""
        from kd.app_helpers import get_compatible_models

        models = get_compatible_models("burgers")
        non_eqgpt = [m for m in models if "EqGPT" not in str(m)]
        for model in non_eqgpt:
            # Non-EqGPT models should be plain strings without annotations
            assert isinstance(model, str)
            assert "built-in" not in model.lower()

    def test_empty_dataset_returns_empty(self):
        """get_compatible_models with empty/None dataset returns []."""
        from kd.app_helpers import get_compatible_models

        assert get_compatible_models("") == []
        assert get_compatible_models(None) == []

    def test_known_dataset_returns_expected_models(self):
        """burgers supports sga, discover, discover_spr, dlga -> non-EqGPT + EqGPT."""
        from kd.app_helpers import get_compatible_models

        models = get_compatible_models("burgers")
        model_strs = [str(m) for m in models]
        # Exact display name matching (not substring)
        assert "KD_SGA" in model_strs
        assert "KD_DSCV" in model_strs
        assert "KD_DSCV_SPR" in model_strs  # burgers supports discover_spr
        assert "KD_DLGA" in model_strs

    def test_dataset_with_limited_models(self):
        """PDE_compound supports only discover -> fewer models + EqGPT."""
        from kd.app_helpers import get_compatible_models

        models = get_compatible_models("PDE_compound")
        model_strs = [str(m) for m in models]
        # SGA is not supported for PDE_compound (unified substring check)
        assert not any("SGA" in str(m) for m in models)
        # EqGPT should still be present
        assert any("EqGPT" in m for m in model_strs)


# ===================================================================
# Integration: _parse_ops + validate_operators
# ===================================================================


class TestParseOpsValidateIntegration:
    """End-to-end: parse text -> validate -> get clear feedback."""

    def test_valid_default_ops(self):
        """The new default ops should all validate successfully."""
        from kd.app_helpers import DSCV_BINARY_DEFAULT, _parse_ops
        from kd.app_helpers import validate_operators

        ops = _parse_ops(DSCV_BINARY_DEFAULT)
        # Build valid set from known Discover operators
        valid_set = {
            "add", "sub", "mul", "div",
            "diff", "diff2", "diff3", "diff4",
            "n2", "n3", "n4", "n5",
            "sin", "cos", "exp", "log", "sqrt",
            "neg", "abs", "lap",
        }
        is_valid, bad = validate_operators(ops, valid_set)
        assert is_valid is True, f"Default ops have invalid entries: {bad}"

    def test_user_typo_caught(self):
        """Simulates user typing 'add, mull' -> validation catches 'mull'."""
        from kd.app_helpers import _parse_ops, validate_operators

        ops = _parse_ops("add, mull")
        valid_set = {"add", "mul", "diff", "diff2"}
        is_valid, bad = validate_operators(ops, valid_set)
        assert is_valid is False
        assert "mull" in bad


# ===================================================================
# Structural: MODEL_KEYS vs registry alignment
# ===================================================================


class TestModelKeysRegistryAlignment:
    """MODEL_KEYS values must match registry model keys."""

    def test_model_keys_align_with_registry(self):
        """Every MODEL_KEYS value (except eqgpt) must exist in at least
        one PDE_REGISTRY entry's 'models' dict."""
        from kd.app_helpers import MODEL_KEYS
        from kd.dataset._registry import PDE_REGISTRY

        all_registry_model_keys: set[str] = set()
        for info in PDE_REGISTRY.values():
            all_registry_model_keys.update(info.get("models", {}).keys())
        for display, key in MODEL_KEYS.items():
            if key == "eqgpt":
                continue
            assert key in all_registry_model_keys, (
                f"MODEL_KEYS['{display}'] = '{key}' not found in registry. "
                f"Available: {all_registry_model_keys}"
            )


# ===================================================================
# Edge cases: _parse_ops
# ===================================================================


class TestParseOpsEdgeCases:
    """Edge-case behavior for _parse_ops."""

    def test_parse_ops_none(self):
        """None input should return empty list, not crash."""
        from kd.app_helpers import _parse_ops

        assert _parse_ops(None) == []

    def test_parse_ops_empty_string(self):
        """Empty string should return empty list."""
        from kd.app_helpers import _parse_ops

        assert _parse_ops("") == []

    def test_parse_ops_trailing_comma(self):
        """Trailing comma should not produce an empty-string element."""
        from kd.app_helpers import _parse_ops

        ops = _parse_ops("add, mul,")
        assert ops == ["add", "mul"]
        assert "" not in ops

    def test_parse_ops_double_comma(self):
        """Double comma should not produce empty-string elements."""
        from kd.app_helpers import _parse_ops

        ops = _parse_ops("add,,mul")
        assert ops == ["add", "mul"]
        assert "" not in ops
