
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast






@dataclass
class MockToken:

    name: str
    arity: int


class MockNode:

    def __init__(self, token: MockToken) -> None:
        self.val: str = token.name
        self.token: MockToken = token
        self.children: list[MockNode] = []

    def __repr__(self) -> str:
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val
        return f"{self.val}({children_repr})"







def build_tree(token_list: list[MockToken]) -> MockNode:
    op = token_list.pop(0)
    node = MockNode(op)
    for _ in range(op.arity):
        node.children.append(build_tree(token_list))
    return node


def tree_to_ir_reference(indices: list[int], vocab: list[MockToken]) -> str:
    tokens = [MockToken(vocab[i].name, vocab[i].arity) for i in indices]
    tree = build_tree(tokens)
    return repr(tree)







VOCAB_LEGACY: list[MockToken] = [
    MockToken("x1", 0),
    MockToken("u1", 0),
    MockToken("add", 2),
    MockToken("mul", 2),
    MockToken("div", 2),
    MockToken("diff", 1),
    MockToken("diff2", 1),
    MockToken("diff3", 1),
    MockToken("n2", 1),
    MockToken("n3", 1),
]


VOCAB_KD: list[MockToken] = [
    MockToken("x", 0),
    MockToken("u", 0),
    MockToken("add", 2),
    MockToken("mul", 2),
    MockToken("div", 2),
    MockToken("diff_x", 1),
    MockToken("diff2_x", 1),
    MockToken("diff3_x", 1),
    MockToken("n2", 1),
    MockToken("n3", 1),
    MockToken("sub", 2),
    MockToken("sin", 1),
    MockToken("cos", 1),
    MockToken("neg", 1),
]







TREE_CASES: dict[str, list[int]] = {
    "single_terminal": [1],
    "coord_terminal": [0],
    "unary_n2": [8, 1],
    "unary_diff": [5, 1],
    "binary_add": [2, 0, 1],
    "binary_mul": [3, 0, 1],
    "binary_div": [4, 1, 0],
    "burgers_gt": [2, 3, 1, 5, 1, 6, 1],
    "deeply_nested": [3, 8, 5, 1, 7, 9, 1],
    "nested_add": [2, 2, 0, 1, 1],
    "chain_unary": [8, 9, 5, 1],
    "div_nested": [4, 3, 1, 0, 6, 1],
    "right_nested_add": [2, 1, 2, 0, 1],
}


KD_EXTRA_CASES: dict[str, list[int]] = {
    "sub_simple": [10, 1, 0],
    "sin_u": [11, 1],
    "cos_x": [12, 0],
    "neg_mul": [13, 3, 1, 0],
    "burgers_with_sub": [10, 13, 3, 1, 5, 1, 6, 1],

    "right_nested_add": [2, 1, 2, 0, 1],
}







ROUNDTRIP_IR_STRINGS: list[str] = [
    "u",
    "x",
    "add(u,x)",
    "mul(u,x)",
    "div(u,x)",
    "sin(u)",
    "n2(u)",
    "diff_x(u)",
    "diff2_x(u)",
    "add(mul(u,diff_x(u)),diff2_x(u))",
    "mul(n2(diff_x(u)),diff3_x(n3(u)))",
    "sub(neg(mul(u,diff_x(u))),diff2_x(u))",
    "add(add(u,x),mul(u,x))",
    "div(mul(u,x),diff2_x(u))",
]







def main() -> None:
    data: dict[str, object] = {}


    legacy_cases: dict[str, dict[str, object]] = {}
    for name, indices in TREE_CASES.items():
        ir_str = tree_to_ir_reference(indices, VOCAB_LEGACY)
        legacy_cases[name] = {
            "tokens": indices,
            "expected_ir": ir_str,
        }
    data["legacy_vocab"] = {
        "names": [t.name for t in VOCAB_LEGACY],
        "arities": [t.arity for t in VOCAB_LEGACY],
        "cases": legacy_cases,
    }


    kd_cases: dict[str, dict[str, object]] = {}
    all_kd_cases = {**TREE_CASES, **KD_EXTRA_CASES}
    for name, indices in all_kd_cases.items():
        ir_str = tree_to_ir_reference(indices, VOCAB_KD)
        kd_cases[name] = {
            "tokens": indices,
            "expected_ir": ir_str,
        }
    data["kd_vocab"] = {
        "names": [t.name for t in VOCAB_KD],
        "arities": [t.arity for t in VOCAB_KD],
        "cases": kd_cases,
    }


    data["roundtrip_ir_strings"] = ROUNDTRIP_IR_STRINGS


    out_path = Path(__file__).parent / "ir_conversion.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


    print("=== Legacy vocab cases ===")
    for name, case in legacy_cases.items():
        token_names = [
            VOCAB_LEGACY[i].name for i in cast("list[int]", case["tokens"])
        ]
        print(f" {name}: {token_names} -> {case['expected_ir']!r}")

    print("\n=== kd-compatible vocab cases ===")
    for name, case in kd_cases.items():
        token_names = [
            VOCAB_KD[i].name for i in cast("list[int]", case["tokens"])
        ]
        print(f" {name}: {token_names} -> {case['expected_ir']!r}")

    print(f"\n=== Round-trip IR strings ({len(ROUNDTRIP_IR_STRINGS)}) ===")
    for s in ROUNDTRIP_IR_STRINGS:
        print(f" {s!r}")

    print(f"\nSaved fixture to {out_path}")


if __name__ == "__main__":
    main()
