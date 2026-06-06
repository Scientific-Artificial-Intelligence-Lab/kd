
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np






@dataclass
class MockToken:

    name: str
    arity: int


class MockNode:

    def __init__(self, token: MockToken) -> None:
        self.val: str = token.name
        self.token: MockToken = token
        self.children: list[MockNode] = []
        self.symbol: int = 1






VOCAB: list[MockToken] = [
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

ARITIES = np.array([t.arity for t in VOCAB], dtype=np.int32)
FIRST_TERMINAL_IDX = 0


def _to_token_objs(indices: list[int]) -> list[MockToken]:
    return [MockToken(VOCAB[i].name, VOCAB[i].arity) for i in indices]


def _token_index(token: MockToken) -> int:
    for i, v in enumerate(VOCAB):
        if v.name == token.name and v.arity == token.arity:
            return i
    raise ValueError(f"Token {token} not in vocabulary")







def build_tree(token_list: list[MockToken]) -> MockNode:
    op = token_list.pop(0)
    node = MockNode(op)
    for _ in range(op.arity):
        node.children.append(build_tree(token_list))
    return node







def preorder_traverse(node: MockNode) -> list[MockToken]:
    result: list[MockToken] = []
    nodes = [node]
    while nodes:
        cur = nodes.pop(0)
        result.append(cur.token)
        if len(cur.children) == 2:
            nodes.insert(0, cur.children[0])
            nodes.insert(1, cur.children[1])
        elif len(cur.children) == 1:
            nodes.insert(0, cur.children[0])
    return result







def max_depth(node: MockNode) -> int:
    d = 0
    for child in node.children:
        d = max(d, max_depth(child))
    return d + 1







def split_sum(root: MockNode) -> list[Any]:
    if root.val not in ("add", "sub", "add_t", "sub_t"):
        return [root]

    if "sub" in root.val:
        if root.symbol == 1:
            root.children[1].symbol *= -1
        else:
            root.children[0].symbol *= -1
            root.children[1].symbol *= -1

    return [split_sum(root.children[0]), split_sum(root.children[1])]


def expand_list(nested: list[Any]) -> list[MockNode]:
    flat: list[MockNode] = []
    queue = list(nested)
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, list):
            queue = cur + queue
        else:
            flat.append(cur)
    return flat


def count_terms(indices: list[int]) -> int:
    tree = build_tree(_to_token_objs(indices))
    terms = expand_list(split_sum(tree))
    return len(terms)








def finish_tokens(
    indices: list[int],
    arities: np.ndarray,
    first_terminal: int,
) -> np.ndarray:
    tokens = np.array(indices, dtype=np.int32)
    a = arities[tokens]
    dangling = 1 + np.cumsum(a - 1)

    if 0 in dangling:

        length = 1 + int(np.argmax(dangling == 0))
        return tokens[:length]
    else:

        n_needed = int(dangling[-1])
        ext = np.full(n_needed, first_terminal, dtype=np.int32)
        return np.concatenate([tokens, ext])







def main() -> None:
    data: dict[str, np.ndarray] = {}


    tree_cases: dict[str, list[int]] = {
        "single_terminal": [1],
        "unary_n2": [8, 1],
        "binary_add": [2, 0, 1],
        "burgers_gt": [2, 3, 1, 5, 1, 6, 1],
        "deeply_nested": [3, 8, 5, 1, 7, 9, 1],
        "nested_add": [2, 2, 0, 1, 1],
    }

    case_names: list[str] = []
    for name, indices in tree_cases.items():
        case_names.append(name)
        tokens_arr = np.array(indices, dtype=np.int32)


        tree = build_tree(_to_token_objs(indices))
        depth = max_depth(tree)
        traversal = preorder_traverse(tree)
        roundtrip = np.array(
            [_token_index(t) for t in traversal], dtype=np.int32
        )


        n_terms = count_terms(indices)

        data[f"{name}__tokens"] = tokens_arr


        data[f"{name}__depth"] = np.int32(depth)
        data[f"{name}__n_terms"] = np.int32(n_terms)
        data[f"{name}__n_nodes"] = np.int32(
            len(indices)
        )
        data[f"{name}__roundtrip"] = roundtrip

    data["tree_case_names"] = np.array(case_names)


    ft_cases: dict[str, list[int]] = {
        "complete": [2, 0, 1],
        "add_only": [2],
        "add_one_child": [2, 1],
        "overlong": [1, 0, 2],
        "unary_incomplete": [5],
        "nested_incomplete": [2, 3],
    }

    ft_names: list[str] = []
    for name, inp in ft_cases.items():
        ft_names.append(name)
        inp_arr = np.array(inp, dtype=np.int32)
        out_arr = finish_tokens(inp, ARITIES, FIRST_TERMINAL_IDX)
        data[f"ft_{name}__input"] = inp_arr
        data[f"ft_{name}__output"] = out_arr

    data["ft_case_names"] = np.array(ft_names)


    data["vocab_names"] = np.array([t.name for t in VOCAB])
    data["vocab_arities"] = ARITIES


    out_path = Path(__file__).parent / "tree_model.npz"
    np.savez(out_path, **data)


    print("=== Tree structure cases ===")
    for name in case_names:
        tokens = data[f"{name}__tokens"]
        token_names = [VOCAB[i].name for i in tokens]
        print(f" {name}: {token_names}")
        print(
            f" depth={data[f'{name}__depth']}, "
            f"n_terms={data[f'{name}__n_terms']}, "
            f"n_nodes={data[f'{name}__n_nodes']}"
        )
        rt = data[f"{name}__roundtrip"].tolist()
        assert rt == tokens.tolist(), f"Round-trip mismatch for {name}"
        print(" roundtrip OK")

    print("\n=== Finish tokens cases ===")
    for name in ft_names:
        saved_inp = data[f"ft_{name}__input"]
        saved_out = data[f"ft_{name}__output"]
        inp_names = [VOCAB[i].name for i in saved_inp]
        out_names = [VOCAB[i].name for i in saved_out]
        print(f" {name}: {inp_names} -> {out_names}")

    print(f"\nSaved {len(data)} arrays to {out_path}")


if __name__ == "__main__":
    main()
