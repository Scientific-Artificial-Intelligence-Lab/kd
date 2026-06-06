
from __future__ import annotations

import ast

from kd.search.discover.core.tree import ExpressionTree, TreeNode
from kd.search.discover.tokens.library import Library, Token

_PARSE_MODE = "eval"


def tree_to_ir(tree: ExpressionTree) -> str:
    return _node_to_ir(tree.root)


def ir_to_tree(ir: str, library: Library) -> ExpressionTree:
    parsed = _parse_ir(ir)
    root = _ast_to_node(parsed.body, library, None)
    return ExpressionTree(root=root, _library=library)


def tokens_to_ir(tokens: list[int], library: Library) -> str:
    return tree_to_ir(ExpressionTree.from_preorder(tokens, library))


def _node_to_ir(root: TreeNode) -> str:
    result_stack: list[str] = []

    work_stack: list[tuple[TreeNode, bool]] = [(root, False)]

    while work_stack:
        node, visited = work_stack.pop()

        if not node.children:





            if node.token.arity != 0:
                raise ValueError(
                    f"_node_to_ir reached a childless operator "
                    f"{node.token.name!r} (arity {node.token.arity}); "
                    "tree is malformed.",
                )
            result_stack.append(node.token.name)
            continue

        if visited:




            n = len(node.children)
            if node.token.arity != n:
                raise ValueError(
                    f"_node_to_ir internal arity mismatch for "
                    f"{node.token.name!r}: token.arity={node.token.arity}, "
                    f"len(children)={n}; tree is malformed.",
                )
            child_fragments = result_stack[len(result_stack) - n:]
            del result_stack[len(result_stack) - n:]
            result_stack.append(f"{node.token.name}({','.join(child_fragments)})")
        else:

            work_stack.append((node, True))
            for child in reversed(node.children):
                work_stack.append((child, False))




    if len(result_stack) != 1:
        raise ValueError(
            f"_node_to_ir expected exactly one root fragment, got "
            f"{len(result_stack)} (tree is malformed).",
        )
    return result_stack[0]


def _parse_ir(ir: str) -> ast.Expression:
    stripped = ir.strip()
    if not stripped:
        raise ValueError("IR string cannot be empty.")

    try:
        parsed = ast.parse(stripped, mode=_PARSE_MODE)
    except SyntaxError as exc:
        raise ValueError(f"Invalid IR syntax: {exc}") from exc
    except RecursionError as exc:





        raise ValueError(
            f"IR too deeply nested for ast.parse: {exc}",
        ) from exc

    if not isinstance(parsed, ast.Expression):
        raise ValueError("IR must parse as a Python expression.")
    return parsed


def _ast_to_node(
    root_ast: ast.expr,
    library: Library,
    root_parent: TreeNode | None,
) -> TreeNode:
    root = _make_tree_node(root_ast, library, root_parent)


    stack: list[tuple[ast.Call, TreeNode]] = []
    if isinstance(root_ast, ast.Call):
        stack.append((root_ast, root))

    while stack:
        call_ast, tree_node = stack.pop()
        children: list[TreeNode] = []
        for arg in call_ast.args:
            child = _make_tree_node(arg, library, tree_node)
            children.append(child)
            if isinstance(arg, ast.Call):
                stack.append((arg, child))
        tree_node.children = children

    return root


def _make_tree_node(
    node: ast.expr,
    library: Library,
    parent: TreeNode | None,
) -> TreeNode:
    if isinstance(node, ast.Call):
        return _call_to_node_shallow(node, library, parent)
    if isinstance(node, ast.Name):
        return _name_to_node(node.id, library, parent)
    if isinstance(node, ast.Constant):
        raise ValueError("Numeric constants are not supported in discover IR.")
    raise ValueError(_unsupported_syntax_message(node))


def _call_to_node_shallow(
    node: ast.Call,
    library: Library,
    parent: TreeNode | None,
) -> TreeNode:
    if node.keywords:
        raise ValueError("Keyword arguments are not allowed in IR.")

    token = _lookup_token(_call_name(node), library)
    if token.arity == 0:
        raise ValueError(f"Terminal token {token.name!r} cannot be called.")
    if len(node.args) != token.arity:
        raise ValueError(
            f"Token {token.name!r} expects {token.arity} args, "
            f"got {len(node.args)}."
        )

    return TreeNode(token=token, parent=parent)


def _name_to_node(
    name: str,
    library: Library,
    parent: TreeNode | None,
) -> TreeNode:
    token = _lookup_token(name, library)
    if token.arity != 0:
        raise ValueError(f"Operator token {name!r} must use call syntax.")
    return TreeNode(token=token, parent=parent)


def _call_name(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ValueError("IR calls must target bare token names.")
    return node.func.id


def _lookup_token(name: str, library: Library) -> Token:
    try:
        return library[name]
    except KeyError as exc:
        raise ValueError(f"Unknown token: {name!r}") from exc


def _unsupported_syntax_message(node: ast.expr) -> str:
    if isinstance(node, ast.BinOp):
        return "Infix operators are not allowed in IR."
    if isinstance(node, ast.UnaryOp):
        return "Unary operators are not allowed in IR."
    if isinstance(node, ast.BoolOp):
        return "Boolean operators are not allowed in IR."
    return f"Unsupported IR syntax: {type(node).__name__}."
