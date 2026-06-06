
import ast


ALLOWED_NODES: set[type] = {
    ast.Expression,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
}


def _is_negative_constant(node: ast.AST) -> bool:
    if not isinstance(node, ast.UnaryOp):
        return False
    if not isinstance(node.op, ast.USub):
        return False
    return isinstance(node.operand, ast.Constant)


def validate_expr(code: str, allowed_funcs: set[str]) -> bool:

    code = code.strip()


    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False



    negative_const_nodes: set[int] = set()
    for node in ast.walk(tree):
        if _is_negative_constant(node):

            negative_const_nodes.add(id(node))
            negative_const_nodes.add(id(node.op))
            negative_const_nodes.add(id(node.operand))


    for node in ast.walk(tree):

        if id(node) in negative_const_nodes:
            continue


        if type(node) not in ALLOWED_NODES:
            return False


        if isinstance(node, ast.Call):

            if not isinstance(node.func, ast.Name):
                return False

            if node.func.id not in allowed_funcs:
                return False

    return True


def get_function_calls(code: str) -> set[str]:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return set()

    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_names.add(node.func.id)

    return func_names
