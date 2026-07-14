
from __future__ import annotations

from typing import Final

from kd.search.eqgpt.vocab import DIV_ID, E_ID, FIRST_TERM_ID, MUL_ID, PLUS_ID, Vocab


TOKEN_IR_ATOM: Final[dict[str, str]] = {
    "u": "u",
    "ux": "u_x",
    "uxx": "u_xx",
    "ux^2": "n2(u_x)",
    "uxxxx": "u_xxxx",
    "(uux)x": "diff_x(mul(u, u_x))",
    "u^2": "n2(u)",
    "uxxx": "u_xxx",
    "u^3": "n3(u)",
    "x": "x",
    "(uux)xx": "diff2_x(mul(u, u_x))",
    "(u^3)xx": "diff2_x(n3(u))",
    "(1/u)xx": "diff2_x(recip(u))",
    "(u^-2*ux)x": "diff_x(mul(recip(n2(u)), u_x))",
    "(uxx+ux/x)^2": "n2(add(u_xx, mul(recip(x), u_x)))",
    "x^4": "n2(n2(x))",
    "(u^4)xx": "diff2_x(n2(n2(u)))",
    "(u(u^2)xx)xx": "diff2_x(mul(u, diff2_x(mul(u, u_x))))",
    "uxxt": "diff_t(u_xx)",
    "ut^2": "n2(u_t)",
    "uxt": "diff_t(u_x)",
    "utt": "u_tt",
    "uxxtt": "diff2_t(u_xx)",
    "uyy": "u_yy",
    "ut^3": "n3(u_t)",
    "uxxxxx": "u_xxxxx",
    "sin(u)": "sin(u)",
    "BiLaplace(u)": "lap(lap(u))",
    "uy": "u_y",
    "Laplace(u)": "lap(u)",
    "y": "y",
    "x^2": "n2(x)",
    "y^2": "n2(y)",
    "uy^2": "n2(u_y)",
    "uxy": "diff_y(u_x)",
    "uz": "u_z",
    "uzz": "u_zz",
    "Laplace(utt)": "lap(u_tt)",
    "(x+y)": "add(x, y)",
    "exp(x)": "exp(x)",
    "uyyt": "diff_t(u_yy)",
    "sint": "sin(t)",
    "sinx": "sin(x)",
    "exp(-y)": "exp(neg(y))",
    "t": "t",
    "uyyy": "u_yyy",
    "(uux)t": "diff_t(mul(u, u_x))",
}


class UnmappedTokenError(ValueError):
    pass


class MalformedSentenceError(ValueError):
    pass


def ir_inexpressible_tokens(vocab: Vocab) -> frozenset[int]:
    return frozenset(
        i
        for i in range(FIRST_TERM_ID, vocab.size)
        if vocab.id2word[i] not in TOKEN_IR_ATOM
    )


def order_masked_tokens(vocab: Vocab, max_order: int) -> frozenset[int]:
    masked: set[int] = set()
    for idx in range(FIRST_TERM_ID, vocab.size):
        word = vocab.id2word[idx]
        atom = TOKEN_IR_ATOM.get(word)
        if atom is None or not atom.startswith("u_"):
            continue
        suffix = atom[2:]
        if suffix and set(suffix) == {"x"} and len(suffix) > max_order:
            masked.add(idx)
    return frozenset(masked)


def token_to_ir_atom(word: str) -> str:
    try:
        return TOKEN_IR_ATOM[word]
    except KeyError:
        raise UnmappedTokenError(word) from None


def _assemble_term(vocab: Vocab, tokens: list[int]) -> str:
    if not tokens or len(tokens) % 2 == 0:
        raise MalformedSentenceError(f"malformed term slice: {tokens}")
    words = vocab.decode(tokens)
    result = token_to_ir_atom(words[0])
    i = 1
    while i < len(words):
        op_id = tokens[i]
        next_atom = token_to_ir_atom(words[i + 1])
        if op_id == MUL_ID:
            result = f"mul({result}, {next_atom})"
        elif op_id == DIV_ID:
            result = f"div({result}, {next_atom})"
        else:
            raise MalformedSentenceError(
                f"unexpected operator {words[i]!r} inside term slice: {tokens}"
            )
        i += 2
    return result


def sentence_to_rhs_terms(
    vocab: Vocab, sentence: list[int], *, start_len: int
) -> list[str]:
    body = list(sentence[start_len:])
    if body and body[-1] == E_ID:
        body = body[:-1]
    if not body or len(body) % 2 == 0:
        raise MalformedSentenceError(f"operator-terminated or empty RHS body: {body}")

    terms: list[str] = []
    current: list[int] = []
    for i, tok in enumerate(body):
        if i % 2 == 1 and tok == PLUS_ID:
            if not current:
                raise MalformedSentenceError("empty term before '+'")
            terms.append(_assemble_term(vocab, current))
            current = []
            continue
        current.append(tok)
    if not current:
        raise MalformedSentenceError("trailing '+' with no following term")
    terms.append(_assemble_term(vocab, current))
    return terms
