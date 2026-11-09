
from __future__ import annotations

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.transforms import Affine2D


_SIZE = 20.0
_PAD = 0.04 * _SIZE




_FONTSET = "dejavusans"

_DUMMY_GLYPH = 0xA4


class MathTypesetError(ValueError):
    pass


def _layout(tex: str) -> Path:
    prop = FontProperties(math_fontfamily=_FONTSET)
    to_path = TextToPath()
    text = f"${tex}$"
    try:
        glyphs = to_path.mathtext_parser.parse(text, to_path.DPI, prop).glyphs
    except ValueError as exc:
        raise MathTypesetError("syntax not supported by mathtext") from exc
    except RecursionError as exc:


        raise MathTypesetError("nesting too deep for mathtext") from exc
    if chr(_DUMMY_GLYPH) not in tex and any(
        code == _DUMMY_GLYPH for _font, _size, code, _x, _y in glyphs
    ):
        raise MathTypesetError("a symbol has no glyph in the mathtext fonts")
    vertices, codes = to_path.get_text_path(prop, text, ismath=True)
    return Path(vertices, codes).transformed(
        Affine2D().scale(_SIZE / to_path.FONT_SCALE)
    )


def _number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


_COMMANDS = {Path.MOVETO: "M", Path.LINETO: "L", Path.CURVE3: "Q", Path.CURVE4: "C"}


def _path_data(path: Path) -> str:
    commands: list[str] = []
    for vertices, code in path.iter_segments(simplify=False, curves=True):
        if code == Path.CLOSEPOLY:
            commands.append("Z")
            continue
        points = " ".join(
            f"{_number(x)} {_number(-y)}"
            for x, y in zip(vertices[::2], vertices[1::2], strict=True)
        )
        commands.append(f"{_COMMANDS[code]}{points}")
    return "".join(commands)


def math_svg(tex: str) -> str:
    path = _layout(tex)
    extents = path.get_extents()
    left = extents.x0 - _PAD
    bottom = extents.y0 - _PAD
    width = extents.width + 2 * _PAD
    height = extents.height + 2 * _PAD
    top = -(bottom + height)
    style = (
        f"width:{_number(width / _SIZE)}em;height:{_number(height / _SIZE)}em;"
        f"vertical-align:{_number(bottom / _SIZE)}em"
    )
    return (
        f'<svg class="math-svg" viewBox="{_number(left)} {_number(top)} '
        f'{_number(width)} {_number(height)}" style="{style}" aria-hidden="true" '
        f'focusable="false"><path fill="currentColor" d="{_path_data(path)}">'
        "</path></svg>"
    )


__all__ = ["MathTypesetError", "math_svg"]
