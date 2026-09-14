
from collections import Counter
from collections.abc import Sequence


def column_indices(
    header: Sequence[object], column_names: Sequence[str], *, kind: str
) -> dict[str, int]:
    counts = Counter(value for value in header if isinstance(value, str))
    ambiguous = [name for name in dict.fromkeys(column_names) if counts[name] > 1]
    if ambiguous:
        raise ValueError(f"ambiguous {kind} columns: {ambiguous}")
    indices = {
        value: index for index, value in enumerate(header) if isinstance(value, str)
    }
    missing = [name for name in column_names if name not in indices]
    if missing:
        available = ", ".join(repr(name) for name in indices)
        raise ValueError(
            f"missing {kind} columns: {missing}; available headers: {available}"
        )
    return {name: indices[name] for name in column_names}
