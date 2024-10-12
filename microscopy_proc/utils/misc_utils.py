from enum import EnumType
from typing import Any, Iterable


def enum2list(my_enum: EnumType) -> list[Any]:
    return [e.value for e in my_enum]


def const2iter(x: Any, n: int) -> Iterable[Any]:
    """
    Iterates the object, `x`, `n` times.
    """
    for _ in range(n):
        yield x


def const2ls(x: Any, n: int) -> list[Any]:
    """
    Iterates the list, `ls`, `n` times.
    """
    return [x for _ in range(n)]
