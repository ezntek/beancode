import math
import sys
import time
import random

from typing import NoReturn

from . import Pos, humanize_index
from .bean_ast import BCArrayType, BCPrimitiveType, BCValue
from .error import BCError

# No list supplied means variadic arguments, no type checking at all
# None specified for a type means any type, no type checking
# Tuple means a union type
Libroutine = list[tuple[BCPrimitiveType | None, ...] | BCPrimitiveType | None] | None
Libroutines = dict[str, Libroutine]

_NUMERIC = (BCPrimitiveType.INTEGER, BCPrimitiveType.REAL)

LIBROUTINES: Libroutines = {
    "ucase": [(BCPrimitiveType.STRING, BCPrimitiveType.CHAR)],
    "lcase": [(BCPrimitiveType.STRING, BCPrimitiveType.CHAR)],
    "div": [_NUMERIC, _NUMERIC],
    "mod": [_NUMERIC, _NUMERIC],
    "substring": [
        BCPrimitiveType.STRING,
        BCPrimitiveType.INTEGER,
        BCPrimitiveType.INTEGER,
    ],
    "round": [BCPrimitiveType.REAL, BCPrimitiveType.INTEGER],
    "sqrt": [_NUMERIC],
    "length": [BCPrimitiveType.STRING],
    "sin": [BCPrimitiveType.REAL],
    "cos": [BCPrimitiveType.REAL],
    "tan": [BCPrimitiveType.REAL],
    "getchar": [],
    "random": [],
    "execute": [BCPrimitiveType.STRING],
    "putchar": [BCPrimitiveType.CHAR],
    "exit": [BCPrimitiveType.INTEGER],
    "sleep": [BCPrimitiveType.REAL],
    "flush": [],
    # complicated stuff goes here
    "typeof": [None],
    "type": [None],
    "clear": [],
    "format": None,
    "initarray": None,
}


def bean_ucase(pos: Pos, txt: BCValue) -> BCValue:
    _ = pos
    if txt.kind == BCPrimitiveType.STRING:
        return BCValue.new_string(txt.get_string().upper())
    else:
        return BCValue.new_char(txt.get_char().upper()[0])


def bean_lcase(pos: Pos, txt: BCValue) -> BCValue:
    _ = pos
    if txt.kind == BCPrimitiveType.STRING:
        return BCValue.new_string(txt.get_string().lower())
    else:
        return BCValue.new_char(txt.get_char().lower()[0])


def bean_substring(pos: Pos, txt: str, begin: int, length: int) -> BCValue:
    if txt == 0:
        raise BCError("cannot SUBSTRING an empty string!", pos)

    txt_len = len(txt)
    if (
        (begin > txt_len)
        or (length > txt_len)
        or (begin < 1)
        or (length < 1)
        or (begin + length - 1 > txt_len)
    ):
        raise BCError(
            f"invalid SUBSTRING from {begin} with length {length} on text with length {txt_len}!",
            pos,
        )

    begin = begin - 1
    s = txt[begin : begin + length]

    if len(s) == 1:
        return BCValue.new_char(s[0])
    else:
        return BCValue.new_string(s)


def bean_length(pos: Pos, txt: str) -> BCValue:
    _ = pos
    return BCValue.new_integer(len(txt))


def bean_round(pos: Pos, val: float, places: int) -> BCValue:
    if places < 0:
        raise BCError("cannot round to negative places!", pos)

    res = round(val, places)
    if places == 0:
        return BCValue.new_integer(int(res))
    else:
        return BCValue.new_real(res)


def bean_getchar(pos: Pos) -> BCValue:
    _ = pos
    s = sys.stdin.read(1)[0]  # get ONE character
    return BCValue.new_char(s)


def bean_putchar(pos: Pos, ch: str):
    _ = pos
    print(ch[0], end="")


def bean_exit(pos: Pos, code: int) -> NoReturn:
    _ = pos
    sys.exit(code)


def bean_div(pos: Pos, lhs: int | float, rhs: int | float) -> BCValue:
    if rhs == 0:
        raise BCError("cannot divide by zero!", pos)

    return BCValue.new_integer(int(lhs // rhs))


def bean_mod(pos: Pos, lhs: int | float, rhs: int | float) -> BCValue:
    if rhs == 0:
        raise BCError("cannot divide by zero!", pos)

    if type(rhs) == float:
        return BCValue.new_real(float(lhs % rhs))
    else:
        return BCValue.new_integer(int(lhs % rhs))


def bean_sqrt(pos: Pos, val: BCValue) -> BCValue:  # type: ignore
    if val.kind == BCPrimitiveType.INTEGER:
        num = val.get_integer()

        if num < 0:
            raise BCError("cannot calculate the square root of a negative!", pos)

        return BCValue.new_real(math.sqrt(num))
    elif val.kind == BCPrimitiveType.REAL:
        num = val.get_real()

        if num < 0:
            raise BCError("cannot calculate the square root of a negative!", pos)

        return BCValue.new_real(math.sqrt(num))


def bean_random(pos: Pos) -> BCValue:
    _ = pos
    return BCValue.new_real(random.random())


def bean_sleep(pos: Pos, duration: float):
    _ = pos
    time.sleep(duration)


def bean_typeof(pos: Pos, val: BCValue) -> BCValue:
    _ = pos
    return BCValue.new_string(str(val.kind).upper())


def bean_format(pos: Pos, args: list[BCValue]) -> BCValue:
    if len(args) == 0:
        raise BCError("Not enough argument supplied to FORMAT", pos)

    fmt = args[0]
    if fmt.kind != BCPrimitiveType.STRING:
        raise BCError("first argument to FORMAT must be a STRING!", pos)

    items = list()
    for idx, itm in enumerate(args[1:]):
        if itm.is_uninitialized():
            raise BCError(
                f"{humanize_index(idx + 2)} argument in format argument list is NULL/uninitialized!",
                pos,
            )
        if isinstance(itm.kind, BCArrayType):
            items.append(str(itm.val))
        elif itm.kind == BCPrimitiveType.NULL:
            items.append("(null)")
        else:
            items.append(itm.val)

    try:
        res = fmt.get_string() % tuple(items)
        return BCValue.new_string(res)
    except TypeError as e:
        pymsg = e.args[0]
        raise BCError(f"format error: {pymsg}", pos)


def bean_initarray(pos: Pos, args: list[BCValue]):
    if len(args) != 2:
        raise BCError("expected 2 arguments to INITARRAY", pos)

    if not isinstance(args[0].kind, BCArrayType):
        raise BCError(
            "first argument supplied to INITARRAY must be an array or a 2D array!", pos
        )

    arr = args[0].get_array()
    val = args[1]

    if arr.typ.inner != val.kind:
        raise BCError(
            f"expected value of type {str(arr.typ.inner).upper()} to INITARRAY, but found {str(args[1].kind).upper()}",
            pos,
        )

    if arr.is_flat():
        for item in arr.get_flat():
            item.val = val.val
    elif arr.is_matrix():
        for outer in arr.get_matrix():
            for item in outer:
                item.val = val.val
