import math
import sys
import time
import random

from typing import NoReturn

from beancode.bean_ast import BCPrimitiveType, BCValue

Libroutine = list[tuple[BCPrimitiveType, ...] | BCPrimitiveType]
Libroutines = dict[str, Libroutine]

_NUMERIC = (BCPrimitiveType.INTEGER, BCPrimitiveType.REAL)

LIBROUTINES: Libroutines = {
    "ucase": [(BCPrimitiveType.STRING, BCPrimitiveType.CHAR)],
    "lcase": [(BCPrimitiveType.STRING, BCPrimitiveType.CHAR)],
    "div": [
        _NUMERIC,
        _NUMERIC
    ],
    "mod": [
        _NUMERIC,
        _NUMERIC
    ],
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
    "help": [BCPrimitiveType.STRING],
    "getchar": [],
    "random": [],
    "execute": [BCPrimitiveType.STRING],
    "format": [],  # stub
}

LIBROUTINES_NORETURN: Libroutines = {
    "putchar": [BCPrimitiveType.CHAR],
    "exit": [BCPrimitiveType.INTEGER],
    "sleep": [BCPrimitiveType.REAL],
    "flush": [],
}


def bean_ucase(txt: BCValue) -> BCValue:
    if txt.kind == BCPrimitiveType.STRING:
        return BCValue.new_string(txt.get_string().upper())
    else:
        return BCValue.new_char(txt.get_char().upper()[0])


def bean_lcase(txt: BCValue) -> BCValue:
    if txt.kind == BCPrimitiveType.STRING:
        return BCValue.new_string(txt.get_string().lower())
    else:
        return BCValue.new_char(txt.get_char().lower()[0])


def bean_substring(txt: str, begin: int, length: int) -> BCValue:
    begin = begin - 1
    s = txt[begin : begin + length]

    if len(s) == 1:
        return BCValue.new_char(s[0])
    else:
        return BCValue.new_string(s)


def bean_length(txt: str) -> BCValue:
    return BCValue.new_integer(len(txt))


def bean_round(val: float, places: int) -> BCValue:
    res = round(val, places)
    if places == 0:
        return BCValue.new_integer(int(res))
    else:
        return BCValue.new_real(res)


def bean_getchar() -> BCValue:
    s = sys.stdin.read(1)[0]  # get ONE character
    return BCValue.new_char(s)


def bean_putchar(ch: str):
    print(ch[0], end="")


def bean_exit(code: int) -> NoReturn:
    sys.exit(code)


def bean_div(lhs: int | float, rhs: int | float) -> BCValue:
    return BCValue.new_integer(int(lhs // rhs))


def bean_mod(lhs: int | float, rhs: int | float) -> BCValue:
    if type(rhs) == float:
        return BCValue.new_real(float(lhs % rhs))
    else:
        return BCValue.new_integer(int(lhs % rhs))


def bean_sqrt(val: BCValue) -> BCValue:  # type: ignore
    if val.kind == BCPrimitiveType.INTEGER:
        num = val.get_integer()
        return BCValue.new_real(math.sqrt(num))
    elif val.kind == BCPrimitiveType.REAL:
        num = val.get_real()
        return BCValue.new_real(math.sqrt(num))


def bean_random() -> BCValue:
    return BCValue.new_real(random.random())


def bean_sleep(duration: float):
    time.sleep(duration)
