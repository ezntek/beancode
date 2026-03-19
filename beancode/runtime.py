# beancode: a portable IGCSE Computer Science (0478, 0984, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

from typing import Any
import beancode
from .bean_ast import BCPrimitiveType, BCValue
from .typechecker import check_binaryexpr


def bean_input():
    inp = input()
    try:
        return float(inp)
    except ValueError:
        pass
    try:
        return int(inp)
    except ValueError:
        pass
    t = inp.strip().lower()
    if t in ("true", "false", "no", "yes"):
        return True if t in ("true", "yes") else False
    return inp


def op_add(lhs: BCValue, rhs: BCValue) -> BCValue:
    if lhs.kind_is_alpha() or rhs.kind_is_alpha():
        return BCValue(BCPrimitiveType.STRING, str(lhs) + str(rhs))
    res = lhs.val + rhs.val  # type: ignore
    return (
        BCValue(BCPrimitiveType.INTEGER, res)
        if type(res) is int
        else BCValue(BCPrimitiveType.REAL, res)
    )


def op_sub(lhs: BCValue, rhs: BCValue) -> BCValue:
    res = lhs.val - rhs.val  # type: ignore
    return (
        BCValue(BCPrimitiveType.INTEGER, res)
        if type(res) is int
        else BCValue(BCPrimitiveType.REAL, res)
    )


def op_mul(lhs: BCValue, rhs: BCValue) -> BCValue:
    res = lhs.val * rhs.val  # type: ignore
    return (
        BCValue(BCPrimitiveType.INTEGER, res)
        if type(res) is int
        else BCValue(BCPrimitiveType.REAL, res)
    )


def op_div(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.REAL, lhs.val / rhs.val)  # type: ignore


def op_equal(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.BOOLEAN, lhs == rhs)


def op_not_equal(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.BOOLEAN, lhs != rhs)


def op_less_than(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(
        BCPrimitiveType.BOOLEAN,
        (
            ord(lhs.val) < ord(rhs.val)  # type: ignore
            if lhs.kind == BCPrimitiveType.CHAR
            else lhs.val < rhs.val  # type: ignore
        ),
    )


def op_greater_than(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(
        BCPrimitiveType.BOOLEAN,
        (
            ord(lhs.val) > ord(rhs.val)  # type: ignore
            if lhs.kind == BCPrimitiveType.CHAR
            else lhs.val > rhs.val  # type: ignore
        ),
    )


def op_less_than_or_equal(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(
        BCPrimitiveType.BOOLEAN,
        (
            ord(lhs.val) <= ord(rhs.val)  # type: ignore
            if lhs.kind == BCPrimitiveType.CHAR
            else lhs.val <= rhs.val  # type: ignore
        ),
    )


def op_greater_than_or_equal(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(
        BCPrimitiveType.BOOLEAN,
        (
            ord(lhs.val) >= ord(rhs.val)  # type: ignore
            if lhs.kind == BCPrimitiveType.CHAR
            else lhs.val >= rhs.val  # type: ignore
        ),
    )


def op_and(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.BOOLEAN, lhs.val and rhs.val)  # type: ignore


def op_or(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.BOOLEAN, lhs.val and rhs.val)  # type: ignore


def op_pow(lhs: BCValue, rhs: BCValue) -> BCValue:
    res = (
        1 << rhs.val
        if (int(lhs.val) == 2 and type(rhs.val) is int)  # type: ignore
        else lhs.val**rhs.val  # type: ignore
    )

    return (
        BCValue(BCPrimitiveType.INTEGER, res)
        if type(res) is int
        else BCValue(BCPrimitiveType.REAL, res)
    )


def op_floor_div(lhs: BCValue, rhs: BCValue) -> BCValue:
    return BCValue(BCPrimitiveType.INTEGER, lhs.val // rhs.val)  # type: ignore


def get_globals() -> dict[str, Any]:
    return {
        "beancode": beancode,
        "V": BCValue,
        "T": BCPrimitiveType,
        "op_add": op_add,
        "op_sub": op_sub,
        "op_mul": op_mul,
        "op_div": op_div,
        "op_equal": op_equal,
        "op_not_equal": op_not_equal,
        "op_less_than": op_less_than,
        "op_greater_than": op_greater_than,
        "op_less_than_or_equal": op_less_than_or_equal,
        "op_greater_than_or_equal": op_greater_than_or_equal,
        "op_and": op_and,
        "op_or": op_or,
        "op_pow": op_pow,
        "op_floor_div": op_floor_div,
        "__builtins__": {"print": print, "input": bean_input},
    }
