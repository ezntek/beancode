# beancode: a portable IGCSE Computer Science (0478, 0984, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

from .bean_ast import BCPrimitiveType, BCValue


def guess_input_type(inp: str) -> BCValue:
    if is_real(inp):
        return BCValue.empty(BCPrimitiveType.REAL)
    elif is_integer(inp):
        return BCValue.empty(BCPrimitiveType.INTEGER)

    if inp.strip().lower() in {"true", "false", "no", "yes"}:
        return BCValue.empty(BCPrimitiveType.BOOLEAN)

    if len(inp.strip()) == 1:
        return BCValue.empty(BCPrimitiveType.CHAR)
    else:
        return BCValue.empty(BCPrimitiveType.STRING)


def is_integer(val: str) -> bool:
    if len(val) == 0:
        return False

    newval = val
    if val[0] == "-":
        newval = val[1:]

    for ch in newval:
        if not ch.isdigit():
            return False
    return True


def is_real(val: str) -> bool:
    if len(val) == 0:
        return False

    if val[0] == "-":
        val = val[1:]

    if is_integer(val):
        return False

    found_decimal = False

    for ch in val:
        if ch == ".":
            if found_decimal:
                return False
            found_decimal = True

    return found_decimal
