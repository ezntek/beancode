# beancode: a portable IGCSE Computer Science (0478, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# NOTE: just some helper functions for type checking
#

from beancode.bean_ast import *

# NOTE: this function propagates BCErrors on failure, and will not
# return a success/failure.
def check_binaryexpr(expr: BinaryExpr, lhs: BCValue, rhs: BCValue):
    if expr.op in {Operator.EQUAL, Operator.NOT_EQUAL}:
        human_kind = "a comparison"
    elif expr.op in {
        Operator.LESS_THAN,
        Operator.LESS_THAN_OR_EQUAL,
        Operator.GREATER_THAN,
        Operator.GREATER_THAN_OR_EQUAL,
    }:
        human_kind = "an ordered comparison"

        if lhs.kind != rhs.kind and not (
            lhs.kind_is_numeric() and rhs.kind_is_numeric()
        ):
            raise BCError(
                f"cannot {expr.op.humanize()} incompatible types {lhs.kind} and {rhs.kind}",
                expr.pos,
            )
    elif expr.op in {
        Operator.AND,
        Operator.OR,
        Operator.NOT,
    }:
        human_kind = "a boolean operation"

        if lhs.kind != rhs.kind:
            raise BCError(
                f"cannot {expr.op.humanize()} incompatible types {lhs.kind} and {rhs.kind}!",
                expr.pos,
            )

        if not (
            lhs.kind == BCPrimitiveType.BOOLEAN
            or rhs.kind == BCPrimitiveType.BOOLEAN
        ):
            raise BCError(
                f"cannot {expr.op.humanize()} between {lhs.kind} and {rhs.kind}!",
                expr.pos,
            )
    else:
        human_kind = "an arithmetic expression"

        # XXX: microoptimizations™
        # we are reducing the number of calls we visit in the Python VM per addition. Addition is a
        # very very common operator and it speeds PrimeTorture up by around 230ms.
        if expr.op != Operator.ADD:
            if expr.op not in {Operator.FLOOR_DIV, Operator.MOD} and not (
                lhs.kind_is_numeric() and rhs.kind_is_numeric()
            ):
                raise BCError(
                    f"cannot {expr.op.humanize()} between BOOLEANs, CHARs and STRINGs!",
                    expr.pos,
                )

    if expr.op != Operator.EQUAL:
        if lhs.is_uninitialized():
            raise BCError(
                f"cannot have NULL in the left hand side of {human_kind}\n"
                + "is your value an uninitialized value/variable?",
                expr.lhs.pos,
            )
        if rhs.is_uninitialized():
            raise BCError(
                f"cannot have NULL in the right hand side of {human_kind}\n"
                + "is your value an uninitialized value/variable?",
                expr.rhs.pos,
            )
