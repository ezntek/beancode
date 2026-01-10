# beancode: a portable IGCSE Computer Science (0478, 0984, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

from dataclasses import dataclass
from typing import Callable, TypedDict
from .bean_ast import BCArrayType, BCType, BCPrimitiveType, BCValue


def array(inner: BCPrimitiveType, low: int, high: int) -> BCArrayType:
    return BCArrayType.new_flat(inner, (low, high))


def matrix(
    inner: BCPrimitiveType,
    low_outer: int,
    high_outer: int,
    low_inner: int,
    high_inner: int,
) -> BCArrayType:
    b = (low_outer, high_outer, low_inner, high_inner)
    return BCArrayType.new_matrix(inner, b)


BCParamSpec = dict[str, BCType]
BCArgsList = dict[str, BCValue]


@dataclass
class BCFunction:  # ffi variant of a function
    name: str
    params: BCParamSpec
    returns: BCPrimitiveType
    fn: Callable[[BCArgsList], BCValue]


@dataclass
class BCProcedure:  # ffi variant of a function
    name: str
    params: BCParamSpec  # spec of arg names and types
    fn: Callable[[BCArgsList], None]


@dataclass
class BCDeclare:
    name: str
    typ: BCType | None = None  # either typ, value or both must be set
    value: BCValue | None = None


@dataclass
class BCConstant:
    name: str
    value: BCValue


class Exports(TypedDict):
    constants: list[BCConstant]
    variables: list[BCDeclare]
    procs: list[BCProcedure]
    funcs: list[BCFunction]
