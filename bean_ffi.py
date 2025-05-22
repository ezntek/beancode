from _typeshed import DataclassInstance
from dataclasses import dataclass
from typing import Callable, TypedDict
from bean_ast import BCArrayType, BCPrimitiveType, BCType, BCValue, Literal

def _int_literal(i: int) -> Literal:
    return Literal("integer", integer=i)

def array(inner: BCPrimitiveType, low: int, high: int) -> BCArrayType:
    b = (_int_literal(low), _int_literal(high))
    return BCArrayType(inner, is_matrix=False, flat_bounds=b)

def matrix(
    inner: BCPrimitiveType,
    low_outer: int,
    high_outer: int,
    low_inner: int,
    high_inner: int,
) -> BCArrayType:
    b = (
        _int_literal(low_outer),
        _int_literal(high_outer),
        _int_literal(low_inner),
        _int_literal(high_inner),
    )
    return BCArrayType(inner, is_matrix=True, matrix_bounds=b)

BCArgsList = dict[str, BCValue]

@dataclass
class BCFunction:
    args: BCArgsList
    returns: BCPrimitiveType
    fn: Callable[[BCArgsList], BCPrimitiveType]

@dataclass
class BCProcedure:
    args: BCArgsList
    fn: Callable[[BCArgsList],]

@dataclass
class BCDeclare:
    name: str
    typ: BCType
    value: BCValue | None = None

@dataclass
class BCConstant:
    name: str
    value: BCValue

@dataclass
class Exports(TypedDict):
    constants: list[BCConstant]
    variables: list[BCDeclare]
    procs: list[BCProcedure]
    funcs: list[BCFunction]

