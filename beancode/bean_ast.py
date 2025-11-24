import typing

from enum import Enum
from typing import IO, Any, Callable
from io import StringIO
from dataclasses import dataclass

from . import Pos
from .error import *


@dataclass
class Expr:
    pos: Pos


class BCPrimitiveType(Enum):
    INTEGER = 0
    REAL = 1
    CHAR = 2
    STRING = 3
    BOOLEAN = 4
    NULL = 5
    
    def __repr__(self):
        return {
            BCPrimitiveType.INTEGER: "integer",
            BCPrimitiveType.REAL: "real",
            BCPrimitiveType.CHAR: "char",
            BCPrimitiveType.STRING: "string",
            BCPrimitiveType.BOOLEAN: "boolean",
            BCPrimitiveType.NULL: "null"
        }[self]

    @classmethod
    def from_string(cls, kind: str):
        TABLE = {
            "integer": BCPrimitiveType.INTEGER,
            "real": BCPrimitiveType.REAL,
            "char": BCPrimitiveType.CHAR,
            "string": BCPrimitiveType.STRING,
            "boolean": BCPrimitiveType.BOOLEAN,
            "null": BCPrimitiveType.NULL
        } 
        res = TABLE.get(kind.lower())
        if not res:
            raise BCError(f"tried to convert invalid string type {kind} to a BCPrimitiveType!")
        return res

@dataclass
class ArrayType:
    """parse-time representation of the array type"""

    inner: BCPrimitiveType
    is_matrix: bool  # true: 2d array
    flat_bounds: tuple["Expr", "Expr"] | None = None  # begin:end
    matrix_bounds: tuple["Expr", "Expr", "Expr", "Expr"] | None = (
        None  # begin:end,begin:end
    )

    def get_flat_bounds(self) -> tuple["Expr", "Expr"]:
        if self.flat_bounds is None:
            raise BCError("tried to access flat bounds on array without flat bounds")
        return self.flat_bounds

    def get_matrix_bounds(self) -> tuple["Expr", "Expr", "Expr", "Expr"]:
        if self.matrix_bounds is None:
            raise BCError(
                "tried to access matrix bounds on array without matrix bounds"
            )
        return self.matrix_bounds

    def __repr__(self) -> str:
        if self.is_matrix:
            return "ARRAY[2D] OF " + str(self.inner).upper()
        else:
            return "ARRAY OF " + str(self.inner).upper()


@dataclass
class BCArrayType:
    """runtime representation of an array type"""

    inner: BCPrimitiveType
    is_matrix: bool
    flat_bounds: tuple[int, int] | None = None
    matrix_bounds: tuple[int, int, int, int] | None = None

    @classmethod
    def new_flat(cls, inner: BCPrimitiveType, bounds: tuple[int, int]) -> "BCArrayType":
        return cls(inner, False, flat_bounds=bounds)

    @classmethod
    def new_matrix(
        cls, inner: BCPrimitiveType, bounds: tuple[int, int, int, int]
    ) -> "BCArrayType":
        return cls(inner, True, matrix_bounds=bounds)

    def get_flat_bounds(self) -> tuple[int, int]:
        if self.flat_bounds is None:
            raise BCError("tried to access flat bounds on array without flat bounds")
        return self.flat_bounds

    def get_matrix_bounds(self) -> tuple[int, int, int, int]:
        if self.matrix_bounds is None:
            raise BCError("tried to access matrixbounds on array without matrix bounds")
        return self.matrix_bounds

    def __repr__(self) -> str:
        s = StringIO()
        s.write("ARRAY[")
        if self.flat_bounds is not None:
            s.write(array_bounds_to_string(self.flat_bounds))
        elif self.matrix_bounds is not None:
            s.write(matrix_bounds_to_string(self.matrix_bounds))
        s.write("] OF ")
        s.write(str(self.inner).upper())
        return s.getvalue()


def array_bounds_to_string(bounds: tuple[int, int]) -> str:
    return f"{bounds[0]}:{bounds[1]}"


def matrix_bounds_to_string(bounds: tuple[int, int, int, int]) -> str:
    return f"{bounds[0]}:{bounds[1]},{bounds[2]}:{bounds[3]}"


@dataclass
class BCArray:
    typ: BCArrayType
    flat: list["BCValue"] | None = None  # must be a BCPrimitiveType
    matrix: list[list["BCValue"]] | None = None  # must be a BCPrimitiveType

    @classmethod
    def new_flat(cls, typ: BCArrayType, flat: list["BCValue"]) -> "BCArray":
        return cls(typ=typ, flat=flat)

    @classmethod
    def new_matrix(cls, typ: BCArrayType, matrix: list[list["BCValue"]]) -> "BCArray":
        return cls(typ=typ, matrix=matrix)

    def get_flat(self) -> list["BCValue"]:
        if self.flat is None:
            raise BCError("tried to access 1D array from a 2D array")
        return self.flat

    def get_matrix(self) -> list[list["BCValue"]]:
        if self.matrix is None:
            raise BCError("tried to access 2D array from a 1D array")
        return self.matrix

    def get_flat_bounds(self) -> tuple[int, int]:
        if self.typ.flat_bounds is None:
            raise BCError("tried to access 1D array from a 2D array")
        return self.typ.flat_bounds

    def get_matrix_bounds(self) -> tuple[int, int, int, int]:
        if self.typ.matrix_bounds is None:
            raise BCError("tried to access 2D array from a 1D array")
        return self.typ.matrix_bounds

    def __repr__(self) -> str:
        if not self.typ.is_matrix:
            return str(self.flat)
        else:
            return str(self.matrix)


# parsetime
Type = ArrayType | BCPrimitiveType

# runtime
BCType = BCArrayType | BCPrimitiveType

BCPayload = int | float | str | bool | BCArray | None

class BCValue:
    __slots__ = ('kind', 'val')
    kind: BCType
    val: BCPayload

    def __init__(self, kind: BCType, value: BCPayload = None):
        self.kind = kind
        self.val = value

    def is_uninitialized(self) -> bool:
        return self.val is None

    def is_null(self) -> bool:
        return self.kind == BCPrimitiveType.NULL or self.val is None 

    def __eq__(self, value: object, /) -> bool:
        if type(self) is not type(value):
            return False

        return self.kind == value.kind and self.val == value.val # type: ignore
    
    def __neq__(self, value: object, /) -> bool:
        return not (self.__eq__(value))

    @classmethod
    def empty(cls, kind: BCType) -> "BCValue":
        return cls(
            kind,
            None
        )

    @classmethod
    def new_null(cls) -> "BCValue":
        return cls(BCPrimitiveType.NULL)

    @classmethod
    def new_integer(cls, i: int) -> "BCValue":
        return cls(BCPrimitiveType.INTEGER, i)

    @classmethod
    def new_real(cls, r: float) -> "BCValue":
        return cls(BCPrimitiveType.REAL, r)

    @classmethod
    def new_boolean(cls, b: bool) -> "BCValue":
        return cls(BCPrimitiveType.BOOLEAN, b)

    @classmethod
    def new_char(cls, c: str) -> "BCValue":
        return cls(BCPrimitiveType.CHAR, c[0])

    @classmethod
    def new_string(cls, s: str) -> "BCValue":
        return cls(BCPrimitiveType.STRING, s)

    @classmethod
    def new_array(cls, a: BCArray) -> "BCValue":
        return cls(a.typ, a)

    def get_integer(self) -> int:
        if self.kind != BCPrimitiveType.INTEGER:
            raise BCError(f"tried to access INTEGER value from BCValue of {str(self.kind)}")

        return self.val  # type: ignore

    def get_real(self) -> float:
        if self.kind != BCPrimitiveType.REAL:
            raise BCError(f"tried to access REAL value from BCValue of {str(self.kind)}")

        return self.val  # type: ignore

    def get_char(self) -> str:
        if self.kind != BCPrimitiveType.CHAR:
            raise BCError(f"tried to access CHAR value from BCValue of {str(self.kind)}")

        return self.val[0]  # type: ignore

    def get_string(self) -> str:
        if self.kind != BCPrimitiveType.STRING:
            raise BCError(f"tried to access STRING value from BCValue of {str(self.kind)}")

        return self.val  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != BCPrimitiveType.BOOLEAN:
            raise BCError(f"tried to access BOOLEAN value from BCValue of {str(self.kind)}")

        return self.val  # type: ignore

    def get_array(self) -> BCArray:
        if not isinstance(self.kind, BCArrayType):
            raise BCError(f"tried to access array value from BCValue of {str(self.kind)}")

        return self.val  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        if self.kind == "array":
            return str(self.val)

        if self.is_uninitialized():
            return "(null)"

        match self.kind:
            case BCPrimitiveType.STRING:
                return self.get_string()
            case BCPrimitiveType.REAL:
                return str(self.get_real())
            case BCPrimitiveType.INTEGER:
                return str(self.get_integer())
            case BCPrimitiveType.CHAR:
                return str(self.get_char())
            case BCPrimitiveType.BOOLEAN:
                return str(self.get_boolean()).upper()
            case BCPrimitiveType.NULL:
                return "(null)"

@dataclass
class File:
    stream: IO[Any] # im lazy
    # read, write, append
    mode: tuple[bool, bool, bool]
    open = True


@dataclass
class FileCallbacks:
    open: Callable[[str, str], IO[Any]]
    close: Callable[[IO[Any]], None]

@dataclass
class Literal(Expr):
    val: BCValue


@dataclass
class Negation(Expr):
    inner: Expr


@dataclass
class Not(Expr):
    inner: Expr


@dataclass
class Grouping(Expr):
    inner: Expr


@dataclass
class Identifier(Expr):
    ident: str


@dataclass
class Typecast(Expr):
    typ: BCPrimitiveType
    expr: Expr


@dataclass
class ArrayLiteral(Expr):
    items: list[Expr]


Operator = typing.Literal[
    "assign",
    "equal",
    "less_than",
    "greater_than",
    "less_than_or_equal",
    "greater_than_or_equal",
    "not_equal",
    "mul",
    "div",
    "add",
    "sub",
    "pow",
    "and",
    "or",
    "not",
]

@dataclass
class BinaryExpr(Expr):
    lhs: Expr
    op: Operator
    rhs: Expr


@dataclass
class ArrayIndex(Expr):
    ident: Identifier
    idx_outer: Expr
    idx_inner: Expr | None = None

Lvalue = Identifier | ArrayIndex

@dataclass
class Statement:
    pos: Pos


@dataclass
class CallStatement(Statement):
    ident: str
    args: list[Expr]


@dataclass
class FunctionCall(Expr):
    ident: str
    args: list[Expr]


@dataclass
class OutputStatement(Statement):
    items: list[Expr]
    newline: bool = True


@dataclass
class InputStatement(Statement):
    ident: Lvalue


@dataclass
class ConstantStatement(Statement):
    ident: Identifier
    value: Expr
    export: bool = False


@dataclass
class DeclareStatement(Statement):
    ident: list[Identifier]
    typ: Type
    export: bool = False
    expr: Expr | None = None


@dataclass
class AssignStatement(Statement):
    ident: Lvalue
    value: Expr


@dataclass
class IfStatement(Statement):
    cond: Expr
    if_block: list["Statement"]
    else_block: list["Statement"]


@dataclass
class CaseofBranch:
    pos: Pos
    expr: Expr
    stmt: "Statement"


@dataclass
class CaseofStatement(Statement):
    expr: Expr
    branches: list[CaseofBranch]
    otherwise: "Statement | None"


@dataclass
class WhileStatement(Statement):
    end_pos: Pos  # for tracing
    cond: Expr
    block: list["Statement"]


@dataclass
class ForStatement(Statement):
    end_pos: Pos  # for tracing
    counter: Identifier
    block: list["Statement"]
    begin: Expr
    end: Expr
    step: Expr | None


@dataclass
class RepeatUntilStatement(Statement):
    end_pos: Pos  # for tracing
    cond: Expr
    block: list["Statement"]


@dataclass
class FunctionArgument:
    pos: Pos
    name: str
    typ: Type


@dataclass
class ProcedureStatement(Statement):
    name: str
    args: list[FunctionArgument]
    block: list["Statement"]
    export: bool = False


@dataclass
class FunctionStatement(Statement):
    name: str
    args: list[FunctionArgument]
    returns: Type
    block: list["Statement"]
    export: bool = False


@dataclass
class ReturnStatement(Statement):
    expr: Expr | None = None


FileMode = typing.Literal["read", "write", "append"]


@dataclass
class OpenfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    # guaranteed to be valid
    mode: tuple[bool, bool, bool]

@dataclass
class ReadfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    target: Lvalue


@dataclass
class WritefileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    src: Expr


@dataclass
class ClosefileStatement(Statement):
    file_ident: Expr | str


# extra statements
@dataclass
class AppendfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    src: Expr


@dataclass
class ScopeStatement(Statement):
    block: list["Statement"]


@dataclass
class IncludeStatement(Statement):
    file: str
    ffi: bool


@dataclass
class TraceStatement(Statement):
    vars: list[str]
    file_name: str | None
    block: list["Statement"]


@dataclass
class ExprStatement(Statement):
    inner: Expr

    @classmethod
    def from_expr(cls, e: Expr) -> "ExprStatement":
        return cls(e.pos, e)


@dataclass
class Program:
    stmts: list[Statement]


@dataclass
class Variable:
    val: BCValue
    const: bool
    export: bool = False

    def is_uninitialized(self) -> bool:
        return self.val.is_uninitialized()

    def is_null(self) -> bool:
        return self.val.is_null()


@dataclass
class CallStackEntry:
    name: str
    rtype: Type | None
    func: bool = False
    proc: bool = False
