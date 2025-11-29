import typing

from enum import IntEnum
from typing import IO, Any, Callable
from io import StringIO
from dataclasses import dataclass

from . import Pos
from .error import *


@dataclass
class Expr:
    pos: Pos


class BCPrimitiveType(IntEnum):
    INTEGER = 1
    REAL = 2
    CHAR = 3
    STRING = 4
    BOOLEAN = 5
    NULL = 6

    def __repr__(self):
        return {
            BCPrimitiveType.INTEGER: "integer",
            BCPrimitiveType.REAL: "real",
            BCPrimitiveType.CHAR: "char",
            BCPrimitiveType.STRING: "string",
            BCPrimitiveType.BOOLEAN: "boolean",
            BCPrimitiveType.NULL: "null",
        }[self]

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, f) -> str:
        _ = f
        return self.__repr__().upper()

    @classmethod
    def from_string(cls, kind: str):
        TABLE = {
            "integer": BCPrimitiveType.INTEGER,
            "real": BCPrimitiveType.REAL,
            "char": BCPrimitiveType.CHAR,
            "string": BCPrimitiveType.STRING,
            "boolean": BCPrimitiveType.BOOLEAN,
            "null": BCPrimitiveType.NULL,
        }
        res = TABLE.get(kind.lower())
        if res is None:
            raise BCError(
                f"tried to convert invalid string type {kind} to a BCPrimitiveType!"
            )
        return res


class ArrayType:
    """parse-time representation of the array type"""

    __slots__ = ("inner", "bounds")

    inner: BCPrimitiveType
    bounds: tuple["Expr", "Expr"] | tuple["Expr", "Expr", "Expr", "Expr"]

    def __init__(
        self,
        inner: BCPrimitiveType,
        bounds: tuple["Expr", "Expr"] | tuple["Expr", "Expr", "Expr", "Expr"],
    ):
        self.inner = inner
        self.bounds = bounds

    def is_flat(self) -> bool:
        return len(self.bounds) == 2

    def is_matrix(self) -> bool:
        return len(self.bounds) == 4

    def get_flat_bounds(self) -> tuple["Expr", "Expr"]:
        if len(self.bounds) != 2:
            raise BCError("tried to access flat bounds on a matrix!")
        return self.bounds

    def get_matrix_bounds(self) -> tuple["Expr", "Expr", "Expr", "Expr"]:
        if len(self.bounds) != 4:
            raise BCError("tried to access matrix bounds on a flat array!")
        return self.bounds

    def __repr__(self) -> str:
        if len(self.bounds) == 2:
            return "ARRAY[2D] OF " + str(self.inner).upper()
        else:
            return "ARRAY OF " + str(self.inner).upper()


class BCArrayType:
    """runtime representation of an array type"""

    __slots__ = ("inner", "bounds")

    inner: BCPrimitiveType
    bounds: tuple[int, int] | tuple[int, int, int, int]

    def __init__(
        self,
        inner: BCPrimitiveType,
        bounds: tuple[int, int] | tuple[int, int, int, int],
    ):
        self.inner = inner
        self.bounds = bounds

    def __eq__(self, value: object, /) -> bool:
        if type(self) is not type(value):
            return False

        return self.inner == value.inner and self.bounds == value.bounds  # type: ignore

    def __neq__(self, value: object, /) -> bool:
        return not (self.__eq__(value))

    def is_flat(self) -> bool:
        return len(self.bounds) == 2

    def is_matrix(self) -> bool:
        return len(self.bounds) == 4

    @classmethod
    def new_flat(cls, inner: BCPrimitiveType, bounds: tuple[int, int]) -> "BCArrayType":
        return cls(inner, bounds)

    @classmethod
    def new_matrix(
        cls, inner: BCPrimitiveType, bounds: tuple[int, int, int, int]
    ) -> "BCArrayType":
        return cls(inner, bounds)

    def get_flat_bounds(self) -> tuple[int, int]:
        if len(self.bounds) != 2:
            raise BCError("tried to access flat bounds on a matrix!")
        return self.bounds

    def get_matrix_bounds(self) -> tuple[int, int, int, int]:
        if len(self.bounds) != 4:
            raise BCError("tried to access flat bounds on a matrix!")
        return self.bounds

    def __repr__(self) -> str:
        s = list()
        s.append("ARRAY[")

        if len(self.bounds) == 2:
            s.append(array_bounds_to_string(self.bounds))
        else:
            s.append(matrix_bounds_to_string(self.bounds))

        s.append("] OF ")
        s.append(str(self.inner).upper())
        return "".join(s)


def array_bounds_to_string(bounds: tuple[int, int]) -> str:
    return f"{bounds[0]}:{bounds[1]}"


def matrix_bounds_to_string(bounds: tuple[int, int, int, int]) -> str:
    return f"{bounds[0]}:{bounds[1]},{bounds[2]}:{bounds[3]}"


class BCArray:
    __slots__ = ("typ", "data")
    typ: BCArrayType
    data: list["BCValue"] | list[list["BCValue"]]

    def __init__(self, typ: BCArrayType, data: list["BCValue"] | list[list["BCValue"]]):
        self.typ = typ
        self.data = data

    @classmethod
    def new_flat(cls, typ: BCArrayType, flat: list["BCValue"]) -> "BCArray":
        return cls(typ, flat)

    @classmethod
    def new_matrix(cls, typ: BCArrayType, matrix: list[list["BCValue"]]) -> "BCArray":
        return cls(typ, matrix)

    def is_flat(self) -> bool:
        return self.typ.is_flat()

    def is_matrix(self) -> bool:
        return self.typ.is_matrix()

    def get_flat(self) -> list["BCValue"]:
        if not self.typ.is_flat():
            raise BCError("tried to access 1D array from a 2D array")
        return self.data  # type: ignore

    def get_matrix(self) -> list[list["BCValue"]]:
        if not self.typ.is_matrix():
            raise BCError("tried to access 1D array from a 2D array")
        return self.data  # type: ignore

    def get_flat_bounds(self) -> tuple[int, int]:
        if not self.typ.is_flat():
            raise BCError("tried to access 1D array from a 2D array")
        return self.typ.bounds  # type: ignore

    def get_matrix_bounds(self) -> tuple[int, int, int, int]:
        if not self.typ.is_matrix():
            raise BCError("tried to access 2D array from a 1D array")
        return self.typ.bounds  # type: ignore

    def __repr__(self) -> str:
        return str(self.data)


# parsetime
Type = ArrayType | BCPrimitiveType

# runtime
BCType = BCArrayType | BCPrimitiveType

BCPayload = int | float | str | bool | BCArray | None


class BCValue:
    __slots__ = ("kind", "val", "is_array")
    kind: BCType
    val: BCPayload
    is_array: bool

    def __init__(self, kind: BCType, value: BCPayload = None, is_array=False):
        self.kind = kind
        self.val = value
        self.is_array = is_array

    def is_uninitialized(self) -> bool:
        return self.val is None

    def is_null(self) -> bool:
        return self.kind == BCPrimitiveType.NULL or self.val is None

    def __eq__(self, value: object, /) -> bool:
        if type(self) is not type(value):
            return False

        return self.kind == value.kind and self.val == value.val  # type: ignore

    def __neq__(self, value: object, /) -> bool:
        return not (self.__eq__(value))

    def kind_is_numeric(self) -> bool:
        return self.kind == BCPrimitiveType.INTEGER or self.kind == BCPrimitiveType.REAL

    def kind_is_alpha(self) -> bool:
        return self.kind == BCPrimitiveType.STRING or self.kind == BCPrimitiveType.CHAR

    @classmethod
    def empty(cls, kind: BCType) -> "BCValue":
        return cls(kind, None)

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
        return cls(a.typ, a, is_array=True)

    def get_integer(self) -> int:
        if self.kind != BCPrimitiveType.INTEGER:
            raise BCError(
                f"tried to access INTEGER value from BCValue of {str(self.kind)}"
            )

        return self.val  # type: ignore

    def get_real(self) -> float:
        if self.kind != BCPrimitiveType.REAL:
            raise BCError(
                f"tried to access REAL value from BCValue of {str(self.kind)}"
            )

        return self.val  # type: ignore

    def get_char(self) -> str:
        if self.kind != BCPrimitiveType.CHAR:
            raise BCError(
                f"tried to access CHAR value from BCValue of {str(self.kind)}"
            )

        return self.val[0]  # type: ignore

    def get_string(self) -> str:
        if self.kind != BCPrimitiveType.STRING:
            raise BCError(
                f"tried to access STRING value from BCValue of {str(self.kind)}"
            )

        return self.val  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != BCPrimitiveType.BOOLEAN:
            raise BCError(
                f"tried to access BOOLEAN value from BCValue of {str(self.kind)}"
            )

        return self.val  # type: ignore

    def get_array(self) -> BCArray:
        if not self.is_array:
            raise BCError(
                f"tried to access array value from BCValue of {str(self.kind)}"
            )

        return self.val  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        if self.is_array:
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
    stream: IO[Any]  # im lazy
    # read, write, append
    mode: tuple[bool, bool, bool]


@dataclass
class FileCallbacks:
    open: Callable[[str, str], IO[Any]]
    close: Callable[[IO[Any]], None]
    # only for when the file has changed
    write: Callable[[str], None]
    append: Callable[[str], None]


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

# !!! INTERNAL USE ONLY !!!
# This is for the purposes of optimization. Library routine calls
# are typed by default, and they are SLOW!
@dataclass
class Sqrt(Expr):
    inner: Expr

@dataclass
class Identifier(Expr):
    ident: str
    libroutine: bool = False


@dataclass
class Typecast(Expr):
    typ: BCPrimitiveType
    expr: Expr


@dataclass
class ArrayLiteral(Expr):
    items: list[Expr]


class Operator(IntEnum):
    ASSIGN = 1
    EQUAL = 2
    LESS_THAN = 3
    GREATER_THAN = 4
    LESS_THAN_OR_EQUAL = 4
    GREATER_THAN_OR_EQUAL = 6
    NOT_EQUAL = 7
    MUL = 8
    DIV = 9
    ADD = 10
    SUB = 11
    POW = 12
    AND = 13
    OR = 14
    NOT = 15

    # !!! INTERNAL USE ONLY !!!
    # This is for the purposes of optimization. Library routine calls
    # are typed by default, and they are SLOW!
    FLOOR_DIV = 16
    MOD = 17

    @classmethod
    def from_str(cls, data: str) -> "Operator":
        return {
            "assign": Operator.ASSIGN,
            "equal": Operator.EQUAL,
            "less_than": Operator.LESS_THAN,
            "greater_than": Operator.GREATER_THAN,
            "less_than_or_equal": Operator.LESS_THAN_OR_EQUAL,
            "greater_than_or_equal": Operator.GREATER_THAN_OR_EQUAL,
            "not_equal": Operator.NOT_EQUAL,
            "mul": Operator.MUL,
            "div": Operator.DIV,
            "add": Operator.ADD,
            "sub": Operator.SUB,
            "pow": Operator.POW,
            "and": Operator.AND,
            "or": Operator.OR,
            "not": Operator.NOT,
            "floor_div": Operator.FLOOR_DIV,
            "mod": Operator.MOD,
        }[data]

    def __repr__(self) -> str:
        return {
            Operator.ASSIGN: "assign",
            Operator.EQUAL: "equal",
            Operator.LESS_THAN: "less_than",
            Operator.GREATER_THAN: "greater_than",
            Operator.LESS_THAN_OR_EQUAL: "less_than_or_equal",
            Operator.GREATER_THAN_OR_EQUAL: "greater_than_or_equal",
            Operator.NOT_EQUAL: "not_equal",
            Operator.MUL: "mul",
            Operator.DIV: "div",
            Operator.ADD: "add",
            Operator.SUB: "sub",
            Operator.POW: "pow",
            Operator.AND: "and",
            Operator.OR: "or",
            Operator.NOT: "not",
            Operator.FLOOR_DIV: "floor_div",
            Operator.MOD: "mod",
        }[self]

    def __str__(self) -> str:
        return self.__repr__()


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
    libroutine: bool = False


@dataclass
class FunctionCall(Expr):
    ident: str
    args: list[Expr]
    libroutine: bool = False


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
    is_ident: bool = True # for optimization


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
