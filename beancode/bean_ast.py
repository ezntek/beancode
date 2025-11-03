import typing

from io import StringIO
from dataclasses import dataclass

from . import Pos
from .error import *


@dataclass
class Expr:
    pos: Pos


BCPrimitiveType = typing.Literal["integer", "real", "char", "string", "boolean", "null"]


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
            return "ARRAY[2D] OF " + self.inner.upper()
        else:
            return "ARRAY OF " + self.inner.upper()


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


@dataclass
class BCValue:
    kind: BCType
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None
    boolean: bool | None = None
    array: BCArray | None = None

    def is_uninitialized(self) -> bool:
        # arrays are always initialized to NULLs
        if self.array:
            return False

        return (
            self.integer is None
            and self.real is None
            and self.char is None
            and self.string is None
            and self.boolean is None
        )

    def is_null(self) -> bool:
        return self.kind == "null" or self.is_uninitialized()

    @classmethod
    def empty(cls, kind: BCType) -> "BCValue":
        return cls(
            kind,
            integer=None,
            real=None,
            char=None,
            string=None,
            boolean=None,
            array=None,
        )

    @classmethod
    def new_null(cls) -> "BCValue":
        return cls("null")

    @classmethod
    def new_integer(cls, i: int) -> "BCValue":
        return cls("integer", integer=i)

    @classmethod
    def new_real(cls, r: float) -> "BCValue":
        return cls("real", real=r)

    @classmethod
    def new_boolean(cls, b: bool) -> "BCValue":
        return cls("boolean", boolean=b)

    @classmethod
    def new_char(cls, c: str) -> "BCValue":
        return cls("char", char=c[0])

    @classmethod
    def new_string(cls, s: str) -> "BCValue":
        return cls("string", string=s)

    @classmethod
    def new_array(cls, a: BCArray) -> "BCValue":
        return cls(a.typ, array=a)

    def get_integer(self) -> int:
        if self.kind != "integer":
            raise BCError(f"tried to access INTEGER value from BCValue of {self.kind}")

        return self.integer  # type: ignore

    def get_real(self) -> float:
        if self.kind != "real":
            raise BCError(f"tried to access REAL value from BCValue of {self.kind}")

        return self.real  # type: ignore

    def get_char(self) -> str:
        if self.kind != "char":
            raise BCError(f"tried to access CHAR value from BCValue of {self.kind}")

        return self.char  # type: ignore

    def get_string(self) -> str:
        if self.kind != "string":
            raise BCError(f"tried to access STRING value from BCValue of {self.kind}")

        return self.string  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != "boolean":
            raise BCError(f"tried to access BOOLEAN value from BCValue of {self.kind}")

        return self.boolean  # type: ignore

    def get_array(self) -> BCArray:
        if not isinstance(self.kind, BCArrayType):
            raise BCError(f"tried to access array value from BCValue of {self.kind}")

        return self.array  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        if isinstance(self.kind, BCArrayType):
            return self.array.__repr__()

        if self.is_uninitialized():
            return "(null)"

        match self.kind:
            case "string":
                return self.get_string()
            case "real":
                return str(self.get_real())
            case "integer":
                return str(self.get_integer())
            case "char":
                return str(self.get_char())
            case "boolean":
                return str(self.get_boolean()).upper()
            case "null":
                return "(null)"


@dataclass
class Literal(Expr):
    kind: BCPrimitiveType
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None
    boolean: bool | None = None

    def to_bcvalue(self) -> BCValue:
        return BCValue(
            self.kind,
            integer=self.integer,
            real=self.real,
            char=self.char,
            string=self.string,
            boolean=self.boolean,
            array=None,
        )


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


@dataclass
class InputStatement(Statement):
    ident: Identifier | ArrayIndex


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
    ident: Identifier | ArrayIndex
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
    modes: list[FileMode]


@dataclass
class ReadfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    src: Identifier | ArrayIndex


@dataclass
class WritefileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    target: Identifier | ArrayIndex


@dataclass
class ClosefileStatement(Statement):
    file_ident: Expr | str


# extra statements
@dataclass
class AppendfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    target: Identifier | ArrayIndex


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
