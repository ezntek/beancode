import typing

from enum import IntEnum
from typing import IO, Any, Callable
from dataclasses import dataclass

from . import Pos, is_case_consistent
from .error import *


class TokenKind(IntEnum):
    DECLARE = 1
    CONSTANT = 2
    OUTPUT = 3
    INPUT = 4
    AND = 5
    OR = 6
    NOT = 7
    IF = 8
    THEN = 9
    ELSE = 10
    ENDIF = 11
    CASE = 12
    OF = 13
    OTHERWISE = 14
    ENDCASE = 15
    WHILE = 16
    DO = 17
    ENDWHILE = 18
    REPEAT = 19
    UNTIL = 20
    FOR = 21
    TO = 22
    STEP = 23
    NEXT = 24
    PROCEDURE = 25
    ENDPROCEDURE = 26
    CALL = 27
    FUNCTION = 28
    RETURN = 29
    RETURNS = 30
    ENDFUNCTION = 31
    OPENFILE = 32
    READFILE = 33
    WRITEFILE = 34
    CLOSEFILE = 35
    READ = 36
    WRITE = 37
    APPEND = 38
    INCLUDE = 39
    INCLUDE_FFI = 40
    EXPORT = 41
    SCOPE = 42
    ENDSCOPE = 43
    PRINT = 44
    TRACE = 45
    ENDTRACE = 46
    ASSIGN = 47
    EQUAL = 48
    LESS_THAN = 49
    GREATER_THAN = 50
    LESS_THAN_OR_EQUAL = 51
    GREATER_THAN_OR_EQUAL = 52
    NOT_EQUAL = 53
    MUL = 54
    DIV = 55
    ADD = 56
    SUB = 57
    POW = 58
    LEFT_PAREN = 59
    RIGHT_PAREN = 60
    LEFT_BRACKET = 61
    RIGHT_BRACKET = 62
    LEFT_CURLY = 63
    RIGHT_CURLY = 64
    COLON = 65
    COMMA = 66
    DOT = 67
    NEWLINE = 68
    LITERAL_STRING = 69
    LITERAL_CHAR = 70
    LITERAL_NUMBER = 71
    TRUE = 72
    FALSE = 73
    NULL = 74
    IDENT = 75
    TYPE = 76
    COMMENT = 77

    @classmethod
    def from_str_or_none(cls, s: str):
        if is_case_consistent(s):
            return cls.__members__.get(s.upper())
        else:
            return None

    @staticmethod
    def from_str(s: str):
        res = TokenKind.from_str_or_none(s)
        if not res:
            raise BCError(f"tried to convert invalid string token type {s}")
        return res

    def __repr__(self) -> str:
        return self.name.lower()

    def __str__(self):
        return self.__repr__()

    def humanize(self) -> str:
        match self:
            case TokenKind.ASSIGN:
                return "'<-'"
            case TokenKind.EQUAL:
                return "'='"
            case TokenKind.LESS_THAN:
                return "'<'"
            case TokenKind.GREATER_THAN:
                return "'>'"
            case TokenKind.LESS_THAN_OR_EQUAL:
                return "'<='"
            case TokenKind.GREATER_THAN_OR_EQUAL:
                return "'>='"
            case TokenKind.NOT_EQUAL:
                return "'<>'"
            case TokenKind.MUL:
                return "'*'"
            case TokenKind.DIV:
                return "'/'"
            case TokenKind.ADD:
                return "'+'"
            case TokenKind.SUB:
                return "'-'"
            case TokenKind.POW:
                return "'^'"
            case TokenKind.LEFT_PAREN:
                return "'('"
            case TokenKind.RIGHT_PAREN:
                return "')'"
            case TokenKind.LEFT_BRACKET:
                return "'['"
            case TokenKind.RIGHT_BRACKET:
                return "']'"
            case TokenKind.LEFT_CURLY:
                return "'{'"
            case TokenKind.RIGHT_CURLY:
                return "'}'"
            case TokenKind.COLON:
                return "':'"
            case TokenKind.COMMA:
                return "','"
            case TokenKind.DOT:
                return "'.'"
            case TokenKind.NEWLINE:
                return "newline"
            case TokenKind.LITERAL_STRING:
                return "string literal"
            case TokenKind.LITERAL_CHAR:
                return "character literal"
            case TokenKind.LITERAL_NUMBER:
                return "number literal"
            case TokenKind.IDENT:
                return "identifier or name"
            case TokenKind.TYPE:
                return "type"
            case _:
                return str(self).upper()


@dataclass(slots=True)
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
        return self.name.lower()

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, f) -> str:
        _ = f
        return self.__repr__().upper()

    @classmethod
    def from_str(cls, kind: str):
        res = cls.__members__.get(kind.upper())
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

    def __hash__(self) -> int:
        return hash((self.inner, self.bounds))

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

    def copy(self):
        if self.typ.is_matrix():
            new = [[v.copy() for v in l] for l in self.data]  # type: ignore
        else:
            new = [v.copy() for v in self.data]  # type: ignore
        return BCArray(self.typ, new)  # type: ignore

    def __eq__(self, value: object, /) -> bool:
        if type(value) is not type(self):
            return False
        return self.typ == value.typ and self.data == value.data  # type: ignore

    def __neq__(self, value: object, /) -> bool:
        return not self.__eq__(value)

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

    def __hash__(self) -> int:
        return hash((self.kind, self.val, self.is_array))

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

    def copy(self) -> "BCValue":
        if self.is_array:
            return BCValue(self.kind, self.val.copy(), True)  # type: ignore
        else:
            return BCValue(self.kind, self.val)

    def replace_inner(self, other: "BCValue"):
        self.kind = other.kind
        self.is_array = other.is_array
        if self.is_array:
            self.val = other.val.copy()  # type: ignore
        else:
            self.val = other.val

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
        if self.is_uninitialized():
            return "(null)"

        match self.kind:
            case BCPrimitiveType.STRING:
                return self.val  # type: ignore
            case BCPrimitiveType.BOOLEAN:
                return str(self.val).upper()
            case BCPrimitiveType.NULL:
                return "(null)"
            case _:
                return str(self.val)


@dataclass(slots=True)
class File:
    stream: IO[Any]  # im lazy
    # read, write, append
    mode: tuple[bool, bool, bool]


@dataclass(slots=True)
class FileCallbacks:
    open: Callable[[str, str], IO[Any]]
    close: Callable[[IO[Any]], None]
    # only for when the file has changed
    write: Callable[[str], None]
    append: Callable[[str], None]


@dataclass(slots=True)
class Literal(Expr):
    val: BCValue


@dataclass(slots=True)
class Negation(Expr):
    inner: Expr


@dataclass(slots=True)
class Not(Expr):
    inner: Expr


@dataclass(slots=True)
class Grouping(Expr):
    inner: Expr


# !!! INTERNAL USE ONLY !!!
# This is for the purposes of optimization. Library routine calls
# are typed by default, and they are SLOW!
@dataclass(slots=True)
class Sqrt(Expr):
    inner: Expr


@dataclass(slots=True)
class Identifier(Expr):
    ident: str
    libroutine: bool = False


@dataclass(slots=True)
class Typecast(Expr):
    typ: BCPrimitiveType
    expr: Expr


@dataclass(slots=True)
class ArrayLiteral(Expr):
    items: list[Expr]


class Operator(IntEnum):
    ASSIGN = 1
    EQUAL = 2
    LESS_THAN = 3
    GREATER_THAN = 4
    LESS_THAN_OR_EQUAL = 5
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
    FLOOR_DIV = 16
    MOD = 17

    @classmethod
    def from_token_kind(cls, token_kind: TokenKind) -> "Operator":
        return cls[token_kind.name]

    def __repr__(self) -> str:
        return self.name.lower()

    def __str__(self) -> str:
        return self.__repr__()

    def as_symbol(self) -> str:
        return {
            Operator.ASSIGN: "<-",
            Operator.EQUAL: "=",
            Operator.LESS_THAN: "<",
            Operator.GREATER_THAN: ">",
            Operator.LESS_THAN_OR_EQUAL: "<=",
            Operator.GREATER_THAN_OR_EQUAL: ">=",
            Operator.NOT_EQUAL: "<>",
            Operator.MUL: "*",
            Operator.DIV: "/",
            Operator.ADD: "+",
            Operator.SUB: "-",
            Operator.POW: "^",
            Operator.AND: "AND",
            Operator.OR: "OR",
            Operator.NOT: "NOT",
            Operator.FLOOR_DIV: "DIV",
            Operator.MOD: "MOD",
        }[self]

    # should return a verb!
    def humanize(self) -> str:
        match self:
            case Operator.MUL:
                return "multiply"
            case Operator.DIV:
                return "divide"
            case Operator.ADD:
                return "add"
            case Operator.SUB:
                return "subtract"
            case Operator.POW:
                return "exponentiate"
            case Operator.AND | Operator.OR | Operator.NOT:
                return self.name.upper()
            case Operator.ASSIGN:
                return "assign"
            case Operator.FLOOR_DIV:
                return "perform DIV()"
            case Operator.MOD:
                return "perform MOD()"
            case Operator.EQUAL | Operator.NOT_EQUAL:
                return "compare"
            case _:
                return f"perform {self.name.lower().replace('_', ' ')} between"


@dataclass(slots=True)
class BinaryExpr(Expr):
    lhs: Expr
    op: Operator
    rhs: Expr


@dataclass(slots=True)
class ArrayIndex(Expr):
    expr: Expr
    idx_outer: Expr
    idx_inner: Expr | None = None


Lvalue = Identifier | ArrayIndex


@dataclass(slots=True)
class Statement:
    pos: Pos


@dataclass(slots=True)
class CallStatement(Statement):
    ident: str
    args: list[Expr]
    libroutine: bool = False


@dataclass(slots=True)
class FunctionCall(Expr):
    ident: str
    args: list[Expr]
    libroutine: bool = False


@dataclass(slots=True)
class OutputStatement(Statement):
    items: list[Expr]
    newline: bool = True


@dataclass(slots=True)
class InputStatement(Statement):
    ident: Lvalue


@dataclass(slots=True)
class ConstantStatement(Statement):
    ident: Identifier
    value: Expr
    export: bool = False


@dataclass(slots=True)
class DeclareStatement(Statement):
    ident: list[Identifier]
    typ: Type
    export: bool = False


@dataclass(slots=True)
class AssignStatement(Statement):
    ident: Lvalue
    value: Expr
    is_ident: bool = True  # for optimization


@dataclass(slots=True)
class IfStatement(Statement):
    cond: Expr
    if_block: list["Statement"]
    else_block: list["Statement"]


@dataclass(slots=True)
class CaseofBranch:
    pos: Pos
    expr: Expr
    stmt: "Statement"


@dataclass(slots=True)
class CaseofStatement(Statement):
    expr: Expr
    # extra possible nodes for a CST for the formatter
    branches: list["CaseofBranch | NewlineStatement | Comment"]
    otherwise: "Statement | None"


@dataclass(slots=True)
class WhileStatement(Statement):
    end_pos: Pos  # for tracing
    cond: Expr
    block: list["Statement"]


@dataclass(slots=True)
class ForStatement(Statement):
    end_pos: Pos  # for tracing
    counter: Identifier
    block: list["Statement"]
    begin: Expr
    end: Expr
    step: Expr | None


@dataclass(slots=True)
class RepeatUntilStatement(Statement):
    end_pos: Pos  # for tracing
    cond: Expr
    block: list["Statement"]


@dataclass(slots=True)
class FunctionArgument:
    pos: Pos
    name: str
    typ: Type


@dataclass(slots=True)
class ProcedureStatement(Statement):
    name: str
    args: list[FunctionArgument]
    block: list["Statement"]
    export: bool = False


@dataclass(slots=True)
class FunctionStatement(Statement):
    name: str
    args: list[FunctionArgument]
    returns: Type
    block: list["Statement"]
    export: bool = False


@dataclass(slots=True)
class ReturnStatement(Statement):
    expr: Expr | None = None


FileMode = typing.Literal["read", "write", "append"]


@dataclass(slots=True)
class OpenfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    # guaranteed to be valid
    mode: tuple[bool, bool, bool]


@dataclass(slots=True)
class ReadfileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    target: Lvalue


@dataclass(slots=True)
class WritefileStatement(Statement):
    # file identifier or path
    file_ident: Expr | str
    src: Expr


@dataclass(slots=True)
class ClosefileStatement(Statement):
    file_ident: Expr | str


@dataclass(slots=True)
class ScopeStatement(Statement):
    block: list["Statement"]


@dataclass(slots=True)
class IncludeStatement(Statement):
    file: str
    ffi: bool


@dataclass(slots=True)
class TraceStatement(Statement):
    vars: list[str]
    file_name: str | None
    block: list["Statement"]


@dataclass(slots=True)
class ExprStatement(Statement):
    inner: Expr

    @classmethod
    def from_expr(cls, e: Expr) -> "ExprStatement":
        return cls(e.pos, e)


class NewlineStatement(Statement):
    pass


@dataclass(slots=True)
class Program:
    stmts: list[Statement]


@dataclass(slots=True)
class Variable:
    val: BCValue
    const: bool
    export: bool = False

    def is_uninitialized(self) -> bool:
        return self.val.is_uninitialized()

    def is_null(self) -> bool:
        return self.val.is_null()


@dataclass(slots=True)
class CallStackEntry:
    name: str
    rtype: Type | None
    func: bool = False
    proc: bool = False


@dataclass(slots=True)
class Comment:
    data: list[str]
    multiline: bool = False
    shebang: bool = False


@dataclass(slots=True)
class CommentStatement(Statement):
    comment: Comment
