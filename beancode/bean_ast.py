import typing
from dataclasses import dataclass
from .error import *

@dataclass
class Expr:
    # location of the token
    pos: tuple[int, int, int] | None


BCPrimitiveType = typing.Literal["integer", "real", "char", "string", "boolean", "null"]


@dataclass
class BCArrayType:
    inner: BCPrimitiveType
    is_matrix: bool  # true: 2d array
    flat_bounds: tuple["Expr", "Expr"] | None = None  # begin:end
    matrix_bounds: tuple["Expr", "Expr", "Expr", "Expr"] | None = (
        None  # begin:end,begin:end
    )

    def has_bounds(self) -> bool:
        return self.flat_bounds is not None or self.matrix_bounds is not None

    def get_flat_bounds(self) -> tuple["Expr", "Expr"]:
        if self.flat_bounds is None:
            raise BCError("tried to access flat bounds on array without flat bounds")
        return self.flat_bounds

    def get_matrix_bounds(self) -> tuple["Expr", "Expr", "Expr", "Expr"]:
        if self.matrix_bounds is None:
            raise BCError("tried to access matrixbounds on array without matrix bounds")
        return self.matrix_bounds

@dataclass
class BCArray:
    typ: BCArrayType
    flat: list["BCValue"] | None = None  # must be a BCPrimitiveType
    matrix: list[list["BCValue"]] | None = None  # must be a BCPrimitiveType
    flat_bounds: tuple[int, int] | None = None
    matrix_bounds: tuple[int, int, int, int] | None = None

    def __repr__(self) -> str:
        if not self.typ.is_matrix:
            return str(self.flat)
        else:
            return str(self.matrix)

    def get_flat(self) -> list["BCValue"]:
        if self.flat is None:
            raise BCError("tried to access flat array from a matrix array")
        return self.flat

    def get_matrix(self) -> list[list["BCValue"]]:
        if self.matrix is None:
            raise BCError("tried to access matrix array from a flat array")
        return self.matrix

    def get_type_str(self) -> str:
        s = "ARRAY["
        if self.flat_bounds is not None:
            s += str(self.flat_bounds[0])
            s += ":"
            s += str(self.flat_bounds[1])
        elif self.matrix_bounds is not None:
            s += str(self.matrix_bounds[0])
            s += ":"
            s += str(self.matrix_bounds[1])
            s += ","
            s += str(self.matrix_bounds[2])
            s += ":"
            s += str(self.matrix_bounds[3])
        s += "] OF "
        s += str(self.typ.inner).upper()
        return s

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
        return (
            self.integer is None
            and self.real is None
            and self.char is None
            and self.string is None
            and self.boolean is None
            and self.array is None
        )

    def is_null(self) -> bool:
        return self.kind == "null"

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

    # arrays later
    def _kind(self) -> str:
        if isinstance(self.kind, BCArrayType):
            return "(array)"
        else:
            return str(self.kind)

    def get_integer(self) -> int:
        if self.kind != "integer":
            raise BCError(f"tried to access integer value from BCValue of {self._kind()}")

        return self.integer  # type: ignore

    def get_real(self) -> float:
        if self.kind != "real":
            raise BCError(f"tried to access real value from BCValue of {self._kind()}")

        return self.real  # type: ignore

    def get_char(self) -> str:
        if self.kind != "char":
            raise BCError(f"tried to access char value from BCValue of {self._kind()}")

        return self.char  # type: ignore

    def get_string(self) -> str:
        if self.kind != "string":
            raise BCError(f"tried to access string value from BCValue of {self._kind()}")

        return self.string  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != "boolean":
            raise BCError(f"tried to access boolean value from BCValue of {self._kind()}")

        return self.boolean  # type: ignore

    def get_array(self) -> BCArray:
        if not isinstance(self.kind, BCArrayType):
            raise BCError(f"tried to access array value from BCValue of {self._kind()}")

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


StatementKind = typing.Literal[
    "declare",
    "output",
    "input",
    "constant",
    "assign",
    "if",
    "caseof",
    "while",
    "for",
    "repeatuntil",
    "function",
    "procedure",
    "call",
    "fncall",
    "return",
    "scope",
    "include",
]


@dataclass
class CallStatement:
    pos: tuple[int, int, int]
    ident: str
    args: list[Expr]


@dataclass
class FunctionCall(Expr):
    ident: str
    args: list[Expr]


@dataclass
class OutputStatement:
    pos: tuple[int, int, int]
    items: list[Expr]


@dataclass
class InputStatement:
    pos: tuple[int, int, int]
    ident: Identifier


@dataclass
class ConstantStatement:
    pos: tuple[int, int, int]
    ident: Identifier
    value: Literal
    export: bool = False


@dataclass
class DeclareStatement:
    pos: tuple[int, int, int]
    ident: list[Identifier]
    typ: BCType
    export: bool = False
    expr: Expr | None = None


@dataclass
class AssignStatement:
    pos: tuple[int, int, int]
    ident: Identifier | ArrayIndex
    value: Expr


@dataclass
class IfStatement:
    pos: tuple[int, int, int]
    cond: Expr
    if_block: list["Statement"]
    else_block: list["Statement"]


@dataclass
class CaseofBranch:
    pos: tuple[int, int, int]
    expr: Expr
    stmt: "Statement"


@dataclass
class CaseofStatement:
    pos: tuple[int, int, int]
    expr: Expr
    branches: list[CaseofBranch]
    otherwise: "Statement | None"


@dataclass
class WhileStatement:
    pos: tuple[int, int, int]
    cond: Expr
    block: list["Statement"]


@dataclass
class ForStatement:
    pos: tuple[int, int, int]
    counter: Identifier
    block: list["Statement"]
    begin: Expr
    end: Expr
    step: Expr | None


@dataclass
class RepeatUntilStatement:
    pos: tuple[int, int, int]
    cond: Expr
    block: list["Statement"]


@dataclass
class FunctionArgument:
    pos: tuple[int, int, int]
    name: str
    typ: BCType


@dataclass
class ProcedureStatement:
    pos: tuple[int, int, int]
    name: str
    args: list[FunctionArgument]
    block: list["Statement"]
    export: bool = False


@dataclass
class FunctionStatement:
    pos: tuple[int, int, int]
    name: str
    args: list[FunctionArgument]
    returns: BCType
    block: list["Statement"]
    export: bool = False


@dataclass
class ReturnStatement:
    pos: tuple[int, int, int]
    expr: Expr | None


@dataclass
class ScopeStatement:
    pos: tuple[int, int, int]
    block: list["Statement"]


@dataclass
class IncludeStatement:
    pos: tuple[int, int, int]
    file: str
    ffi: bool


@dataclass
class Statement:
    kind: StatementKind
    declare: DeclareStatement | None = None
    output: OutputStatement | None = None
    input: InputStatement | None = None
    constant: ConstantStatement | None = None
    assign: AssignStatement | None = None
    if_s: IfStatement | None = None
    caseof: CaseofStatement | None = None
    while_s: WhileStatement | None = None
    for_s: ForStatement | None = None
    repeatuntil: RepeatUntilStatement | None = None
    function: FunctionStatement | None = None
    procedure: ProcedureStatement | None = None
    call: CallStatement | None = None
    fncall: FunctionCall | None = None  # Impostor! expr as statement?!
    return_s: ReturnStatement | None = None
    scope: ScopeStatement | None = None
    include: IncludeStatement | None = None

    def __repr__(self) -> str:
        match self.kind:
            case "declare":
                return self.declare.__repr__()
            case "output":
                return self.output.__repr__()
            case "input":
                return self.input.__repr__()
            case "constant":
                return self.constant.__repr__()
            case "assign":
                return self.assign.__repr__()
            case "if":
                return self.if_s.__repr__()
            case "caseof":
                return self.caseof.__repr__()
            case "while":
                return self.while_s.__repr__()
            case "for":
                return self.for_s.__repr__()
            case "repeatuntil":
                return self.repeatuntil.__repr__()
            case "function":
                return self.function.__repr__()
            case "procedure":
                return self.procedure.__repr__()
            case "call":
                return self.call.__repr__()
            case "return":
                return self.return_s.__repr__()
            case "fncall":
                return self.fncall.__repr__()
            case "scope":
                return self.scope.__repr__()
            case "include":
                return self.include.__repr__()


@dataclass
class Program:
    stmts: list[Statement]
