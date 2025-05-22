import typing as t
import lexer as l
from dataclasses import dataclass
from util import *
import typing

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

    def get_integer(self) -> int:
        if self.kind != "integer":
            raise BCError(f"tried to access integer value from BCValue of {self.kind}")

        return self.integer  # type: ignore

    def get_real(self) -> float:
        if self.kind != "real":
            raise BCError(f"tried to access real value from BCValue of {self.kind}")

        return self.real  # type: ignore

    def get_char(self) -> str:
        if self.kind != "char":
            raise BCError(f"tried to access char value from BCValue of {self.kind}")

        return self.char  # type: ignore

    def get_string(self) -> str:
        if self.kind != "string":
            raise BCError(f"tried to access string value from BCValue of {self.kind}")

        return self.string  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != "boolean":
            raise BCError(f"tried to access boolean value from BCValue of {self.kind}")

        return self.boolean  # type: ignore

    def get_array(self) -> BCArray:
        if self.kind != "array":
            raise BCError(f"tried to access array value from BCValue of {self.kind}")

        return self.array  # type: ignore

    def __repr__(self) -> str:  # type: ignore
        if isinstance(self.kind, BCArrayType):
            raise BCError("BCValue of array can only be represented at runtime")

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
                return str(self.get_boolean())
            case "null":
                return "(null)"


@dataclass
class Expr:
    pass


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


Operator = t.Literal[
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


StatementKind = t.Literal[
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
    ident: str
    args: list[Expr]


@dataclass
class FunctionCall(Expr):
    ident: str
    args: list[Expr]


@dataclass
class OutputStatement:
    items: list[Expr]


@dataclass
class InputStatement:
    ident: Identifier


@dataclass
class ConstantStatement:
    ident: Identifier
    value: Literal
    export: bool = False


@dataclass
class DeclareStatement:
    ident: Identifier
    typ: BCType
    export: bool = False
    expr: Expr | None = None

@dataclass
class AssignStatement:
    ident: Identifier | ArrayIndex
    value: Expr


@dataclass
class IfStatement:
    cond: Expr
    if_block: list["Statement"]
    else_block: list["Statement"]


@dataclass
class CaseofBranch:
    expr: Expr
    stmt: "Statement"


@dataclass
class CaseofStatement:
    expr: Expr
    branches: list[CaseofBranch]
    otherwise: "Statement | None"


@dataclass
class WhileStatement:
    cond: Expr
    block: list["Statement"]


@dataclass
class ForStatement:
    counter: Identifier
    block: list["Statement"]
    begin: Expr
    end: Expr
    step: Expr | None


@dataclass
class RepeatUntilStatement:
    cond: Expr
    block: list["Statement"]


@dataclass
class FunctionArgument:
    name: str
    typ: BCType


@dataclass
class ProcedureStatement:
    name: str
    args: list[FunctionArgument]
    block: list["Statement"]
    export: bool = False


@dataclass
class FunctionStatement:
    name: str
    args: list[FunctionArgument]
    returns: BCType
    block: list["Statement"]
    export: bool = False


@dataclass
class ReturnStatement:
    expr: Expr | None


@dataclass
class ScopeStatement:
    block: list["Statement"]

@dataclass
class IncludeStatement:
    file: str

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


class Parser:
    tokens: list[l.Token]
    cur: int

    def __init__(self, tokens: list[l.Token]) -> None:
        self.cur = 0
        self.tokens = tokens

    def check(self, tok: tuple[l.TokenType, str]) -> bool:
        if self.cur == len(self.tokens):
            return False

        peek = self.peek()
        if tok[0] != peek.kind:
            return False

        match tok[0]:
            case "type":
                return tok[1] == peek.typ
            case "ident":
                return tok[1] == peek.ident
            case "keyword":
                return tok[1] == peek.keyword
            case "literal":
                return tok[1] == peek.literal
            case "operator":
                return tok[1] == peek.operator
            case "separator":
                return tok[1] == peek.separator

        return False

    def advance(self) -> l.Token:
        if self.cur < len(self.tokens):
            self.cur += 1

        return self.prev()

    def consume_newlines(self):
        while self.peek().kind == "newline":
            self.advance()

    def check_newline(self, s: str):
        nl = self.advance()
        if nl.kind != "newline":
            raise BCError(f"expected newline after {s}, but found `{self.prev()}`", nl)

    def prev(self) -> l.Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> l.Token:
        return self.tokens[self.cur]

    def peek_next(self) -> l.Token:
        return self.tokens[self.cur + 1]

    def match(self, typs: list[tuple[l.TokenType, str]]) -> bool:
        for typ in typs:
            if self.check(typ):
                self.advance()
                return True
        return False

    def is_integer(self, val: str) -> bool:
        if val[0] == "-" and val[1].isdigit():
            val = val[1:]

        for ch in val:
            if not ch.isdigit() and ch != "_":
                return False
        return True

    def is_real(self, val: str) -> bool:
        if not self.is_integer(val):
            return False

        found_decimal = False

        for ch in val:
            if ch == "." and found_decimal:
                return False
            elif ch == ".":
                found_decimal = True

        return found_decimal

    def literal(self) -> Expr | None:
        c = self.advance()

        if c.kind != "literal":
            return None

        lit: l.Literal
        lit = c.literal  # type: ignore

        match lit.kind:
            case "char":
                val = lit.value
                if val[0] == "\\":
                    if len(val) == 1:
                        return Literal("char", char="\\")
                    c = ""
                    match val[1]:
                        case "n":
                            c = "\n"
                        case "r":
                            c = "\r"
                        case "e":
                            c = "\033"
                        case "a":
                            c = "\a"
                        case "b":
                            c = "\b"
                        case "f":
                            c = "\f"
                        case "v":
                            c = "\v"
                        case "\\":
                            c = "\\"
                    return Literal("char", char=c)
                else:
                    if len(val) > 1:
                        raise BCError(
                            f"more than 1 character in char literal `{lit}`", c
                        )
                    return Literal("char", char=val[0])
            case "string":
                val = lit.value
                return Literal("string", string=val)
            case "boolean":
                val = lit.value.lower()
                if val == "true":
                    return Literal("boolean", boolean=True)
                elif val == "false":
                    return Literal("boolean", boolean=False)
                else:
                    raise BCError(f"invalid boolean literal `{lit.value}`", c)
            case "number":
                val = lit.value

                if self.is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        raise BCError(f"invalid number literal `{val}`", c)

                    return Literal("real", real=res)
                elif self.is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        raise BCError(f"invalid number literal `{val}`", c)

                    return Literal("integer", integer=res)
                else:
                    raise BCError(f"invalid number literal `{val}`", c)

    def typ(self) -> BCType | None:
        PRIM_TYPES = ["integer", "real", "boolean", "char", "string"]

        adv = self.advance()

        if adv.kind == "type" and adv.typ != "array":
            if adv.typ not in PRIM_TYPES:
                return None

            t: BCPrimitiveType = adv.typ  # type: ignore
            return t
        elif adv.kind == "type" and adv.typ == "array":
            flat_bounds = None
            matrix_bounds = None
            is_matrix = False
            inner: BCPrimitiveType

            left_bracket = self.advance()
            if left_bracket.separator == "left_bracket":
                begin = self.expression()
                if begin is None:
                    raise BCError(
                        "invalid expression as beginning value of array declaration",
                        begin,
                    )

                colon = self.advance()
                if colon.kind != "separator" and colon.separator != "colon":
                    raise BCError(
                        "expected colon after beginning value of array declaration",
                        colon,
                    )

                end = self.expression()
                if end is None:
                    raise BCError(
                        "invalid expression as ending value of array declaration", end
                    )

                flat_bounds = (begin, end)

                right_bracket = self.advance()
                if right_bracket.separator == "right_bracket":
                    pass
                elif (
                    right_bracket.kind == "separator"
                    and right_bracket.separator == "comma"
                ):
                    inner_begin = self.expression()
                    if inner_begin is None:
                        raise BCError(
                            "invalid expression as beginning value of array declaration",
                            inner_begin,
                        )

                    inner_colon = self.advance()
                    if (
                        inner_colon.kind != "separator"
                        and inner_colon.separator != "colon"
                    ):
                        raise BCError(
                            "expected colon after beginning value of array declaration",
                            inner_colon,
                        )

                    inner_end = self.expression()
                    if inner_end is None:
                        raise BCError(
                            "invalid expression as ending value of array declaration",
                            inner_end,
                        )

                    matrix_bounds = (
                        flat_bounds[0],
                        flat_bounds[1],
                        inner_begin,
                        inner_end,
                    )

                    flat_bounds = None

                    right_bracket = self.advance()
                    if right_bracket.separator != "right_bracket":
                        raise BCError(
                            "expected ending right bracket after matrix length declaration",
                            right_bracket,
                        )

                    is_matrix = True
                else:
                    raise BCError(
                        "expected right bracket or comma after array bounds declaration",
                        right_bracket,
                    )

            of = self.advance()
            if of.kind != "keyword" and of.keyword != "of":
                raise BCError("expected `OF` after `ARRAY` and/or size declaration", of)

            # TODO: refactor
            arrtyp = self.advance()

            if arrtyp.typ == "array":
                raise BCError(
                    "cannot have array as array element type, please use the matrix syntax instead",
                    arrtyp,
                )

            if arrtyp.typ not in PRIM_TYPES:
                raise BCError("invalid type used as array element type", arrtyp)

            inner = arrtyp.typ  # type: ignore

            return BCArrayType(
                is_matrix=is_matrix,
                flat_bounds=flat_bounds,
                matrix_bounds=matrix_bounds,
                inner=inner,
            )

    def ident(self) -> Expr:
        c = self.advance()

        return Identifier(c.ident)  # type: ignore

    def array_index(self) -> Expr | None:
        pn = self.peek_next()
        if pn.kind != "separator" and pn.separator != "left_bracket":
            return None

        ident = self.ident()

        leftb = self.advance()
        if leftb.separator != "left_bracket":
            raise BCError("expected left_bracket after ident in array index", leftb)

        exp = self.expression()
        if exp is None:
            raise BCError("expected expression as array index", exp)

        rightb = self.advance()
        exp_inner = None
        if rightb.separator == "right_bracket":
            pass
        elif rightb.separator == "comma":
            exp_inner = self.expression()
            if exp_inner is None:
                raise BCError("expected expression as array index", exp_inner)

            rightb = self.advance()
            if rightb.separator != "right_bracket":
                raise BCError(
                    "expected right_bracket after expression in array index", rightb
                )
        else:
            raise BCError(
                "expected right_bracket after expression in array index", rightb
            )

        return ArrayIndex(ident=ident, idx_outer=exp, idx_inner=exp_inner)  # type: ignore

    def operator(self) -> l.Operator | None:
        o = self.advance()
        return o.operator

    def function_call(self) -> Expr | None:
        # avoid consuming tokens
        ident = self.peek()
        if ident.kind != "ident":
            return None

        leftb = self.peek_next()
        if leftb.separator != "left_paren":
            return None

        self.advance()
        self.advance()

        args = []

        while self.peek().separator != "right_paren":
            expr = self.expression()
            if expr is None:
                raise BCError("invalid expression as function argument", expr)

            args.append(expr)

            comma = self.peek()
            if comma.separator != "comma" and comma.separator != "right_paren":
                raise BCError(
                    "expected comma after argument in function call argument list",
                    comma,
                )
            elif comma.separator == "comma":
                self.advance()

        rightb = self.advance()
        if rightb.separator != "right_paren":
            raise BCError(
                "expected right paren after arg list in function call", rightb
            )

        return FunctionCall(ident=ident.ident, args=args)  # type: ignore

    def unary(self) -> Expr | None:
        p = self.peek()
        if p.kind == "literal":
            return self.literal()
        elif p.kind == "ident":
            pn = self.peek_next()
            if pn.kind == "separator" and pn.separator == "left_bracket":
                return self.array_index()

            if pn.kind == "separator" and pn.separator == "left_paren":
                return self.function_call()

            return self.ident()
        elif p.kind == "separator" and p.separator == "left_paren":
            self.advance()
            e = self.expression()
            if e is None:
                raise BCError("invalid expression inside grouping", e)

            end = self.advance()

            if end.separator != "right_paren":
                raise BCError("expected ending ) delimiter after (", end)

            return Grouping(inner=e)
        elif p.kind == "operator" and p.operator == "sub":
            self.advance()
            e = self.expression()
            if e is None:
                raise BCError("invalid expression for negation", e)
            return Negation(e)
        elif p.kind == "keyword" and p.keyword == "not":
            self.advance()
            e = self.expression()
            if e is None:
                raise BCError("invalid expression for logical NOT", e)
            return Not(e)
        else:
            return None

    def factor(self) -> Expr | None:
        expr = self.unary()
        if expr is None:
            return None

        while self.match([("operator", "mul"), ("operator", "div")]):
            op = self.prev().operator

            if op is None:
                raise BCError("factor: op is None", op)

            right = self.unary()

            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if expr is None:
            return None

        while self.match([("operator", "add"), ("operator", "sub")]):
            op = self.prev().operator

            if op is None:
                raise BCError("term: no operator provided", op)

            right = self.factor()
            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def comparison(self) -> Expr | None:
        # > < >= <=
        expr = self.term()
        if expr is None:
            return None

        while self.match(
            [
                ("operator", "greater_than"),
                ("operator", "less_than"),
                ("operator", "greater_than_or_equal"),
                ("operator", "less_than_or_equal"),
            ]
        ):
            op = self.prev().operator
            if op is None:
                raise BCError("comparison: no operator provided", op)

            right = self.term()
            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def equality(self) -> Expr | None:
        expr = self.comparison()

        if expr is None:
            return None

        while self.match(
            [
                ("operator", "not_equal"),
                ("operator", "equal"),
            ]
        ):
            op = self.prev().operator
            if op is None:
                raise BCError("equality: no operator provided", op)

            right = self.comparison()
            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)

        return expr

    def logical_comparison(self) -> Expr | None:
        expr = self.equality()
        if expr is None:
            return None

        while self.match([("keyword", "and"), ("keyword", "or")]):
            kw = self.prev().keyword
            if kw is None:
                raise BCError("logical_comparison: no keyword provided", kw)

            right = self.equality()

            if right is None:
                return None

            op: Operator = ""  # type: ignore
            if kw == "and":
                op = "and"
            elif kw == "or":
                op = "or"

            expr = BinaryExpr(expr, op, right)  # kw must be and or or

        return expr

    def expression(self) -> Expr | None:
        return self.logical_comparison()

    def output_stmt(self) -> Statement | None:
        exprs = []
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "output":
            return None

        self.advance()
        initial = self.expression()
        if initial is None:
            raise BCError("found OUTPUT but no expression that follows", self.peek())

        exprs.append(initial)

        while self.match([("separator", "comma")]):
            new = self.expression()
            if new is None:
                break

            exprs.append(new)

        self.check_newline("OUTPUT")

        res = OutputStatement(items=exprs)
        return Statement("output", output=res)

    def input_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "input":
            return None

        self.advance()

        ident = self.ident()
        if not isinstance(ident, Identifier) or Identifier is None:
            raise BCError(
                f"expected identifier after `INPUT` but found {ident}", self.peek()
            )

        self.check_newline("INPUT")

        res = InputStatement(ident)
        return Statement("input", input=res)

    def return_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "return":
            return None

        self.advance()

        expr = self.expression()
        if expr is None:
            raise BCError("invalid expression used as RETURN expression", self.peek())

        return Statement("return", return_s=ReturnStatement(expr))

    def call_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "call":
            return

        self.advance()

        # CALL <ident>(<expr>, <expr>)
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid ident after procedure call", self.peek())

        leftb = self.peek()
        args = []
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            self.advance()
            while self.peek().separator != "right_paren":
                expr = self.expression()
                if expr is None:
                    raise BCError(
                        "invalid expression as procedure argument", self.peek()
                    )

                args.append(expr)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    raise BCError(
                        "expected comma after argument in procedure call argument list",
                        self.peek(),
                    )
                elif comma.separator == "comma":
                    self.advance()

            rightb = self.advance()
            if rightb.separator != "right_paren":
                raise BCError(
                    "expected right paren after arg list in procedure call", self.peek()
                )

        self.check_newline("procedure call")

        res = CallStatement(ident=ident.ident, args=args)
        return Statement("call", call=res)

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            export = True
            begin = self.peek_next()

        # combining the conditions does NOT WORK.
        if begin.keyword != "declare":
            return None

        # consume the keyword
        self.advance()
        if export == True:
            self.advance()

        ident = self.advance()
        if ident.ident is None:
            raise BCError("expected ident after declare stmt", self.peek())

        typ = None
        expr = None

        print(self.peek())
        if self.peek().separator == "colon":
            self.advance()
            
            typ = self.typ()
            if typ is None:
                raise BCError("invalid type after DECLARE", self.peek())
        
        print(self.peek())
        if self.peek().operator == "assign":
            self.advance()

            expr = self.expression()
            if expr is None:
                raise BCError("invalid expression after assign in declare", self.peek())
        
        print(f"{typ} {expr}")
        if typ is None and expr is None:
            raise BCError("must have either a type declaration, expression to assign as, or both")

        self.check_newline("variable declaration (DECLARE)")

        res = DeclareStatement(ident=Identifier(ident.ident), typ=typ, expr=expr, export=export)  # type: ignore
        return Statement("declare", declare=res)

    def constant_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            export = True

        if begin.kind != "keyword":
            return None

        if begin.keyword != "constant":
            return None

        # consume the kw
        self.advance()
        if export == True:
            self.advance()

        ident: Identifier | None = self.ident()  # type: ignore
        if ident.ident is None or not isinstance(ident, Identifier):  # type: ignore
            raise BCError("expected ident after constant stmt", self.peek())

        arrow = self.advance()
        if arrow.kind != "operator" and arrow.operator != "assign":
            raise BCError(
                "expected `<-` after variable name in constant declaration", self.peek()
            )

        literal: Literal | None = self.literal()  # type: ignore
        if literal is None:
            raise BCError(
                "expected literal after `<-` in constant declaration", self.peek()
            )

        self.check_newline("constant declaration (CONSTANT)")

        res = ConstantStatement(ident, literal, export=export)
        return Statement("constant", constant=res)

    def assign_stmt(self) -> Statement | None:
        p = self.peek_next()

        if p.separator == "left_bracket":
            temp_idx = self.cur
            while self.tokens[temp_idx].separator != "right_bracket":
                temp_idx += 1

            p = self.tokens[temp_idx + 1]

            if p.kind != "operator" and p.operator != "assign":
                return None
        elif p.operator != "assign":
            return None

        ident = self.array_index()
        if ident is None:
            ident = self.ident()

        self.advance()  # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            raise BCError("expected expression after `<-` in assignment", self.peek())

        self.check_newline("assignment")

        res = AssignStatement(ident, expr)  # type: ignore
        return Statement("assign", assign=res)

    # multiline statements go here
    def if_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "if":
            return

        self.advance()  # byebye `IF`

        cond = self.expression()
        if cond is None:
            raise BCError("found invalid expression for if condition", self.peek())

        # allow stupid igcse shit
        if self.peek().kind == "newline":
            self.clean_newlines()

        then = self.advance()
        if then.keyword != "then":
            raise BCError("expected `THEN` after if condition", self.peek())

        # dont enforce newline after then
        if self.peek().kind == "newline":
            self.clean_newlines()

        if_stmts = []
        else_stmts = []

        while self.peek().keyword not in ["else", "endif"]:
            if_stmts.append(self.scan_one_statement())

        if self.peek().keyword == "else":
            self.advance()  # byebye else

            # dont enforce newlines after else
            if self.peek().kind == "newline":
                self.clean_newlines()

            while self.peek().keyword != "endif":
                else_stmts.append(self.scan_one_statement())

        self.advance()  # byebye endif
        self.check_newline("ENDIF")

        res = IfStatement(cond=cond, if_block=if_stmts, else_block=else_stmts)
        return Statement("if", if_s=res)

    def caseof_stmt(self) -> Statement | None:
        case = self.peek()

        # if case.kind == "keyword" and case.keyword == "case":
        #    raise BCError("case of not implemented", self.peek())

        if case.keyword != "case":
            return
        self.advance()

        if self.peek().keyword != "of":
            return
        self.advance()

        main_expr = self.expression()
        if main_expr is None:
            raise BCError("found invalid expression for case of value", self.peek())

        self.check_newline("after case of expression")

        branches: list[CaseofBranch] = []
        otherwise: Statement | None = None
        next_expr: Expr | None = None
        while self.peek().keyword != "endcase":
            is_otherwise = self.peek().keyword == "otherwise"

            if not is_otherwise:
                expr = self.expression() if next_expr is None else next_expr
                if not expr:
                    raise BCError("invalid expression for case of branch", self.peek())

                colon = self.advance()
                if colon.separator != "colon":
                    raise BCError(
                        "expected colon after case of branch expression", self.prev()
                    )
            else:
                self.advance()

            stmt = self.stmt()

            if stmt is None:
                raise BCError("expected statement for case of branch block")

            if is_otherwise:
                otherwise = stmt
            else:
                branches.append(CaseofBranch(expr, stmt))  # type: ignore
        self.advance()

        res = CaseofStatement(main_expr, branches, otherwise)
        return Statement(kind="caseof", caseof=res)

    def while_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "while":
            return

        # byebye `WHILE`
        self.advance()

        expr = self.expression()
        if expr is None:
            raise BCError(
                "found invalid expression for while loop condition", self.peek()
            )

        do = self.advance()
        if do.keyword != "do":
            raise BCError("expected `DO` after while loop condition", self.peek())

        # oh dear
        self.check_newline("while loop declaration")

        if self.peek().kind == "newline":
            self.clean_newlines()

        stmts = []
        while self.peek().keyword != "endwhile":
            stmts.append(self.scan_one_statement())

        self.advance()  # byebye `ENDWHILE`

        self.check_newline("after while loop declaration")

        res = WhileStatement(expr, stmts)
        return Statement("while", while_s=res)

    def for_stmt(self):
        # FOR <counter> <assign> <lit> TO <lit> [STEP <lit>]
        initial = self.peek()

        if initial.keyword != "for":
            return

        self.advance()

        counter: Identifier | None = self.ident()  # type: ignore
        if counter.ident is None or not isinstance(counter, Identifier):  # type: ignore
            raise BCError("expected ident before `<-` in a for loop", self.peek())

        assign = self.advance()
        if assign.operator != "assign":
            raise BCError(
                "expected assignment operator `<-` after counter in a for loop", assign
            )

        begin = self.expression()
        if begin is None:
            raise BCError("invalid expression as begin in for loop", self.peek())

        to = self.advance()
        if to.keyword != "to":
            raise BCError("expected TO after beginning value in for loop", to)

        end = self.expression()
        if end is None:
            raise BCError("invalid expression as end in for loop", self.peek())

        step: Expr | None = None
        if self.peek().keyword == "step":
            self.advance()
            step = self.expression()
            if step is None:
                raise BCError("invalid expression as step in for loop", self.peek())

        self.check_newline("after for loop declaration")

        stmts = []
        while self.peek().keyword != "next":
            stmts.append(self.scan_one_statement())

        self.advance()  # byebye NEXT

        next_counter: Identifier | None = self.ident()  # type: ignore
        if next_counter is None or not isinstance(counter, Identifier):
            raise BCError("expected identifier after NEXT in a for loop", self.peek())
        elif counter.ident != next_counter.ident:
            raise BCError(
                f"initialized counter as {counter.ident} but used {next_counter.ident} after loop",
                self.prev(),
            )

        # thanks python for not having proper null handling
        res = ForStatement(counter=counter, block=stmts, begin=begin, end=end, step=step)  # type: ignore
        return Statement("for", for_s=res)

    def repeatuntil_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "repeat":
            return

        # byebye `REPEAT`
        self.advance()

        self.check_newline("repeat-until loop declaration")

        self.clean_newlines()

        stmts = []
        while self.peek().keyword != "until":
            stmts.append(self.scan_one_statement())

        self.advance()  # byebye `UNTIL`

        expr = self.expression()
        if expr is None:
            raise BCError(
                "found invalid expression for repeat-until loop condition", self.peek()
            )

        self.check_newline("after repeat-until loop declaration")

        res = RepeatUntilStatement(expr, stmts)
        return Statement("repeatuntil", repeatuntil=res)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier for function arg", self.peek())

        colon = self.advance()
        if colon.kind != "separator" and colon.separator != "colon":
            raise BCError(
                "expected colon after ident in function argument", self.peek()
            )

        typ = self.typ()
        if typ is None:
            raise BCError("invalid type after colon in function argument", self.peek())

        return FunctionArgument(name=ident.ident, typ=typ)

    def procedure_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            export = True

        if begin.keyword != "procedure":
            return

        self.advance()  # byebye PROCEDURE
        if export == True:
            self.advance()

        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier after PROCEDURE declaration", self.peek())

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.advance()
            while self.peek().separator != "right_paren":
                arg = self.function_arg()
                if arg is None:
                    raise BCError("invalid function argument", self.peek())

                args.append(arg)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    raise BCError(
                        "expected comma after procedure argument in list", self.peek()
                    )

                if comma.separator == "comma":
                    self.advance()

            rightb = self.advance()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                raise BCError(
                    f"expected right paren after arg list in procedure declaration, found {rightb}",
                    self.peek(),
                )

        self.check_newline("PROCEDURE")

        stmts = []
        while self.peek().keyword != "endprocedure":
            stmts.append(self.scan_one_statement())

        self.advance()  # bye bye ENDPROCEDURE

        res = ProcedureStatement(
            name=ident.ident, args=args, block=stmts, export=export
        )
        return Statement("procedure", procedure=res)

    def function_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            export = True

        if begin.keyword != "function":
            return None

        self.advance()  # byebye FUNCTION
        if export == True:
            self.advance()

        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier after FUNCTION declaration", self.peek())

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.advance()
            while self.peek().separator != "right_paren":
                arg = self.function_arg()
                if arg is None:
                    raise BCError("invalid function argument", self.peek())

                args.append(arg)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    raise BCError(
                        "expected comma after function argument in list", self.peek()
                    )

                if comma.separator == "comma":
                    self.advance()

            rightb = self.advance()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                raise BCError(
                    f"expected right paren after arg list in function declaration, found {rightb}",
                    self.peek(),
                )

        returns = self.advance()
        if returns.keyword != "returns":
            raise BCError("expected RETURNS after function arguments", self.peek())

        typ = self.typ()
        if typ is None:
            raise BCError(
                "invalid type after RETURNS for function return value", self.peek()
            )

        self.check_newline("FUNCTION")

        stmts = []
        while self.peek().keyword != "endfunction":
            stmt = self.scan_one_statement()
            stmts.append(stmt)

        self.advance()  # bye bye ENDFUNCTION

        res = FunctionStatement(
            name=ident.ident, args=args, returns=typ, block=stmts, export=export
        )
        return Statement("function", function=res)

    def scope_stmt(self) -> Statement | None:
        scope = self.peek()
        if scope.keyword != "scope":
            return
        self.advance()

        self.check_newline("manual scope declaration")

        self.clean_newlines()

        stmts = []
        while self.peek().keyword != "endscope":
            stmts.append(self.scan_one_statement())

        self.advance()
        self.check_newline("after scope declaration end")
        res = ScopeStatement(stmts)
        return Statement("scope", scope=res)

    def include_stmt(self) -> Statement | None:
        include = self.peek()
        if include.keyword != "include":
            return
        self.advance()

        name = self.advance()
        if name.kind != "literal":
            raise BCError("Include must be followed by a literal of the name of the file to include")

        if name.literal.kind != "string": # type: ignore
            raise BCError("literal for include must be a string!")

        res = IncludeStatement(name.literal.value) # type: ignore
        return Statement("include", include=res)

    def clean_newlines(self):
        while self.cur < len(self.tokens) and self.peek().kind == "newline":
            self.advance()

    def stmt(self) -> Statement | None:
        assign = self.assign_stmt()
        if assign is not None:
            return assign

        constant = self.constant_stmt()
        if constant is not None:
            return constant

        output = self.output_stmt()
        if output is not None:
            return output

        inp = self.input_stmt()
        if inp is not None:
            return inp

        proc_call = self.call_stmt()
        if proc_call is not None:
            return proc_call

        fncall: FunctionCall | None = self.function_call()  # type: ignore
        if fncall is not None:
            return Statement("fncall", fncall=fncall)

        return_s = self.return_stmt()
        if return_s is not None:
            return return_s

        include = self.include_stmt()
        if include is not None:
            return include

        declare = self.declare_stmt()
        if declare is not None:
            return declare

        if_s = self.if_stmt()
        if if_s is not None:
            return if_s

        caseof = self.caseof_stmt()
        if caseof is not None:
            return caseof

        while_s = self.while_stmt()
        if while_s is not None:
            return while_s

        for_s = self.for_stmt()
        if for_s is not None:
            return for_s

        repeatuntil_s = self.repeatuntil_stmt()
        if repeatuntil_s is not None:
            return repeatuntil_s

        procedure = self.procedure_stmt()
        if procedure is not None:
            return procedure

        function = self.function_stmt()
        if function is not None:
            return function

        scope = self.scope_stmt()
        if scope is not None:
            return scope

        cur = self.peek()
        expr = self.expression()
        if expr:
            raise BCWarning("unused expression", cur, data=expr)
        else:
            raise BCError("invalid statement or expression", cur)

    def scan_one_statement(self) -> Statement:
        s = self.stmt()

        if s is not None:
            self.clean_newlines()
            return s
        else:
            p = self.peek()
            raise BCError(f"found invalid statement at `{p}`", p)

    def program(self) -> tuple[Program, list[BCWarning]]:
        stmts = []
        warnings = []

        try:
            while self.cur < len(self.tokens):
                self.clean_newlines()
                stmt = self.scan_one_statement()
                stmts.append(stmt)
        except BCWarning as warn:
            warnings.append(warn)

        return (Program(stmts=stmts), warnings)
