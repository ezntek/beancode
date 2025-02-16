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
    matrix_bounds: tuple["Expr", "Expr", "Expr", "Expr"] | None = None # begin:end,begin:end

    def has_bounds(self) -> bool:
        return self.flat_bounds is not None or self.matrix_bounds is not None

    def get_flat_bounds(self) -> tuple["Expr", "Expr"]:
        if self.flat_bounds is None:
            panic("tried to access flat bounds on array without flat bounds")
        return self.flat_bounds

    def get_matrix_bounds(self) -> tuple["Expr", "Expr", "Expr", "Expr"]:
        if self.matrix_bounds is None:
            panic("tried to access matrixbounds on array without matrix bounds")
        return self.matrix_bounds


@dataclass
class BCArray:
    typ: BCArrayType
    flat: list["BCValue"] | None = None # must be the BCPrimitiveType
    matrix: list[list["BCValue"]] | None = None  # must be the BCPrimitiveType
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

    @classmethod
    def empty(cls, kind: BCType) -> 'BCValue':
        return cls(kind, integer=None, real=None, char=None, string=None, boolean=None, array=None)

    def get_integer(self) -> int:
        if self.kind != "integer":
            panic(f"tried to access integer value from BCValue of {self.kind}")

        return self.integer  # type: ignore

    def get_real(self) -> float:
        if self.kind != "real":
            panic(f"tried to access real value from BCValue of {self.kind}")

        return self.real  # type: ignore

    def get_char(self) -> str:
        if self.kind != "char":
            panic(f"tried to access char value from BCValue of {self.kind}")

        return self.char  # type: ignore

    def get_string(self) -> str:
        if self.kind != "string":
            panic(f"tried to access string value from BCValue of {self.kind}")

        return self.string  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != "boolean":
            panic(f"tried to access boolean value from BCValue of {self.kind}")

        return self.boolean  # type: ignore

    def get_array(self) -> BCArray:
        if self.kind != "array":
            panic(f"tried to access array value from BCValue of {self.kind}")

        return self.array  # type: ignore

    def __repr__(self) -> str: # type: ignore
        if isinstance(self.kind, BCArrayType):
            panic("BCValue of array can only be represented at runtime")

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
                return "(uninitialized)"
    

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
    "while",
    "for",
    "repeatuntil",
    "function",
    "procedure",
    "call",
    "fncall",
    "return",
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
class DeclareStatement:
    ident: Identifier
    typ: BCType


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

@dataclass
class FunctionStatement:
    name: str
    args: list[FunctionArgument]
    returns: BCType
    block: list["Statement"]

@dataclass
class ReturnStatement:
    expr: Expr | None

@dataclass
class Statement:
    kind: StatementKind
    declare: DeclareStatement | None = None
    output: OutputStatement | None = None
    input: InputStatement | None = None
    constant: ConstantStatement | None = None
    assign: AssignStatement | None = None
    if_s: IfStatement | None = None
    while_s: WhileStatement | None = None
    for_s: ForStatement | None = None
    repeatuntil: RepeatUntilStatement | None = None
    function: FunctionStatement | None = None
    procedure: ProcedureStatement | None = None
    call: CallStatement | None = None
    fncall: FunctionCall | None = None # Impostor! expr as statement?!
    return_s: ReturnStatement | None = None

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

@dataclass
class Program:
    stmts: list[Statement]


class Parser:
    tokens: list[l.Token]
    cur: int

    def __init__(self, tokens: list[l.Token]) -> None:
        self.cur = 0
        self.tokens = tokens

    def check(self, tok: l.Token) -> bool:
        if self.cur == len(self.tokens):
            return False

        peek = self.peek()
        if tok.kind != peek.kind:
            return False

        match tok.kind:
            case "type":
                return tok.typ == peek.typ
            case "ident":
                return tok.ident == peek.ident
            case "keyword":
                return tok.keyword == peek.keyword
            case "literal":
                return tok.literal == peek.literal
            case "operator":
                return tok.operator == peek.operator
            case "separator":
                return tok.separator == peek.separator

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
            panic(f"expected newline after {s}, but found `{self.prev()}`")

    def prev(self) -> l.Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> l.Token:
        return self.tokens[self.cur]

    def peek_next(self) -> l.Token:
        return self.tokens[self.cur + 1]

    def match(self, typs: list[l.Token]) -> bool:
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
                if len(val) > 1:
                    panic(f"more than 1 character in char literal `{lit}`")
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
                    panic(f"invalid boolean literal `{lit.value}`")
            case "number":
                val = lit.value

                if self.is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        panic(f"invalid number literal `{val}`")

                    return Literal("real", real=res)
                elif self.is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        panic(f"invalid number literal `{val}`")

                    return Literal("integer", integer=res)
                else:
                    panic(f"invalid number literal `{val}`")

    def type(self) -> BCType | None:
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
                    panic("invalid expression as beginning value of array declaration")

                colon = self.advance()
                if colon.kind != "separator" and colon.separator != "colon":
                    panic("expected colon after beginning value of array declaration")

                end = self.expression()
                if end is None:
                    panic("invalid expression as ending value of array declaration")

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
                        panic(
                            "invalid expression as beginning value of array declaration"
                        )

                    inner_colon = self.advance()
                    if (
                        inner_colon.kind != "separator"
                        and inner_colon.separator != "colon"
                    ):
                        panic(
                            "expected colon after beginning value of array declaration"
                        )

                    inner_end = self.expression()
                    if inner_end is None:
                        panic("invalid expression as ending value of array declaration")

                    matrix_bounds = (
                        flat_bounds[0],
                        flat_bounds[1],
                        inner_begin,
                        inner_end,
                    )

                    flat_bounds = None

                    right_bracket = self.advance()
                    if right_bracket.separator != "right_bracket":
                        panic("expected ending right bracket after matrix length declaration")
                    
                    is_matrix = True
                else:
                    panic(
                        "expected right bracket or comma after array bounds declaration"
                    )

            of = self.advance()
            if of.kind != "keyword" and of.keyword != "of":
                panic("expected `OF` after `ARRAY` and/or size declaration")

            # TODO: refactor
            arrtyp = self.advance()

            if arrtyp.typ == "array":
                panic(
                    "cannot have array as array element type, please use the matrix syntax instead"
                )

            if arrtyp.typ not in PRIM_TYPES:
                panic("invalid type used as array element type")

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
            panic("expected left_bracket after ident in array index")

        exp = self.expression()
        if exp is None:
            panic("expected expression as array index")

        rightb = self.advance()
        exp_inner = None
        if rightb.separator == "right_bracket":
            pass
        elif rightb.separator == "comma":
            exp_inner = self.expression()
            if exp_inner is None:
                panic("expected expression as array index")

            rightb = self.advance()
            if rightb.separator != "right_bracket":
                panic("expected right_bracket after expression in array index")
        else:
            panic("expected right_bracket after expression in array index")

        return ArrayIndex(ident=ident, idx_outer=exp, idx_inner=exp_inner) # type: ignore 

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
        
        if self.peek_next().separator != "right_paren":
            while self.peek().separator != "right_paren":
                expr = self.expression()
                if expr is None:
                    panic("invalid expression as function argument")

                args.append(expr)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    panic("expected comma after argument in function call argument list")
                elif comma.separator == "comma":
                    self.advance()

            rightb = self.advance()
            if rightb.separator != "right_paren":
                panic("expected right paren after arg list in function call")
        
        return FunctionCall(ident=ident.ident, args=args) # type: ignore

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
                panic("invalid expression inside grouping")

            end = self.advance()

            if end != l.Token("separator", separator="right_paren"):
                panic("expected ending ) delimiter after (")

            return Grouping(inner=e)
        elif p.kind == "operator" and p.operator == "sub":
            self.advance()
            e = self.expression()
            if e is None:
                panic("invalid expression for negation")
            return Negation(e)
        elif p.kind == "keyword" and p.keyword == "not":
            self.advance()
            e = self.expression()
            if e is None:
                panic("invalid expression for logical NOT")
            return Not(e)
        else:
            return None

    def factor(self) -> Expr | None:
        expr = self.unary()
        if expr is None:
            return None

        while self.match(
            [l.Token("operator", operator="mul"), l.Token("operator", operator="div")]
        ):
            op = self.prev().operator

            if op is None:
                panic("factor: op is None")

            right = self.unary()

            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if expr is None:
            return None

        while self.match(
            [l.Token("operator", operator="add"), l.Token("operator", operator="sub")]
        ):
            op = self.prev().operator

            if op is None:
                panic("term: op is is None")

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
                l.Token("operator", operator="greater_than"),
                l.Token("operator", operator="less_than"),
                l.Token("operator", operator="greater_than_or_equal"),
                l.Token("operator", operator="less_than_or_equal"),
            ]
        ):
            op = self.prev().operator
            if op is None:
                panic("comparison: op is None")

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
                l.Token("operator", operator="not_equal"),
                l.Token("operator", operator="equal"),
            ]
        ):
            op = self.prev().operator
            if op is None:
                panic("equality: op is None")

            right = self.comparison()
            if right is None:
                return None

            expr = BinaryExpr(expr, op, right)

        return expr

    def logical_comparison(self) -> Expr | None:
        expr = self.equality()
        if expr is None: return None

        while self.match([l.Token("keyword", keyword="and"), l.Token("keyword", keyword="or")]):
            kw = self.prev().keyword
            if kw is None:
                panic("logical_comparison: kw is None")

            right = self.equality()

            if right is None:
                return None

            op: Operator = "" # type: ignore
            if kw == "and":
                op = "and"
            elif kw == "or":
                op = "or"

            expr = BinaryExpr(expr, op, right) # kw must be and or or

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
            panic("found OUTPUT but no expression that follows")

        exprs.append(initial)

        comma = l.Token(kind="separator", separator="comma")
        while self.match([comma]):
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
            panic(f"expected identifier after `INPUT` but found {ident}")

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
            panic("invalid expression used as RETURN expression")

        return Statement("return", return_s=ReturnStatement(expr))

    def call_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "call":
            return

        self.advance()

        # CALL <ident>(<expr>, <expr>)
        ident = self.ident()
        if not isinstance(ident, Identifier):
            panic("invalid ident after procedure call")

        leftb = self.advance()
        args = []
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            while self.peek().separator != "right_paren":
                expr = self.expression()
                if expr is None:
                    panic("invalid expression as procedure argument")

                args.append(expr)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    panic("expected comma after argument in procedure call argument list")
                elif comma.separator == "comma":
                    self.advance()

            rightb = self.advance()
            if rightb.separator != "right_paren":
                panic("expected right paren after arg list in procedure call")

        self.check_newline("procedure call")

        res = CallStatement(ident=ident.ident, args=args)
        return Statement("call", call=res)

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()

        # combining the conditions does NOT WORK.
        if begin.kind != "keyword":
            return None

        if begin.keyword != "declare":
            return None

        # consume the keyword
        self.advance()

        ident = self.advance()
        if ident.ident is None:
            panic("expected ident after declare stmt")

        colon = self.advance()
        if colon.kind != "separator" and colon.separator != "colon":
            panic("expected colon `:` after ident in declare")

        typ = self.type()
        if typ is None:
            panic("invalid type after DECLARE")

        self.check_newline("variable declaration (DECLARE)")

        res = DeclareStatement(ident=Identifier(ident.ident), typ=typ)  # type: ignore
        return Statement("declare", declare=res)

    def constant_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "constant":
            return None

        # consume the kw
        self.advance()

        ident: Identifier | None = self.ident()  # type: ignore
        if ident.ident is None or not isinstance(ident, Identifier):  # type: ignore
            panic("expected ident after constant stmt")

        arrow = self.advance()
        if arrow.kind != "operator" and arrow.operator != "assign":
            panic("expected `<-` after variable name in constant declaration")

        literal: Literal | None = self.literal()  # type: ignore
        if literal is None:
            panic("expected literal after `<-` in constant declaration")

        self.check_newline("constant declaration (CONSTANT)")

        res = ConstantStatement(ident, literal)
        return Statement("constant", constant=res)

    def assign_stmt(self) -> Statement | None:
        p = self.peek_next()

        if p.kind == "separator" and p.separator == "left_bracket":
            temp_idx = self.cur
            while self.tokens[temp_idx].separator != "right_bracket":
                temp_idx += 1

            p = self.tokens[temp_idx + 1]

            if p.kind != "operator" and p.operator != "assign":
                return None
        elif p.kind != "operator" and p.operator != "assign":
            return None

        ident = self.array_index()
        if ident is None:
            ident = self.ident()

        self.advance()  # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            panic("expected expression after `<-` in assignment")

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
            panic("found invalid expression for if condition")

        # allow stupid igcse shit
        if self.peek().kind == "newline":
            self.clean_newlines()

        then = self.advance()
        if then.keyword != "then":
            panic("expected `THEN` after if condition")

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

    def caseof_stmt(self) -> None:
        case = self.peek()

        if case.kind == "keyword" and case.keyword == "case":
            panic("case of not implemented")

    def while_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "while":
            return

        # byebye `WHILE`
        self.advance()

        expr = self.expression()
        if expr is None:
            panic("found invalid expression for while loop condition")

        do = self.advance()
        if do.keyword != "do":
            panic("expected `DO` after while loop condition")

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
            panic("expected ident before `<-` in a for loop")

        assign = self.advance()
        if assign.operator != "assign":
            panic("expected assignment operator `<-` after counter in a for loop")

        begin = self.expression()
        if begin is None:
            panic("invalid expression as begin in for loop")

        to = self.advance()
        if to.keyword != "to":
            panic("expected TO after beginning value in for loop")

        end = self.expression()
        if end is None:
            panic("invalid expression as end in for loop")

        step: Expr | None = None
        if self.peek().keyword == "step":
            self.advance()
            step = self.expression()
            if step is None:
                panic("invalid expression as step in for loop")

        self.check_newline("after for loop declaration")

        stmts = []
        while self.peek().keyword != "next":
            stmts.append(self.scan_one_statement())

        self.advance()  # byebye NEXT

        next_counter: Identifier | None = self.ident()  # type: ignore
        if next_counter is None or not isinstance(counter, Identifier):
            panic("expected identifier after NEXT in a for loop")
        elif counter.ident != next_counter.ident:
            panic(
                f"initialized counter as {counter.ident} but used {next_counter.ident} after loop"
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
            panic("found invalid expression for repeat-until loop condition")

        self.check_newline("after repeat-until loop declaration")

        res = RepeatUntilStatement(expr, stmts)
        return Statement("repeatuntil", repeatuntil=res)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.ident()
        if not isinstance(ident, Identifier):
            panic("invalid identifier for function arg")

        colon = self.advance()
        if colon.kind != "separator" and colon.separator != "colon":
            panic("expected colon after ident in function argument")

        typ = self.type()
        if typ is None:
            panic("invalid type after colon in function argument")

        return FunctionArgument(name=ident.ident, typ=typ)

    def procedure_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "procedure":
            return None

        self.advance() # byebye PROCEDURE
        
        ident = self.ident()
        if not isinstance(ident, Identifier):
            panic("invalid identifier after PROCEDURE declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.advance()
            while self.peek().separator != "right_paren":
                arg = self.function_arg()
                if arg is None:
                    panic("invalid function argument")
                
                args.append(arg)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    panic("expected comma after procedure argument in list")
                
                if comma.separator == "comma":
                    self.advance()
            
            rightb = self.advance()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                panic(f"expected right paren after arg list in procedure declaration, found {rightb}")

        self.check_newline("PROCEDURE")
 
        stmts = []
        while self.peek().keyword != "endprocedure":
            stmts.append(self.scan_one_statement())

        self.advance() # bye bye ENDPROCEDURE

        res = ProcedureStatement(name=ident.ident, args=args, block=stmts)
        return Statement("procedure", procedure=res)

    def function_stmt(self) -> Statement | None:
        begin = self.peek()        

        if begin.keyword != "function":
            return None

        self.advance() # byebye FUNCTION
        
        ident = self.ident()
        if not isinstance(ident, Identifier):
            panic("invalid identifier after FUNCTION declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.advance()
            while self.peek().separator != "right_paren":
                arg = self.function_arg()
                if arg is None:
                    panic("invalid function argument")
                
                args.append(arg)

                comma = self.peek()
                if comma.separator != "comma" and comma.separator != "right_paren":
                    panic("expected comma after function argument in list")
                
                if comma.separator == "comma":
                    self.advance()
            
            rightb = self.advance()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                panic(f"expected right paren after arg list in function declaration, found {rightb}")

        returns = self.advance()
        if returns.keyword != "returns":
            panic("expected RETURNS after function arguments")

        typ = self.type()
        if typ is None:
            panic("invalid type after RETURNS for function return value")

        self.check_newline("FUNCTION")
 
        stmts = []
        returned = False
        while self.peek().keyword != "endfunction":
            stmt = self.scan_one_statement()
            # avoid writing to the variable for every statement
            if isinstance(stmt, ReturnStatement):
                returned = True
            stmts.append(stmt)

        if not returned:
            panic("functions must return at least one value once!")
        
        self.advance() # bye bye ENDFUNCTION

        res = FunctionStatement(name=ident.ident, args=args, returns=typ, block=stmts)
        return Statement("function", function=res)

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

        fncall: FunctionCall | None = self.function_call() # type: ignore
        if fncall is not None:
            return Statement("fncall", fncall=fncall) 

        return_s = self.return_stmt()
        if return_s is not None:
            return return_s

        declare = self.declare_stmt()
        if declare is not None:
            return declare

        if_s = self.if_stmt()
        if if_s is not None:
            return if_s

        self.caseof_stmt()

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

    def scan_one_statement(self) -> Statement:
        s = self.stmt()

        if s is not None:
            self.clean_newlines()
            return s
        else:
            panic(f"found invalid statement at `{self.peek()}`")

    def program(self) -> Program:
        stmts = []

        while self.cur < len(self.tokens):
            self.clean_newlines()
            stmt = self.scan_one_statement()
            stmts.append(stmt)

        return Program(stmts=stmts)
