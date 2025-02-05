from typing import NoReturn
import typing as t
import lexer as l
from dataclasses import dataclass


def panic(s: str) -> NoReturn:
    print(s)
    exit(1)


@dataclass
class Expr:
    pass


@dataclass
class Grouping(Expr):
    inner: Expr


@dataclass
class Identifier(Expr):
    ident: str


@dataclass
class BinaryExpr(Expr):
    lhs: Expr
    op: l.Operator
    rhs: Expr


Type = t.Literal["integer", "real", "char", "string", "boolean"]


@dataclass
class Literal(Expr):
    kind: Type
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None
    boolean: bool | None = None
    # array not implemented


StatementKind = t.Literal["declare", "output", "constant", "assign"]


@dataclass
class DeclareStatement:
    ident: str
    typ: Type


@dataclass
class OutputStatement:
    items: list[Expr]

@dataclass
class ConstantStatement:
    ident: str
    value: Literal

@dataclass
class AssignStatement:
    ident: str
    value: Expr

@dataclass
class Statement:
    kind: StatementKind
    declare: DeclareStatement | None = None
    output: OutputStatement | None = None
    constant: ConstantStatement | None = None
    assign: AssignStatement | None = None

    def __repr__(self) -> str:
        match self.kind:
            case "declare":
                return self.declare.__repr__()
            case "output":
                return self.output.__repr__()
            case "constant":
                return self.constant.__repr__()
            case "assign":
                return self.assign.__repr__()


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
            case "array":
                panic("not implemented")
            case "char":
                val = lit.value
                if len(val) > 1:
                    panic(f"more than 1 character in char literal `{lit}`")
                return Literal("char", char=val[0])
            case "string":
                val = lit.value
                return Literal("string", string=val)
            case "boolean":
                val = lit.value
                if val == "TRUE":
                    return Literal("boolean", boolean=True)
                elif val == "FALSE":
                    return Literal("boolean", boolean=False)
                else:
                    panic(f"invalid boolean literal `{val}`")
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

    def ident(self) -> Expr:
        c = self.advance()

        return Identifier(c.ident)  # type: ignore

    def operator(self) -> l.Operator | None:
        o = self.advance()
        return o.operator

    def unary(self) -> Expr | None:
        p = self.peek()
        if p.kind == "literal":
            return self.literal()
        elif p.kind == "ident":
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
                print("factor: op is None")

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
                print("term: op is is None")

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
                print("comparison: op is None")

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

    def expression(self) -> Expr | None:
        return self.equality()

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

        print(self.peek())
        self.check_newline("OUTPUT")

        res = OutputStatement(items=exprs)
        return Statement("output", output=res)

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

        typ = self.advance()

        if typ.kind != "type":
            panic("expected type after `:` in declare")

        self.check_newline("variable declaration (DECLARE)")

        res = DeclareStatement(ident=Identifier(ident.ident), typ=typ.typ)  # type: ignore
        return Statement("declare", declare=res)

    def constant_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None
            
        if begin.keyword != "constant":
            return None

        # consume the kw
        self.advance()

        ident: Identifier | None = self.ident() # type: ignore
        if ident.ident is None or not isinstance(ident, Identifier): # type: ignore
            panic("expected ident after constant stmt")

        arrow = self.advance()
        if arrow.kind != "operator" and arrow.operator != "assign":
            panic("expected `<-` after variable name in constant declaration")

        literal: Literal | None = self.literal() # type: ignore
        if literal is None:
            panic("expected literal after `<-` in constant declaration")

        self.check_newline("constant declaration (CONSTANT)")

        res = ConstantStatement(ident.ident, literal)
        return Statement("constant", constant=res) 

    def assign_stmt(self) -> Statement | None:
        p = self.peek_next()

        if p.kind != "operator" and p.operator != "assign":
            return None

        ident: Identifier | None = self.ident() # type: ignore
        if ident.ident is None or not isinstance(ident, Identifier): # type: ignore
            panic("expected ident before `<-` in assignment")

        self.advance() # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            panic("expected expression after `<-` in assignment")

        self.check_newline("assignment")

        res = AssignStatement(ident.ident, expr)
        return Statement("assign", assign=res)

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

        declare = self.declare_stmt()
        if declare is not None:
            return declare


    def program(self) -> Program:
        stmts = []

        while self.cur < len(self.tokens):
            print(f"begin {self.peek()}")
            s = self.stmt()

            if s is not None:
                self.clean_newlines()
                stmts.append(s)
            else:
                print(f"found invalid statement at `{self.peek()}`")
                break

        return Program(stmts=stmts)
