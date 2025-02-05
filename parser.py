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


StatementKind = t.Literal["declare", "output"]


@dataclass
class DeclareStatement:
    name: str
    typ: Type


@dataclass
class OutputStatement:
    items: list[Expr]


@dataclass
class Statement:
    kind: StatementKind
    declare: DeclareStatement | None = None
    output: OutputStatement | None = None

    def __repr__(self) -> str:
        match self.kind:
            case "declare":
                return self.declare.__repr__()
            case "output":
                return self.output.__repr__()


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

    def consume(self) -> l.Token:
        if self.cur < len(self.tokens):
            self.cur += 1

        return self.prev()

    def prev(self) -> l.Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> l.Token:
        return self.tokens[self.cur]

    def match(self, typs: list[l.Token]) -> bool:
        for typ in typs:
            if self.check(typ):
                self.consume()
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

    def literal(self) -> Expr:
        c = self.consume()

        if c.kind != "literal":
            panic("passed nonliteral to literal function")

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
        c = self.consume()

        return Identifier(c.ident)  # type: ignore

    def operator(self) -> l.Operator | None:
        o = self.consume()
        return o.operator

    def unary(self) -> Expr | None:
        p = self.peek()
        if p.kind == "literal":
            return self.literal()
        elif p.kind == "ident":
            return self.ident()
        elif p.kind == "separator" and p.separator == "left_paren":
            self.consume()
            e = self.expression()
            if e == None:
                panic("invalid expression inside grouping")

            end = self.consume()

            if end != l.Token("separator", separator="right_paren"):
                panic("expected ending ) delimiter after (")

            return Grouping(inner=e)
        else:
            return None

    def factor(self) -> Expr | None:
        expr = self.unary()
        if expr == None:
            return None

        while self.match(
            [l.Token("operator", operator="mul"), l.Token("operator", operator="div")]
        ):
            op = self.prev().operator

            if op == None:
                print("factor: op is None")

            right = self.unary()

            if right == None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if expr == None:
            return None

        while self.match(
            [l.Token("operator", operator="add"), l.Token("operator", operator="sub")]
        ):
            op = self.prev().operator

            if op == None:
                print("term: op is is None")

            right = self.factor()
            if right == None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def comparison(self) -> Expr | None:
        # > < >= <=
        expr = self.term()
        if expr == None:
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
            if op == None:
                print("comparison: op is None")

            right = self.term()
            if right == None:
                return None

            expr = BinaryExpr(expr, op, right)  # type: ignore

        return expr

    def equality(self) -> Expr | None:
        expr = self.comparison()

        if expr == None:
            return None

        while self.match(
            [
                l.Token("operator", operator="not_equal"),
                l.Token("operator", operator="equal"),
            ]
        ):
            op = self.prev().operator
            if op == None:
                panic("equality: op is None")

            right = self.comparison()
            if right == None:
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

        self.consume()
        initial = self.expression()
        if initial == None:
            panic("found OUTPUT but no expression that follows")

        exprs.append(initial)

        comma = l.Token(kind="separator", separator="comma")
        while self.match([comma]):
            new = self.expression()
            if new == None:
                break

            exprs.append(new)

        res = OutputStatement(items=exprs)
        return Statement("output", output=res)

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "declare":
            return None

        # consume the keyword
        self.consume()

        ident = self.consume()
        if ident.ident == None:
            panic("expected ident after declare stmt")

        colon = self.consume()

        if colon.kind != "separator" and colon.separator != "colon":
            panic("expected colon `:` after ident in declare")

        typ = self.consume()

        if typ.kind != "type":
            panic("expected type after `:` in declare")

        res = DeclareStatement(name=Identifier(ident.ident), typ=typ.typ)  # type: ignore
        return Statement("declare", declare=res)

    def stmt(self) -> Statement | None:
        output = self.output_stmt()
        if output != None:
            return output

        declare = self.declare_stmt()
        if declare != None:
            return declare

    def program(self) -> Program:
        stmts = []

        while self.cur < len(self.tokens):
            s = self.stmt()

            if s == None:
                print(f"found invalid statement at `{self.peek()}`")
                continue

            stmts.append(s)

        return Program(stmts=stmts)
