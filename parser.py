from typing import NoReturn
import lexer as l
from dataclasses import dataclass

def panic(s: str) -> NoReturn:
    print(s)
    exit(1)

@dataclass
class Expr:
    pass

@dataclass
class Identifier(Expr):
    ident: str

@dataclass
class BinaryExpr(Expr):
    lhs: Expr
    op: l.Operator
    rhs: Expr

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
        return self.tokens[self.cur-1]

    def peek(self) -> l.Token:
        return self.tokens[self.cur]

    def match(self, typs: list[l.Token]) -> bool:
        for typ in typs:
            if self.check(typ):
                self.consume()
                return True
        return False

    def ident(self) -> Expr:
        c = self.consume()

        if c.ident == None:
            panic("aeaeaea")
        
        return Identifier(c) # type: ignore
         
    def operator(self) -> l.Operator | None:
        o = self.consume()
        return o.operator

    def term(self) -> Expr:
        return self.ident()

    def comparison(self) -> Expr:
        # > < >= <=
        expr: Expr = self.term()

        while self.match([l.Token("operator",operator="greater_than"), l.Token("operator",operator="less_than"), l.Token("operator",operator="greater_than_or_equal"), l.Token("operator",operator="less_than_or_equal")]):
            op = self.prev().operator
            if op == None:
                print("comparison: op is None")

            right = self.term()
            expr = BinaryExpr(expr, op, right)

        return expr

    def equality(self) -> Expr:
        expr: Expr = self.comparison()

        if expr == None:
            panic(f"malformed expression: expected ident got {expr}")

        while self.match([l.Token("operator", operator="not_equal"), l.Token("operator", operator="equal")]):
            op = self.prev().operator
            if op == None:
                panic("op is None")

            right = self.comparison()
            expr = BinaryExpr(expr, op, right)

        return expr

    def expression(self) -> Expr:
        e = self.equality() 
        if e == None:
            panic("whoops malformed")
        
        return e # type: ignore
