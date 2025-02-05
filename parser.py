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
class Identifier(Expr):
    ident: str

@dataclass
class BinaryExpr(Expr):
    lhs: Expr
    op: l.Operator
    rhs: Expr

@dataclass
class Literal(Expr):
    kind: t.Literal["integer", "real", "char", "string", "boolean"]
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None 
    boolean: bool | None = None
    # array not implemented

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

    def is_integer(self, val: str) -> bool:
        for ch in val:
            if not ch.isdigit() and ch != '_':
                return False
        return True

    def is_real(self, val: str) -> bool:
        if not self.is_integer(val):
            return False

        found_decimal = False

        for ch in val:
            if ch == '.' and found_decimal:
                return False
            elif ch == '.':
                found_decimal = True
        
        return found_decimal

    def literal(self) -> Expr:
        c = self.consume()

        if c.kind != "literal":
            panic("passed nonliteral to literal function")

        lit: l.Literal
        lit = c.literal # type: ignore
        
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
        
        return Identifier(c) # type: ignore
         
    def operator(self) -> l.Operator | None:
        o = self.consume()
        return o.operator

    def unary(self) -> Expr:
        k = self.peek().kind
        if k == "literal":
            return self.literal()
        elif k == "ident":
            return self.ident()
        else:
            panic("expected ident or literal, found garbage")

    def factor(self) -> Expr:
        expr: Expr = self.unary()

        while self.match([l.Token("operator", operator="mul"), l.Token("operator", operator="div")]):
            op = self.prev().operator

            if op == None:
                print("factor: op is None")

            right = self.unary()
            expr = BinaryExpr(expr, op, right) # type: ignore

        return expr
    def term(self) -> Expr:
        expr: Expr = self.factor()

        while self.match([l.Token("operator", operator="add"), l.Token("operator", operator="sub")]):
            op = self.prev().operator

            if op == None:
                print("term: op is is None")

            right = self.factor()
            expr = BinaryExpr(expr, op, right) # type: ignore

        return expr

    def comparison(self) -> Expr:
        # > < >= <=
        expr: Expr = self.term() 

        while self.match([l.Token("operator",operator="greater_than"), l.Token("operator",operator="less_than"), l.Token("operator",operator="greater_than_or_equal"), l.Token("operator",operator="less_than_or_equal")]):
            op = self.prev().operator
            if op == None:
                print("comparison: op is None")

            right = self.term()
            expr = BinaryExpr(expr, op, right) # type: ignore

        return expr

    def equality(self) -> Expr:
        expr: Expr = self.comparison()

        while self.match([l.Token("operator", operator="not_equal"), l.Token("operator", operator="equal")]):
            op = self.prev().operator
            if op == None:
                panic("equality: op is None")

            right = self.comparison()
            expr = BinaryExpr(expr, op, right)

        return expr

    def expression(self) -> Expr:
        e = self.equality() 
        

        return e # type: ignore
