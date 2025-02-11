from abc import abstractmethod
import typing as t
import lexer as l
from dataclasses import dataclass
from bctypes import *
from util import *


@dataclass
class Expr:
    pass

@dataclass
class Negation(Expr):
    inner: Expr

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


@dataclass
class Literal(Expr):
    kind: BCType
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None
    boolean: bool | None = None
    # array not implemented

    def to_bcvalue(self) -> BCValue:
        return BCValue(
            self.kind,
            integer=self.integer,
            real=self.real,
            char=self.char,
            string=self.string,
            boolean=self.boolean,
        )


StatementKind = t.Literal[
    "declare", "output", "input", "constant", "assign", "if", "while", "for", "repeatuntil"
]

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
    ident: Identifier
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
    repeatuntil : RepeatUntilStatement | None = None

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
        if val[0] == '-' and val[1].isdigit():
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
        elif p.kind == "operator" and p.operator == "sub":
            self.advance()
            e = self.expression()
            if e is None:
                panic("invalid expression for negation")
            return Negation(e) 
        else:
            return None

    def factor(self)-> Expr | None:
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

        if p.kind != "operator" and p.operator != "assign":
            return None

        ident: Identifier | None = self.ident()  # type: ignore
        if ident.ident is None or not isinstance(ident, Identifier):  # type: ignore
            panic("expected ident before `<-` in assignment")

        self.advance()  # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            panic("expected expression after `<-` in assignment")

        self.check_newline("assignment")

        res = AssignStatement(ident, expr)
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

        declare = self.declare_stmt()
        if declare is not None:
            return declare

        if_s = self.if_stmt()
        if if_s is not None:
            return if_s

        while_s = self.while_stmt()
        if while_s is not None:
            return while_s

        for_s = self.for_stmt()
        if for_s is not None:
            return for_s

        repeatuntil_s = self.repeatuntil_stmt()
        if repeatuntil_s is not None:
            return repeatuntil_s

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
            stmts.append(self.scan_one_statement())

        return Program(stmts=stmts)
