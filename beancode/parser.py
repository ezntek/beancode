from . import lexer as l
from . import *

from .bean_ast import *

from .error import *
from . import __version__

class Parser:
    tokens: list[l.Token]
    cur: int

    def __init__(self, tokens: list[l.Token]) -> None:
        self.cur = 0
        self.tokens = tokens
        self.PRIM_TYPES = ["integer", "real", "boolean", "char", "string"]

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

    def consume(self) -> l.Token:
        if self.cur < len(self.tokens):
            self.cur += 1

        return self.prev()

    def consume_newlines(self):
        while self.peek().kind == "newline":
            self.consume()

    def check_newline(self, s: str):
        nl = self.consume()
        if nl.kind != "newline":
            raise BCError(f"expected newline after {s}, but found `{self.prev()}`", nl)

    def prev(self) -> l.Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> l.Token:
        if self.cur >= len(self.tokens):
            raise BCError(
                f"unexpected end of file", self.tokens[len(self.tokens) - 1], eof=True
            )

        return self.tokens[self.cur]

    def peek_next(self) -> l.Token | None:
        if self.cur + 1 >= len(self.tokens):
            return None

        return self.tokens[self.cur + 1]

    def match(self, typs: list[tuple[l.TokenType, str]]) -> bool:
        for typ in typs:
            if self.check(typ):
                self.consume()
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
        if self.is_integer(val):
            return False

        found_decimal = False

        for ch in val:
            if ch == ".":
                if found_decimal:
                    return False
                found_decimal = True

        return found_decimal

    def array_literal(self, nested=False) -> Expr | None:
        lbrace = self.consume()  # TODO: allow other braced literals later
        if lbrace.separator != "left_curly":
            raise BCError("expected left curly brace for array or matrix literal!")

        exprs = []
        while self.peek().separator != "right_curly":
            self.clean_newlines()

            if self.peek().separator == "left_curly":
                if nested:
                    raise BCError("cannot nest array literals over 2 dimensions!")
                arrlit = self.array_literal(nested=True)
                exprs.append(arrlit)
            else:
                expr = self.expression()
                if expr is None:
                    raise BCError(
                        "invalid expression supplied as argument to array literal"
                    )
                exprs.append(expr)

            self.clean_newlines()
            comma = self.peek()
            if comma.separator == "right_curly":
                break
            elif comma.separator != "comma":
                raise BCError(
                    f"expected comma after expression in array literal, found {comma.kind}",
                    comma,
                )
            self.consume()

            self.clean_newlines()  # allow newlines

        if len(exprs) == 0:
            raise BCError(
                f"array literals may not have no elements, as the resulting array has no space",
                self.peek(),
            )

        # allow for trailing comma
        if self.peek().separator == "right_curly":
            self.consume()

        return ArrayLiteral(lbrace.pos, exprs)

    def literal(self) -> Expr | None:
        tok = self.consume()

        if tok.kind != "literal":
            return None

        lit: l.Literal
        lit = tok.literal  # type: ignore

        match lit.kind:
            case "char":
                val = lit.value
                if val[0] == "\\":
                    if len(val) == 1:
                        return Literal(tok.pos, "char", char="\\")
                    ch = ""
                    match val[1]:
                        case "n":
                            ch = "\n"
                        case "r":
                            ch = "\r"
                        case "e":
                            ch = "\033"
                        case "a":
                            ch = "\a"
                        case "b":
                            ch = "\b"
                        case "f":
                            ch = "\f"
                        case "v":
                            ch = "\v"
                        case "\\":
                            ch = "\\"
                    return Literal(tok.pos, "char", char=ch)
                else:
                    if len(val) > 1:
                        raise BCError(
                            f"more than 1 character in char literal `{lit}`", tok
                        )
                    return Literal(tok.pos, "char", char=val[0])
            case "string":
                val = lit.value
                return Literal(tok.pos, "string", string=val)
            case "boolean":
                val = lit.value.lower()
                if val == "true":
                    return Literal(tok.pos, "boolean", boolean=True)
                elif val == "false":
                    return Literal(tok.pos, "boolean", boolean=False)
                else:
                    raise BCError(f"invalid boolean literal `{lit.value}`", tok)
            case "number":
                val = lit.value

                if self.is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        raise BCError(f"invalid number literal `{val}`", tok)

                    return Literal(tok.pos, "real", real=res)
                elif self.is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        raise BCError(f"invalid number literal `{val}`", tok)

                    return Literal(tok.pos, "integer", integer=res)
                else:
                    raise BCError(f"invalid number literal `{val}`", tok)

    def typ(self) -> BCType | None:
        adv = self.consume()

        if adv.kind == "newline":
            raise BCError("unexpected newline when scanning for type", adv)

        if adv.kind == "type" and adv.typ != "array":
            if adv.typ not in self.PRIM_TYPES:
                return None

            t: BCPrimitiveType = adv.typ  # type: ignore
            return t
        elif adv.kind == "type" and adv.typ == "array":
            flat_bounds = None
            matrix_bounds = None
            is_matrix = False
            inner: BCPrimitiveType

            left_bracket = self.consume()
            if left_bracket.separator == "left_bracket":
                begin = self.expression()
                if begin is None:
                    raise BCError(
                        "invalid expression as beginning value of array declaration",
                        begin,
                    )

                colon = self.consume()
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

                right_bracket = self.consume()
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

                    inner_colon = self.consume()
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

                    right_bracket = self.consume()
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

            of = self.consume()
            if of.kind != "keyword" and of.keyword != "of":
                raise BCError("expected `OF` after `ARRAY` and/or size declaration", of)

            # TODO: refactor
            arrtyp = self.consume()

            if arrtyp.typ == "array":
                raise BCError(
                    "cannot have array as array element type, please use the matrix syntax instead",
                    arrtyp,
                )

            if arrtyp.typ not in self.PRIM_TYPES:
                raise BCError("invalid type used as array element type", arrtyp)

            inner = arrtyp.typ  # type: ignore

            return BCArrayType(
                is_matrix=is_matrix,
                flat_bounds=flat_bounds,
                matrix_bounds=matrix_bounds,
                inner=inner,
            )

    def ident(self) -> Expr:
        c = self.consume()

        return Identifier(c.pos, c.ident)  # type: ignore

    def array_index(self) -> Expr | None:
        pn = self.peek_next()
        if pn is None:
            return None

        if pn.kind != "separator" and pn.separator != "left_bracket":
            return None

        ident = self.ident()

        leftb = self.consume()
        if leftb.separator != "left_bracket":
            raise BCError("expected left_bracket after ident in array index", leftb)

        exp = self.expression()
        if exp is None:
            raise BCError("expected expression as array index", exp)

        rightb = self.consume()
        exp_inner = None
        if rightb.separator == "right_bracket":
            pass
        elif rightb.separator == "comma":
            exp_inner = self.expression()
            if exp_inner is None:
                raise BCError("expected expression as array index", exp_inner)

            rightb = self.consume()
            if rightb.separator != "right_bracket":
                raise BCError(
                    "expected right_bracket after expression in array index", rightb
                )
        else:
            raise BCError(
                "expected right_bracket after expression in array index", rightb
            )

        return ArrayIndex(ident.pos, ident=ident, idx_outer=exp, idx_inner=exp_inner)  # type: ignore

    def operator(self) -> l.Operator | None:
        o = self.consume()
        return o.operator

    def function_call(self) -> Expr | None:
        # avoid consuming tokens
        ident = self.peek()
        if ident.kind != "ident":
            return None

        leftb = self.peek_next()
        if leftb is None:
            return None

        if leftb.separator != "left_paren":
            return None

        self.consume()
        self.consume()

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
                self.consume()

        rightb = self.consume()
        if rightb.separator != "right_paren":
            raise BCError(
                "expected right paren after arg list in function call", rightb
            )
        return FunctionCall(ident.pos, ident=ident.ident, args=args)  # type: ignore

    def typecast(self) -> Typecast | None:
        typ = self.consume()
        if typ.typ is None:
            raise BCError("invalid type supplied for type cast", typ)

        if typ.typ not in self.PRIM_TYPES:
            raise BCError("array type supplied for type cast", typ)

        t: BCPrimitiveType = typ.typ  # type: ignore

        self.consume()  # checked already

        expr = self.expression()
        if expr is None:
            raise BCError("invalid expression supplied for type cast", expr)

        rbracket = self.consume()
        if rbracket.separator != "right_paren":
            raise BCError("expected right paren after type cast expression", rbracket)

        return Typecast(typ.pos, t, expr)

    def unary(self) -> Expr | None:
        p = self.peek()
        if p.kind == "literal":
            return self.literal()
        elif p.separator == "left_curly":
            return self.array_literal()
        elif p.kind == "ident":
            pn = self.peek_next()
            if pn is None:
                return None

            if pn.kind == "separator" and pn.separator == "left_bracket":
                return self.array_index()

            if pn.kind == "separator" and pn.separator == "left_paren":
                return self.function_call()

            return self.ident()
        elif p.kind == "type":
            pn = self.peek_next()
            if pn is None:
                return None

            if pn.kind == "separator" and pn.separator == "left_paren":
                return self.typecast()
        elif p.kind == "separator" and p.separator == "left_paren":
            begin = self.consume()
            e = self.expression()
            if e is None:
                raise BCError("invalid expression inside grouping", e)

            end = self.consume()

            if end.separator != "right_paren":
                raise BCError("expected ending ) delimiter after (", begin)

            return Grouping(begin.pos, inner=e)
        elif p.kind == "operator" and p.operator == "sub":
            begin = self.consume()
            e = self.unary()
            if e is None:
                raise BCError("invalid expression for negation", begin)
            return Negation(begin.pos, e)
        elif p.kind == "keyword" and p.keyword == "not":
            begin = self.consume()
            e = self.expression()
            if e is None:
                raise BCError("invalid expression for logical NOT", begin)
            return Not(begin.pos, e)
        else:
            return None

    def pow(self) -> Expr | None:
        expr = self.unary()
        if expr is None:
            return None

        while self.match([("operator", "pow")]):
            op = self.prev().operator

            if op is None:
                raise BCError("pow: op is None", op)

            right = self.unary()

            if right is None:
                return None

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore

        return expr

    def factor(self) -> Expr | None:
        expr = self.pow()
        if expr is None:
            return None

        while self.match([("operator", "mul"), ("operator", "div")]):
            op = self.prev().operator

            if op is None:
                raise BCError("factor: op is None", op)

            right = self.pow()

            if right is None:
                return None

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore

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

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore

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

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore
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

            expr = BinaryExpr(expr.pos, expr, op, right)

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

            expr = BinaryExpr(expr.pos, expr, op, right)  # kw must be and or or

        return expr

    def expression(self) -> Expr | None:
        return self.logical_comparison()

    def output_stmt(self) -> Statement | None:
        exprs = []
        begin = self.peek()

        if begin.keyword != "output" and begin.keyword != "print":
            return None

        self.consume()
        initial = self.expression()
        if initial is None:
            raise BCError(
                "found OUTPUT but an invalid expression that follows", self.peek()
            )

        exprs.append(initial)

        while self.match([("separator", "comma")]):
            new = self.expression()
            if new is None:
                break

            exprs.append(new)

        res = OutputStatement(begin.pos, items=exprs)
        return Statement("output", output=res)

    def input_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "input":
            return None

        begin = self.consume()

        ident = self.ident()
        if not isinstance(ident, Identifier) or ident.ident is None:
            raise BCError(f"expected identifier after `INPUT` but found {ident}", begin)

        res = InputStatement(begin.pos, ident)
        return Statement("input", input=res)

    def return_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.kind != "keyword":
            return None

        if begin.keyword != "return":
            return None

        self.consume()

        expr = self.expression()
        if expr is None:
            raise BCError("invalid expression used as RETURN expression", self.peek())

        return Statement("return", return_s=ReturnStatement(begin.pos, expr))

    def call_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "call":
            return

        self.consume()

        # CALL <ident>(<expr>, <expr>)
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid ident after procedure call", self.peek())

        leftb = self.peek()
        args = []
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            self.consume()
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
                    self.consume()

            rightb = self.consume()
            if rightb.separator != "right_paren":
                raise BCError(
                    "expected right paren after arg list in procedure call", self.peek()
                )

        self.check_newline("procedure call")

        res = CallStatement(begin.pos, ident=ident.ident, args=args)
        return Statement("call", call=res)

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            export = True
            begin = self.peek_next()
            if begin is None:
                raise BCError(
                    "expected token following export, but got end of file", begin
                )

        # combining the conditions does NOT WORK.
        if begin.keyword != "declare":
            return None

        # consume the keyword
        declare_kw = self.consume()
        if export == True:
            self.consume()

        idents = []
        ident = self.consume()
        if ident.ident is None:
            raise BCError(
                f"expected ident after declare stmt, found `{ident.__repr__()}`",
                self.peek(),
            )
        idents.append(Identifier(ident.pos, ident.ident))

        while self.peek().separator == "comma":
            self.consume()  # consume the sep
            if self.peek().separator == "colon":
                break

            ident = self.consume()
            if ident.ident is None:
                raise BCError(
                    f"invalid identifier after comma in declare statement: `{ident.__repr__()}`",
                    self.peek(),
                )
            idents.append(Identifier(ident.pos, ident.ident))

        typ = None
        expr = None

        if self.peek().separator == "colon":
            self.consume()

            typ = self.typ()
            if typ is None:
                raise BCError("invalid type after DECLARE", self.peek())

        if self.peek().operator == "assign":
            tok = self.consume()
            if len(idents) > 1:
                raise BCError(
                    "cannot have assignment in declaration of multiple variables",
                    tok.pos,
                )

            expr = self.expression()
            if expr is None:
                raise BCError("invalid expression after assign in declare", tok.pos)

        if typ is None and expr is None:
            raise BCError(
                "must have either a type declaration, expression to assign as, or both",
                declare_kw,
            )

        self.check_newline("variable declaration (DECLARE)")

        res = DeclareStatement(begin.pos, ident=idents, typ=typ, expr=expr, export=export)  # type: ignore
        return Statement("declare", declare=res)

    def constant_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            if begin is None:
                raise BCError(
                    "expected token following export, but got end of file", begin
                )
            export = True

        if begin.kind != "keyword":
            return None

        if begin.keyword != "constant":
            return None

        # consume the kw
        self.consume()
        if export == True:
            self.consume()

        ident = self.consume()
        if ident.ident is None:
            raise BCError(
                f"expected ident after constant declaration, found `{ident.__repr__()}`",
                self.peek(),
            )

        arrow = self.consume()
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

        res = ConstantStatement(begin.pos, Identifier(ident.pos, ident.ident), literal, export=export)
        return Statement("constant", constant=res)

    def assign_stmt(self) -> Statement | None:
        p = self.peek_next()
        if p is None:
            return None

        if p.separator == "left_bracket":
            temp_idx = self.cur
            while self.tokens[temp_idx].separator != "right_bracket":
                temp_idx += 1
                if temp_idx == len(self.tokens):
                    raise BCError(
                        "reached end of file while searching for end delimiter `]`",
                        self.tokens[temp_idx - 1],
                    )

            p = self.tokens[temp_idx + 1]
            if p.kind != "operator":
                return None
            elif p.operator != "assign":
                return None
        elif p.operator != "assign":
            return None

        ident = self.array_index()
        if ident is None:
            ident = self.ident()

        self.consume()  # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            raise BCError("expected expression after `<-` in assignment", self.peek())

        self.check_newline("assignment")

        res = AssignStatement(ident.pos, ident, expr)  # type: ignore
        return Statement("assign", assign=res)

    # multiline statements go here
    def if_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "if":
            return

        self.consume()  # byebye `IF`

        cond = self.expression()
        if cond is None:
            raise BCError("found invalid expression for if condition", self.peek())

        # allow stupid igcse stuff
        if self.peek().kind == "newline":
            self.clean_newlines()

        then = self.peek()
        if then.keyword != "then":
            raise BCError(
                f"expected `THEN` after if condition, but found `{str(then)}`", then
            )
        self.consume()

        # dont enforce newline after then
        if self.peek().kind == "newline":
            self.clean_newlines()

        if_stmts = []
        else_stmts = []

        while self.peek().keyword not in ["else", "endif"]:
            if_stmts.append(self.scan_one_statement())

        if self.peek().keyword == "else":
            self.consume()  # byebye else

            # dont enforce newlines after else
            if self.peek().kind == "newline":
                self.clean_newlines()

            while self.peek().keyword != "endif":
                else_stmts.append(self.scan_one_statement())

        self.consume()  # byebye endif

        res = IfStatement(
            begin.pos, cond=cond, if_block=if_stmts, else_block=else_stmts
        )
        return Statement("if", if_s=res)

    def caseof_stmt(self) -> Statement | None:
        case = self.peek()

        # if case.kind == "keyword" and case.keyword == "case":
        #    raise BCError("case of not implemented", self.peek())

        if case.keyword != "case":
            return
        self.consume()

        if self.peek().keyword != "of":
            return
        self.consume()

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

                colon = self.consume()
                if colon.separator != "colon":
                    raise BCError(
                        "expected colon after case of branch expression", self.prev()
                    )
            else:
                self.consume()

            stmt = self.stmt()

            if stmt is None:
                raise BCError("expected statement for case of branch block")

            if is_otherwise:
                otherwise = stmt
            else:
                branches.append(CaseofBranch(expr.pos, expr, stmt))  # type: ignore

            self.check_newline("CASE OF branch")
            self.consume_newlines()
        self.consume()

        res = CaseofStatement(case.pos, main_expr, branches, otherwise)
        return Statement(kind="caseof", caseof=res)

    def while_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "while":
            return

        # byebye `WHILE`
        self.consume()

        expr = self.expression()
        if expr is None:
            raise BCError(
                "found invalid expression for while loop condition", self.peek()
            )

        if self.peek().kind == "newline":
            self.clean_newlines()

        do = self.peek()
        if do.keyword != "do":
            raise BCError(
                f"expected `DO` after while loop condition, but found {str(do)}",
                self.peek(),
            )
        self.consume()

        if self.peek().kind == "newline":
            self.clean_newlines()

        stmts = []
        while self.peek().keyword != "endwhile":
            stmts.append(self.scan_one_statement())

        self.consume()  # byebye `ENDWHILE`

        res = WhileStatement(begin.pos, expr, stmts)
        return Statement("while", while_s=res)

    def for_stmt(self):
        # FOR <counter> <assign> <lit> TO <lit> [STEP <lit>]
        initial = self.peek()

        if initial.keyword != "for":
            return

        self.consume()

        counter: Identifier | None = self.ident()  # type: ignore
        if counter.ident is None or not isinstance(counter, Identifier):  # type: ignore
            raise BCError("expected ident before `<-` in a for loop", self.peek())

        assign = self.peek()
        if assign.operator != "assign":
            raise BCError(
                "expected assignment operator `<-` after counter in a for loop", assign
            )
        self.consume()

        begin = self.expression()
        if begin is None:
            raise BCError("invalid expression as begin in for loop", self.peek())

        to = self.peek()
        if to.keyword != "to":
            raise BCError("expected TO after beginning value in for loop", to)
        self.consume()

        end = self.expression()
        if end is None:
            raise BCError("invalid expression as end in for loop", self.peek())

        step: Expr | None = None
        if self.peek().keyword == "step":
            self.consume()
            step = self.expression()
            if step is None:
                raise BCError("invalid expression as step in for loop", self.peek())

        stmts = []
        while self.peek().keyword != "next":
            stmts.append(self.scan_one_statement())

        self.consume()  # byebye NEXT

        next_counter: Identifier | None = self.ident()  # type: ignore
        if next_counter is None or not isinstance(counter, Identifier):
            raise BCError("expected identifier after NEXT in a for loop", self.peek())
        elif counter.ident != next_counter.ident:
            raise BCError(
                f"initialized counter as {counter.ident} but used {next_counter.ident} after loop",
                self.prev(),
            )

        # thanks python for not having proper null handling
        res = ForStatement(begin.pos, counter=counter, block=stmts, begin=begin, end=end, step=step)  # type: ignore
        return Statement("for", for_s=res)

    def repeatuntil_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "repeat":
            return

        # byebye `REPEAT`
        self.consume()

        self.clean_newlines()

        stmts = []
        while self.peek().keyword != "until":
            stmts.append(self.scan_one_statement())

        self.consume()  # byebye `UNTIL`

        expr = self.expression()
        if expr is None:
            raise BCError(
                "found invalid expression for repeat-until loop condition", self.peek()
            )

        res = RepeatUntilStatement(begin.pos, expr, stmts)
        return Statement("repeatuntil", repeatuntil=res)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier for function arg", self.peek())

        colon = self.consume()
        if colon.kind != "separator" and colon.separator != "colon":
            raise BCError(
                "expected colon after ident in function argument", self.peek()
            )

        typ = self.typ()
        if typ is None:
            raise BCError("invalid type after colon in function argument", self.peek())

        return FunctionArgument(
            pos=ident.pos if ident.pos is not None else (0, 0, 0),
            name=ident.ident,
            typ=typ,
        )

    def procedure_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            if begin is None:
                raise BCError(
                    "expected token following export, but got end of file", begin
                )
            export = True

        if begin.keyword != "procedure":
            return

        self.consume()  # byebye PROCEDURE
        if export == True:
            self.consume()

        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier after PROCEDURE declaration", self.peek())

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.consume()
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
                    self.consume()

            rightb = self.consume()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                raise BCError(
                    f"expected right paren after arg list in procedure declaration, found {rightb}",
                    self.peek(),
                )

        self.consume_newlines()

        stmts = []
        while self.peek().keyword != "endprocedure":
            stmts.append(self.scan_one_statement())

        self.consume()  # bye bye ENDPROCEDURE

        res = ProcedureStatement(
            begin.pos, name=ident.ident, args=args, block=stmts, export=export
        )
        return Statement("procedure", procedure=res)

    def function_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.keyword == "export":
            begin = self.peek_next()
            if begin is None:
                raise BCError(
                    "expected token following export, but got end of file", begin
                )
            export = True

        if begin.keyword != "function":
            return None

        self.consume()  # byebye FUNCTION
        if export == True:
            self.consume()

        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier after FUNCTION declaration", self.peek())

        args = []
        leftb = self.peek()
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            # there is an arg list
            self.consume()
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
                    self.consume()

            rightb = self.consume()
            if rightb.kind != "separator" and rightb.separator != "right_paren":
                raise BCError(
                    f"expected right paren after arg list in function declaration, found {rightb}",
                    self.peek(),
                )

        returns = self.consume()
        if returns.keyword != "returns":
            raise BCError("expected RETURNS after function arguments", self.peek())

        typ = self.typ()
        if typ is None:
            raise BCError(
                "invalid type after RETURNS for function return value", self.peek()
            )

        self.consume_newlines()
        
        stmts = []
        while self.peek().keyword != "endfunction":
            stmt = self.scan_one_statement()
            stmts.append(stmt)

        self.consume()  # bye bye ENDFUNCTION

        res = FunctionStatement(
            begin.pos,
            name=ident.ident,
            args=args,
            returns=typ,
            block=stmts,
            export=export,
        )
        return Statement("function", function=res)

    def scope_stmt(self) -> Statement | None:
        scope = self.peek()
        if scope.keyword != "scope":
            return
        self.consume()

        self.clean_newlines()

        stmts = []
        while self.peek().keyword != "endscope":
            stmts.append(self.scan_one_statement())

        self.consume()
        res = ScopeStatement(scope.pos, stmts)
        return Statement("scope", scope=res)

    def include_stmt(self) -> Statement | None:
        include = self.peek()
        if include.keyword != "include" and include.keyword != "include_ffi":
            return

        ffi = False
        c = self.consume()
        if c.keyword == "include_ffi":
            ffi = True

        name = self.consume()
        if name.kind != "literal":
            raise BCError(
                "Include must be followed by a literal of the name of the file to include"
            )

        if name.literal.kind != "string":  # type: ignore
            raise BCError("literal for include must be a string!")

        res = IncludeStatement(include.pos, name.literal.value, ffi=ffi)  # type: ignore
        return Statement("include", include=res)

    def clean_newlines(self):
        while self.cur < len(self.tokens) and self.peek().kind == "newline":
            self.consume()

    def stmt(self) -> Statement | None:
        self.clean_newlines()

        if self.cur + 1 >= len(self.tokens):
            self.cur += 1
            return None

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
        if expr is not None:
            raise BCWarning("unused expression", cur, data=expr)
        else:
            raise BCError("invalid statement or expression", cur)

    def scan_one_statement(self) -> Statement | None:
        s = self.stmt()

        if s is not None:
            self.clean_newlines()
            return s
        else:
            if self.cur >= len(self.tokens):
                return None

            p = self.peek()
            raise BCError(f"found invalid statement at `{p}`", p)

    def reset(self):
        self.cur = 0

    def program(self) -> Program:
        stmts = []

        while self.cur < len(self.tokens):
            self.clean_newlines()
            stmt = self.scan_one_statement()
            if stmt is None:  # this has to be an EOF
                continue
            stmts.append(stmt)

        self.cur = 0

        return Program(stmts=stmts)
