import copy

from . import *
from .lexer import *
from .bean_ast import *

from .error import *
from . import __version__


def _convert_escape_code(ch: str) -> str | None:
    match ch:
        case "n":
            return "\n"
        case "r":
            return "\r"
        case "e":
            return "\033"
        case "a":
            return "\a"
        case "b":
            return "\b"
        case "f":
            return "\f"
        case "v":
            return "\v"
        case "0":
            return "\0"
        case "\\":
            return "\\"
        case _:
            return


class Parser:
    tokens: list[Token]
    cur: int

    def __init__(self, tokens: list[Token]) -> None:
        self.cur = 0
        self.tokens = tokens

    def prev(self) -> Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> Token:
        if self.cur >= len(self.tokens):
            raise BCError(f"unexpected end of file", self.tokens[-1].pos, eof=True)

        return self.tokens[self.cur]

    def pos(self) -> Pos:
        return self.peek().pos

    def peek_next(self) -> Token | None:
        if self.cur + 1 >= len(self.tokens):
            return

        return self.tokens[self.cur + 1]

    def peek_and_expect(self, expected: TokenKind, ctx=str(), help=str()) -> Token:
        tok = self.peek()

        s = " " + ctx if ctx else str()
        h = "\n" + help if help else str()

        if tok.kind != expected:
            raise BCError(
                f"expected token {humanize_token_kind(expected)}{s}, but got {tok.to_humanized_string()}{h}",
                tok.pos,
            )
        return tok

    def peek_next_and_expect(self, expected: TokenKind, ctx=str(), help=str()) -> Token:
        tok = self.peek_next()

        s = " " + ctx if ctx else str()
        h = "\n" + help if help else str()

        if not tok:
            raise BCError(
                f"expected token {humanize_token_kind(expected)}{s}, but reached end of file{h}",
                eof=True,
            )

        if tok.kind != expected:
            raise BCError(
                f"expected token {humanize_token_kind(expected)}{s}, but got {tok.to_humanized_string()}{h}",
                tok.pos,
            )

        return tok

    def check(self, tok: TokenKind) -> bool:
        """peek and check"""
        if self.cur >= len(self.tokens):
            return False

        return self.peek().kind == tok

    def check_and_consume(self, expected: TokenKind) -> Token | None:
        if self.check(expected):
            return self.consume()
        else:
            return

    def consume_and_expect(self, expected: TokenKind, ctx=str(), help=str()) -> Token:
        cons = self.consume()
        if cons.kind != expected:
            s = " " + ctx if ctx else str()
            h = "\n" + help if help else str()

            if help:
                h = "\n" + help
            raise BCError(
                f"expected token {humanize_token_kind(expected)}{s}, but got {cons.to_humanized_string()}{h}",
                cons.pos,
            )
        return cons

    def consume(self) -> Token:
        if self.cur < len(self.tokens):
            self.cur += 1
        else:
            raise BCError("reached end of file", eof=True)

        return self.prev()

    def consume_newlines(self):
        while self.cur < len(self.tokens) and self.peek().kind == "newline":
            self.consume()

    def check_newline(self, s: str):
        self.consume_and_expect("newline", ctx=f"after {s}")

    def match(self, *typs: TokenKind) -> bool:
        for typ in typs:
            if self.check(typ):
                self.consume()
                return True
        return False

    def check_many(self, *typs: TokenKind) -> bool:
        """peek and check many"""
        return self.peek().kind in typs

    def array_literal(self, nested=False) -> Expr | None:
        lbrace = self.consume_and_expect("left_curly", "for array or matrix literal")

        exprs = []
        while not self.check("right_curly"):
            self.clean_newlines()

            if self.check("left_curly"):
                if nested:
                    raise BCError(
                        "cannot nest array literals over 2 dimensions!", self.pos()
                    )
                arrlit = self.array_literal(nested=True)
                exprs.append(arrlit)
            else:
                expr = self.expression()
                if not expr:
                    raise BCError(
                        "invalid or no expression supplied as argument to array literal",
                        self.pos(),
                    )
                exprs.append(expr)

            self.clean_newlines()
            comma = self.peek()
            if comma.kind == "right_curly":
                break
            elif comma.kind != "comma":
                raise BCError(
                    f"expected comma after expression in array literal, found {comma.kind}",
                    comma.pos,
                )
            self.consume()

            self.clean_newlines()  # allow newlines

        if len(exprs) == 0:
            raise BCError(
                f"array literals may not have no elements, as the resulting array has no space",
                self.pos(),
            )

        self.check_and_consume("right_curly")

        return ArrayLiteral(lbrace.pos, exprs)

    def literal(self) -> Expr | None:
        if not self.check_many(
            "literal_string", "literal_char", "literal_number", "true", "false", "null"
        ):
            return

        lit = self.consume()
        match lit.kind:
            case "literal_char":
                if len(lit.data) == 0:  # type: ignore
                    raise BCError(
                        "CHAR literal cannot have no characters in it!", lit.pos
                    )

                val: str = lit.data  # type: ignore
                if val[0] == "\\":
                    if len(val) == 1:
                        return Literal(lit.pos, "char", char="\\")

                    ch = _convert_escape_code(val[1])
                    if not ch:
                        raise BCError(
                            f"invalid escape sequence in literal '{val}'",
                            lit.pos,
                        )

                    return Literal(lit.pos, "char", char=ch)
                else:
                    if len(val) > 1:
                        raise BCError(
                            f"more than 1 character in char literal '{val}'", lit.pos
                        )
                    return Literal(lit.pos, "char", char=val[0])
            case "literal_string":
                val: str = lit.data  # type: ignore
                res = StringIO()
                i = 0

                while i < len(val):
                    if val[i] == "\\":
                        if i == len(val) - 1:
                            res.write("\\")
                        else:
                            i += 1
                            ch = _convert_escape_code(val[i])
                            if not ch:
                                pos = copy.copy(lit.pos)
                                pos.col += i
                                pos.span = 2
                                raise BCError(
                                    f'invalid escape sequence in literal "{val}"',
                                    pos,
                                )
                            res.write(ch)
                    else:
                        res.write(val[i])
                    i += 1

                return Literal(lit.pos, "string", string=res.getvalue())
            case "literal_number":
                val: str = lit.data  # type: ignore

                if is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        raise BCError(f'invalid number literal "{val}"', lit.pos)

                    return Literal(lit.pos, "real", real=res)
                elif is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        raise BCError(f'invalid number literal "{val}"', lit.pos)

                    return Literal(lit.pos, "integer", integer=res)
                else:
                    raise BCError(f'invalid number literal "{val}"', lit.pos)
            case "true":
                return Literal(lit.pos, "boolean", boolean=True)
            case "false":
                return Literal(lit.pos, "boolean", boolean=False)
            case "null":
                return Literal(lit.pos, "null")

    def _array_type(self) -> Type:
        flat_bounds = None
        matrix_bounds = None
        is_matrix = False
        inner: BCPrimitiveType

        self.consume_and_expect("left_bracket", "for array type declaration")
        begin = self.expression()
        if not begin:
            raise BCError(
                "invalid or no expression as beginning value of array declaration",
                begin,
            )

        self.consume_and_expect("colon", "after beginning value of array declaration")

        end = self.expression()
        if not end:
            raise BCError(
                "invalid or no expression as ending value of array declaration",
                end,
            )

        flat_bounds = (begin, end)

        right_bracket = self.consume()
        if right_bracket.kind == "right_bracket":
            pass
        elif right_bracket.kind == "comma":
            inner_begin = self.expression()
            if not inner_begin:
                raise BCError(
                    "invalid or no expression as beginning value of array declaration",
                    inner_begin,
                )

            self.consume_and_expect(
                "colon", "after beginning value of array declaration"
            )

            inner_end = self.expression()
            if not inner_end:
                raise BCError(
                    "invalid or no expression as ending value of array declaration",
                    inner_end,
                )

            matrix_bounds = (
                flat_bounds[0],
                flat_bounds[1],
                inner_begin,
                inner_end,
            )

            flat_bounds = None

            self.consume_and_expect("right_bracket", "after matrix length declaration")

            is_matrix = True
        else:
            raise BCError(
                "expected right bracket or comma after array bounds declaration",
                right_bracket.pos,
            )

        self.consume_and_expect("of", "after array size declaration")

        arrtyp = self.consume_and_expect("type", "after array size declaration")
        if arrtyp.data == "array":
            raise BCError(
                "cannot have array as array element type, please use the matrix syntax instead",
                arrtyp.pos,
            )

        inner = arrtyp.data  # type: ignore

        return ArrayType(
            is_matrix=is_matrix,
            flat_bounds=flat_bounds,
            matrix_bounds=matrix_bounds,
            inner=inner,
        )

    def typ(self) -> Type:
        adv = self.consume_and_expect("type")
        if adv.data == "array":
            return self._array_type()
        else:
            t: BCPrimitiveType = adv.data  # type: ignore
            return t

    def ident(self) -> Identifier:
        c = self.consume_and_expect("ident")
        return Identifier(c.pos, c.data)  # type: ignore

    def expect_ident(self, ctx=str()) -> Identifier:
        c = self.consume_and_expect("ident", ctx)
        return Identifier(c.pos, c.data)  # type: ignore

    def array_index(self) -> Expr | None:
        pn = self.peek_next()
        if not pn:
            return
        if pn.kind != "left_bracket":
            return

        ident = self.ident()

        leftb = self.consume_and_expect("left_bracket")
        exp = self.expression()
        if not exp:
            raise BCError("expected expression as array index", leftb.pos)

        rightb = self.consume()
        exp_inner = None
        if rightb.kind == "right_bracket":
            pass
        elif rightb.kind == "comma":
            exp_inner = self.expression()
            if not exp_inner:
                raise BCError("expected expression as array index", exp_inner)

            self.consume_and_expect("right_bracket", "after expression in array index")
        else:
            raise BCError(
                "expected right_bracket or comma after expression in array index",
                rightb.pos,
            )

        return ArrayIndex(leftb.pos, ident=ident, idx_outer=exp, idx_inner=exp_inner)

    def function_call(self) -> Expr | None:
        # avoid consuming tokens
        ident = self.peek()
        if ident.kind != "ident":
            return

        leftb = self.peek_next()
        if not leftb:
            return
        if leftb.kind != "left_paren":
            return

        # consume both the ident and left_paren
        self.consume()
        self.consume()

        args = []
        while self.peek().kind != "right_paren":
            expr = self.expression()
            if not expr:
                raise BCError(
                    "invalid or no expression as function argument", leftb.pos
                )

            args.append(expr)

            comma = self.peek()
            if comma.kind not in ("comma", "right_paren"):
                raise BCError(
                    "expected comma or right parenthesis after argument in function call argument list",
                    comma.pos,
                )
            elif comma.kind == "comma":
                self.consume()

        self.consume_and_expect("right_paren", "after argument list in function call")
        return FunctionCall(leftb.pos, ident=str(ident.data), args=args)

    def typecast(self) -> Typecast | None:
        typ = self.consume_and_expect("type", "for typecast")
        if typ.data == "array":
            raise BCError("cannot typecast to an array!")

        t: BCPrimitiveType = typ.data  # type: ignore
        self.consume()  # checked already

        expr = self.expression()
        if not expr:
            raise BCError("invalid or no expression supplied for type cast", expr)

        self.consume_and_expect("right_paren", "after type cast expression")

        return Typecast(typ.pos, t, expr)

    def grouping(self) -> Expr | None:
        begin = self.consume_and_expect("left_paren", "in grouping")
        e = self.expression()
        if not e:
            raise BCError("invalid or no expression inside grouping", e)

        self.consume_and_expect("right_paren", "after expression in grouping")
        return Grouping(begin.pos, inner=e)

    def unary(self) -> Expr | None:
        p = self.peek()

        lit = self.literal()
        if lit:
            return lit

        if p.kind == "left_curly":
            return self.array_literal()
        elif p.kind == "ident":
            pn = self.peek_next()
            if pn is not None:
                if pn.kind == "left_bracket":
                    return self.array_index()
                elif pn.kind == "left_paren":
                    return self.function_call()
            return self.ident()
        elif p.kind == "type":
            pn = self.peek_next()
            if not pn:
                return

            if pn.kind == "left_paren":
                return self.typecast()
        elif p.kind == "left_paren":
            return self.grouping()
        elif p.kind == "sub":
            begin = self.consume()
            e = self.unary()
            if not e:
                raise BCError("invalid or no expression for negation", begin.pos)
            return Negation(begin.pos, e)
        elif p.kind == "not":
            begin = self.consume()
            e = self.expression()
            if not e:
                raise BCError("invalid or no expression for logical NOT", begin.pos)
            return Not(begin.pos, e)
        else:
            return

    def pow(self) -> Expr | None:
        expr = self.unary()
        if not expr:
            return

        if self.match("pow"):
            op = self.prev()
            right = self.pow()

            if not right:
                return

            expr = BinaryExpr(op.pos, expr, "pow", right)

        return expr

    def factor(self) -> Expr | None:
        expr = self.pow()
        if not expr:
            return

        while self.match("mul", "div"):
            op = self.prev()

            right = self.pow()

            if not right:
                return

            expr = BinaryExpr(op.pos, expr, op.kind, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if not expr:
            return

        while self.match("add", "sub"):
            op = self.prev()

            right = self.factor()
            if not right:
                return

            expr = BinaryExpr(op.pos, expr, op.kind, right)  # type: ignore

        return expr

    def comparison(self) -> Expr | None:
        # > < >= <=
        expr = self.term()
        if not expr:
            return

        while self.match(
            "greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal"
        ):
            op = self.prev()

            right = self.term()
            if not right:
                return

            expr = BinaryExpr(op.pos, expr, op.kind, right)  # type: ignore

        return expr

    def equality(self) -> Expr | None:
        expr = self.comparison()

        if not expr:
            return

        while self.match("equal", "not_equal"):
            op = self.prev()

            right = self.comparison()
            if not right:
                return

            expr = BinaryExpr(op.pos, expr, op.kind, right)  # type: ignore

        return expr

    def logical_comparison(self) -> Expr | None:
        expr = self.equality()
        if not expr:
            return

        while self.match("and", "or"):
            op = self.prev()  # type: ignore

            right = self.equality()

            if not right:
                return

            expr = BinaryExpr(op.pos, expr, op.kind, right)  # type: ignore

        return expr

    def expression(self) -> Expr | None:
        return self.logical_comparison()

    def output_stmt(self) -> Statement | None:
        exprs = []
        begin = self.peek()

        if begin.kind not in ("output", "print"):
            return

        self.consume()
        initial = self.expression()
        if not initial:
            raise BCError(
                "found OUTPUT but an invalid or no expression that follows", begin.pos
            )

        exprs.append(initial)

        while self.match("comma"):
            new = self.expression()
            if not new:
                break

            exprs.append(new)

        return OutputStatement(begin.pos, items=exprs)

    def input_stmt(self) -> Statement | None:
        begin = self.check_and_consume("input")
        if not begin:
            return

        ident: ArrayIndex | Identifier
        array_index = self.array_index()
        if not array_index:
            ident = self.expect_ident("after INPUT")
        else:
            ident = array_index  # type: ignore

        return InputStatement(begin.pos, ident)

    def return_stmt(self) -> Statement | None:
        begin = self.check_and_consume("return")
        if not begin:
            return

        if self.check("newline"):
            return ReturnStatement(begin.pos)

        expr = self.expression()
        if not expr:
            raise BCError(
                "invalid or no expression used as RETURN expression", begin.pos
            )

        return ReturnStatement(begin.pos, expr)

    def call_stmt(self) -> Statement | None:
        begin = self.check_and_consume("call")
        if not begin:
            return

        ident = self.expect_ident("after procedure call")

        leftb = self.peek()
        args = []
        if leftb.kind == "left_paren":
            self.consume()
            while self.peek().kind != "right_paren":
                expr = self.expression()
                if not expr:
                    raise BCError(
                        "invalid or no expression as procedure argument", leftb.pos
                    )

                args.append(expr)

                comma = self.peek()
                if comma.kind != "comma" and comma.kind != "right_paren":
                    raise BCError(
                        "expected comma after argument in procedure call argument list",
                        comma.pos,
                    )
                elif comma.kind == "comma":
                    self.consume()

            self.consume_and_expect("right_paren", "after arg list in procedure call")

        self.consume_newlines()

        return CallStatement(begin.pos, ident=ident.ident, args=args)

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.kind == "export":
            export = True
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )

        if begin.kind != "declare":
            return

        # consume the keyword
        self.consume()
        if export == True:
            self.consume()

        idents = []
        ident = self.consume_and_expect("ident", "after declare statement")
        idents.append(Identifier(ident.pos, str(ident.data)))

        while self.check("comma"):
            self.consume()  # consume the sep
            if self.check("colon"):
                break

            ident = self.consume_and_expect("ident", "after comma in declare statement")
            idents.append(Identifier(ident.pos, str(ident.data)))

        typ = None
        expr = None

        colon = self.peek()
        if self.check("colon"):
            self.consume()

            typ = self.typ()
            if not typ:
                raise BCError("invalid type after DECLARE", colon.pos)

        if self.check("assign"):
            tok = self.consume()
            if len(idents) > 1:
                raise BCError(
                    "cannot have assignment in declaration of multiple variables",
                    tok.pos,
                )

            expr = self.expression()
            if not expr:
                raise BCError(
                    "invalid or no expression after assign in declare", tok.pos
                )

        if not typ and not expr:
            raise BCError(
                "must have either a type declaration, expression to assign as, or both",
                colon.pos,
            )

        self.check_newline("variable declaration (DECLARE)")

        return DeclareStatement(begin.pos, ident=idents, typ=typ, expr=expr, export=export)  # type: ignore

    def constant_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if self.check("export"):
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != "constant":
            return

        # consume the kw
        self.consume()
        if export == True:
            self.consume()

        ident = self.consume_and_expect("ident", "after constant declaration")
        self.consume_and_expect("assign", "after variable name in constant declaration")

        expr = self.expression()
        if not expr:
            raise BCError(
                "invalid or no expression for constant declaration", self.pos()
            )

        self.check_newline("constant declaration (CONSTANT)")

        return ConstantStatement(
            begin.pos, Identifier(ident.pos, str(ident.data)), expr, export=export
        )

    def assign_stmt(self) -> Statement | None:
        p = self.peek_next()
        if not p:
            return

        if p.kind == "left_bracket":
            temp_idx = self.cur
            while self.tokens[temp_idx].kind != "right_bracket":
                temp_idx += 1
                if temp_idx == len(self.tokens):
                    raise BCError(
                        "reached end of file while searching for end delimiter `]`",
                        self.tokens[-1].pos,
                        eof=True,
                    )

            p = self.tokens[temp_idx + 1]

        if p.kind != "assign":
            return

        ident = self.array_index()
        if not ident:
            ident = self.expect_ident("for left hand side of assignment")

        self.consume()  # go past the arrow

        expr: Expr | None = self.expression()
        if not expr:
            raise BCError("expected expression after `<-` in assignment", p.pos)

        self.check_newline("assignment")

        return AssignStatement(ident.pos, ident, expr)  # type: ignore

    # multiline statements go here
    def block(self, delim: TokenKind) -> list[Statement]:
        res = list()
        while not self.check(delim):
            res.append(self.scan_one_statement())
        return res

    def if_stmt(self) -> Statement | None:
        begin = self.check_and_consume("if")
        if not begin:
            return

        cond = self.expression()
        if not cond:
            raise BCError("found invalid or no expression for if condition", self.pos())

        # allow stupid igcse stuff
        if self.peek().kind == "newline":
            self.clean_newlines()

        self.consume_and_expect("then", "after if condition")
        self.clean_newlines()

        if_stmts = []
        else_stmts = []

        while not self.check_many("else", "endif"):
            if_stmts.append(self.scan_one_statement())

        if self.check_and_consume("else"):
            self.clean_newlines()
            else_stmts = self.block("endif")

        self.consume()  # byebye endif

        return IfStatement(
            begin.pos, cond=cond, if_block=if_stmts, else_block=else_stmts
        )

    def caseof_stmt(self) -> Statement | None:
        case = self.check_and_consume("case")
        if not case:
            return

        self.consume_and_expect("of", "after CASE keyword")

        main_expr = self.expression()
        if not main_expr:
            raise BCError(
                "found invalid or no expression for case of value", self.pos()
            )

        self.check_newline("after case of expression")

        branches: list[CaseofBranch] = []
        otherwise: Statement | None = None
        while not self.check("endcase"):
            is_otherwise = self.check("otherwise")
            if not is_otherwise:
                expr = self.expression()
                if not expr:
                    raise BCError(
                        "invalid or no expression for case of branch", self.pos()
                    )

                self.consume_and_expect("colon", "after case of branch expression")
            else:
                self.consume()

            stmt = self.stmt()
            self.consume_newlines()

            if not stmt:
                raise BCError("expected statement for case of branch block")

            if is_otherwise:
                otherwise = stmt
            else:
                branches.append(CaseofBranch(expr.pos, expr, stmt))  # type: ignore

        self.consume()

        return CaseofStatement(case.pos, main_expr, branches, otherwise)

    def while_stmt(self) -> Statement | None:
        begin = self.check_and_consume("while")
        if not begin:
            return

        expr = self.expression()
        if not expr:
            raise BCError(
                "found invalid or no expression for while loop condition", self.pos()
            )

        self.clean_newlines()
        self.consume_and_expect("do", "after while loop condition")
        self.clean_newlines()

        stmts = self.block("endwhile")
        end = self.consume()

        return WhileStatement(begin.pos, end.pos, expr, stmts)

    def for_stmt(self):
        initial = self.check_and_consume("for")
        if not initial:
            return

        counter = self.expect_ident("for for loop counter")

        self.consume_and_expect("assign", "after counter in for loop")

        begin = self.expression()
        if not begin:
            raise BCError("invalid or no expression as begin in for loop", self.pos())

        self.consume_and_expect("to", "after beginning value in for loop")

        end = self.expression()
        if not end:
            raise BCError("invalid or no expression as end in for loop", self.pos())

        step: Expr | None = None
        if self.check("step"):
            self.consume()
            step = self.expression()
            if not step:
                raise BCError(
                    "invalid or no expression as step in for loop", self.pos()
                )

        self.clean_newlines()
        stmts = self.block("next")
        next = self.consume()

        next_counter = self.expect_ident("after NEXT in for loop")

        if counter.ident != next_counter.ident:
            raise BCError(
                f"initialized counter as {counter.ident} but used {next_counter.ident} after loop",
                self.prev().pos,
            )

        return ForStatement(
            initial.pos,
            next.pos,
            counter=counter,
            block=stmts,
            begin=begin,
            end=end,
            step=step,
        )

    def repeatuntil_stmt(self) -> Statement | None:
        begin = self.check_and_consume("repeat")
        if not begin:
            return

        self.clean_newlines()

        stmts = self.block("until")
        until = self.consume()

        expr = self.expression()
        if not expr:
            raise BCError(
                "found invalid or no expression for repeat-until loop condition",
                self.pos(),
            )

        return RepeatUntilStatement(begin.pos, until.pos, expr, stmts)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.expect_ident("for function argument")
        colon = self.consume_and_expect(
            "colon", "after identifier in function argument"
        )

        typ = self.typ()
        if not typ:
            raise BCError("invalid type after colon in function argument", colon.pos)

        return FunctionArgument(
            pos=ident.pos,
            name=ident.ident,
            typ=typ,
        )

    def procedure_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.kind == "export":
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != "procedure":
            return

        self.consume()  # byebye PROCEDURE
        if export == True:
            self.consume()

        ident = self.expect_ident("after PROCEDURE declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == "left_paren":
            # there is an arg list
            self.consume()
            while not self.check("right_paren"):
                arg = self.function_arg()
                if not arg:
                    raise BCError("invalid function argument", self.pos())

                args.append(arg)

                if not self.check_many("comma", "right_paren"):
                    raise BCError(
                        "expected comma after procedure argument list", self.pos()
                    )

                if self.check("comma"):
                    self.consume()

            self.consume_and_expect(
                "right_paren", "after argument list in procedure declaration"
            )

        self.consume_newlines()

        stmts = self.block("endprocedure")

        self.consume()

        return ProcedureStatement(
            begin.pos, name=ident.ident, args=args, block=stmts, export=export
        )

    def function_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.kind == "export":
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != "function":
            return

        self.consume()  # byebye FUNCTION
        if export == True:
            self.consume()

        ident = self.expect_ident("after FUNCTION declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == "left_paren":
            # there is an arg list
            self.consume()
            while not self.check("right_paren"):
                arg = self.function_arg()
                if not arg:
                    raise BCError("invalid function argument", self.pos())

                args.append(arg)

                if not self.check_many("comma", "right_paren"):
                    raise BCError(
                        "expected comma after function argument in list", self.pos()
                    )

                if self.check("comma"):
                    self.consume()

            self.consume_and_expect(
                "right_paren", "after argument list in function declaration"
            )

        self.consume_and_expect("returns", "after function arguments")

        typ = self.typ()
        if not typ:
            raise BCError(
                "invalid type after RETURNS for function return value", self.pos()
            )

        self.consume_newlines()

        stmts = self.block("endfunction")
        self.consume()

        return FunctionStatement(
            begin.pos,
            name=ident.ident,
            args=args,
            returns=typ,
            block=stmts,
            export=export,
        )

    def scope_stmt(self) -> Statement | None:
        begin = self.check_and_consume("scope")
        if not begin:
            return

        self.clean_newlines()
        stmts = self.block("endscope")
        self.consume()

        return ScopeStatement(begin.pos, stmts)

    def include_stmt(self) -> Statement | None:
        if not self.check_many("include", "include_ffi"):
            return
        include = self.consume()

        ffi = False
        if include.kind == "include_ffi":
            ffi = True

        name = self.consume()
        if name.kind != "literal_string":
            raise BCError(
                "include must be followed by a literal of the name of the file to include",
                name.pos,
            )

        return IncludeStatement(include.pos, str(name.data), ffi=ffi)  # type: ignore

    def trace_stmt(self) -> Statement | None:
        begin = self.check_and_consume("trace")
        if not begin:
            return

        self.consume_and_expect("left_paren", "after TRACE keyword")

        vars = list()
        while not self.check("right_paren"):
            ident = self.expect_ident("in variable list in TRACE statement")

            vars.append(ident.ident)  # type: ignore

            if not self.check_many("comma", "right_paren"):
                raise BCError(
                    "expected comma after procedure argument list", self.pos()
                )
            elif self.check("comma"):
                self.consume()

        self.consume_and_expect("right_paren", "after variable list in TRACE statement")

        file_name: str | None = None
        if self.check_and_consume("to"):
            lit: Literal | None = self.literal()  # type: ignore
            if not lit:
                raise BCError(
                    "expected valid literal after TO keyword in TRACE statement!",
                    self.pos(),
                )

            val = lit.to_bcvalue()
            if val.kind != "string":
                raise BCError(
                    "expected string literal after TO keyword in TRACE statement\n"
                    + "pass the file name of the output trace table in a string.",
                    self.pos(),
                )
            file_name = val.get_string()

        self.consume_newlines()
        block = self.block("endtrace")
        self.consume()  # byebye ENDTRACE

        return TraceStatement(begin.pos, vars, file_name, block)

    def openfile_stmt(self) -> Statement | None:
        if self.check("openfile"):
            raise BCError("File I/O has not been implemented yet!", self.pos())

    def readfile_stmt(self) -> Statement | None:
        if self.check("readfile"):
            raise BCError("File I/O has not been implemented yet!", self.pos())

    def writefile_stmt(self) -> Statement | None:
        if self.check("writefile"):
            raise BCError("File I/O has not been implemented yet!", self.pos())

    def appendfile_stmt(self) -> Statement | None:
        if self.check("appendfile"):
            raise BCError("File I/O has not been implemented yet!", self.pos())

    def closefile_stmt(self) -> Statement | None:
        if self.check("closefile"):
            raise BCError("File I/O has not been implemented yet!", self.pos())

    def clean_newlines(self):
        while self.cur < len(self.tokens):
            if not self.check("newline"):
                break
            self.consume()

    def stmt(self) -> Statement | None:
        self.clean_newlines()

        constant = self.constant_stmt()
        if constant:
            return constant

        declare = self.declare_stmt()
        if declare:
            return declare

        output = self.output_stmt()
        if output:
            return output

        inp = self.input_stmt()
        if inp:
            return inp

        assign = self.assign_stmt()
        if assign:
            return assign

        proc_call = self.call_stmt()
        if proc_call:
            return proc_call

        return_s = self.return_stmt()
        if return_s:
            return return_s

        include = self.include_stmt()
        if include:
            return include

        trace = self.trace_stmt()
        if trace:
            return trace

        openfile = self.openfile_stmt()
        if openfile:
            return openfile

        readfile = self.readfile_stmt()
        if readfile:
            return readfile

        writefile = self.writefile_stmt()
        if writefile:
            return writefile

        appendfile = self.appendfile_stmt()
        if appendfile:
            return appendfile

        closefile = self.closefile_stmt()
        if closefile:
            return closefile

        if_s = self.if_stmt()
        if if_s:
            return if_s

        caseof = self.caseof_stmt()
        if caseof:
            return caseof

        while_s = self.while_stmt()
        if while_s:
            return while_s

        for_s = self.for_stmt()
        if for_s:
            return for_s

        repeatuntil_s = self.repeatuntil_stmt()
        if repeatuntil_s:
            return repeatuntil_s

        procedure = self.procedure_stmt()
        if procedure:
            return procedure

        function = self.function_stmt()
        if function:
            return function

        scope = self.scope_stmt()
        if scope:
            return scope

        cur = self.peek()
        expr = self.expression()
        if expr:
            return ExprStatement.from_expr(expr)
        else:
            raise BCError("invalid statement or expression", cur.pos)

    def scan_one_statement(self) -> Statement | None:
        s = self.stmt()

        if s:
            self.clean_newlines()
            return s
        else:
            if self.cur >= len(self.tokens):
                return

            p = self.peek()
            raise BCError(f"found invalid statement at `{p}`", p.pos)

    def reset(self):
        self.cur = 0

    def program(self) -> Program:
        stmts = []

        while self.cur < len(self.tokens):
            self.clean_newlines()
            if self.cur >= len(self.tokens):
                break

            stmt = self.scan_one_statement()
            if not stmt:  # this has to be an EOF
                continue
            stmts.append(stmt)

        self.cur = 0

        return Program(stmts=stmts)
