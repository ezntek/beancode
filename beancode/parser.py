from . import *
from .lexer import *
from .bean_ast import *
from .libroutines import LIBROUTINES
from .error import *


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

    def peek_and_expect(self, expected: TokenKind, ctx="", help="") -> Token:
        tok = self.peek()

        s = " " + ctx if ctx else ""
        h = "\n" + help if help else ""

        if tok.kind != expected:
            raise BCError(
                f"expected {expected.humanize()}{s}, but got {tok.to_humanized_string()}{h}",
                tok.pos,
            )
        return tok

    def peek_next_and_expect(self, expected: TokenKind, ctx="", help="") -> Token:
        tok = self.peek_next()

        s = " " + ctx if ctx else ""
        h = "\n" + help if help else ""

        if not tok:
            raise BCError(
                f"expected {expected.humanize()}{s}, but reached end of file{h}",
                self.pos(),
                eof=True,
            )

        if tok.kind != expected:
            raise BCError(
                f"expected {expected.humanize()}{s}, but got {tok.to_humanized_string()}{h}",
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

    def consume_and_expect(self, expected: TokenKind, ctx="", help="") -> Token:
        cons = self.consume()
        if cons.kind != expected:
            s = " " + ctx if ctx else ""
            h = "\n" + help if help else ""

            if help:
                h = "\n" + help
            raise BCError(
                f"expected {expected.humanize()}{s}, but got {cons.to_humanized_string()}{h}",
                cons.pos,
            )
        return cons

    def consume(self) -> Token:
        if self.cur < len(self.tokens):
            self.cur += 1
        else:
            prevpos = None
            if len(self.tokens) > 0:
                prevpos = self.prev().pos

            raise BCError("reached end of file", prevpos, eof=True)

        return self.prev()

    def consume_newlines(self):
        while self.cur < len(self.tokens) and self.peek().kind == TokenKind.NEWLINE:
            self.consume()

    def check_newline(self, s: str):
        self.consume_and_expect(TokenKind.NEWLINE, ctx=f"after {s}")

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
        lbrace = self.consume_and_expect(
            TokenKind.LEFT_CURLY, "for array or matrix literal"
        )

        exprs = []
        while not self.check(TokenKind.RIGHT_CURLY):
            self.clean_newlines()

            if self.check(TokenKind.LEFT_CURLY):
                if nested:
                    raise BCError(
                        "cannot nest array literals over 2 dimensions!", self.pos()
                    )
                arrlit = self.array_literal(nested=True)
                exprs.append(arrlit)
            else:
                expr = self.expr()
                if not expr:
                    raise BCError(
                        "invalid or no expression supplied as argument to array literal",
                        self.pos(),
                    )
                exprs.append(expr)

            self.clean_newlines()
            comma = self.peek()
            if comma.kind == TokenKind.RIGHT_CURLY:
                break
            elif comma.kind != TokenKind.COMMA:
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

        self.check_and_consume(TokenKind.RIGHT_CURLY)

        return ArrayLiteral(lbrace.pos, exprs)

    def literal(self) -> Expr | None:
        if not self.check_many(
            TokenKind.LITERAL_STRING,
            TokenKind.LITERAL_CHAR,
            TokenKind.LITERAL_NUMBER,
            TokenKind.TRUE,
            TokenKind.FALSE,
            TokenKind.NULL,
        ):
            return

        lit = self.consume()
        match lit.kind:
            case TokenKind.LITERAL_CHAR:
                if len(lit.data) == 0:  # type: ignore
                    raise BCError(
                        "CHAR literal cannot have no characters in it!", lit.pos
                    )

                val: str = lit.data  # type: ignore
                if val[0] == "\\":
                    if len(val) == 1:
                        return Literal(lit.pos, BCValue.new_char("\\"))

                    ch = _convert_escape_code(val[1])
                    if not ch:
                        raise BCError(
                            f"invalid escape sequence in literal '{val}'",
                            lit.pos,
                        )

                    return Literal(lit.pos, BCValue.new_char(ch))
                else:
                    if len(val) > 1:
                        raise BCError(
                            f"more than 1 character in char literal '{val}'", lit.pos
                        )
                    return Literal(lit.pos, BCValue.new_char(val[0]))
            case TokenKind.LITERAL_STRING:
                val: str = lit.data  # type: ignore
                res = list()
                i = 0

                while i < len(val):
                    if val[i] == "\\":
                        if i == len(val) - 1:
                            res.append("\\")
                        else:
                            i += 1
                            ch = _convert_escape_code(val[i])
                            if not ch:
                                pos = lit.pos.copy()
                                pos.col += i
                                pos.span = 2
                                raise BCError(
                                    f'invalid escape sequence in literal "{val}"',
                                    pos,
                                )
                            res.append(ch)
                    else:
                        res.append(val[i])
                    i += 1

                return Literal(lit.pos, BCValue.new_string("".join(res)))
            case TokenKind.LITERAL_NUMBER:
                val: str = lit.data  # type: ignore

                if is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        raise BCError(f'invalid number literal "{val}"', lit.pos)

                    return Literal(lit.pos, BCValue.new_real(res))
                elif is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        raise BCError(f'invalid number literal "{val}"', lit.pos)

                    return Literal(lit.pos, BCValue.new_integer(res))
                else:
                    raise BCError(f'invalid number literal "{val}"', lit.pos)
            case TokenKind.TRUE:
                return Literal(lit.pos, BCValue.new_boolean(True))
            case TokenKind.FALSE:
                return Literal(lit.pos, BCValue.new_boolean(False))
            case TokenKind.NULL:
                return Literal(lit.pos, BCValue.new_null())

    def _array_type(self) -> Type:
        inner: BCPrimitiveType

        self.consume_and_expect(TokenKind.LEFT_BRACKET, "for array type declaration")
        begin = self.expr()
        if not begin:
            raise BCError(
                "invalid or no expression as beginning value of array declaration",
                begin,
            )

        self.consume_and_expect(
            TokenKind.COLON, "after beginning value of array declaration"
        )

        end = self.expr()
        if not end:
            raise BCError(
                "invalid or no expression as ending value of array declaration",
                end,
            )

        flat_bounds = (begin, end)
        matrix_bounds = None

        right_bracket = self.consume()
        if right_bracket.kind == TokenKind.RIGHT_BRACKET:
            pass
        elif right_bracket.kind == TokenKind.COMMA:
            inner_begin = self.expr()
            if not inner_begin:
                raise BCError(
                    "invalid or no expression as beginning value of array declaration",
                    inner_begin,
                )

            self.consume_and_expect(
                TokenKind.COLON, "after beginning value of array declaration"
            )

            inner_end = self.expr()
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

            self.consume_and_expect(
                TokenKind.RIGHT_BRACKET, "after matrix length declaration"
            )
        else:
            raise BCError(
                "expected right bracket or comma after array bounds declaration",
                right_bracket.pos,
            )

        self.consume_and_expect(TokenKind.OF, "after array size declaration")

        arrtyp = self.consume_and_expect(TokenKind.TYPE, "after array size declaration")
        if arrtyp.data == "array":
            raise BCError(
                "cannot have array as array element type, please use the matrix syntax instead",
                arrtyp.pos,
            )

        inner = BCPrimitiveType.from_str(arrtyp.data)  # type: ignore

        bounds = matrix_bounds if matrix_bounds else flat_bounds
        return ArrayType(inner, bounds)  # type: ignore

    def typ(self) -> Type:
        adv = self.consume_and_expect(TokenKind.TYPE)
        if adv.data == "array":
            return self._array_type()
        else:
            return BCPrimitiveType.from_str(adv.data)  # type: ignore

    def ident(self, ctx="", function=False) -> Identifier:
        c = self.consume_and_expect(TokenKind.IDENT, ctx=ctx)
        libroutine = False
        if not function and is_case_consistent(c.data) and c.data in LIBROUTINES:  # type: ignore
            libroutine = True
        return Identifier(c.pos, c.data, libroutine=libroutine)  # type: ignore

    def function_call(self) -> Expr | None:
        # avoid consuming tokens
        ident = self.peek()
        if ident.kind != TokenKind.IDENT:
            return

        leftb = self.peek_next()
        if not leftb:
            return
        if leftb.kind != TokenKind.LEFT_PAREN:
            return

        # consume both the ident and left_paren
        self.consume()
        self.consume()

        args = []
        while self.peek().kind != TokenKind.RIGHT_PAREN:
            expr = self.expr()
            if not expr:
                raise BCError(
                    "invalid or no expression as function argument", leftb.pos
                )

            args.append(expr)

            comma = self.peek()
            if comma.kind not in (TokenKind.COMMA, TokenKind.RIGHT_PAREN):
                raise BCError(
                    "expected comma or right parenthesis after argument in function call argument list",
                    comma.pos,
                )
            elif comma.kind == TokenKind.COMMA:
                self.consume()

        dat = str(ident.data)
        libroutine = False
        if is_case_consistent(dat) and dat.lower() in LIBROUTINES:
            dat = dat.lower()
            libroutine = True

        self.consume_and_expect(
            TokenKind.RIGHT_PAREN, "after argument list in function call"
        )
        return FunctionCall(leftb.pos, ident=dat, args=args, libroutine=libroutine)

    def typecast(self) -> Typecast | None:
        typ = self.consume_and_expect(TokenKind.TYPE, "for typecast")
        if typ.data == "array":
            # should be unreachable
            raise BCError("cannot typecast to an array!", typ.pos)

        t = BCPrimitiveType.from_str(typ.data)  # type: ignore
        self.consume()  # checked already

        expr = self.expr()
        if not expr:
            raise BCError("invalid or no expression supplied for type cast", typ.pos)

        self.consume_and_expect(TokenKind.RIGHT_PAREN, "after type cast expression")

        return Typecast(typ.pos, t, expr)

    def grouping(self) -> Expr | None:
        begin = self.consume_and_expect(TokenKind.LEFT_PAREN, "in grouping")
        e = self.expr()
        if not e:
            raise BCError("invalid or no expression inside grouping", begin.pos)

        self.consume_and_expect(TokenKind.RIGHT_PAREN, "after expression in grouping")
        return Grouping(begin.pos, inner=e)

    def unary(self) -> Expr | None:
        p = self.peek()

        lit = self.literal()
        if lit:
            return lit

        match p.kind:
            case TokenKind.NULL:
                return Literal(p.pos, BCValue.new_null())
            case TokenKind.LEFT_CURLY:
                return self.array_literal()
            case TokenKind.IDENT:
                pn = self.peek_next()
                if pn and pn.kind == TokenKind.LEFT_PAREN:
                    return self.function_call()
                return self.ident()
            case TokenKind.TYPE:
                pn = self.peek_next()
                if not pn:
                    return

                if pn.kind == TokenKind.LEFT_PAREN:
                    return self.typecast()
            case TokenKind.LEFT_PAREN:
                return self.grouping()
            case TokenKind.SUB:
                begin = self.consume()
                e = self.unary()
                if not e:
                    raise BCError("invalid or no expression for negation", begin.pos)
                return Negation(begin.pos, e)
            case TokenKind.NOT:
                begin = self.consume()
                e = self.expr()
                if not e:
                    raise BCError("invalid or no expression for logical NOT", begin.pos)
                return Not(begin.pos, e)
            case _:
                return

    def array_index(self) -> Expr | None:
        expr = self.unary()

        leftb = self.check_and_consume(TokenKind.LEFT_BRACKET)
        if not leftb:
            return expr

        exp = self.expr()
        if not exp:
            raise BCError("expected expression as array index", leftb.pos)

        rightb = self.consume()
        exp_inner = None
        if rightb.kind == TokenKind.RIGHT_BRACKET:
            pass
        elif rightb.kind == TokenKind.COMMA:
            exp_inner = self.expr()
            if not exp_inner:
                raise BCError("expected expression as array index", exp_inner)

            self.consume_and_expect(
                TokenKind.RIGHT_BRACKET, "after expression in array index"
            )
        else:
            raise BCError(
                "expected right_bracket or comma after expression in array index",
                rightb.pos,
            )

        return ArrayIndex(leftb.pos, expr, idx_outer=exp, idx_inner=exp_inner)  # type: ignore

    def array_index_or_none(self) -> ArrayIndex | None:
        saved_point = self.cur
        arridx = self.array_index()
        if not isinstance(arridx, ArrayIndex):
            self.cur = saved_point
            return
        else:
            return arridx

    def pow(self) -> Expr | None:
        expr = self.array_index()
        if not expr:
            return

        if self.match(TokenKind.POW):
            op_tok = self.prev()
            right = self.pow()

            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, Operator.POW, right)

        return expr

    def factor(self) -> Expr | None:
        expr = self.pow()
        if not expr:
            return

        if self.check(TokenKind.IDENT) and is_case_consistent(s := self.peek().data.lower()):  # type: ignore
            if s == "div":
                raise BCError(
                    "DIV is not an infix operator!\nPlease use the DIV(a, b) library routine instead of a DIV b!",
                    self.pos(),
                )
            elif s == "mod":
                raise BCError(
                    "MOD is not an infix operator!\nPlease use the MOD(a, b) library routine instead of a MOD b!",
                    self.pos(),
                )

        while self.match(TokenKind.MUL, TokenKind.DIV):
            op_tok = self.prev()
            op = Operator.from_token_kind(op_tok.kind)

            right = self.pow()

            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, op, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if not expr:
            return

        while self.match(TokenKind.ADD, TokenKind.SUB):
            op_tok = self.prev()
            op = Operator.from_token_kind(op_tok.kind)

            right = self.factor()
            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, op, right)  # type: ignore

        return expr

    def comparison(self) -> Expr | None:
        # > < >= <=
        expr = self.term()
        if not expr:
            return

        while self.match(
            TokenKind.GREATER_THAN,
            TokenKind.LESS_THAN,
            TokenKind.GREATER_THAN_OR_EQUAL,
            TokenKind.LESS_THAN_OR_EQUAL,
        ):
            op_tok = self.prev()
            op = Operator.from_token_kind(op_tok.kind)

            right = self.term()
            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, op, right)  # type: ignore

        return expr

    def equality(self) -> Expr | None:
        expr = self.comparison()

        if not expr:
            return

        while self.match(TokenKind.EQUAL, TokenKind.NOT_EQUAL):
            op_tok = self.prev()
            op = Operator.from_token_kind(op_tok.kind)

            right = self.comparison()
            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, op, right)  # type: ignore

        return expr

    def logical_comparison(self) -> Expr | None:
        expr = self.equality()
        if not expr:
            return

        while self.match(TokenKind.AND, TokenKind.OR):
            op_tok = self.prev()
            op = Operator.from_token_kind(op_tok.kind)  # type: ignore

            right = self.equality()

            if not right:
                return

            expr = BinaryExpr(op_tok.pos, expr, op, right)  # type: ignore

        return expr

    def expr(self) -> Expr | None:
        return self.logical_comparison()

    def output_stmt(self) -> Statement | None:
        exprs = []
        begin = self.peek()

        if begin.kind not in (TokenKind.OUTPUT, TokenKind.PRINT):
            return

        newline = True
        if begin.kind == TokenKind.PRINT:
            newline = False

        self.consume()
        initial = self.expr()
        if not initial:
            raise BCError(
                "found OUTPUT but an invalid or no expression that follows", begin.pos
            )

        exprs.append(initial)

        while self.match(TokenKind.COMMA):
            new = self.expr()
            if not new:
                break

            exprs.append(new)

        return OutputStatement(begin.pos, items=exprs, newline=newline)

    def input_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.INPUT)
        if not begin:
            return

        ident: ArrayIndex | Identifier
        array_index = self.array_index_or_none()
        if not array_index:
            ident = self.ident("after INPUT")
        else:
            ident = array_index  # type: ignore

        return InputStatement(begin.pos, ident)

    def return_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.RETURN)
        if not begin:
            return

        if self.check(TokenKind.NEWLINE):
            return ReturnStatement(begin.pos)

        expr = self.expr()
        if not expr:
            raise BCError(
                "invalid or no expression used as RETURN expression", begin.pos
            )

        return ReturnStatement(begin.pos, expr)

    def call_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.CALL)
        if not begin:
            return

        ident = self.ident("after procedure call")

        leftb = self.peek()
        args = []
        if leftb.kind == TokenKind.LEFT_PAREN:
            self.consume()
            while self.peek().kind != TokenKind.RIGHT_PAREN:
                expr = self.expr()
                if not expr:
                    raise BCError(
                        "invalid or no expression as procedure argument", leftb.pos
                    )

                args.append(expr)

                comma = self.peek()
                if (
                    comma.kind != TokenKind.COMMA
                    and comma.kind != TokenKind.RIGHT_PAREN
                ):
                    raise BCError(
                        "expected comma after argument in procedure call argument list",
                        comma.pos,
                    )
                elif comma.kind == TokenKind.COMMA:
                    self.consume()

            self.consume_and_expect(
                TokenKind.RIGHT_PAREN, "after arg list in procedure call"
            )

        self.consume_newlines()

        libroutine = False
        if is_case_consistent(ident.ident) and ident.ident.lower() in LIBROUTINES:
            libroutine = True

        return CallStatement(
            begin.pos, ident=ident.ident, args=args, libroutine=libroutine
        )

    def declare_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.kind == TokenKind.EXPORT:
            export = True
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )

        if begin.kind != TokenKind.DECLARE:
            return

        # consume the keyword
        self.consume()
        if export == True:
            self.consume()

        idents = []
        ident = self.consume_and_expect(TokenKind.IDENT, "after declare statement")
        idents.append(Identifier(ident.pos, str(ident.data)))

        while self.check(TokenKind.COMMA):
            self.consume()  # consume the sep
            if self.check(TokenKind.COLON):
                break

            ident = self.consume_and_expect(
                TokenKind.IDENT, "after comma in declare statement"
            )
            idents.append(Identifier(ident.pos, str(ident.data)))

        typ = None
        expr = None

        colon = self.peek()
        if self.check(TokenKind.COLON):
            self.consume()

            typ = self.typ()
            if not typ:
                raise BCError("invalid type after DECLARE", colon.pos)

        if self.check(TokenKind.ASSIGN):
            tok = self.consume()
            if len(idents) > 1:
                raise BCError(
                    "cannot have assignment in declaration of multiple variables",
                    tok.pos,
                )

            expr = self.expr()
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

        if self.check(TokenKind.EXPORT):
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != TokenKind.CONSTANT:
            return

        # consume the kw
        self.consume()
        if export == True:
            self.consume()

        ident = self.consume_and_expect(TokenKind.IDENT, "after constant declaration")
        self.consume_and_expect(
            TokenKind.ASSIGN, "after variable name in constant declaration"
        )

        expr = self.expr()
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

        if p.kind == TokenKind.LEFT_BRACKET:
            temp_idx = self.cur
            while self.tokens[temp_idx].kind != TokenKind.RIGHT_BRACKET:
                temp_idx += 1
                if temp_idx == len(self.tokens):
                    raise BCError(
                        "reached end of file while searching for end delimiter `]`",
                        self.tokens[-1].pos,
                        eof=True,
                    )

            p = self.tokens[temp_idx + 1]

        if p.kind != TokenKind.ASSIGN:
            return

        ident = self.array_index_or_none()
        if not ident:
            ident = self.ident("for left hand side of assignment")

        self.consume()  # go past the arrow

        expr: Expr | None = self.expr()
        if not expr:
            raise BCError("expected expression after `<-` in assignment", p.pos)

        self.check_newline("assignment")

        is_ident = False
        if isinstance(ident, Identifier):
            is_ident = True

        return AssignStatement(ident.pos, ident, expr, is_ident=is_ident)  # type: ignore

    # multiline statements go here
    def block(self, delim: TokenKind) -> list[Statement]:
        res = list()
        while not self.check(delim):
            res.append(self.scan_one_statement())
        return res

    def if_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.IF)
        if not begin:
            return

        cond = self.expr()
        if not cond:
            raise BCError("found invalid or no expression for if condition", self.pos())

        # allow stupid igcse stuff
        if self.peek().kind == TokenKind.NEWLINE:
            self.clean_newlines()

        self.consume_and_expect(TokenKind.THEN, "after if condition")
        self.clean_newlines()

        if_stmts = []
        else_stmts = []

        while not self.check_many(TokenKind.ELSE, TokenKind.ENDIF):
            if_stmts.append(self.scan_one_statement())

        if self.check_and_consume(TokenKind.ELSE):
            self.clean_newlines()
            else_stmts = self.block(TokenKind.ENDIF)

        self.consume()  # byebye endif

        return IfStatement(
            begin.pos, cond=cond, if_block=if_stmts, else_block=else_stmts
        )

    def caseof_stmt(self) -> Statement | None:
        case = self.check_and_consume(TokenKind.CASE)
        if not case:
            return

        self.consume_and_expect(TokenKind.OF, "after CASE keyword")

        main_expr = self.expr()
        if not main_expr:
            raise BCError(
                "found invalid or no expression for case of value", self.pos()
            )

        self.check_newline("after case of expression")

        branches: list[CaseofBranch] = []
        otherwise: Statement | None = None
        while not self.check(TokenKind.ENDCASE):
            is_otherwise = self.check(TokenKind.OTHERWISE)
            if not is_otherwise:
                expr = self.expr()
                if not expr:
                    raise BCError(
                        "invalid or no expression for case of branch", self.pos()
                    )

                self.consume_and_expect(
                    TokenKind.COLON, "after case of branch expression"
                )
            else:
                self.consume()
                if self.check(TokenKind.COLON):
                    raise BCError(
                        'a colon ":" after OTHERWISE is A-level Pseudocode syntax!\nRemove this colon.',
                        self.pos(),
                    )

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
        begin = self.check_and_consume(TokenKind.WHILE)
        if not begin:
            return

        expr = self.expr()
        if not expr:
            raise BCError(
                "found invalid or no expression for while loop condition", self.pos()
            )

        self.clean_newlines()
        self.consume_and_expect(TokenKind.DO, "after while loop condition")
        self.clean_newlines()

        stmts = self.block(TokenKind.ENDWHILE)
        end = self.consume()

        return WhileStatement(begin.pos, end.pos, expr, stmts)

    def for_stmt(self):
        initial = self.check_and_consume(TokenKind.FOR)
        if not initial:
            return

        counter = self.ident("for for loop counter")

        self.consume_and_expect(TokenKind.ASSIGN, "after counter in for loop")

        begin = self.expr()
        if not begin:
            raise BCError("invalid or no expression as begin in for loop", self.pos())

        self.consume_and_expect(TokenKind.TO, "after beginning value in for loop")

        end = self.expr()
        if not end:
            raise BCError("invalid or no expression as end in for loop", self.pos())

        step: Expr | None = None
        if self.check(TokenKind.STEP):
            self.consume()
            step = self.expr()
            if not step:
                raise BCError(
                    "invalid or no expression as step in for loop", self.pos()
                )

        self.clean_newlines()
        stmts = self.block(TokenKind.NEXT)
        next = self.consume()

        next_counter = self.ident("after NEXT in for loop")

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
        begin = self.check_and_consume(TokenKind.REPEAT)
        if not begin:
            return

        self.clean_newlines()

        stmts = self.block(TokenKind.UNTIL)
        until = self.consume()

        expr = self.expr()
        if not expr:
            raise BCError(
                "found invalid or no expression for repeat-until loop condition",
                self.pos(),
            )

        return RepeatUntilStatement(begin.pos, until.pos, expr, stmts)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.ident("for function argument")
        colon = self.consume_and_expect(
            TokenKind.COLON, "after identifier in function argument"
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

        if begin.kind == TokenKind.EXPORT:
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != TokenKind.PROCEDURE:
            return

        self.consume()  # byebye PROCEDURE
        if export == True:
            self.consume()

        ident = self.ident("after PROCEDURE declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == TokenKind.LEFT_PAREN:
            # there is an arg list
            self.consume()
            while not self.check(TokenKind.RIGHT_PAREN):
                arg = self.function_arg()
                if not arg:
                    raise BCError("invalid function argument", self.pos())

                args.append(arg)

                if not self.check_many(TokenKind.COMMA, TokenKind.RIGHT_PAREN):
                    raise BCError(
                        "expected comma after procedure argument list", self.pos()
                    )

                if self.check(TokenKind.COMMA):
                    self.consume()

            self.consume_and_expect(
                TokenKind.RIGHT_PAREN, "after argument list in procedure declaration"
            )

        self.consume_newlines()

        stmts = self.block(TokenKind.ENDPROCEDURE)

        self.consume()

        return ProcedureStatement(
            begin.pos, name=ident.ident, args=args, block=stmts, export=export
        )

    def function_stmt(self) -> Statement | None:
        begin = self.peek()
        export = False

        if begin.kind == TokenKind.EXPORT:
            begin = self.peek_next()
            if not begin:
                raise BCError(
                    "expected token following export, but got end of file",
                    begin,
                    eof=True,
                )
            export = True

        if begin.kind != TokenKind.FUNCTION:
            return

        self.consume()  # byebye FUNCTION
        if export == True:
            self.consume()

        ident = self.ident("after FUNCTION declaration")

        args = []
        leftb = self.peek()
        if leftb.kind == TokenKind.LEFT_PAREN:
            # there is an arg list
            self.consume()
            while not self.check(TokenKind.RIGHT_PAREN):
                arg = self.function_arg()
                if not arg:
                    raise BCError("invalid function argument", self.pos())

                args.append(arg)

                if not self.check_many(TokenKind.COMMA, TokenKind.RIGHT_PAREN):
                    raise BCError(
                        "expected comma after function argument in list", self.pos()
                    )

                if self.check(TokenKind.COMMA):
                    self.consume()

            self.consume_and_expect(
                TokenKind.RIGHT_PAREN, "after argument list in function declaration"
            )

        self.consume_and_expect(TokenKind.RETURNS, "after function arguments")

        typ = self.typ()
        if not typ:
            raise BCError(
                "invalid type after RETURNS for function return value", self.pos()
            )

        self.consume_newlines()

        stmts = self.block(TokenKind.ENDFUNCTION)
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
        begin = self.check_and_consume(TokenKind.SCOPE)
        if not begin:
            return

        self.clean_newlines()
        stmts = self.block(TokenKind.ENDSCOPE)
        self.consume()

        return ScopeStatement(begin.pos, stmts)

    def include_stmt(self) -> Statement | None:
        if not self.check_many(TokenKind.INCLUDE, TokenKind.INCLUDE_FFI):
            return
        include = self.consume()

        ffi = False
        if include.kind == TokenKind.INCLUDE_FFI:
            ffi = True

        name = self.consume()
        if name.kind != TokenKind.LITERAL_STRING:
            raise BCError(
                "include must be followed by a literal of the name of the file to include",
                name.pos,
            )

        return IncludeStatement(include.pos, str(name.data), ffi=ffi)  # type: ignore

    def trace_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.TRACE)
        if not begin:
            return

        self.consume_and_expect(TokenKind.LEFT_PAREN, "after TRACE keyword")

        vars = list()
        while not self.check(TokenKind.RIGHT_PAREN):
            ident = self.ident("in variable list in TRACE statement")

            vars.append(ident.ident)  # type: ignore

            if not self.check_many(TokenKind.COMMA, TokenKind.RIGHT_PAREN):
                raise BCError(
                    "expected comma after procedure argument list", self.pos()
                )
            elif self.check(TokenKind.COMMA):
                self.consume()

        self.consume_and_expect(
            TokenKind.RIGHT_PAREN, "after variable list in TRACE statement"
        )

        file_name: str | None = None
        if self.check_and_consume(TokenKind.TO):
            lit: Literal | None = self.literal()  # type: ignore
            if not lit:
                raise BCError(
                    "expected valid literal after TO keyword in TRACE statement\n"
                    + "pass the file name of the output trace table in a string.",
                    self.pos(),
                )

            val = lit.val
            if val.kind != BCPrimitiveType.STRING:
                raise BCError(
                    "expected string literal after TO keyword in TRACE statement\n"
                    + "pass the file name of the output trace table in a string.",
                    self.pos(),
                )
            file_name = val.get_string()

        self.consume_newlines()
        block = self.block(TokenKind.ENDTRACE)
        self.consume()  # byebye ENDTRACE

        return TraceStatement(begin.pos, vars, file_name, block)

    def _file_id(self, ctx: str) -> Expr | str:
        s = self.check_and_consume(TokenKind.LITERAL_STRING)
        if not s:
            expr = self.expr()
            if not expr:
                raise BCError(
                    f"expected expression, file identifier or string literal after {ctx}!\n"
                    + "pass the name of the file as a string literal or bare file name.",
                    self.pos(),
                )
            return expr
        else:
            return str(s.data)

    def openfile_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.OPENFILE)
        if not begin:
            return

        filename: Expr | str = self._file_id("OPENFILE")

        self.consume_and_expect(TokenKind.FOR)

        read = False
        write = False
        append = False

        while not self.check(TokenKind.NEWLINE):
            mode = self.consume()
            if mode.kind in (TokenKind.READ, TokenKind.WRITE, TokenKind.APPEND):
                match mode.kind:
                    case TokenKind.READ:
                        if read:
                            raise BCError("duplicate file mode READ", mode.pos)
                        read = True
                    case TokenKind.WRITE:
                        if write:
                            raise BCError("duplicate file mode WRITE", mode.pos)
                        write = True
                    case TokenKind.APPEND:
                        if append:
                            raise BCError("duplicate file mode APPEND", mode.pos)
                        append = True

                if append and write:
                    raise BCError(
                        f"you cannot open a file for APPEND and WRITE!", mode.pos
                    )
            else:
                raise BCError(
                    f"unrecognized file mode!\n"
                    + "supported modes are READ, WRITE and APPEND.",
                    self.pos(),
                )

            self.check_and_consume(TokenKind.AND)

        if not (read or write or append):
            raise BCError("No file modes specified!", begin.pos)

        return OpenfileStatement(begin.pos, filename, (read, write, append))

    def readfile_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.READFILE)
        if not begin:
            return

        fileid: Expr | str = self._file_id("READFILE")

        self.consume_and_expect(TokenKind.COMMA)

        val: ArrayIndex | Identifier | None = self.array_index_or_none()  # type: ignore
        if not val:
            val = self.ident()

        return ReadfileStatement(begin.pos, fileid, val)  # type: ignore

    def writefile_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.WRITEFILE)
        if not begin:
            return

        fileid: Expr | str = self._file_id("WRITEFILE")

        self.consume_and_expect(TokenKind.COMMA)

        val = self.expr()
        if not val:
            raise BCError(
                "invalid or no expression after comma in WRITEFILE statement",
                self.pos(),
            )

        return WritefileStatement(begin.pos, fileid, val)

    def appendfile_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.APPENDFILE)
        if not begin:
            return

        fileid: Expr | str = self._file_id("APPENDFILE")

        self.consume_and_expect(TokenKind.COMMA)

        val = self.expr()
        if not val:
            raise BCError(
                "invalid or no expression after comma in APPENDFILE statement",
                self.pos(),
            )

        return AppendfileStatement(begin.pos, fileid, val)

    def closefile_stmt(self) -> Statement | None:
        begin = self.check_and_consume(TokenKind.CLOSEFILE)
        if not begin:
            return

        fileid: Expr | str = self._file_id("CLOSEFILE")

        return ClosefileStatement(begin.pos, fileid)

    def clean_newlines(self):
        while self.cur < len(self.tokens):
            if not self.check(TokenKind.NEWLINE):
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
        expr = self.expr()
        if expr:
            return ExprStatement.from_expr(expr)
        else:
            DIDNT_END = "did you forget to end a statement (if, while, etc.) earlier?"
            if cur.kind == TokenKind.NEXT:
                raise BCError("cannot end FOR loop now!\n" + DIDNT_END, cur.pos)
            elif str(cur.kind)[:3] == "end":
                end = str(cur.kind)[3:].upper()
                if end == "CASE":
                    end += " OF"
                raise BCError(
                    f"cannot end a {end} statement now!\n" + DIDNT_END, cur.pos
                )
            else:
                raise BCError(f"unexpected token: {cur.kind.humanize()}", cur.pos)

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
