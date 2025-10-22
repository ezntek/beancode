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
            return None


class Parser:
    tokens: list[Token]
    cur: int

    def __init__(self, tokens: list[Token]) -> None:
        self.cur = 0
        self.tokens = tokens
        self.PRIM_TYPES = ["integer", "real", "boolean", "char", "string"]

    def prev(self) -> Token:
        return self.tokens[self.cur - 1]

    def peek(self) -> Token:
        if self.cur >= len(self.tokens):
            raise BCError(
                f"unexpected end of file", self.tokens[-1].pos, eof=True
            )

        return self.tokens[self.cur]

    def peek_next(self) -> Token | None:
        if self.cur + 1 >= len(self.tokens):
            return None

        return self.tokens[self.cur + 1]

    def peek_and_expect(self, expected: TokenKind) -> Token:
        tok = self.peek()
        if tok.kind != expected:
            raise BCError(f"expected token {expected}, but got {tok}", tok.pos)
        return tok

    def check(self, tok: TokenKind) -> bool:
        if self.cur >= len(self.tokens):
            return False

        return self.peek().kind == tok 

    def check_and_consume(self, expected: TokenKind) -> Token | None:
        if self.check(expected):
            return self.consume()
        else:
            return None

    def consume_and_expect(self, expected: TokenKind, ctx=str(), help=str()) -> Token:
        cons = self.consume()
        if cons.kind != expected:
            s = str()
            if ctx is not None:
                s = " " + ctx
            h = str()
            if help is not None:
                h = '\n' + help
            raise BCError(f"expected token {expected}{s}, but got {cons}\n{h}", cons.pos)
        return cons
       
    def check_and_expect(self, expected: TokenKind, ctx=str(), help=str()) -> Token:
        """check_and_consume, then expect"""

        t = self.check_and_consume(expected)
        if t is None:
            s = str()
            if ctx is not None:
                s = ' ' + ctx

            h = str()
            if help is not None:
                h = '\n' + help
            
            raise BCError(f"expected token {expected}{s}, but got {t}\n{h}", self.peek().pos)
        return t

    def consume(self) -> Token:
        if self.cur < len(self.tokens):
            self.cur += 1

        return self.prev()

    def consume_newlines(self):
        while self.cur != len(self.tokens) - 1 and self.peek().kind == "newline":
            self.consume()

    def check_newline(self, s: str):
        self.consume_and_expect("newline", ctx=f"after {s}")

    def match(self, typs: list[TokenKind]) -> bool:
        for typ in typs:
            if self.check(typ):
                self.consume()
                return True
        return False

    def is_one_of(self, typs: list[TokenKind]) -> bool:
        for typ in typs:
            if self.check(typ):
                return True
        return False

    def is_integer(self, val: str) -> bool:
        for ch in val:
            if not ch.isdigit():
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
        lbrace = self.consume_and_expect("left_curly", "for array or matrix literal")

        exprs = []
        while self.peek().kind != "right_curly":
            self.clean_newlines()

            if self.peek().kind == "left_curly":
                if nested:
                    raise BCError(
                        "cannot nest array literals over 2 dimensions!", self.peek().pos
                    )
                arrlit = self.array_literal(nested=True)
                exprs.append(arrlit)
            else:
                expr = self.expression()
                if expr is None:
                    raise BCError(
                        "invalid or no expression supplied as argument to array literal",
                        self.peek().pos,
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
                self.peek().pos,
            )

        self.check_and_consume("right_curly")

        return ArrayLiteral(lbrace.pos, exprs)

    def literal(self) -> Expr | None:
        if not self.match(["literal_string", "literal_char", "literal_number", "true", "false", "null"]):
            return None

        lit = self.consume()

        match lit.kind:
            case "literal_char":
                if len(lit.data) == 0: # type: ignore
                    raise BCError(
                        "CHAR literal cannot have no characters in it!", lit.pos
                    )

                val: str = lit.data # type: ignore 
                if val[0] == "\\":
                    if len(val) == 1:
                        return Literal(lit.pos, "char", char="\\")

                    ch = _convert_escape_code(val[1])
                    if ch is None:
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
                val: str = lit.data # type: ignore
                res = StringIO()
                i = 0

                while i < len(val):
                    if val[i] == "\\":
                        if i == len(val) - 1:
                            res.write("\\")
                        else:
                            i += 1
                            ch = _convert_escape_code(val[i])
                            if ch is None:
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
                val: str = lit.data # type: ignore

                if self.is_real(val):
                    try:
                        res = float(val)
                    except ValueError:
                        raise BCError(f"invalid number literal \"{val}\"", lit.pos)

                    return Literal(lit.pos, "real", real=res)
                elif self.is_integer(val):
                    try:
                        res = int(val)
                    except ValueError:
                        raise BCError(f"invalid number literal \"{val}\"", lit.pos)

                    return Literal(lit.pos, "integer", integer=res)
                else:
                    raise BCError(f"invalid number literal \"{val}\"", lit.pos)
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
        if begin is None:
            raise BCError(
                "invalid or no expression as beginning value of array declaration",
                begin,
            )

        self.consume_and_expect("colon", "after beginning value of array declaration")

        end = self.expression()
        if end is None:
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
            if inner_begin is None:
                raise BCError(
                    "invalid or no expression as beginning value of array declaration",
                    inner_begin,
                )

            self.consume_and_expect("colon", "after beginning value of array declaration")

            inner_end = self.expression()
            if inner_end is None:
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
        if arrtyp.data not in self.PRIM_TYPES:
            raise BCError("invalid type used as array element type", arrtyp.pos)

        inner = arrtyp.typ  # type: ignore

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
            t: BCPrimitiveType = adv.data # type: ignore
            return t

    def ident(self) -> Identifier:
        c = self.consume_and_expect("ident")
        return Identifier(c.pos, c.data) # type: ignore

    def array_index(self) -> Expr | None:
        pn = self.peek_next()
        if pn is None:
            return None
        if pn.kind != "left_bracket":
            return None

        ident = self.ident()

        leftb = self.consume_and_expect("left_bracket")
        exp = self.expression()
        if exp is None:
            raise BCError("expected expression as array index", leftb.pos)

        rightb = self.consume()
        exp_inner = None
        if rightb.kind == "right_bracket":
            pass
        elif rightb.kind == "comma":
            exp_inner = self.expression()
            if exp_inner is None:
                raise BCError("expected expression as array index", exp_inner)

            self.consume_and_expect("right_bracket", "after expression in array index")
        else:
            raise BCError(
                "expected right_bracket or comma after expression in array index", rightb.pos
            )

        return ArrayIndex(leftb.pos, ident=ident, idx_outer=exp, idx_inner=exp_inner) 

    def function_call(self) -> Expr | None:
        # avoid consuming tokens
        ident = self.peek()
        if ident.kind != "ident":
            return None

        leftb = self.peek_next()
        if leftb is None:
            return None
        if leftb.kind != "left_paren":
            return None

        # consume both the ident and left_paren
        self.consume()
        self.consume()

        args = []
        while self.peek().kind != "right_paren":
            expr = self.expression()
            if expr is None:
                raise BCError("invalid or no expression as function argument", leftb)

            args.append(expr)

            comma = self.peek()
            if comma.kind != "comma" and comma.kind != "right_paren":
                raise BCError(
                    "expected comma or right parenthesis after argument in function call argument list",
                    comma.pos,
                )
            elif comma.kind == "comma":
                self.consume()

        self.consume_and_expect("right_paren", "after argument list in function call")
        return FunctionCall(leftb.pos, ident=str(ident.data), args=args) 

    def typecast(self) -> Typecast | None:
        typ = self.check_and_expect("type", "for typecast")
        if typ.data == "array":
            raise BCError("cannot typecast to an array!")

        t: BCPrimitiveType = typ.data # type: ignore
        self.consume()  # checked already

        expr = self.expression()
        if expr is None:
            raise BCError("invalid or no expression supplied for type cast", expr)

        self.consume_and_expect("right_paren", "after type cast expression")

        return Typecast(typ.pos, t, expr)

    def grouping(self) -> Expr | None:
        begin = self.consume_and_expect("left_paren", "in grouping")
        e = self.expression()
        if e is None:
            raise BCError("invalid or no expression inside grouping", e)

        self.consume_and_expect("right_paren", "after expression in grouping")
        return Grouping(begin.pos, inner=e)

    def unary(self) -> Expr | None:
        p = self.peek()
        if p.kind == "literal":
            return self.literal()
        elif p.kind == "left_curly":
            return self.array_literal()
        elif p.kind == "ident":
            pn = self.peek_next()
            if pn is None:
                return None

            if pn.kind == "left_bracket":
                return self.array_index()
            elif pn.kind == "left_paren":
                return self.function_call()

            return self.ident()
        elif p.kind == "type":
            pn = self.peek_next()
            if pn is None:
                return None

            if pn.kind == "left_paren":
                return self.typecast()
        elif p.kind == "left_paren":
            return self.grouping()
        elif p.kind == "sub":
            begin = self.consume()
            e = self.unary()
            if e is None:
                raise BCError("invalid or no expression for negation", begin.pos)
            return Negation(begin.pos, e)
        elif p.kind == "not":
            begin = self.consume()
            e = self.expression()
            if e is None:
                raise BCError("invalid or no expression for logical NOT", begin.pos)
            return Not(begin.pos, e)
        else:
            return None

    def pow(self) -> Expr | None:
        expr = self.unary()
        if expr is None:
            return None

        if self.peek().kind == "pow":
            right = self.pow()
            expr = BinaryExpr(expr.pos, expr, "pow", right)  # type: ignore

        return expr

    def factor(self) -> Expr | None:
        expr = self.pow()
        if expr is None:
            return None

        while self.match(["mul", "div"]):
            op: Operator = self.prev().kind # type: ignore

            right = self.pow()

            if right is None:
                return None

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore

        return expr

    def term(self) -> Expr | None:
        expr = self.factor()

        if expr is None:
            return None

        while self.match(["add", "sub"]):
            op: Operator = self.prev().kind # type: ignore

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
                "greater_than",
                "less_than",
                "greater_than_or_equal",
                "less_than_or_equal",
            ]
        ):
            op: Operator = self.prev().kind # type: ignore

            right = self.term()
            if right is None:
                return None

            expr = BinaryExpr(expr.pos, expr, op, right)  # type: ignore
        return expr

    def equality(self) -> Expr | None:
        expr = self.comparison()

        if expr is None:
            return None

        while self.match(["equal", "not_equal"]):
            op: Operator = self.prev().kind # type: ignore

            right = self.comparison()
            if right is None:
                return None

            expr = BinaryExpr(expr.pos, expr, op, right)

        return expr

    def logical_comparison(self) -> Expr | None:
        expr = self.equality()
        if expr is None:
            return None

        while self.match(["and", "or"]):
            op: Operator = self.prev().kind # type: ignore

            right = self.equality()

            if right is None:
                return None

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
                "found OUTPUT but an invalid or no expression that follows", begin
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

        ident: ArrayIndex | Identifier

        array_index = self.array_index()
        if array_index is None:
            ident_exp = self.ident()
            if not isinstance(ident_exp, Identifier) or ident_exp.ident is None:
                raise BCError(f"found invalid identifier after INPUT", begin)
            ident = ident_exp
        else:
            ident = array_index  # type: ignore

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
            raise BCError("invalid or no expression used as RETURN expression", begin)

        return Statement("return", return_s=ReturnStatement(begin.pos, expr))

    def call_stmt(self) -> Statement | None:
        begin = self.peek()

        if begin.keyword != "call":
            return

        self.consume()

        # CALL <ident>(<expr>, <expr>)
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid ident after procedure call", begin)

        leftb = self.peek()
        args = []
        if leftb.kind == "separator" and leftb.separator == "left_paren":
            self.consume()
            while self.peek().separator != "right_paren":
                expr = self.expression()
                if expr is None:
                    raise BCError(
                        "invalid or no expression as procedure argument", leftb
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
                    "expected right paren after arg list in procedure call", leftb
                )

        self.consume_newlines()

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
        self.consume()
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

        colon = self.peek()
        if colon.separator == "colon":
            self.consume()

            typ = self.typ()
            if typ is None:
                raise BCError("invalid type after DECLARE", colon)

        if self.peek().operator == "assign":
            tok = self.consume()
            if len(idents) > 1:
                raise BCError(
                    "cannot have assignment in declaration of multiple variables",
                    tok.pos,
                )

            expr = self.expression()
            if expr is None:
                raise BCError(
                    "invalid or no expression after assign in declare", tok.pos
                )

        if typ is None and expr is None:
            raise BCError(
                "must have either a type declaration, expression to assign as, or both",
                colon,
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
                "expected `<-` after variable name in constant declaration", begin
            )

        expr = self.expression()
        if expr is None:
            raise BCError(
                "invalid or no expression for constant declaration", self.peek()
            )

        self.check_newline("constant declaration (CONSTANT)")

        res = ConstantStatement(
            begin.pos, Identifier(ident.pos, ident.ident), expr, export=export
        )
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

            if ident is None:
                raise BCError("invalid left hand side of assignment", p)

        self.consume()  # go past the arrow

        expr: Expr | None = self.expression()
        if expr is None:
            raise BCError("expected expression after `<-` in assignment", p)

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
            raise BCError(
                "found invalid or no expression for if condition", self.peek()
            )

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

        if case.keyword != "case":
            return
        self.consume()

        if self.peek().keyword != "of":
            return
        self.consume()

        main_expr = self.expression()
        if main_expr is None:
            raise BCError(
                "found invalid or no expression for case of value", self.peek()
            )

        self.check_newline("after case of expression")

        branches: list[CaseofBranch] = []
        otherwise: Statement | None = None
        next_expr: Expr | None = None
        while self.peek().keyword != "endcase":
            is_otherwise = self.peek().keyword == "otherwise"

            if not is_otherwise:
                expr = self.expression() if next_expr is None else next_expr
                if not expr:
                    raise BCError(
                        "invalid or no expression for case of branch", self.peek()
                    )

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

            # self.check_newline("CASE OF branch")
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
                "found invalid or no expression for while loop condition", self.peek()
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

        end = self.consume()  # byebye `ENDWHILE`

        res = WhileStatement(begin.pos, end.pos, expr, stmts)
        return Statement("while", while_s=res)

    def for_stmt(self):
        # FOR <counter> <assign> <lit> TO <lit> [STEP <lit>]
        initial = self.peek()

        if initiakeyword != "for":
            return

        self.consume()

        counter: Identifier | None = self.ident()  # type: ignore
        if counter is None:
            raise BCError("invalid identifier as counter in for loop", initial)

        assign = self.peek()
        if assign.operator != "assign":
            raise BCError(
                "expected assignment operator `<-` after counter in a for loop", assign
            )
        self.consume()

        begin = self.expression()
        if begin is None:
            raise BCError("invalid or no expression as begin in for loop", self.peek())

        to = self.peek()
        if to.keyword != "to":
            raise BCError("expected TO after beginning value in for loop", to)
        self.consume()

        end = self.expression()
        if end is None:
            raise BCError("invalid or no expression as end in for loop", self.peek())

        step: Expr | None = None
        if self.peek().keyword == "step":
            self.consume()
            step = self.expression()
            if step is None:
                raise BCError(
                    "invalid or no expression as step in for loop", self.peek()
                )

        stmts = []
        while self.peek().keyword != "next":
            stmts.append(self.scan_one_statement())

        next = self.consume()  # byebye NEXT

        next_counter: Identifier | None = self.ident()  # type: ignore
        if next_counter is None:
            raise BCError("invalid identifier after NEXT in a for loop", self.peek())

        if counter.ident != next_counter.ident:
            raise BCError(
                f"initialized counter as {counter.ident} but used {next_counter.ident} after loop",
                self.prev(),
            )

        # thanks python for not having proper null handling
        res = ForStatement(begin.pos, next.pos, counter=counter, block=stmts, begin=begin, end=end, step=step)  # type: ignore
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

        until = self.consume()  # byebye `UNTIL`

        expr = self.expression()
        if expr is None:
            raise BCError(
                "found invalid or no expression for repeat-until loop condition",
                self.peek(),
            )

        res = RepeatUntilStatement(begin.pos, untipos, expr, stmts)
        return Statement("repeatuntil", repeatuntil=res)

    def function_arg(self) -> FunctionArgument | None:
        # ident : type
        ident = self.ident()
        if not isinstance(ident, Identifier):
            raise BCError("invalid identifier for function arg", ident)

        colon = self.consume()
        if colon.kind != "separator" and colon.separator != "colon":
            raise BCError(
                "expected colon after ident in function argument", self.peek()
            )

        typ = self.typ()
        if typ is None:
            raise BCError("invalid type after colon in function argument", colon)

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
            raise BCError("invalid identifier after PROCEDURE declaration", begin)

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
            raise BCError("invalid identifier after FUNCTION declaration", begin)

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
            raise BCError("expected RETURNS after function arguments", begin)

        typ = self.typ()
        if typ is None:
            raise BCError("invalid type after RETURNS for function return value", begin)

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

        if name.literakind != "string":  # type: ignore
            raise BCError("literal for include must be a string!")

        res = IncludeStatement(include.pos, name.literavalue, ffi=ffi)  # type: ignore
        return Statement("include", include=res)

    def trace_stmt(self) -> Statement | None:
        begin = self.peek()
        if begin.keyword != "trace":
            return
        self.consume()

        lparen = self.consume()
        if lparen.separator != "left_paren":
            raise BCError(
                "expected opening parenthesis after TRACE keyword!", self.peek()
            )

        vars = list()
        while self.peek().separator != "right_paren":
            ident: Identifier | None = self.ident()  # type: ignore
            if ident is None:
                raise BCError(
                    "expected valid identifier after left paren in TRACE keyword!",
                    lparen,
                )

            vars.append(ident.ident)  # type: ignore

            comma = self.peek()
            if comma.separator != "comma" and comma.separator != "right_paren":
                raise BCError(
                    "expected comma or right parenthesis after argument in function call argument list",
                    comma,
                )
            elif comma.separator == "comma":
                self.consume()

        rightb = self.consume()
        if rightb.separator != "right_paren":
            raise BCError(
                "expected right paren after variable list in TRACE statement!", rightb
            )

        stmt: CallStatement | FunctionCall
        call = self.call_stmt()
        if call is not None:
            stmt = calcall  # type: ignore
        else:
            fncall = self.function_call()
            if fncall is None:
                raise BCError(
                    "invalid function or procedure call in trace statement!", begin
                )
            stmt = fncall  # type: ignore

        to = self.consume()
        if to.keyword != "to":
            raise BCError(
                "expected TO keyword after call/function call in TRACE statement!", to
            )

        lit: Literal | None = self.literal()  # type: ignore
        if lit is None:
            raise BCError(
                "expected valid literal after TO keyword in TRACE statement!",
                self.peek(),
            )

        file_name = lit.to_bcvalue()
        if file_name.kind != "string":
            raise BCError(
                "expected string literal after TO keyword in TRACE statement!\n"
                + "pass the file name of the output trace table in a string.",
                self.peek(),
            )

        res = TraceStatement(begin.pos, vars, stmt, file_name.get_string())
        return Statement("trace", trace=res)

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

        return_s = self.return_stmt()
        if return_s is not None:
            return return_s

        include = self.include_stmt()
        if include is not None:
            return include

        trace = self.trace_stmt()
        if trace is not None:
            return trace

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
            return Statement("expr", expr=expr)
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
