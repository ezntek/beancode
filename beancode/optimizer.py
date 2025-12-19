from . import is_case_consistent, prefix_string_with_article
from .bean_ast import *
from .libroutines import *

def block_empty(blk: list[Statement]):
    if not blk:
        return True

    for itm in blk:
        if not (isinstance(itm, CommentStatement) or isinstance(itm, NewlineStatement)):
            return False

    return True

class Optimizer:
    # TODO: use
    constants: list[dict[str, BCValue]]
    ignore_constants: list[list[str]]
    block: list[Statement]
    cur_stmt: int
    active_constants: set[str]
    elided_procedures: set[str] 
    remove_cur: bool

    def __init__(self, block: list[Statement]):
        self.block = block
        self.constants = list()
        self.active_constants = set()
        self.elided_procedures = set()
        self.ignore_constants = list()
        self.cur_stmt = 0
        self.remove_cur = False

    def _update_active_constants(self):
        self.active_constants = self.active_constants.union(
            *(d.keys() for d in self.constants)
        )
        self.active_constants -= {x for row in self.ignore_constants for x in row}

    def _typecast_string(self, inner: BCValue, pos: Pos) -> BCValue | None:
        _ = pos  # shut up the type checker
        s = ""

        if isinstance(inner.kind, BCArrayType):
            return
        else:
            match inner.kind:
                case BCPrimitiveType.NULL:
                    s = "(null)"
                case BCPrimitiveType.BOOLEAN:
                    if inner.get_boolean():
                        s = "true"
                    else:
                        s = "false"
                case BCPrimitiveType.INTEGER:
                    s = str(inner.get_integer())
                case BCPrimitiveType.REAL:
                    s = str(inner.get_real())
                case BCPrimitiveType.CHAR:
                    s = str(inner.get_char()[0])
                case BCPrimitiveType.STRING:
                    return inner

        return BCValue.new_string(s)

    def _typecast_integer(self, inner: BCValue, pos: Pos) -> BCValue | None:
        i = 0
        match inner.kind:
            case BCPrimitiveType.STRING:
                s = inner.get_string()
                try:
                    i = int(s.strip())
                except ValueError:
                    raise BCError(f'impossible to convert "{s}" to an INTEGER!', pos)
            case BCPrimitiveType.INTEGER:
                return inner
            case BCPrimitiveType.REAL:
                i = int(inner.get_real())
            case BCPrimitiveType.CHAR:
                i = ord(inner.get_char()[0])
            case BCPrimitiveType.BOOLEAN:
                i = 1 if inner.get_boolean() else 0

        return BCValue.new_integer(i)

    def _typecast_real(self, inner: BCValue, pos: Pos) -> BCValue | None:
        r = 0.0

        match inner.kind:
            case BCPrimitiveType.STRING:
                s = inner.get_string()
                try:
                    r = float(s.strip())
                except ValueError:
                    raise BCError(f'impossible to convert "{s}" to a REAL!', pos)
            case BCPrimitiveType.INTEGER:
                r = float(inner.get_integer())
            case BCPrimitiveType.REAL:
                return inner
            case BCPrimitiveType.CHAR:
                raise BCError(f"impossible to convert a REAL to a CHAR!", pos)
            case BCPrimitiveType.BOOLEAN:
                r = 1.0 if inner.get_boolean() else 0.0

        return BCValue.new_real(r)

    def _typecast_char(self, inner: BCValue, pos: Pos) -> BCValue | None:
        c = ""

        match inner.kind:
            case BCPrimitiveType.STRING:
                raise BCError(
                    f"cannot convert a STRING to a CHAR! use SUBSTRING(str, begin, 1) to get a character.",
                    pos,
                )
            case BCPrimitiveType.INTEGER:
                c = chr(inner.get_integer())
            case BCPrimitiveType.REAL:
                raise BCError(f"impossible to convert a CHAR to a REAL!", pos)
            case BCPrimitiveType.CHAR:
                return inner
            case BCPrimitiveType.BOOLEAN:
                raise BCError(f"impossible to convert a BOOLEAN to a CHAR!", pos)

        return BCValue.new_char(c)

    def _typecast_boolean(self, inner: BCValue) -> BCValue | None:
        b = False

        match inner.kind:
            case BCPrimitiveType.STRING:
                b = inner.get_string() != ""
            case BCPrimitiveType.INTEGER:
                b = inner.get_integer() != 0
            case BCPrimitiveType.REAL:
                b = inner.get_real() != 0.0
            case BCPrimitiveType.CHAR:
                b = ord(inner.get_char()) != 0
            case BCPrimitiveType.BOOLEAN:
                return inner

        return BCValue.new_boolean(b)

    def visit_typecast(self, tc: Typecast) -> BCValue | None:
        inner = self.fold_expr(tc.expr)

        if not inner:
            return

        if inner.kind == BCPrimitiveType.NULL:
            raise BCError("cannot cast NULL to anything!", tc.pos)

        if isinstance(inner.kind, BCArrayType) and tc.typ != BCPrimitiveType.STRING:
            raise BCError(f"cannot cast an array to a {tc.typ}", tc.pos)

        match tc.typ:
            case BCPrimitiveType.STRING:
                return self._typecast_string(inner, tc.pos)
            case BCPrimitiveType.INTEGER:
                return self._typecast_integer(inner, tc.pos)
            case BCPrimitiveType.REAL:
                return self._typecast_real(inner, tc.pos)
            case BCPrimitiveType.CHAR:
                return self._typecast_char(inner, tc.pos)
            case BCPrimitiveType.BOOLEAN:
                return self._typecast_boolean(inner)

    def visit_identifier(self, id: Identifier) -> BCValue | None:
        if id.ident not in self.active_constants:
            return

        for d in reversed(self.constants):
            if id.ident in d:
                return d[id.ident]

    def visit_array_literal(self, expr: ArrayLiteral):
        for i in range(len(expr.items)):
            opt = self.fold_expr(expr.items[i])
            if not opt:
                continue
            expr.items[i] = Literal(expr.items[i].pos, opt)

    def visit_binaryexpr(self, expr: BinaryExpr):
        should_return = False
        lhs = self.fold_expr(expr.lhs)  # type: ignore
        if not lhs:
            should_return = True
        else:
            expr.lhs = Literal(expr.lhs.pos, lhs)

        rhs = self.fold_expr(expr.rhs)  # type: ignore
        if not rhs:
            should_return = True
        else:
            expr.rhs = Literal(expr.rhs.pos, rhs)

        if should_return:
            return

        lhs: BCValue
        rhs: BCValue

        if expr.op in {Operator.EQUAL, Operator.NOT_EQUAL}:
            human_kind = "a comparison"
        elif expr.op in {
            Operator.LESS_THAN,
            Operator.LESS_THAN_OR_EQUAL,
            Operator.GREATER_THAN,
            Operator.GREATER_THAN_OR_EQUAL,
        }:
            human_kind = "an ordered comparison"

            if lhs.kind != rhs.kind and not (
                lhs.kind_is_numeric() and rhs.kind_is_numeric()
            ):
                raise BCError(
                    f"cannot {expr.op.humanize()} incompatible types {lhs.kind} and {rhs.kind}",
                    expr.pos,
                )
        elif expr.op in {
            Operator.AND,
            Operator.OR,
            Operator.NOT,
        }:
            human_kind = "a boolean operation"

            if lhs.kind != rhs.kind:
                raise BCError(
                    f"cannot {expr.op.humanize()} incompatible types {lhs.kind} and {rhs.kind}!",
                    expr.pos,
                )

            if not (
                lhs.kind == BCPrimitiveType.BOOLEAN
                or rhs.kind == BCPrimitiveType.BOOLEAN
            ):
                raise BCError(
                    f"cannot {expr.op.humanize()} between {lhs.kind} and {rhs.kind}!",
                    expr.pos,
                )
        else:
            human_kind = "an arithmetic expression"

            if expr.op not in {Operator.ADD, Operator.FLOOR_DIV, Operator.MOD} and not (
                lhs.kind_is_numeric() and rhs.kind_is_numeric()
            ):
                raise BCError(
                    f"cannot {expr.op.humanize()} between BOOLEANs, CHARs and STRINGs!",
                    expr.pos,
                )

        if expr.op != Operator.EQUAL:
            if lhs.is_uninitialized():
                raise BCError(
                    f"cannot have NULL in the left hand side of {human_kind}\n"
                    + "is your value an uninitialized value/variable?",
                    expr.lhs.pos,
                )
            if rhs.is_uninitialized():
                raise BCError(
                    f"cannot have NULL in the right hand side of {human_kind}\n"
                    + "is your value an uninitialized value/variable?",
                    expr.rhs.pos,
                )

        match expr.op:
            case Operator.ASSIGN:
                raise ValueError("impossible to have assign in binaryexpr")
            case Operator.EQUAL:
                return BCValue(BCPrimitiveType.BOOLEAN, lhs == rhs)
            case Operator.NOT_EQUAL:
                return BCValue(BCPrimitiveType.BOOLEAN, lhs != rhs)
            case Operator.GREATER_THAN:
                return BCValue(
                    BCPrimitiveType.BOOLEAN,
                    (
                        ord(lhs.val) > ord(rhs.val)  # type: ignore
                        if lhs.kind == BCPrimitiveType.CHAR
                        else lhs.val > rhs.val  # type: ignore
                    ),
                )
            case Operator.LESS_THAN:
                return BCValue(
                    BCPrimitiveType.BOOLEAN,
                    (
                        ord(lhs.val) < ord(rhs.val)  # type: ignore
                        if lhs.kind == BCPrimitiveType.CHAR
                        else lhs.val < rhs.val  # type: ignore
                    ),
                )
            case Operator.GREATER_THAN_OR_EQUAL:
                return BCValue(
                    BCPrimitiveType.BOOLEAN,
                    (
                        ord(lhs.val) >= ord(rhs.val)  # type: ignore
                        if lhs.kind == BCPrimitiveType.CHAR
                        else lhs.val >= rhs.val  # type: ignore
                    ),
                )
            case Operator.LESS_THAN_OR_EQUAL:
                return BCValue(
                    BCPrimitiveType.BOOLEAN,
                    (
                        ord(lhs.val) <= ord(rhs.val)  # type: ignore
                        if lhs.kind == BCPrimitiveType.CHAR
                        else lhs.val <= rhs.val  # type: ignore
                    ),
                )
            case Operator.POW:
                lhs_num: int | float = lhs.val  # type: ignore
                rhs_num: int | float = rhs.val  # type: ignore

                res = (
                    1 << rhs_num
                    if (int(lhs_num) == 2 and type(rhs_num) is int)
                    else lhs_num**rhs_num
                )

                return (
                    BCValue(BCPrimitiveType.INTEGER, res)
                    if type(res) is int
                    else BCValue(BCPrimitiveType.REAL, res)
                )
            case Operator.MUL:
                res = lhs.val * rhs.val  # type: ignore
                return (
                    BCValue(BCPrimitiveType.INTEGER, res)
                    if type(res) is int
                    else BCValue(BCPrimitiveType.REAL, res)
                )
            case Operator.DIV:
                if rhs.val == 0:
                    raise BCError("cannot divide by zero!", expr.rhs.pos)

                res = lhs.val / rhs.val  # type: ignore
                return (
                    BCValue(BCPrimitiveType.INTEGER, res)
                    if type(res) is int
                    else BCValue(BCPrimitiveType.REAL, res)
                )
            case Operator.ADD:
                if (
                    lhs.kind == BCPrimitiveType.BOOLEAN
                    or rhs.kind == BCPrimitiveType.BOOLEAN
                ):
                    raise BCError("cannot add BOOLEANs!", expr.pos)

                if lhs.kind_is_alpha() or rhs.kind_is_alpha():
                    return BCValue(BCPrimitiveType.STRING, str(lhs) + str(rhs))
                else:
                    res = lhs.val + rhs.val  # type: ignore
                    return (
                        BCValue(BCPrimitiveType.INTEGER, res)
                        if type(res) is int
                        else BCValue(BCPrimitiveType.REAL, res)
                    )
            case Operator.SUB:
                res = lhs.val - rhs.val  # type: ignore
                return (
                    BCValue(BCPrimitiveType.INTEGER, res)
                    if type(res) is int
                    else BCValue(BCPrimitiveType.REAL, res)
                )
            case Operator.FLOOR_DIV:
                if not (lhs.kind_is_numeric() or rhs.kind_is_numeric()):
                    raise BCError(
                        "Cannot DIV() between BOOLEANs, CHARs and STRINGs!",
                        expr.pos,
                    )

                if rhs.val == 0:
                    raise BCError("cannot divide by zero!", expr.rhs.pos)

                return BCValue(BCPrimitiveType.INTEGER, int(lhs.val // rhs.val))  # type: ignore
            case Operator.MOD:
                if not (lhs.kind_is_numeric() or rhs.kind_is_numeric()):
                    raise BCError(
                        "Cannot MOD() between BOOLEANs, CHARs and STRINGs!",
                        expr.pos,
                    )

                if rhs.val == 0:
                    raise BCError("cannot divide by zero!", expr.rhs.pos)

                res = lhs.val % rhs.val  # type: ignore
                return (
                    BCValue(BCPrimitiveType.INTEGER, res)
                    if type(res) is int
                    else BCValue(BCPrimitiveType.REAL, res)
                )
            case Operator.AND:
                return BCValue(BCPrimitiveType.BOOLEAN, lhs.val and rhs.val)  # type: ignore
            case Operator.OR:
                return BCValue(BCPrimitiveType.BOOLEAN, lhs.val or rhs.val)  # type: ignore

    def visit_array_index(self, expr: ArrayIndex):
        _ = expr

    def _eval_libroutine_args(
        self,
        args: list[Expr],
        lr: Libroutine,
        name: str,
        pos: Pos | None,
    ) -> list[BCValue] | None:
        if lr and len(args) < len(lr):
            raise BCError(
                f"expected {len(lr)} args, but got {len(args)} in call to library routine {name.upper()}",
                pos,
            )

        evargs: list[BCValue] = []
        if lr:
            for idx, (arg, arg_type) in enumerate(zip(args, lr)):
                new = self.fold_expr(arg)
                if not new:
                    return

                mismatch = False
                if isinstance(arg_type, tuple):
                    if new.kind not in arg_type:
                        mismatch = True
                elif not arg_type:
                    pass
                elif arg_type != new.kind:
                    mismatch = True

                if mismatch and new.is_null():
                    raise BCError(
                        f"{humanize_index(idx + 1)} argument in call to library routine {name.upper()} is NULL!",
                        pos,
                    )

                if mismatch:
                    err_base = f"expected {humanize_index(idx + 1)} argument to library routine {name.upper()} to be "
                    if isinstance(arg_type, tuple):
                        err_base += "either "

                        for i, expected in enumerate(arg_type):
                            if i == len(arg_type) - 1:
                                err_base += "or "

                            err_base += prefix_string_with_article(
                                str(expected).upper()
                            )
                            err_base += " "
                    else:
                        if str(new.kind)[0] in "aeiou":
                            err_base += "a "
                        else:
                            err_base += "an "

                        err_base += prefix_string_with_article(str(arg_type).upper())
                        err_base += " "

                    wanted = str(new.kind).upper()
                    err_base += f"but found {wanted}"
                    raise BCError(err_base, pos)

                evargs.append(new)
        else:
            evargs = list()
            for e in args:
                evaled = self.fold_expr(e)
                if not evaled:
                    return
                evargs.append(evaled)

        return evargs

    def visit_libroutine(self, stmt: FunctionCall) -> BCValue | None:  # type: ignore
        name = stmt.ident.lower()
        lr = LIBROUTINES[name.lower()]

        evargs = self._eval_libroutine_args(stmt.args, lr, name, stmt.pos)
        if not evargs:
            return

        try:
            match name.lower():
                case "initarray":
                    return None  # runtime only
                case "format":
                    return None  # runtime only
                case "typeof" | "type":
                    return bean_typeof(stmt.pos, evargs[0])
                case "ucase":
                    [txt, *_] = evargs
                    return bean_ucase(stmt.pos, txt)
                case "lcase":
                    [txt, *_] = evargs
                    return bean_ucase(stmt.pos, txt)
                case "substring":
                    [txt, begin, length, *_] = evargs

                    return bean_substring(
                        stmt.pos,
                        txt.get_string(),
                        begin.get_integer(),
                        length.get_integer(),
                    )
                case "div":
                    [lhs, rhs, *_] = evargs

                    lhs_val = (
                        lhs.get_integer()
                        if lhs.kind == BCPrimitiveType.INTEGER
                        else lhs.get_real()
                    )
                    rhs_val = (
                        rhs.get_integer()
                        if rhs.kind == BCPrimitiveType.INTEGER
                        else rhs.get_real()
                    )

                    return bean_div(stmt.pos, lhs_val, rhs_val)
                case "mod":
                    [lhs, rhs, *_] = evargs

                    lhs_val = (
                        lhs.get_integer()
                        if lhs.kind == BCPrimitiveType.INTEGER
                        else lhs.get_real()
                    )
                    rhs_val = (
                        rhs.get_integer()
                        if rhs.kind == BCPrimitiveType.INTEGER
                        else rhs.get_real()
                    )

                    return bean_mod(stmt.pos, lhs_val, rhs_val)
                case "length":
                    [txt, *_] = evargs
                    return bean_length(stmt.pos, txt.get_string())
                case "round":
                    [val_r, places, *_] = evargs
                    return bean_round(stmt.pos, val_r.get_real(), places.get_integer())
                case "sqrt":
                    [val, *_] = evargs
                    return bean_sqrt(stmt.pos, val)
                case "getchar":
                    return None  # runtime only
                case "random":
                    return None  # runtime only
                case "sin":
                    [val, *_] = evargs
                    return BCValue.new_real(math.sin(val.get_real()))
                case "cos":
                    [val, *_] = evargs
                    return BCValue.new_real(math.cos(val.get_real()))
                case "tan":
                    [val, *_] = evargs
                    return BCValue.new_real(math.tan(val.get_real()))
                case "execute":
                    return None  # runtime only
                case "putchar":
                    return None  # runtime only
                case "exit":
                    return None  # runtime only
                case "sleep":
                    return None  # runtime only
                case "flush":
                    return None  # runtime only
        except BCError as e:
            e.pos = stmt.pos
            raise e

    def visit_fncall(self, expr: FunctionCall):
        if is_case_consistent(expr.ident) and expr.ident.lower() in LIBROUTINES:
            return self.visit_libroutine(expr)

        for i, itm in enumerate(expr.args):
            val = self.fold_expr(itm)
            if val:
                expr.args[i] = Literal(itm.pos, val)

    def fold_expr(self, expr: Expr) -> BCValue | None:
        match expr:
            case Typecast():
                return self.visit_typecast(expr)
            case Grouping():
                return self.fold_expr(expr.inner)
            case Negation():
                inner = self.fold_expr(expr.inner)
                if not inner:
                    return

                if inner.kind == BCPrimitiveType.INTEGER:
                    return BCValue.new_integer(-inner.get_integer())  # type: ignore
                elif inner.kind == BCPrimitiveType.REAL:
                    return BCValue.new_real(-inner.get_real())  # type: ignore
            case Not():
                inner = self.fold_expr(expr.inner)
                if not inner:
                    return

                if inner.kind != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot perform logical NOT on {inner.kind}",
                        expr.inner.pos,
                    )

                return BCValue.new_boolean(not inner.get_boolean())
            case Identifier():
                return self.visit_identifier(expr)
            case Literal():
                return expr.val
            case ArrayLiteral():
                return self.visit_array_literal(expr)
            case BinaryExpr():
                return self.visit_binaryexpr(expr)
            case ArrayIndex():
                return self.visit_array_index(expr)
            case FunctionCall():
                return self.visit_fncall(expr)
        raise BCError(
            "whoops something is very wrong. this is a rare error, please report it to the developers."
        )

    def fold_expr_if_possible(self, expr: Expr) -> Expr:
        res = self.fold_expr(expr)
        if res:
            return Literal(expr.pos, res)
        else:
            return expr

    def visit_expr(self, expr: Expr) -> Expr:
        default = self.fold_expr_if_possible(expr)
        if isinstance(default, Literal):
            return default  # always favor static folding

        match expr:
            case Typecast():
                expr.expr = self.visit_expr(expr.expr)
            case Grouping():
                expr.inner = self.visit_expr(expr.inner)
            case Negation():
                expr.inner = self.visit_expr(expr.inner)
            case Not():
                expr.inner = self.visit_expr(expr.inner)
            case Identifier():
                pass  # nothing inside to optimize
            case Literal():
                pass  # unreachable
            case ArrayLiteral():
                for i, itm in enumerate(expr.items):
                    expr.items[i] = self.visit_expr(itm)
            case BinaryExpr():
                expr.lhs = self.visit_expr(expr.lhs)
                expr.rhs = self.visit_expr(expr.rhs)
            case ArrayIndex():
                expr.idx_outer = self.visit_expr(expr.idx_outer)
                if expr.idx_inner:
                    expr.idx_inner = self.visit_expr(expr.idx_inner)
            case FunctionCall():
                if not expr.libroutine:
                    return default

                if expr.ident in {"div", "mod"}:
                    if len(expr.args) != 2:
                        return default

                    lhs = self.visit_expr(expr.args[0])
                    rhs = self.visit_expr(expr.args[1])
                    op = Operator.FLOOR_DIV if expr.ident == "div" else Operator.MOD
                    return BinaryExpr(expr.pos, lhs, op, rhs)
                elif expr.ident == "sqrt":
                    if len(expr.args) != 1:
                        return default

                    arg = self.visit_expr(expr.args[0])
                    return Sqrt(expr.pos, arg)

        return default

    def visit_type(self, typ: Type):
        if isinstance(typ, ArrayType):
            new = list()
            for itm in typ.bounds:
                new.append(self.visit_expr(itm))
            typ.bounds = tuple(new)

    def visit_if_stmt(self, stmt: IfStatement):
        stmt.cond = self.visit_expr(stmt.cond)
        stmt.if_block = self.visit_block(stmt.if_block)
        stmt.else_block = self.visit_block(stmt.else_block)

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        for b in stmt.branches:
            if not isinstance(b, CaseofStatement):
                continue

            b.expr = self.visit_expr(b.expr)
            self.visit_stmt(b.stmt) # type: ignore

    def visit_for_stmt(self, stmt: ForStatement):
        stmt.begin = self.visit_expr(stmt.begin)
        stmt.end = self.visit_expr(stmt.end)
        if stmt.step:
            stmt.step = self.visit_expr(stmt.step)

        stmt.block = self.visit_block(stmt.block, ignore=[stmt.counter.ident])

    def visit_while_stmt(self, stmt: WhileStatement):
        stmt.cond = self.visit_expr(stmt.cond)
        stmt.block = self.visit_block(stmt.block)

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        stmt.cond = self.visit_expr(stmt.cond)
        stmt.block = self.visit_block(stmt.block)

    def visit_output_stmt(self, stmt: OutputStatement):
        i = 0
        new_items: list[Expr] = list()
        cur_item = ""
        while i < len(stmt.items):
            if cur_item == "":
                first_pos = stmt.items[i].pos
            itm = self.visit_expr(stmt.items[i])
            if not isinstance(itm, Literal):
                new_items.append(Literal(first_pos, BCValue.new_string(cur_item)))  # type: ignore
                new_items.append(itm)
                cur_item = ""
            else:
                cur_item += str(itm.val)
            i += 1

        if cur_item != "":
            new_items.append(Literal(first_pos, BCValue.new_string(cur_item)))  # type: ignore

        useless = set()
        for i, itm in enumerate(new_items):
            if isinstance(itm, Literal) and str(itm.val.val) == "":
                useless.add(i)

        stmt.items = [itm for i, itm in enumerate(new_items) if i not in useless]

    def visit_input_stmt(self, stmt: InputStatement):
        _ = stmt

    def visit_return_stmt(self, stmt: ReturnStatement):
        if stmt.expr:
            stmt.expr = self.visit_expr(stmt.expr)

    def visit_procedure(self, stmt: ProcedureStatement):
        for arg in stmt.args:
            self.visit_type(arg.typ)
        
        stmt.block = self.visit_block(stmt.block, ignore=[arg.name for arg in stmt.args])

    def visit_function(self, stmt: FunctionStatement):
        for arg in stmt.args:
            self.visit_type(arg.typ)

        stmt.block = self.visit_block(stmt.block, ignore=[arg.name for arg in stmt.args])

    def visit_scope_stmt(self, stmt: ScopeStatement):
        stmt.block = self.visit_block(stmt.block)

    def visit_include_stmt(self, stmt: IncludeStatement):
        _ = stmt

    def visit_call(self, stmt: CallStatement):
        for i, itm in enumerate(stmt.args):
            stmt.args[i] = self.visit_expr(itm)

    def visit_assign_stmt(self, stmt: AssignStatement):
        stmt.value = self.visit_expr(stmt.value)

    def visit_constant_stmt(self, stmt: ConstantStatement):
        val = self.fold_expr(stmt.value)
        if not val:
            # try optimizing the expr instead
            stmt.value = self.visit_expr(stmt.value)
            return
        self.constants[-1][stmt.ident.ident] = val
        self._update_active_constants()
        self.remove_cur = True

    def visit_declare_stmt(self, stmt: DeclareStatement):
        self.visit_type(stmt.typ)

    def visit_trace_stmt(self, stmt: TraceStatement):
        _ = stmt

    def visit_openfile_stmt(self, stmt: OpenfileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

    def visit_readfile_stmt(self, stmt: ReadfileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

    def visit_writefile_stmt(self, stmt: WritefileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

        stmt.src = self.visit_expr(stmt.src)

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

    def visit_stmt(self, stmt: Statement) -> list[Statement]:
        match stmt:
            case IfStatement():
                self.visit_if_stmt(stmt)
                if isinstance(stmt.cond, Literal) and stmt.cond.val.kind == BCPrimitiveType.BOOLEAN:
                    v = bool(stmt.cond.val.val)
                    if v:
                        return stmt.if_block
                    elif not v and stmt.else_block:
                        return stmt.else_block
                    else:
                        return []

                if block_empty(stmt.if_block):
                    if block_empty(stmt.else_block):
                        return []
                    else:
                        stmt.cond = Negation(stmt.cond.pos, stmt.cond)
                        stmt.if_block = stmt.else_block
            case CaseofStatement():
                self.visit_caseof_stmt(stmt)
            case ForStatement():
                self.visit_for_stmt(stmt)
                if block_empty(stmt.block):
                    return []
            case WhileStatement():
                self.visit_while_stmt(stmt)

                if block_empty(stmt.block):
                    return []

                if isinstance(stmt.cond, Literal) and stmt.cond.val.kind == BCPrimitiveType.BOOLEAN:
                    v = bool(stmt.cond.val.val)
                    if not v:
                        return [] # just remove the whole block
            case RepeatUntilStatement():
                self.visit_repeatuntil_stmt(stmt)

                if block_empty(stmt.block):
                    return []

                if isinstance(stmt.cond, Literal) and stmt.cond.val.kind == BCPrimitiveType.BOOLEAN:
                    v = bool(stmt.cond.val.val)
                    if not v:
                        return stmt.block # turn it into one block
            case OutputStatement():
                self.visit_output_stmt(stmt)
            case InputStatement():
                self.visit_input_stmt(stmt)
            case ReturnStatement():
                self.visit_return_stmt(stmt)
            case ProcedureStatement():
                self.visit_procedure(stmt)
                if block_empty(stmt.block):
                    self.elided_procedures.add(stmt.name)
                    return []
            case FunctionStatement():
                self.visit_function(stmt)
            case ScopeStatement():
                self.visit_scope_stmt(stmt)
                if block_empty(stmt.block):
                    return []
            case IncludeStatement():
                self.visit_include_stmt(stmt)
            case CallStatement():
                self.visit_call(stmt)
                if stmt.ident in self.elided_procedures:
                    return []
            case AssignStatement():
                self.visit_assign_stmt(stmt)
            case ConstantStatement():
                self.visit_constant_stmt(stmt)
            case DeclareStatement():
                self.visit_declare_stmt(stmt)
            case TraceStatement():
                self.visit_trace_stmt(stmt)
                if block_empty(stmt.block):
                    return []
            case OpenfileStatement():
                self.visit_openfile_stmt(stmt)
            case ReadfileStatement():
                self.visit_readfile_stmt(stmt)
            case WritefileStatement():
                self.visit_writefile_stmt(stmt)
            case ClosefileStatement():
                self.visit_closefile_stmt(stmt)
            case ExprStatement():
                stmt.inner = self.visit_expr(stmt.inner)
                if not isinstance(stmt.inner, FunctionCall):
                    return []

        return [stmt]

    def visit_program(self, program: Program):
        self.visit_block(program.stmts)

    def visit_block(self, block: list[Statement] | None, ignore: list[str] | None = None) -> list[Statement]:
        blk = block if block is not None else self.block
        cur = 0
        self.constants.append(dict())
        self.ignore_constants.append(ignore if ignore else list())
        saved_elided_procedures = self.elided_procedures.copy()
        self._update_active_constants()
        new_block = []
        while cur < len(blk):
            stmt = blk[cur]
            self.cur_stmt = cur
            res = self.visit_stmt(stmt)
            if self.remove_cur:
                self.remove_cur = False
            else:
                for itm in res:
                    new_block.append(itm)
            cur += 1
        self.constants.pop()
        self.ignore_constants.pop()
        self.elided_procedures = saved_elided_procedures
        self._update_active_constants()
        return new_block
