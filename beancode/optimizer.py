from . import is_case_consistent, prefix_string_with_article
from .bean_ast import *
from .libroutines import *


@dataclass
class OptimizerConfig:
    inline_constants = True
    fold_constant_expressions = True
    simplify_math = True
    inline_library_routines = True  # TODO: implement
    inline_functions = True  # TODO: implement


class Optimizer:
    # TODO: use
    config: OptimizerConfig
    constants: list[dict[str, BCValue]]
    block: list[Statement]
    unwanted_items: list[list[int]]
    cur_stmt: int
    active_constants: set[str]

    def __init__(self, block: list[Statement], config: OptimizerConfig | None = None):
        self.block = block
        self.constants = list()
        self.unwanted_items = list()
        self.active_constants = set()
        self.cur_stmt = 0
        if config:
            self.config = config
        else:
            self.config = OptimizerConfig()

    def _update_active_constants(self):
        self.active_constants = self.active_constants.union(
            *(d.keys() for d in self.constants)
        )

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

        human_kind = ""
        if expr.op in {Operator.EQUAL, Operator.NOT_EQUAL}:
            human_kind = "a comparison"
        elif expr.op in {
            Operator.LESS_THAN,
            Operator.LESS_THAN_OR_EQUAL,
            Operator.GREATER_THAN,
            Operator.GREATER_THAN_OR_EQUAL,
        }:
            human_kind = "an ordered comparison"
        elif expr.op in {
            Operator.AND,
            Operator.OR,
            Operator.NOT,
        }:
            human_kind = "a boolean operation"
        else:
            human_kind = "an arithmetic expression"

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
                if lhs.is_uninitialized() and rhs.is_uninitialized():
                    return BCValue(BCPrimitiveType.BOOLEAN, True)

                if lhs.kind != rhs.kind:
                    raise BCError(
                        f"cannot compare incompatible types {lhs.kind} and {rhs.kind}!",
                        expr.pos,
                    )

                res = lhs == rhs
                return BCValue(BCPrimitiveType.BOOLEAN, res)
            case Operator.NOT_EQUAL:
                if lhs.is_uninitialized() and rhs.is_uninitialized():
                    return BCValue(BCPrimitiveType.BOOLEAN, True)

                if lhs.kind != rhs.kind:
                    raise BCError(
                        f"cannot compare incompatible types {lhs.kind} and {rhs.kind}!",
                        expr.pos,
                    )

                res = not (lhs == rhs)  # python is RIDICULOUS
                return BCValue(BCPrimitiveType.BOOLEAN, res)
            case Operator.GREATER_THAN:
                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind_is_numeric():
                    lhs_num = lhs.val  # type: ignore

                    if not rhs.kind_is_numeric():
                        raise BCError(
                            f"impossible to perform greater_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val  # type: ignore

                    return BCValue(BCPrimitiveType.BOOLEAN, (lhs_num > rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCError(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.BOOLEAN:
                        raise BCError(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.STRING:
                        return BCValue(BCPrimitiveType.BOOLEAN, lhs.get_string() > rhs.get_string())
            case Operator.LESS_THAN:
                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind_is_numeric():
                    lhs_num = lhs.val  # type: ignore

                    if not rhs.kind_is_numeric():
                        raise BCError(
                            f"impossible to perform less_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val  # type: ignore

                    return BCValue(BCPrimitiveType.BOOLEAN, (lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCError(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.BOOLEAN:
                        raise BCError(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.STRING:
                        return BCValue(BCPrimitiveType.BOOLEAN, lhs.get_string() < rhs.get_string())
            case Operator.GREATER_THAN_OR_EQUAL:
                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind_is_numeric():
                    lhs_num = lhs.val  # type: ignore

                    if not rhs.kind_is_numeric():
                        raise BCError(
                            f"impossible to perform greater_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val  # type: ignore

                    return BCValue(BCPrimitiveType.BOOLEAN, (lhs_num >= rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCError(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.BOOLEAN:
                        raise BCError(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.STRING:
                        return BCValue(BCPrimitiveType.BOOLEAN, lhs.get_string() >= rhs.get_string())
            case Operator.LESS_THAN_OR_EQUAL:
                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind_is_numeric():
                    lhs_num = lhs.val  # type: ignore

                    if not rhs.kind_is_numeric():
                        raise BCError(
                            f"impossible to perform less_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val  # type: ignore

                    return BCValue(BCPrimitiveType.BOOLEAN, (lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCError(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.BOOLEAN:
                        raise BCError(f"illegal to compare booleans", expr.lhs.pos)
                    elif lhs.kind == BCPrimitiveType.STRING:
                        return BCValue(BCPrimitiveType.BOOLEAN, lhs.get_string() <= rhs.get_string())
            case Operator.POW:
                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot exponentiate BOOLEANs, CHARs and STRINGs!",
                        expr.lhs.pos,
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot exponentiate BOOLEANs, CHARs and STRINGs!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float = lhs.val # type: ignore
                rhs_num: int | float = rhs.val # type: ignore

                if int(lhs_num) == 2 and type(rhs_num) is int:
                    res = 1 << rhs_num
                else:
                    res = lhs_num**rhs_num

                return BCValue(BCPrimitiveType.INTEGER, res) if type(res) is int else BCValue(BCPrimitiveType.REAL, res)
            case Operator.MUL:
                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot multiply between BOOLEANs, CHARs and STRINGs!",
                        expr.lhs.pos,
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot multiply between BOOLEANs, CHARs and STRINGs!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float = lhs.val # type: ignore
                rhs_num: int | float = rhs.val # type: ignore

                res = lhs_num * rhs_num

                return BCValue(BCPrimitiveType.INTEGER, res) if type(res) is int else BCValue(BCPrimitiveType.REAL, res)
            case Operator.DIV:
                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot divide between BOOLEANs, CHARs and STRINGs!",
                        expr.lhs.pos,
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot divide between BOOLEANs, CHARs and STRINGs!",
                        expr.rhs.pos,
                    )

                lhs_num: int | float = lhs.val # type: ignore
                rhs_num: int | float = rhs.val # type: ignore

                res = lhs_num / rhs_num

                return BCValue(BCPrimitiveType.INTEGER, res) if type(res) is int else BCValue(BCPrimitiveType.REAL, res)
            case Operator.ADD:
                if lhs.kind_is_alpha() or rhs.kind_is_alpha():
                    # concatenate instead
                    lhs_str_or_char: str = str()
                    rhs_str_or_char: str = str()

                    if lhs.kind == BCPrimitiveType.STRING:
                        lhs_str_or_char = lhs.get_string()
                    elif lhs.kind == BCPrimitiveType.CHAR:
                        lhs_str_or_char = lhs.get_char()
                    else:
                        lhs_str_or_char = str(lhs)

                    if rhs.kind == BCPrimitiveType.STRING:
                        rhs_str_or_char = rhs.get_string()
                    elif rhs.kind == BCPrimitiveType.CHAR:
                        rhs_str_or_char = rhs.get_char()
                    else:
                        rhs_str_or_char = str(rhs)

                    res = str(lhs_str_or_char + rhs_str_or_char)
                    return BCValue(BCPrimitiveType.STRING, res)

                if (
                    lhs.kind == BCPrimitiveType.BOOLEAN
                    or rhs.kind == BCPrimitiveType.BOOLEAN
                ):
                    raise BCError("Cannot add BOOLEANs, CHARs and STRINGs!", expr.pos)

                lhs_num: int | float = lhs.val # type: ignore
                rhs_num: int | float = rhs.val # type: ignore

                res = lhs_num + rhs_num

                return BCValue(BCPrimitiveType.INTEGER, res) if type(res) is int else BCValue(BCPrimitiveType.REAL, res)
            case Operator.SUB:
                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError("Cannot subtract BOOLEANs, CHARs and STRINGs!")

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError("Cannot subtract BOOLEANs, CHARs and STRINGs!")

                lhs_num: int | float = lhs.val # type: ignore
                rhs_num: int | float = rhs.val # type: ignore

                res = lhs_num - rhs_num

                return BCValue(BCPrimitiveType.INTEGER, res) if type(res) is int else BCValue(BCPrimitiveType.REAL, res)
            # FLOOR_DIV and MOD are impossible here
            case Operator.AND:
                if lhs.kind != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot perform logical AND on value with type {lhs.kind}",
                        expr.lhs.pos,
                    )

                if rhs.kind != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot perform logical AND on value with type {lhs.kind}",
                        expr.rhs.pos,
                    )

                lhs_b: bool = lhs.val # type: ignore
                rhs_b: bool = rhs.val # type: ignore

                res = lhs_b and rhs_b
                return BCValue(BCPrimitiveType.BOOLEAN, res)
            case Operator.OR:
                if lhs.kind != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot perform logical OR on value with type {lhs.kind}",
                        expr.lhs.pos,
                    )

                if rhs.kind != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot perform logical OR on value with type {lhs.kind}",
                        expr.rhs.pos,
                    )

                lhs_b: bool = lhs.val # type: ignore
                rhs_b: bool = rhs.val # type: ignore

                res = lhs_b or rhs_b

                return BCValue(BCPrimitiveType.BOOLEAN, res)

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
                        f"{humanize_index(idx+1)} argument in call to library routine {name.upper()} is NULL!",
                        pos,
                    )

                if mismatch:
                    err_base = f"expected {humanize_index(idx+1)} argument to library routine {name.upper()} to be "
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
                case "help":
                    return None  # runtime only
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
                        f"cannot perform logical NOT on value of type {inner.kind}",
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
            return default # always favor static folding

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
                pass # nothing inside to optimize
            case Literal():
                pass # unreachable
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
            b.expr = self.visit_expr(b.expr)
            self.visit_stmt(b.stmt)

    def visit_for_stmt(self, stmt: ForStatement):
        stmt.begin = self.visit_expr(stmt.begin)
        stmt.end = self.visit_expr(stmt.end)
        if stmt.step:
            stmt.step = self.visit_expr(stmt.step)

        stmt.block = self.visit_block(stmt.block)

    def visit_while_stmt(self, stmt: WhileStatement):
        stmt.cond = self.visit_expr(stmt.cond)
        stmt.block = self.visit_block(stmt.block)

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        stmt.cond = self.visit_expr(stmt.cond)
        stmt.block = self.visit_block(stmt.block)

    def visit_output_stmt(self, stmt: OutputStatement):
        for i, itm in enumerate(stmt.items):
            stmt.items[i] = self.visit_expr(itm)

    def visit_input_stmt(self, stmt: InputStatement):
        _ = stmt

    def visit_return_stmt(self, stmt: ReturnStatement):
        if stmt.expr:
            stmt.expr = self.visit_expr(stmt.expr)

    def visit_procedure(self, stmt: ProcedureStatement):
        for arg in stmt.args:
            self.visit_type(arg.typ)

        stmt.block = self.visit_block(stmt.block)

    def visit_function(self, stmt: FunctionStatement):
        for arg in stmt.args:
            self.visit_type(arg.typ)

        stmt.block = self.visit_block(stmt.block)

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
        self.unwanted_items[-1].append(self.cur_stmt)

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

    def visit_appendfile_stmt(self, stmt: AppendfileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

        stmt.src = self.visit_expr(stmt.src)

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        if isinstance(stmt.file_ident, Expr):
            stmt.file_ident = self.visit_expr(stmt.file_ident)

    def visit_stmt(self, stmt: Statement):
        match stmt:
            case IfStatement():
                self.visit_if_stmt(stmt)
            case CaseofStatement():
                self.visit_caseof_stmt(stmt)
            case ForStatement():
                self.visit_for_stmt(stmt)
            case WhileStatement():
                self.visit_while_stmt(stmt)
            case RepeatUntilStatement():
                self.visit_repeatuntil_stmt(stmt)
            case OutputStatement():
                self.visit_output_stmt(stmt)
            case InputStatement():
                self.visit_input_stmt(stmt)
            case ReturnStatement():
                self.visit_return_stmt(stmt)
            case ProcedureStatement():
                self.visit_procedure(stmt)
            case FunctionStatement():
                self.visit_function(stmt)
            case ScopeStatement():
                self.visit_scope_stmt(stmt)
            case IncludeStatement():
                self.visit_include_stmt(stmt)
            case CallStatement():
                self.visit_call(stmt)
            case AssignStatement():
                self.visit_assign_stmt(stmt)
            case ConstantStatement():
                self.visit_constant_stmt(stmt)
            case DeclareStatement():
                self.visit_declare_stmt(stmt)
            case TraceStatement():
                self.visit_trace_stmt(stmt)
            case OpenfileStatement():
                self.visit_openfile_stmt(stmt)
            case ReadfileStatement():
                self.visit_readfile_stmt(stmt)
            case WritefileStatement():
                self.visit_writefile_stmt(stmt)
            case AppendfileStatement():
                self.visit_appendfile_stmt(stmt)
            case ClosefileStatement():
                self.visit_closefile_stmt(stmt)
            case ExprStatement():
                stmt.inner = self.visit_expr(stmt.inner)

    def visit_program(self, program: Program):
        self.visit_block(program.stmts)

    def visit_block(self, block: list[Statement] | None) -> list[Statement]:
        blk = block if block is not None else self.block
        cur = 0
        self.constants.append(dict())
        self.unwanted_items.append(list())
        while cur < len(blk):
            stmt = blk[cur]
            self.cur_stmt = cur
            self.visit_stmt(stmt)
            cur += 1
        self.constants.pop()
        self._update_active_constants()
        res = [itm for i, itm in enumerate(blk) if i not in self.unwanted_items[-1]]
        self.unwanted_items.pop()
        return res
