from .bean_ast import *


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
    constants: dict[str, BCValue]
    block: list[Statement]
    unwanted_items: list[int]
    cur_stmt: int

    def __init__(self, block: list[Statement], config: OptimizerConfig | None = None):
        self.block = block
        self.constants = dict()
        self.unwanted_items = list()
        self.cur_stmt = 0
        if config:
            self.config = config
        else:
            self.config = OptimizerConfig()

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
        inner = self.visit_expr(tc.expr)

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
        if id.ident not in self.constants:
            return

        return self.constants[id.ident]

    def visit_array_literal(self, expr: ArrayLiteral):
        for i in range(len(expr.items)):
            opt = self.visit_expr(expr.items[i])
            if not opt:
                continue
            expr.items[i] = Literal(expr.items[i].pos, opt)

    def visit_binaryexpr(self, expr: BinaryExpr):
        should_return = False
        lhs = self.visit_expr(expr.lhs) # type: ignore
        if not lhs:
            should_return = True
        else:
            expr.lhs = Literal(expr.lhs.pos, lhs)

        rhs = self.visit_expr(expr.rhs) # type: ignore
        if not rhs:
            should_return = True
        else:
            expr.rhs = Literal(expr.rhs.pos, rhs)

        if should_return:
            return
    
        lhs: BCValue
        rhs: BCValue

        match expr.op:
            case "assign":
                raise ValueError("impossible to have assign in binaryexpr")
            case "equal":
                if lhs.kind != rhs.kind:
                    return BCValue.new_boolean(False)

                # a BCValue(INTEGER, NULL) is not a BCValue(NULL, NULL)
                if lhs.is_null() and rhs.is_null():
                    return BCValue.new_boolean(True)

                res = lhs == rhs
                return BCValue.new_boolean(res)
            case "not_equal":
                if lhs.is_null() and rhs.is_null():
                    return BCValue.new_boolean(True)

                res = not (lhs == rhs)  # python is RIDICULOUS
                return BCValue.new_boolean(res)
            case "greater_than":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an ordered comparison!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an ordered comparison!",
                        expr.rhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                    lhs_num = lhs.val if lhs.val is not None else lhs.val  # type: ignore

                    if rhs.kind not in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                        raise BCError(
                            f"impossible to perform greater_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val if rhs.val is not None else rhs.val  # type: ignore

                    return BCValue.new_boolean((lhs_num > rhs_num))
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
                        return BCValue.new_boolean(lhs.get_string() > rhs.get_string())
            case "less_than":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an ordered comparison!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an ordered comparison!",
                        expr.rhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                    lhs_num = lhs.val if lhs.val is not None else lhs.val  # type: ignore

                    if rhs.kind not in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                        raise BCError(
                            f"impossible to perform less_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val if rhs.val is not None else rhs.val  # type: ignore

                    return BCValue.new_boolean((lhs_num < rhs_num))
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
                        return BCValue.new_boolean(lhs.get_string() < rhs.get_string())
            case "greater_than_or_equal":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an ordered comparison!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an ordered comparison!",
                        expr.rhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                    lhs_num = lhs.val if lhs.val is not None else lhs.val  # type: ignore

                    if rhs.kind not in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                        raise BCError(
                            f"impossible to perform greater_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val if rhs.val is not None else rhs.val  # type: ignore

                    return BCValue.new_boolean((lhs_num >= rhs_num))
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
                        return BCValue.new_boolean(lhs.get_string() >= rhs.get_string())
            case "less_than_or_equal":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an ordered comparison!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an ordered comparison!",
                        expr.rhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                    lhs_num = lhs.val if lhs.val is not None else lhs.val  # type: ignore

                    if rhs.kind not in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                        raise BCError(
                            f"impossible to perform less_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.val if rhs.val is not None else rhs.val  # type: ignore

                    return BCValue.new_boolean((lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCError(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == BCPrimitiveType.BOOLEAN:
                        raise BCError(f"illegal to compare booleans", expr.lhs.pos)
                    elif lhs.kind == BCPrimitiveType.STRING:
                        return BCValue.new_boolean(lhs.get_string() <= rhs.get_string())
            # add sub mul div
            case "pow":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an arithmetic expression!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an arithmetic expression!",
                        expr.rhs.pos,
                    )

                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot exponentiate bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot exponentiate bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind == BCPrimitiveType.INTEGER:
                    lhs_num = lhs.get_integer()
                elif lhs.kind == BCPrimitiveType.REAL:
                    lhs_num = lhs.get_real()

                if rhs.kind == BCPrimitiveType.INTEGER:
                    rhs_num = rhs.get_integer()
                elif rhs.kind == BCPrimitiveType.REAL:
                    rhs_num = rhs.get_real()

                res = lhs_num**rhs_num

                if isinstance(res, int):
                    return BCValue.new_integer(res)
                elif isinstance(res, float):
                    return BCValue.new_real(res)
            case "mul":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an arithmetic expression!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an arithmetic expression!",
                        expr.rhs.pos,
                    )

                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot multiply between bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot multiply between bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind == BCPrimitiveType.INTEGER:
                    lhs_num = lhs.get_integer()
                elif lhs.kind == BCPrimitiveType.REAL:
                    lhs_num = lhs.get_real()

                if rhs.kind == BCPrimitiveType.INTEGER:
                    rhs_num = rhs.get_integer()
                elif rhs.kind == BCPrimitiveType.REAL:
                    rhs_num = rhs.get_real()

                res = lhs_num * rhs_num

                if isinstance(res, int):
                    return BCValue.new_integer(res)
                elif isinstance(res, float):
                    return BCValue.new_real(res)
            case "div":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an arithmetic expression!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an arithmetic expression!",
                        expr.rhs.pos,
                    )

                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot divide between bools, chars, and strings!", expr.lhs.pos
                    )

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError(
                        "Cannot divide between bools, chars, and strings!", expr.rhs.pos
                    )

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind == BCPrimitiveType.INTEGER:
                    lhs_num = lhs.get_integer()
                elif lhs.kind == BCPrimitiveType.REAL:
                    lhs_num = lhs.get_real()

                if rhs.kind == BCPrimitiveType.INTEGER:
                    rhs_num = rhs.get_integer()
                elif rhs.kind == BCPrimitiveType.REAL:
                    rhs_num = rhs.get_real()

                if rhs_num == 0:
                    raise BCError("cannot divide by zero!", expr.rhs.pos)

                res = lhs_num / rhs_num

                if isinstance(res, int):
                    return BCValue.new_integer(res)
                elif isinstance(res, float):
                    return BCValue.new_real(res)
            case "add":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of an arithmetic expression!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of an arithmetic expression!",
                        expr.rhs.pos,
                    )

                if lhs.kind in {
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                } or rhs.kind in {BCPrimitiveType.CHAR, BCPrimitiveType.STRING}:
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
                    return BCValue.new_string(res)

                if BCPrimitiveType.BOOLEAN in [lhs.kind, rhs.kind]:
                    raise BCError("Cannot add bools, chars, and strings!", expr.pos)

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind == BCPrimitiveType.INTEGER:
                    lhs_num = lhs.get_integer()
                elif lhs.kind == BCPrimitiveType.REAL:
                    lhs_num = lhs.get_real()

                if rhs.kind == BCPrimitiveType.INTEGER:
                    rhs_num = rhs.get_integer()
                elif rhs.kind == BCPrimitiveType.REAL:
                    rhs_num = rhs.get_real()

                res = lhs_num + rhs_num

                if isinstance(res, int):
                    return BCValue.new_integer(res)
                elif isinstance(res, float):
                    return BCValue.new_real(res)
            case "sub":
                if lhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the left hand side of a binary expression!",
                        expr.lhs.pos,
                    )
                elif rhs.is_uninitialized():
                    raise BCError(
                        "cannot have NULL in the right hand side of a binary expression!",
                        expr.rhs.pos,
                    )

                if lhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError("Cannot subtract bools, chars, and strings!")

                if rhs.kind in {
                    BCPrimitiveType.BOOLEAN,
                    BCPrimitiveType.CHAR,
                    BCPrimitiveType.STRING,
                }:
                    raise BCError("Cannot subtract bools, chars, and strings!")

                lhs_num: int | float = 0
                rhs_num: int | float = 0

                if lhs.kind == BCPrimitiveType.INTEGER:
                    lhs_num = lhs.get_integer()
                elif lhs.kind == BCPrimitiveType.REAL:
                    lhs_num = lhs.get_real()

                if rhs.kind == BCPrimitiveType.INTEGER:
                    rhs_num = rhs.get_integer()
                elif rhs.kind == BCPrimitiveType.REAL:
                    rhs_num = rhs.get_real()

                res = lhs_num - rhs_num

                if isinstance(res, int):
                    return BCValue.new_integer(res)
                elif isinstance(res, float):
                    return BCValue.new_real(res)
            case "and":
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

                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()

                if lhs_b == None:
                    raise BCError(
                        "left hand side in boolean operation is null", expr.lhs.pos
                    )

                if rhs_b == None:
                    raise BCError(
                        "right hand side in boolean operation is null", expr.rhs.pos
                    )

                res = lhs_b and rhs_b
                return BCValue.new_boolean(res)
            case "or":
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

                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()

                if lhs_b == None:
                    raise BCError(
                        "left hand side in boolean operation is null", expr.lhs.pos
                    )

                if rhs_b == None:
                    raise BCError(
                        "right hand side in boolean operation is null", expr.rhs.pos
                    )

                res = lhs_b or rhs_b

                return BCValue.new_boolean(res)

    def visit_array_index(self, expr: ArrayIndex):
        return

    def visit_fncall(self, expr: FunctionCall):
        return

    def visit_expr(self, expr: Expr) -> BCValue | None:
        match expr:
            case Typecast():
                return self.visit_typecast(expr)
            case Grouping():
                return self.visit_expr(expr.inner)
            case Negation():
                inner = self.visit_expr(expr.inner)
                if not inner:
                    return

                if inner.kind == BCPrimitiveType.INTEGER:
                    return BCValue.new_integer(-inner.get_integer())  # type: ignore
                elif inner.kind == BCPrimitiveType.REAL:
                    return BCValue.new_real(-inner.get_real())  # type: ignore
            case Not():
                inner = self.visit_expr(expr.inner)
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

    def visit_if_stmt(self, stmt: IfStatement):
        # TODO: detect always true/false
        pass

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        pass

    def visit_for_stmt(self, stmt: ForStatement):
        pass

    def visit_while_stmt(self, stmt: WhileStatement):
        pass

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        pass

    def visit_output_stmt(self, stmt: OutputStatement):
        pass

    def visit_input_stmt(self, stmt: InputStatement):
        pass

    def visit_return_stmt(self, stmt: ReturnStatement):
        pass

    def visit_procedure(self, stmt: ProcedureStatement):
        pass

    def visit_function(self, stmt: FunctionStatement):
        pass

    def visit_scope_stmt(self, stmt: ScopeStatement):
        pass

    def visit_include_stmt(self, stmt: IncludeStatement):
        pass

    def visit_call(self, stmt: CallStatement):
        pass

    def visit_assign_stmt(self, stmt: AssignStatement):
        pass

    def visit_constant_stmt(self, stmt: ConstantStatement):
        opt = self.visit_expr(stmt.value)
        if not opt:
            return
        self.constants[stmt.ident.ident] = opt
        self.unwanted_items.append(self.cur_stmt)

    def visit_declare_stmt(self, stmt: DeclareStatement):
        pass

    def visit_trace_stmt(self, stmt: TraceStatement):
        pass

    def visit_openfile_stmt(self, stmt: OpenfileStatement):
        pass

    def visit_readfile_stmt(self, stmt: ReadfileStatement):
        pass

    def visit_writefile_stmt(self, stmt: WritefileStatement):
        pass

    def visit_appendfile_stmt(self, stmt: AppendfileStatement):
        pass

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        pass

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
                val = self.visit_expr(stmt.inner)
                if val:
                    stmt.inner = Literal(stmt.pos, val)

    def visit_program(self, program: Program):
        if program is not None:
            self.visit_block(program.stmts)

    def visit_block(self, block: list[Statement] | None) -> list[Statement]:
        blk = block if block is not None else self.block
        cur = 0
        while cur < len(blk):
            stmt = blk[cur]
            self.cur_stmt = cur
            self.visit_stmt(stmt)
            cur += 1

        return [itm for i, itm in enumerate(blk) if i not in self.unwanted_items]
