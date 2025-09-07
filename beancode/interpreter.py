import os
import sys
import typing as t
import importlib
import random
import time
import copy
import math

from .bean_ffi import BCFunction, BCProcedure, Exports
from .lexer import Lexer
from .parser import *
from .error import *
from . import __version__

@dataclass
class Variable:
    val: BCValue
    const: bool
    export: bool = False

    def is_uninitialized(self) -> bool:
        return self.val.is_uninitialized()

    def is_null(self) -> bool:
        return self.val.is_null()


BlockType = t.Literal[
    "if", "while", "for", "repeatuntil", "function", "procedure", "scope"
]

LIBROUTINES = {
    "ucase": 1,
    "lcase": 1,
    "div": 2,
    "mod": 2,
    "substring": 3,
    "round": 2,
    "sqrt": 1,
    "length": 1,
    "getchar": 0,
    "random": 0,
}

LIBROUTINES_NORETURN = {"putchar": 1, "exit": 1, "sleep": 1, "flush": 0}


class Interpreter:
    block: list[Statement]
    variables: dict[str, Variable]
    functions: dict[
        str, ProcedureStatement | FunctionStatement | BCProcedure | BCFunction
    ]
    calls: list[tuple[BlockType, str | None]]
    func: bool
    proc: bool
    loop: bool
    toplevel: bool
    retval: BCValue | None = None
    _returned: bool

    def __init__(
        self, block: list[Statement], func=False, proc=False, loop=False
    ) -> None:
        self.block = block
        self.calls = list()
        self.variables = dict()
        self.functions = dict()
        self.func = func
        self.proc = proc
        self.loop = loop
        self._returned = False
        self.cur_stmt = 0

        self.variables["null"] = Variable(BCValue("null"), True)
        self.variables["NULL"] = Variable(BCValue("null"), True)

    @classmethod
    def new(cls, block: list[Statement], func=False, proc=False, loop=False) -> "Interpreter":  # type: ignore
        return cls(block, func=func, proc=proc, loop=loop)  # type: ignore

    def can_return(self) -> tuple[bool, bool]:
        proc = False
        func = False

        for item in reversed(self.calls):
            if item[0] == "procedure":
                proc = True
                break
            elif item[0] == "function":
                func = True
                break

        return (proc, func)

    def error(self, msg: str, pos: t.Any = None) -> t.NoReturn:
        proc = None
        func = None
        for item in reversed(self.calls):
            if item[0] == "procedure":
                proc = item[1]
                break
            elif item[0] == "function":
                func = item[1]
                break

        raise BCError(msg, pos, proc=proc, func=func)
    
    def visit_binaryexpr(self, expr: BinaryExpr) -> BCValue:  # type: ignore
        match expr.op:
            case "assign":
                raise ValueError("impossible to have assign in binaryexpr")
            case "equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                res = lhs == rhs
                return BCValue("boolean", boolean=res)
            case "not_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                res = lhs != rhs
                return BCValue("boolean", boolean=res)
            case "greater_than":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                lhs_num: int | float | None
                rhs_num: int | float | None | None

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real  # type: ignore

                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"impossible to perform greater_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real  # type: ignore
                    if lhs_num == None:
                        self.error(
                            "left hand side in comparison operation is null!",
                            expr.lhs.pos,
                        )
                    if rhs_num == None:
                        self.error(
                            "right hand side in comparison operation is null!",
                            expr.rhs.pos,
                        )

                    return BCValue("boolean", boolean=(lhs_num > rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        self.error(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "boolean":
                        self.error(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "string":
                        return BCValue(
                            "boolean", boolean=(lhs.get_string() > rhs.get_string())
                        )
            case "less_than":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                lhs_num: int | float | None
                rhs_num: int | float | None | None

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real  # type: ignore

                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"impossible to perform less_than between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real  # type: ignore
                    if lhs_num == None:
                        self.error(
                            "left hand side in comparison operation is null!",
                            expr.lhs.pos,
                        )
                    if rhs_num == None:
                        self.error(
                            "right hand side in comparison operation is null!",
                            expr.rhs.pos,
                        )

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        self.error(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "boolean":
                        self.error(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "string":
                        return BCValue(
                            "boolean", boolean=(lhs.get_string() < rhs.get_string())
                        )
            case "greater_than_or_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                lhs_num: int | float | None
                rhs_num: int | float | None | None

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real  # type: ignore

                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"impossible to perform greater_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real  # type: ignore
                    if lhs_num == None:
                        self.error(
                            "left hand side in comparison operation is null!",
                            expr.lhs.pos,
                        )
                    if rhs_num == None:
                        self.error(
                            "right hand side in comparison operation is null!",
                            expr.rhs.pos,
                        )

                    return BCValue("boolean", boolean=(lhs_num >= rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        self.error(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "boolean":
                        self.error(
                            f"illegal to compare booleans with inequality comparisons",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "string":
                        return BCValue(
                            "boolean", boolean=(lhs.get_string() >= rhs.get_string())
                        )
            case "less_than_or_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                lhs_num: int | float | None
                rhs_num: int | float | None | None

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real  # type: ignore

                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"impossible to perform less_than_or_equal between {lhs.kind} and {rhs.kind}",
                            expr.rhs.pos,
                        )

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real  # type: ignore
                    if lhs_num == None:
                        self.error(
                            "left hand side in comparison operation is null!",
                            expr.lhs.pos,
                        )
                    if rhs_num == None:
                        self.error(
                            "right hand side in comparison operation is null!",
                            expr.rhs.pos,
                        )

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        self.error(
                            f"cannot compare incompatible types {lhs.kind} and {rhs.kind}",
                            expr.lhs.pos,
                        )
                    elif lhs.kind == "boolean":
                        self.error(f"illegal to compare booleans", expr.lhs.pos)
                    elif lhs.kind == "string":
                        return BCValue(
                            "boolean", boolean=(lhs.get_string() <= rhs.get_string())
                        )
            # add sub mul div
            case "mul":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot multiply between bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                if rhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot multiply between bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float | None = 0
                rhs_num: int | float | None = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()

                if lhs_num == None:
                    self.error(
                        "left hand side in numerical operation is null!", expr.lhs.pos
                    )
                if rhs_num == None:
                    self.error(
                        "right hand side in numerical operation is null!", expr.rhs.pos
                    )

                res = lhs_num * rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "pow":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot exponentiate bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                if rhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot exponentiate bools, chars, and strings!",
                        expr.lhs.pos,
                    )

                lhs_num: int | float | None = 0
                rhs_num: int | float | None = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()

                if lhs_num == None:
                    self.error(
                        "left hand side in numerical operation is null!", expr.lhs.pos
                    )
                if rhs_num == None:
                    self.error(
                        "right hand side in numerical operation is null!", expr.rhs.pos
                    )

                res = lhs_num**rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)

            case "div":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot divide between bools, chars, and strings!", expr.lhs.pos
                    )

                if rhs.kind in ["boolean", "char", "string"]:
                    self.error(
                        "Cannot divide between bools, chars, and strings!", expr.rhs.pos
                    )

                lhs_num: int | float | None = 0
                rhs_num: int | float | None | None = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()

                if lhs_num == None:
                    self.error(
                        "left hand side in numerical operation is null!", expr.lhs.pos
                    )
                if rhs_num == None:
                    self.error(
                        "right hand side in numerical operation is null!", expr.rhs.pos
                    )

                res = lhs_num / rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "add":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["char", "string"] or rhs.kind in ["char", "string"]:
                    # concatenate instead
                    lhs_str_or_char: str = str()
                    rhs_str_or_char: str = str()

                    if lhs.kind == "string":
                        lhs_str_or_char = lhs.get_string()
                    elif lhs.kind == "char":
                        lhs_str_or_char = lhs.get_char()

                    if rhs.kind == "string":
                        rhs_str_or_char = rhs.get_string()
                    elif rhs.kind == "char":
                        rhs_str_or_char = rhs.get_char()

                    if lhs_str_or_char == None:
                        self.error(
                            "left hand side in string/char concatenation is null!",
                            expr.lhs.pos,
                        )

                    if rhs_str_or_char == None:
                        self.error(
                            "right hand side in string/char concatenation is null!",
                            expr.rhs.pos,
                        )

                    res = str(lhs_str_or_char + rhs_str_or_char)
                    return BCValue("string", string=res)

                if "boolean" in [lhs.kind, rhs.kind]:
                    self.error("Cannot add bools, chars, and strings!")

                lhs_num: int | float | None = 0
                rhs_num: int | float | None = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()

                if lhs_num == None:
                    self.error(
                        "left hand side in numerical operation is null!", expr.lhs.pos
                    )
                if rhs_num == None:
                    self.error(
                        "right hand side in numerical operation is null!", expr.rhs.pos
                    )
                res = lhs_num + rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "sub":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["boolean", "char", "string"]:
                    self.error("Cannot subtract bools, chars, and strings!")

                if rhs.kind in ["boolean", "char", "string"]:
                    self.error("Cannot subtract bools, chars, and strings!")

                lhs_num: int | float | None = 0
                rhs_num: int | float | None = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()

                if lhs_num == None:
                    self.error(
                        "left hand side in numerical operation is null!", expr.lhs.pos
                    )
                if rhs_num == None:
                    self.error(
                        "right hand side in numerical operation is null!", expr.rhs.pos
                    )
                res = lhs_num - rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "and":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind != "boolean":
                    self.error(
                        f"cannot perform logical AND on value with type {lhs.kind}",
                        expr.lhs.pos,
                    )

                if rhs.kind != "boolean":
                    self.error(
                        f"cannot perform logical AND on value with type {lhs.kind}",
                        expr.rhs.pos,
                    )

                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()

                if lhs_b == None:
                    self.error(
                        "left hand side in boolean operation is null", expr.lhs.pos
                    )

                if rhs_b == None:
                    self.error(
                        "right hand side in boolean operation is null", expr.rhs.pos
                    )

                res = lhs_b and rhs_b
                return BCValue("boolean", boolean=res)
            case "or":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind != "boolean":
                    self.error(
                        f"cannot perform logical OR on value with type {lhs.kind}",
                        expr.lhs.pos,
                    )

                if rhs.kind != "boolean":
                    self.error(
                        f"cannot perform logical OR on value with type {lhs.kind}",
                        expr.rhs.pos,
                    )

                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()

                if lhs_b == None:
                    self.error(
                        "left hand side in boolean operation is null", expr.lhs.pos
                    )

                if rhs_b == None:
                    self.error(
                        "right hand side in boolean operation is null", expr.rhs.pos
                    )

                # python does: False or True = False... <redacted> you
                res = lhs_b or rhs_b

                return BCValue("boolean", boolean=res)

    def _get_array_index(self, ind: ArrayIndex) -> tuple[int, int | None]:
        index = self.visit_expr(ind.idx_outer).integer

        if index is None:
            self.error("found (null) for array index", ind.idx_outer.pos)

        v = self.variables[ind.ident.ident].val

        if isinstance(v.kind, BCArrayType):
            a: BCArray = v.array  # type: ignore

            if a.typ.is_matrix:
                if ind.idx_inner is None:
                    self.error("expected 2 indices for matrix indexing", ind.pos)

                inner_index = self.visit_expr(ind.idx_inner).integer
                if inner_index is None:
                    self.error(
                        "found (null) for inner array index", ind.idx_inner.pos
                    )

                return (index, inner_index)
            else:
                return (index, None)
        else:
            if v.kind == "string":
                self.error("cannot index a string! please use the SUBSTRING library routine instead.", ind.ident.pos)
            else:
                self.error(f"attempted to index {v.kind}", ind.ident.pos)

    def visit_array_index(self, ind: ArrayIndex) -> BCValue:  # type: ignore
        index = self.visit_expr(ind.idx_outer).integer

        if index is None:
            self.error("found (null) for array index", ind.idx_outer.pos)

        v = self.variables[ind.ident.ident].val

        if isinstance(v.kind, BCArrayType):
            if v.array is None:
                return BCValue("null")

            a: BCArray = v.array  # type: ignore

            tup = self._get_array_index(ind)
            if a.typ.is_matrix:
                inner = tup[1]
                if inner is None:
                    self.error(
                        "second index not present for matrix index", ind.ident.pos
                    )

                if tup[0] not in range(a.matrix_bounds[0], a.matrix_bounds[1] + 1):  # type: ignore
                    self.error(
                        f"attempted to access out of bounds array element `{tup[0]}`",
                        ind.idx_outer.pos,
                    )

                if tup[1] not in range(a.matrix_bounds[2], a.matrix_bounds[3] + 1):  # type: ignore
                    self.error(
                        f"attempted to access out of bounds array element `{tup[1]}`", ind.idx_inner.pos  # type: ignore
                    )

                res = a.matrix[tup[0] - a.matrix_bounds[0]][inner - a.matrix_bounds[2]]  # type: ignore

                if res.is_uninitialized():
                    return BCValue("null")
                else:
                    return res
            else:
                if tup[0] not in range(a.flat_bounds[0], a.flat_bounds[1] + 1):  # type: ignore
                    if tup[0] == 0:
                        self.error(
                            "attemped to access the 0th array element, which is disallowed in pseudocode",
                            ind.idx_outer.pos,
                        )
                    else:
                        self.error(
                            f"attempted to access out of bounds array element {tup[0]}",
                            ind.idx_outer.pos,
                        )

                res = a.flat[tup[0] - a.flat_bounds[0]]  # type: ignore
                if res.is_uninitialized():
                    return BCValue("null")
                else:
                    return res
        else:
            if v.kind == "string":
                self.error(f"cannot index a string! please use SUBSTRING({ind.ident.ident}, {index}, 1) instead.", ind.ident.pos)
            else:
                self.error(f"attempted to index {v.kind}", ind.ident.pos)

    def visit_ucase(self, txt: str) -> BCValue:
        return BCValue("string", string=txt.upper())

    def visit_lcase(self, txt: str) -> BCValue:
        return BCValue("string", string=txt.lower())

    def visit_substring(self, txt: str, begin: int, length: int) -> BCValue:
        begin = begin - 1
        s = txt[begin : begin + length]
        if len(s) == 0:
            return BCValue("null")
        if len(s) == 1:
            return BCValue("char", char=s[0])
        else:
            return BCValue("string", string=s)

    def visit_length(self, txt: str) -> BCValue:
        return BCValue("integer", integer=len(txt))

    def visit_round(self, val: float, places: int) -> BCValue:
        return BCValue("real", real=round(val, places))

    def visit_getchar(self) -> BCValue:
        s = sys.stdin.read(1)[0]  # get ONE character
        return BCValue("char", char=s)

    def visit_putchar(self, ch: str):
        print(ch[0], end="")

    def visit_exit(self, code: int) -> t.NoReturn:
        sys.exit(code)

    def visit_div(self, lhs: int | float, rhs: int | float) -> BCValue:
        return BCValue("integer", integer=int(lhs // rhs))

    def visit_mod(self, lhs: int | float, rhs: int | float) -> BCValue:
        if type(rhs) == float:
            return BCValue("real", real=float(rhs % rhs))
        else:
            return BCValue("integer", integer=int(lhs % rhs))

    def visit_sqrt(self, val: BCValue) -> BCValue:  # type: ignore
        if val.kind == "integer":
            num = val.get_integer()
            return BCValue("real", real=math.sqrt(num))
        elif val.kind == "real":
            num = val.get_real()
            return BCValue("real", real=math.sqrt(num))

    def visit_random(self) -> BCValue:
        return BCValue("real", real=random.random())

    def visit_sleep(self, duration: float):
        time.sleep(duration)

    def visit_libroutine(self, stmt: FunctionCall) -> BCValue:  # type: ignore
        name = stmt.ident.lower()
        args = stmt.args
        nargs = LIBROUTINES[name.lower()]

        if len(args) < nargs:
            self.error(
                f"expected {nargs} args, but got {len(args)} in call to library routine {name}",
                stmt.pos,
            )

        evargs: list[BCValue] = []
        for _, arg in zip(range(nargs), args):
            evargs.append(self.visit_expr(arg))

        try:
            match name.lower():
                case "ucase":
                    [txt, *_] = evargs
                    if txt.kind == "char":
                        return self.visit_ucase(txt.get_char())
                    elif txt.kind == "string":
                        return self.visit_ucase(txt.get_string())
                    else:
                        self.error(f"cannot call UCASE on a {txt.kind}", stmt.pos)
                case "lcase":
                    [txt, *_] = evargs
                    if txt.kind == "char":
                        return self.visit_lcase(txt.get_char())
                    elif txt.kind == "string":
                        return self.visit_lcase(txt.get_string())
                    else:
                        self.error(f"cannot call LCASE on a {txt.kind}", stmt.pos)
                case "substring":
                    [txt, begin, length, *_] = evargs
                    return self.visit_substring(
                        txt.get_string(), begin.get_integer(), length.get_integer()
                    )
                case "div":
                    [lhs, rhs, *_] = evargs

                    if lhs.kind not in ["integer", "real"]:
                        self.error(
                            f"expected INTEGER or REAL for the lhs of DIV, get {lhs.kind}",
                            stmt.pos,
                        )
                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"expected INTEGER or REAL for the rhs of DIV, get {rhs.kind}",
                            stmt.pos,
                        )

                    lhs_val = lhs.get_integer() if lhs.kind == "integer" else lhs.get_real()
                    rhs_val = rhs.get_integer() if lhs.kind == "integer" else rhs.get_real()

                    return self.visit_div(lhs_val, rhs_val)
                case "mod":
                    [lhs, rhs, *_] = evargs

                    if lhs.kind not in ["integer", "real"]:
                        self.error(
                            f"expected INTEGER or REAL for the lhs of MOD, get {lhs.kind}",
                            stmt.pos,
                        )
                    if rhs.kind not in ["integer", "real"]:
                        self.error(
                            f"expected INTEGER or REAL for the rhs of MOD, get {rhs.kind}",
                            stmt.pos,
                        )

                    lhs_val = lhs.get_integer() if lhs.kind == "integer" else lhs.get_real()
                    rhs_val = rhs.get_integer() if lhs.kind == "integer" else rhs.get_real()

                    return self.visit_mod(lhs_val, rhs_val)
                case "length":
                    [txt, *_] = evargs
                    if isinstance(txt.kind, BCArrayType):
                        self.error("cannot call LENGTH on an array! please use the end index instead.", stmt.pos)
                    if txt.kind != "string":
                        self.error(f"cannot call LENGTH on a {txt.kind}", stmt.pos)

                    return self.visit_length(txt.get_string())
                case "round":
                    [val_r, places, *_] = evargs
                    return self.visit_round(val_r.get_real(), places.get_integer())
                case "sqrt":
                    [val, *_] = evargs
                    if val.kind not in ["integer", "real"]:
                        self.error(
                            f"cannot perform a square root on object of type {val.kind}",
                            stmt.pos,
                        )
                    return self.visit_sqrt(val)
                case "getchar":
                    return self.visit_getchar()
                case "random":
                    return self.visit_random()
        except BCError as e:
            e.pos = stmt.pos
            raise e

    def visit_libroutine_noreturn(self, stmt: CallStatement):
        name = stmt.ident.lower()
        args = stmt.args
        nargs = LIBROUTINES_NORETURN[name.lower()]

        if len(args) < nargs:
            self.error(
                f"expected {nargs} args, but got {len(args)} in call to library routine {name}"
            )

        evargs: list[BCValue] = []
        for _, arg in zip(range(nargs), args):
            evargs.append(self.visit_expr(arg))

        match name:
            case "putchar":
                [ch, *_] = evargs
                self.visit_putchar(ch.get_char())
            case "exit":
                [code, *_] = evargs
                self.visit_exit(code.get_integer())
            case "sleep":
                [duration, *_] = evargs
                self.visit_sleep(duration.get_real())
            case "flush":
                sys.stdout.flush()

    def visit_ffi_fncall(self, func: BCFunction, stmt: FunctionCall) -> BCValue:
        if len(func.params) != len(stmt.args):
            self.error(
                # TODO: better error msg
                f"FFI function {func.name} declares {len(func.params)} variables but only found {len(stmt.args)} in function call",
                stmt.pos,
            )

        args = {}
        for param, arg in zip(func.params, stmt.args):
            args[param] = self.visit_expr(arg)

        retval = func.fn(args)

        if retval.is_null() or retval.is_uninitialized():
            self.error(
                f"FFI function {func.name} returns {func.returns.upper()} but returned a null/uninitialized value.",
                stmt.pos,
            )

        return retval

    def visit_typeof(self, stmt: FunctionCall) -> BCValue:
        if len(stmt.args) > 1:
            self.error(f"cannot get the type of more than one value!", stmt.pos)

        if len(stmt.args) == 0:
            self.error(f"insufficient arguments passed to TYPE/TYPEOF", stmt.pos)

        s = str()
        val = self.visit_expr(stmt.args[0])
        kind = val.kind
        if isinstance(kind, BCArrayType):
            s = val.get_array().get_type_str()
        else:
            s = str(kind.upper())
        return BCValue("string", string=s)

    def visit_fncall(self, stmt: FunctionCall) -> BCValue:
        if stmt.ident not in self.functions and stmt.ident.lower() in LIBROUTINES:
            return self.visit_libroutine(stmt)

        if stmt.ident.lower() in ["typeof", "type"]:
            return self.visit_typeof(stmt)

        try:
            func = self.functions[stmt.ident]
        except KeyError:
            self.error(f"no function named {stmt.ident} exists", stmt.pos)

        if isinstance(func, ProcedureStatement):
            self.error("cannot call procedure without CALL!", stmt.pos)

        if isinstance(func, BCProcedure):
            self.error("cannot call FFI procedure without CALL!", stmt.pos)

        if isinstance(func, BCFunction):
            return self.visit_ffi_fncall(func, stmt)

        intp = self.new(func.block, func=True)
        intp.calls = self.calls
        intp.calls.append(("function", func.name))
        vars = self.variables

        if len(func.args) != len(stmt.args):
            self.error(
                f"function {func.name} declares {len(func.args)} variables but only found {len(stmt.args)} in procedure call",
                stmt.pos,
            )

        for argdef, argval in zip(func.args, stmt.args):
            val = self.visit_expr(argval)
            vars[argdef.name] = Variable(val=val, const=False, export=False)

        intp.variables = dict(vars)
        intp.functions = self.functions

        intp.visit_block(func.block)
        if intp._returned is False:
            self.error(f"function did not return a value!", stmt.pos)

        if intp.retval is None:
            self.error(f"function's return value is None!", stmt.pos)
        else:
            return intp.retval  # type: ignore

    def visit_ffi_call(self, proc: BCProcedure, stmt: CallStatement):
        if len(proc.params) != len(stmt.args):
            self.error(
                # TODO: better error msg
                f"FFI procedure {proc.name} declares {len(proc.params)} variables but only found {len(stmt.args)} in procedure call",
                stmt.pos,
            )

        args = {}
        for param, arg in zip(proc.params, stmt.args):
            args[param] = self.visit_expr(arg)

        proc.fn(args)

    def visit_call(self, stmt: CallStatement):
        if (
            stmt.ident not in self.functions
            and stmt.ident.lower() in LIBROUTINES_NORETURN
        ):
            return self.visit_libroutine_noreturn(stmt)

        try:
            proc = self.functions[stmt.ident]
        except KeyError:
            self.error(f"no procedure named {stmt.ident} exists", stmt.pos)

        if isinstance(proc, FunctionStatement):
            self.error(
                "cannot run CALL on a function! please call the function without the CALL keyword instead.",
                stmt.pos,
            )

        if isinstance(proc, BCFunction):
            self.error(
                "cannot run CALL on an FFI function! please call the function without the CALL keyword instaed.",
                stmt.pos,
            )

        if isinstance(proc, BCProcedure):
            return self.visit_ffi_call(proc, stmt)

        intp = self.new(proc.block, proc=True)
        intp.calls = self.calls
        intp.calls.append(("procedure", proc.name))
        vars = self.variables

        if len(proc.args) != len(stmt.args):
            self.error(
                f"procedure {proc.name} declares {len(proc.args)} variables but only found {len(stmt.args)} in procedure call",
                stmt.pos,
            )

        for argdef, argval in zip(proc.args, stmt.args):
            val = self.visit_expr(argval)
            vars[argdef.name] = Variable(val=val, const=False, export=False)

        intp.variables = dict(vars)
        intp.functions = dict(self.functions)

        intp.visit_block(proc.block)

    def _typecast_string(self, inner: BCValue, pos: tuple[int, int, int]) -> BCValue:
        s = ""
       
        if isinstance(inner.kind, BCArrayType):
            arr = inner.get_array()
            s = self._display_array(arr)
        else:
            match inner.kind:
                case "null":
                    s = "(null)"
                case "boolean":
                    if inner.get_boolean():
                        s = "true"
                    else:
                        s = "false"
                case "integer":
                    s = str(inner.get_integer())
                case "real":
                    s = str(inner.get_real())
                case "char":
                    s = str(inner.get_char()[0])
                case "string":
                    return inner

        return BCValue("string", string=s)

    def _typecast_integer(self, inner: BCValue, pos: tuple[int, int, int]) -> BCValue:
        i = 0
        match inner.kind:
            case "string":
                s = inner.get_string()
                try:
                    i = int(s.strip())
                except ValueError:
                    self.error(f"impossible to convert `{s}` to an INTEGER!", pos)
            case "integer":
                return inner
            case "real":
                i = int(inner.get_real())
            case "char":
                i = ord(inner.get_char()[0])
            case "boolean":
                i = 1 if inner.get_boolean() else 0

        return BCValue("integer", integer=i)

    def _typecast_real(self, inner: BCValue, pos: tuple[int, int, int]) -> BCValue:
        r = 0.0

        match inner.kind:
            case "string":
                s = inner.get_string()
                try:
                    r = float(s.strip())
                except ValueError:
                    self.error(f"impossible to convert `{s}` to a REAL!", pos)
            case "integer":
                r = float(inner.get_integer())
            case "real":
                return inner
            case "char":
                self.error(f"impossible to convert a REAL to a CHAR!", pos)
            case "boolean":
                r = 1.0 if inner.get_boolean() else 0.0

        return BCValue("real", real=r)

    def _typecast_char(self, inner: BCValue, pos: tuple[int, int, int]) -> BCValue:
        c = ""

        match inner.kind:
            case "string":
                self.error(
                    f"cannot convert a STRING to a CHAR! use SUBSTRING(str, begin, 1) to get a character.",
                    pos,
                )
            case "integer":
                c = chr(inner.get_integer())
            case "real":
                self.error(f"impossible to convert a CHAR to a REAL!", pos)
            case "char":
                return inner
            case "boolean":
                self.error(f"impossible to convert a BOOLEAN to a CHAR!", pos)

        return BCValue("char", char=c)

    def _typecast_boolean(self, inner: BCValue) -> BCValue:
        b = False

        match inner.kind:
            case "string":
                b = inner.get_string() != ""
            case "integer":
                b = inner.get_integer() != 0
            case "real":
                b = inner.get_real() != 0.0
            case "char":
                b = ord(inner.get_char()) != 0
            case "boolean":
                return inner

        return BCValue("boolean", boolean=b)

    def visit_typecast(self, tc: Typecast) -> BCValue:  # type: ignore
        inner = self.visit_expr(tc.expr)

        if inner.kind == "null":
            self.error("cannot cast anything to NULL!", tc.pos)

        if isinstance(inner.kind, BCArrayType) and tc.typ != "string":
            self.error(f"cannot cast an array to a {tc.typ}", tc.pos)

        match tc.typ:
            case "string":
                return self._typecast_string(inner, tc.pos)  # type: ignore
            case "integer":
                return self._typecast_integer(inner, tc.pos)  # type: ignore
            case "real":
                return self._typecast_real(inner, tc.pos)  # type: ignore
            case "char":
                return self._typecast_char(inner, tc.pos)  # type: ignore
            case "boolean":
                return self._typecast_boolean(inner, tc.pos)  # type: ignore

    def visit_matrix_literal(self, expr: ArrayLiteral) -> BCValue:
        first_matrix_elem: Expr = expr.items[0].items[0]  # type: ignore
        matrix = []
        typ = self.visit_expr(first_matrix_elem).kind
        inner_arr_len = len(expr.items[0].items)  # type: ignore

        outer_arr: list[ArrayLiteral] = expr.items  # type: ignore
        for arr_lit in outer_arr:
            arr = []
            if len(arr_lit.items) != inner_arr_len:
                self.error("all matrix row lengths must be consistent!", arr_lit.pos)

            for val in arr_lit.items:
                newval = self.visit_expr(val)
                if newval.kind != typ:
                    self.error(
                        "matrix literal may not contain items of multiple types!",
                        val.pos,
                    )
                arr.append(newval)

            matrix.append(arr)

        bounds = (1, len(matrix), 1, inner_arr_len)
        arrtyp = BCArrayType(inner=typ, is_matrix=True, matrix_bounds=None)  # type: ignore
        return BCValue(
            kind=arrtyp, array=BCArray(typ=arrtyp, matrix=matrix, matrix_bounds=bounds)
        )

    def visit_array_literal(self, expr: ArrayLiteral) -> BCValue:
        if isinstance(expr.items[0], ArrayLiteral):
            return self.visit_matrix_literal(expr)

        vals = [self.visit_expr(expr.items[0])]
        typ = vals[0].kind

        for val in expr.items[1:]:
            newval = self.visit_expr(val)
            if newval.kind != typ:
                self.error(
                    "array literal may not contain items of multiple types!", val.pos
                )
            vals.append(newval)

        bounds = (1, len(vals))

        arrtyp = BCArrayType(inner=typ, is_matrix=False, flat_bounds=None)  # type: ignore
        return BCValue(
            kind=arrtyp, array=BCArray(typ=arrtyp, flat=vals, flat_bounds=bounds)
        )

    def visit_expr(self, expr: Expr) -> BCValue:  # type: ignore
        if isinstance(expr, Typecast):
            return self.visit_typecast(expr)
        if isinstance(expr, Grouping):
            return self.visit_expr(expr.inner)
        elif isinstance(expr, Negation):
            inner = self.visit_expr(expr.inner)
            if inner.kind not in ["integer", "real"]:
                self.error(
                    f"attemped to negate a value of type {inner.kind}", expr.inner.pos
                )

            if inner.kind == "integer":
                return BCValue("integer", integer=-inner.integer)  # type: ignore
            elif inner.kind == "real":
                return BCValue("real", real=-inner.real)  # type: ignore
        elif isinstance(expr, Not):
            inner = self.visit_expr(expr.inner)
            if inner.kind != "boolean":
                self.error(
                    f"attempted to perform logical NOT on value of type {inner.kind}",
                    expr.inner.pos,
                )

            return BCValue("boolean", boolean=not inner.get_boolean())
        elif isinstance(expr, Identifier):
            try:
                var = self.variables[expr.ident]
            except KeyError:
                self.error(
                    f"attempted to access nonexistent variable `{expr.ident}`", expr.pos
                )
            return var.val
        elif isinstance(expr, Literal):
            return expr.to_bcvalue()
        elif isinstance(expr, ArrayLiteral):
            return self.visit_array_literal(expr)
        elif isinstance(expr, BinaryExpr):
            return self.visit_binaryexpr(expr)
        elif isinstance(expr, ArrayIndex):
            return self.visit_array_index(expr)
        elif isinstance(expr, FunctionCall):
            return self.visit_fncall(expr)
        else:
            raise ValueError("expr is very corrupted whoops")

    def _display_array(self, arr: BCArray) -> str:
        if not arr.typ.is_matrix:
            res = "["
            flat: list[BCValue] = arr.flat  # type: ignore
            for idx, item in enumerate(flat):
                if item.is_uninitialized():
                    res += "(null)"
                else:
                    res += str(item)

                if idx != len(flat) - 1:
                    res += ", "
            res += "]"

            return res
        else:
            matrix: list[list[BCValue]] = arr.matrix  # type: ignore
            outer_res = "["
            for oidx, a in enumerate(matrix):
                res = "["
                for iidx, item in enumerate(a):
                    if item.is_uninitialized():
                        res += "(null)"
                    else:
                        res += str(item)

                    if iidx != len(a) - 1:
                        res += ", "
                res += "]"

                outer_res += res
                if oidx != len(matrix) - 1:
                    outer_res += ", "
            outer_res += "]"

            return outer_res

    def visit_output_stmt(self, stmt: OutputStatement):
        res = ""
        for item in stmt.items:
            evaled = self.visit_expr(item)
            if isinstance(evaled.kind, BCArrayType):
                res += self._display_array(evaled.array)  # type: ignore
            else:
                res += str(evaled)
        print(res)

    def _guess_input_type(self, inp: str) -> BCValue:
        p = Parser([])
        if p.is_real(inp):
            return BCValue(kind="real")
        elif p.is_integer(inp):
            return BCValue(kind="integer")

        if inp.strip().lower() in ["true", "false", "no", "yes"]:
            return BCValue(kind="boolean")

        if len(inp.strip()) == 1:
            return BCValue(kind="char")
        else:
            return BCValue(kind="string")

    def visit_input_stmt(self, stmt: InputStatement):
        inp = input()
        id = stmt.ident.ident

        data: Variable | None = self.variables.get(id)

        if data is None:
            val = self._guess_input_type(inp)
            data = Variable(val, False, export=False)
            self.variables[id] = data

        if data.const:
            self.error(
                f"attempted to call `INPUT` into constant {id}", stmt.ident.pos
            )

        if type(data.val.kind) == BCArrayType:
            self.error(f"attempted to call `INPUT` on an array", stmt.ident.pos)

        if inp.strip() == "":
            self.error(f"empty string supplied into variable with type `{data.val.kind.upper()}`", stmt.pos)  # type: ignore

        match data.val.kind:
            case "string":
                self.variables[id].val.kind = "string"
                self.variables[id].val.string = inp
            case "char":
                if len(inp) > 1:
                    self.error(
                        f"expected single character but got `{inp}` for CHAR", stmt.pos
                    )

                self.variables[id].val.kind = "char"
                self.variables[id].val.char = inp
            case "boolean":
                if inp.lower() not in ["true", "false", "yes", "no"]:
                    self.error(
                        f"expected TRUE, FALSE, YES or NO including lowercase for BOOLEAN but got `{inp}`",
                        stmt.pos,
                    )

                inp = inp.lower()
                if inp in ["true", "yes"]:
                    self.variables[id].val.kind = "boolean"
                    self.variables[id].val.boolean = True
                elif inp in ["false", "no"]:
                    self.variables[id].val.kind = "boolean"
                    self.variables[id].val.boolean = False
            case "integer":
                inp = inp.lower().strip()
                p = Parser([])
                if p.is_integer(inp):
                    try:
                        res = int(inp)
                        self.variables[id].val.kind = "integer"
                        self.variables[id].val.integer = res
                    except ValueError:
                        self.error("expected INTEGER for INPUT", stmt.ident.pos)
                else:
                    self.error("expected INTEGER for INPUT", stmt.ident.pos)
            case "real":
                inp = inp.lower().strip()
                p = Parser([])
                if p.is_real(inp) or p.is_integer(inp):
                    try:
                        res = float(inp)
                        self.variables[id].val.kind = "real"
                        self.variables[id].val.real = res
                    except ValueError:
                        self.error("expected REAL for INPUT", stmt.ident.pos)
                else:
                    self.error("expected REAL for INPUT", stmt.ident.pos)

    def visit_return_stmt(self, stmt: ReturnStatement):
        proc, func = self.can_return()

        if not proc and not func:
            self.error(
                f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                stmt.pos,
            )

        if func:
            if stmt.expr is None:
                self.error("you must return something from a function!", stmt.pos)

            res = self.visit_expr(stmt.expr)
            self.retval = res
            self._returned = True
        elif proc:
            if stmt.expr is not None:
                self.error("you cannot return a value from a procedure!", stmt.pos)

            self._returned = True

    def visit_include_ffi_stmt(self, stmt: IncludeStatement):
        # XXX: this is probably the most scuffed code in existence.
        try:
            mod: Exports = importlib.import_module(f"beancode.modules.{stmt.file}").EXPORTS
        except ModuleNotFoundError:
            self.error(f"failed to include module {stmt.file}", stmt.pos)

        for const in mod["constants"]:
            self.variables[const.name] = Variable(val=const.value, const=True)

        for var in mod["variables"]:
            val = var.value
            if val is not None:
                self.variables[var.name] = Variable(val=val, const=False)
            else:
                if var.typ is None:
                    self.error(
                        "must have either typ, value or both be set in ffi export",
                        stmt.pos,
                    )
                self.variables[var.name] = Variable(BCValue(kind=var.typ), const=False)

        for proc in mod["procs"]:
            self.functions[proc.name] = proc

        for func in mod["funcs"]:
            self.functions[func.name] = func

    def visit_include_stmt(self, stmt: IncludeStatement):
        if stmt.ffi:
            return self.visit_include_ffi_stmt(stmt)

        filename = stmt.file
        path = os.path.join("./", filename)

        # TODO: abstract this stuff into another file
        if not os.path.exists(path):
            error(f"file {filename} does not exist!")

        with open(filename, "r+") as f:
            file_content = f.read()

        lexer = Lexer(file_content)
        toks = lexer.tokenize()
        parser = Parser(toks)
        try:
            program = parser.program()
        except BCError as err:
            err.print(filename, file_content)
            exit(1)
        except BCWarning as warn:
            warn.print(filename, file_content)
            exit(1)

        intp = self.new(program.stmts)
        intp.visit_block(None)

        for name, var in intp.variables.items():
            if var.export:
                self.variables[name] = var

        for name, fn in intp.functions.items():
            if fn.export:  # type: ignore
                self.functions[name] = fn

    def visit_if_stmt(self, stmt: IfStatement):
        cond: BCValue = self.visit_expr(stmt.cond)

        if cond.boolean:
            intp: Interpreter = self.new(stmt.if_block)
        else:
            intp: Interpreter = self.new(stmt.else_block)

        intp.variables = dict(self.variables)
        intp.functions = dict(self.functions)
        intp.calls = self.calls
        intp.calls.append("if")  # type: ignore
        intp.visit_block(None)
        if intp._returned:
            proc, func = self.can_return()

            if not proc and not func:
                # FIXME: is this even a possible branch?!
                self.error(
                    f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                    stmt.pos,
                )

            self._returned = True
            self.retval = intp.retval

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        value: BCValue = self.visit_expr(stmt.expr)

        for branch in stmt.branches:
            rhs = self.visit_expr(branch.expr)
            if value == rhs:
                self.visit_stmt(branch.stmt)
                return

        if stmt.otherwise is not None:
            self.visit_stmt(stmt.otherwise)

    def visit_while_stmt(self, stmt: WhileStatement):
        cond: Expr = stmt.cond  # type: ignore

        block: list[Statement] = stmt.block  # type: ignore

        intp = self.new(block, loop=True)
        intp.variables = dict(self.variables)  # scope
        intp.functions = dict(self.functions)

        while self.visit_expr(cond).boolean:
            intp.visit_block(block)
            # FIXME: barbaric aah
            intp.variables = self.variables.copy()
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    self.error(
                        f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                        stmt.pos,
                    )

                self._returned = True
                self.retval = intp.retval
                return

    def visit_for_stmt(self, stmt: ForStatement):
        begin = self.visit_expr(stmt.begin)

        if begin.kind != "integer":
            self.error(
                "non-integer expression used for for loop begin", stmt.begin.pos
            )

        end = self.visit_expr(stmt.end)

        if end.kind != "integer":
            self.error("non-integer expression used for for loop end", stmt.end.pos)

        if stmt.step is None:
            step = 1
        else:
            step = self.visit_expr(stmt.step).get_integer()

        intp = self.new(stmt.block, loop=True)
        intp.calls = self.calls
        intp.calls.append(("for", None))
        intp.variables = self.variables.copy()
        intp.functions = self.functions.copy()

        counter = begin

        var_existed = stmt.counter.ident in intp.variables
        if var_existed:
            var_prev_value = intp.variables[stmt.counter.ident]

        intp.variables[stmt.counter.ident] = Variable(counter, const=False)

        if step > 0:
            while counter.get_integer() <= end.get_integer():
                intp.visit_block(None)
                #  FIXME: barbaric
                # clear declared variables
                c = intp.variables[stmt.counter.ident]
                intp.variables = self.variables.copy()
                intp.variables[stmt.counter.ident] = c
                if intp._returned:
                    proc, func = self.can_return()

                    if not proc and not func:
                        self.error(
                            f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                            stmt.pos,
                        )

                    self._returned = True
                    self.retval = intp.retval
                    return

                counter.integer = counter.integer + step  # type: ignore
        elif step < 0:
            while counter.get_integer() >= end.get_integer():
                intp.visit_block(None)
                # FIXME:
                # clear declared variables (barbaric)
                c = intp.variables[stmt.counter.ident]
                intp.variables = self.variables.copy()
                intp.variables[stmt.counter.ident] = c
                if intp._returned:
                    proc, func = self.can_return()

                    if not proc and not func:
                        self.error(
                            f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                            stmt.pos,
                        )

                    self._returned = True
                    self.retval = intp.retval
                    return

                counter.integer = counter.integer + step  # type: ignore

        if not var_existed:
            intp.variables.pop(stmt.counter.ident)
        else:
            intp.variables[stmt.counter.ident] = var_prev_value  # type: ignore

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        cond: Expr = stmt.cond  # type: ignore
        intp = self.new(stmt.block, loop=True)
        intp.calls = self.calls
        intp.calls.append(("repeatuntil", None))
        intp.variables = dict(self.variables)
        intp.functions = dict(self.functions)

        while True:
            intp.visit_block(None)
            # FIXME: barbaric
            intp.variables = self.variables.copy()
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    self.error(
                        f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}",
                        stmt.pos,
                    )

                self._returned = True
                self.retval = intp.retval
                return

            if self.visit_expr(cond).boolean:
                break

    def visit_scope_stmt(self, stmt: ScopeStatement):
        intp = self.new(stmt.block, loop=False)
        intp.calls.append(("scope", None))
        intp.variables = dict(self.variables)
        intp.functions = dict(self.functions)
        intp.visit_block(None)

        for name, var in intp.variables.items():
            if var.export:
                self.variables[name] = var

        for name, fn in intp.functions.items():
            if fn.export:  # type: ignore
                self.functions[name] = fn

    def visit_procedure(self, stmt: ProcedureStatement):
        self.functions[stmt.name] = stmt

    def visit_function(self, stmt: FunctionStatement):
        self.functions[stmt.name] = stmt

    def visit_assign_stmt(self, s: AssignStatement):
        if isinstance(s.ident, ArrayIndex):
            key = s.ident.ident.ident

            if self.variables[key].val.array is None:
                self.error(
                    f"tried to index a variable of type {self.variables[key].val.kind} like an array",
                    s.ident.pos,
                )

            tup = self._get_array_index(s.ident)
            if tup[1] is None and self.variables[key].val.array.typ.is_matrix:  # type: ignore
                self.error(f"not enough indices for matrix", s.ident.idx_outer.pos)

            val = self.visit_expr(s.value)
            a: BCArray = self.variables[key].val.array  # type: ignore

            if a.typ.is_matrix:  # type: ignore
                if tup[0] not in range(a.matrix_bounds[0], a.matrix_bounds[1] + 1):  # type: ignore
                    self.error(
                        f"tried to access out of bounds array index {tup[0]}",
                        s.ident.idx_outer.pos,
                    )

                if tup[1] not in range(a.matrix_bounds[2], a.matrix_bounds[3] + 1):  # type: ignore
                    self.error(f"tried to access out of bounds array index {tup[1]}", s.ident.idx_inner.pos)  # type: ignore

                a.matrix[tup[0] - a.matrix_bounds[0]][tup[1] - a.matrix_bounds[2]] = copy.deepcopy(val)  # type: ignore
            else:
                if tup[0] not in range(a.flat_bounds[0], a.flat_bounds[1] + 1):  # type: ignore
                    self.error(
                        f"tried to access out of bounds array index {tup[0]}",
                        s.ident.idx_outer.pos,
                    )

                a.flat[tup[0] - a.flat_bounds[0]] = copy.deepcopy(val)  # type: ignore
        else:
            key = s.ident.ident

            exp = self.visit_expr(s.value)
            var = self.variables.get(key)

            if var is None:
                var = Variable(exp, False, export=False)
                self.variables[key] = var

            if self.variables[key].const:
                self.error(f"attemped to write to constant {key}", s.ident.pos)

            if isinstance(exp.kind, BCArrayType):
                if exp.array.typ.is_matrix and exp.array.matrix_bounds != var.val.array.matrix_bounds:  # type: ignore
                    self.error(
                        f"mismatched matrix sizes in matrix assignment", s.pos
                    )
                elif not exp.array.typ.matrix_bounds and exp.array.flat_bounds != var.val.array.flat_bounds:  # type: ignore
                    self.error(f"mismatched array sizes in array assignment", s.pos)
            elif var.val.kind != exp.kind:
                self.error(
                    f"cannot assign {exp.kind} to {var.val.kind}", s.ident.pos
                )
            self.variables[key].val = copy.deepcopy(exp)

    def visit_constant_stmt(self, c: ConstantStatement):
        key = c.ident.ident

        if key in self.variables:
            self.error(f"variable {key} declared!", c.pos)

        self.variables[key] = Variable(c.value.to_bcvalue(), True, export=c.export)

    def _declare_array(self, d: DeclareStatement, key: str):
        atype: BCArrayType = d.typ  # type: ignore
        inner_type = atype.inner
        if atype.is_matrix:
            inner_end = self.visit_expr(atype.matrix_bounds[3])  # type: ignore
            if inner_end.kind != "integer":
                self.error(
                    f"cannot use type of {inner_end.kind} as array bound!", d.pos
                )

            outer_end = self.visit_expr(atype.matrix_bounds[1])  # type: ignore
            if outer_end.kind != "integer":
                self.error(
                    f"cannot use type of {outer_end.kind} as array bound!", d.pos
                )

            outer_begin = self.visit_expr(atype.matrix_bounds[0])  # type: ignore
            if outer_begin.kind != "integer":
                self.error(
                    f"cannot use type of {outer_begin.kind} as array bound!", d.pos
                )

            inner_begin = self.visit_expr(atype.matrix_bounds[2])  # type: ignore
            if inner_begin.kind != "integer":
                self.error(
                    f"cannot use type of {inner_begin.kind} as array bound!", d.pos
                )

            # Directly setting the result of the comprehension results in multiple pointers pointing to the same list
            in_size = inner_end.get_integer() - inner_begin.get_integer()
            out_size = outer_end.get_integer() - outer_begin.get_integer()
            # array bound declarations are inclusive
            outer_arr = [[BCValue(inner_type) for _ in range(in_size + 1)] for _ in range(out_size + 1)]  # type: ignore

            bounds = (outer_begin.integer, outer_end.integer, inner_begin.integer, inner_end.integer)  # type: ignore
            atype.is_matrix = True
            res = BCArray(typ=atype, matrix=outer_arr, matrix_bounds=bounds)  # type: ignore
        else:
            begin = self.visit_expr(atype.flat_bounds[0])  # type: ignore
            if begin.kind != "integer":
                self.error(f"cannot use type of {begin.kind} as array bound!", d.pos)

            end = self.visit_expr(atype.flat_bounds[1])  # type: ignore
            if end.kind != "integer":
                self.error(f"cannot use type of {end.kind} as array bound!", d.pos)

            size = end.get_integer() - begin.get_integer()
            arr: BCValue = [BCValue(atype) for _ in range(size + 1)]  # type: ignore

            bounds = (begin.integer, end.integer)  # type: ignore
            atype.is_matrix = False
            res = BCArray(typ=atype, flat=arr, flat_bounds=bounds)  # type: ignore

        self.variables[key] = Variable(
            BCValue(kind=d.typ, array=res), False, export=d.export
        )

    def visit_declare_stmt(self, d: DeclareStatement):
        for ident in d.ident:
            key: str = ident.ident
            if key in self.variables:
                self.error(f"variable {key} declared!", d.pos)

            if isinstance(d.typ, BCArrayType):
                self._declare_array(d, key)
            else:
                self.variables[key] = Variable(
                    BCValue(kind=d.typ), False, export=d.export
                )
                if d.expr is not None:
                    expr = self.visit_expr(d.expr)
                    self.variables[key].val = expr

    def visit_stmt(self, stmt: Statement):
        match stmt.kind:
            case "if":
                self.visit_if_stmt(stmt.if_s)  # type: ignore
            case "caseof":
                self.visit_caseof_stmt(stmt.caseof)  # type: ignore
            case "for":
                self.visit_for_stmt(stmt.for_s)  # type: ignore
            case "while":
                self.visit_while_stmt(stmt.while_s)  # type: ignore
            case "repeatuntil":
                self.visit_repeatuntil_stmt(stmt.repeatuntil)  # type: ignore
            case "output":
                self.visit_output_stmt(stmt.output)  # type: ignore
            case "input":
                self.visit_input_stmt(stmt.input)  # type: ignore
            case "return":
                self.visit_return_stmt(stmt.return_s)  # type: ignore
            case "procedure":
                self.visit_procedure(stmt.procedure)  # type: ignore
            case "function":
                self.visit_function(stmt.function)  # type: ignore
            case "scope":
                self.visit_scope_stmt(stmt.scope)  # type: ignore
            case "include":
                self.visit_include_stmt(stmt.include)  # type: ignore
            case "call":
                self.visit_call(stmt.call)  # type: ignore
            case "fncall":
                self.visit_fncall(stmt.fncall)  # type: ignore
            case "assign":
                self.visit_assign_stmt(stmt.assign)  # type: ignore
            case "constant":
                self.visit_constant_stmt(stmt.constant)  # type: ignore
            case "declare":
                self.visit_declare_stmt(stmt.declare)  # type: ignore

    def visit_block(self, block: list[Statement] | None):
        blk = block if block is not None else self.block
        cur = 0
        while cur < len(blk):
            stmt = self.block[cur]
            self.cur_stmt = cur
            self.visit_stmt(stmt)
            if self._returned:
                return
            cur += 1

    def visit_program(self, program: Program):
        if program is not None:
            self.visit_block(program.stmts)

    def reset(self):
        self.cur_stmt = 0

    def reset_all(self):
        self.block = list()
        self.calls = list()
        self.variables = dict()
        self.functions = dict()
