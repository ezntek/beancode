import os
import sys
import importlib
import math
import subprocess

from typing import Any, NoReturn

from .bean_ffi import BCFunction, BCProcedure, Exports
from .lexer import Lexer
from .parser import *
from .error import *
from .libroutines import *
from . import __version__, Pos
from .tracer import *


def _get_file_mode(read: bool, write: bool, append: bool) -> str | None:
    if read and write:
        return "w+"
    elif append and read:
        return "a+"
    elif read:
        return "r"
    elif write:
        return "w"
    elif append:
        return "a"


class Interpreter:
    block: list[Statement]
    variables: dict[str, Variable]
    functions: dict[
        str, ProcedureStatement | FunctionStatement | BCProcedure | BCFunction
    ]
    calls: list[CallStackEntry]
    func: bool
    proc: bool
    loop: bool
    toplevel: bool
    retval: BCValue | None = None
    _returned: bool
    tracer: Tracer | None = None
    tracer_inputs: list[str] | None = None
    tracer_outputs: list[str] | None = None
    tracer_open = False  # open generated html or not by default
    files: dict[str, File]

    def __init__(
        self,
        block: list[Statement],
        func=False,
        proc=False,
        loop=False,
        tracer=None,
        tracer_open=False,
        toplevel=True,
    ) -> None:
        self.block = block
        self.func = func
        self.proc = proc
        self.loop = loop
        self.tracer = tracer
        self.tracer_open = tracer_open
        self.files = dict()
        self.toplevel = toplevel
        self.reset_all()

    @classmethod
    def new(
        cls, block: list[Statement], func=False, proc=False, loop=False, tracer=None
    ) -> "Interpreter":  # type: ignore
        return cls(
            block, func=func, proc=proc, loop=loop, tracer=tracer, toplevel=False
        )  # type: ignore

    def _make_new_interpreter(self, block: list[Statement]) -> "Interpreter":
        intp = self.new(block, loop=False, tracer=self.tracer)
        intp.calls = self.calls
        intp.variables = self.variables.copy()
        intp.functions = self.functions.copy()
        intp.files = self.files.copy()
        return intp

    def reset(self):
        self.cur_stmt = 0

    def reset_all(self):
        self.calls = list()
        self.variables = dict()
        self.functions = dict()

        if self.tracer is not None:
            self.tracer_outputs = list()
            self.tracer_inputs = list()

        self._returned = False
        self.cur_stmt = 0

        for f in self.files.values():
            f.stream.close()

        self.files = dict()
        for file in self.files.values():
            file.stream.close()

    def can_return(self) -> tuple[bool, bool]:
        proc = False
        func = False

        for item in reversed(self.calls):
            if item.func:
                func = True
                break
            else:
                proc = True
                break

        return (proc, func)

    def error(self, msg: str, pos: Any = None) -> NoReturn:
        proc = None
        func = None
        for item in reversed(self.calls):
            if not item.func:
                proc = item.name
                break
            else:
                func = item.name
                break

        raise BCError(msg, pos, proc=proc, func=func)

    def trace(
        self,
        line_num: int,
        loop_trace=False,
    ) -> None:
        if self.tracer is None:
            return

        # loop_trace: manually invoked trace in a loop
        # (every statement in a loop should not be traced)
        if self.loop and not loop_trace:
            return

        self.tracer.collect_new(
            self.variables,
            line_num,
            inputs=self.tracer_inputs,
            outputs=self.tracer_outputs,
        )

        if self.tracer_inputs is not None:
            self.tracer_inputs.clear()  # type: ignore

        if self.tracer_outputs is not None:
            self.tracer_outputs.clear()  # type: ignore

    def get_return_type(self) -> Type | None:
        return self.calls[-1].rtype

    def visit_array_type(self, t: ArrayType) -> BCArrayType:
        if t.is_matrix():
            s_bounds = t.get_matrix_bounds()
            ob = self.visit_expr(s_bounds[0])
            oe = self.visit_expr(s_bounds[1])
            ib = self.visit_expr(s_bounds[2])
            ie = self.visit_expr(s_bounds[3])
            for i, k in enumerate((ob.kind, oe.kind, ib.kind, ie.kind)):
                if k != BCPrimitiveType.INTEGER:
                    self.error(
                        f"cannot use type {k} as array bound!",
                        s_bounds[i].pos,
                    )

            bounds = (ob.val, oe.val, ib.val, ie.val)  # type: ignore
            return BCArrayType.new_matrix(t.inner, bounds)  # type: ignore
        else:
            s_bounds = t.get_flat_bounds()
            begin = self.visit_expr(s_bounds[0])
            if begin.kind != BCPrimitiveType.INTEGER:
                self.error(
                    f"cannot use type {begin.kind} as array bound!",
                    s_bounds[0].pos,
                )

            end = self.visit_expr(s_bounds[1])  # type: ignore
            if end.kind != BCPrimitiveType.INTEGER:
                self.error(
                    f"cannot use type {end.kind} as array bound!",
                    s_bounds[1].pos,
                )

            bounds = (begin.val, end.val)  # type: ignore
            return BCArrayType.new_flat(t.inner, bounds)  # type: ignore

    def visit_type(self, t: Type) -> BCType:
        if isinstance(t, ArrayType):
            return self.visit_array_type(t)
        else:
            return t

    def visit_binaryexpr(self, expr: BinaryExpr) -> BCValue:  # type: ignore
        lhs = self.visit_expr(expr.lhs)
        rhs = self.visit_expr(expr.rhs)

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
                self.error(
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
                self.error(
                    f"cannot {expr.op.humanize()} incompatible types {lhs.kind} and {rhs.kind}!",
                    expr.pos,
                )

            if not (
                lhs.kind == BCPrimitiveType.BOOLEAN
                or rhs.kind == BCPrimitiveType.BOOLEAN
            ):
                self.error(
                    f"cannot {expr.op.humanize()} between {lhs.kind} and {rhs.kind}!",
                    expr.pos,
                )
        else:
            human_kind = "an arithmetic expression"

            # XXX: microoptimizations™
            # we are reducing the number of calls we visit in the Python VM per addition. Addition is a
            # very very common operator and it speeds PrimeTorture up by around 230ms.
            if expr.op != Operator.ADD:
                if expr.op not in {Operator.FLOOR_DIV, Operator.MOD} and not (
                    lhs.kind_is_numeric() and rhs.kind_is_numeric()
                ):
                    self.error(
                        f"cannot {expr.op.humanize()} between BOOLEANs, CHARs and STRINGs!",
                        expr.pos,
                    )

        if expr.op != Operator.EQUAL:
            if lhs.is_uninitialized():
                self.error(
                    f"cannot have NULL in the left hand side of {human_kind}\n"
                    + "is your value an uninitialized value/variable?",
                    expr.lhs.pos,
                )
            if rhs.is_uninitialized():
                self.error(
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
                    self.error("cannot divide by zero!", expr.rhs.pos)

                return BCValue(BCPrimitiveType.REAL, lhs.val / rhs.val) # type: ignore
            case Operator.ADD:
                if (
                    lhs.kind == BCPrimitiveType.BOOLEAN
                    or rhs.kind == BCPrimitiveType.BOOLEAN
                ):
                    self.error("cannot add BOOLEANs!", expr.pos)

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
                    self.error(
                        "Cannot DIV() between BOOLEANs, CHARs and STRINGs!",
                        expr.pos,
                    )

                if rhs.val == 0:
                    self.error("cannot divide by zero!", expr.rhs.pos)

                return BCValue(BCPrimitiveType.INTEGER, int(lhs.val // rhs.val))  # type: ignore
            case Operator.MOD:
                if not (lhs.kind_is_numeric() or rhs.kind_is_numeric()):
                    self.error(
                        "Cannot MOD() between BOOLEANs, CHARs and STRINGs!",
                        expr.pos,
                    )

                if rhs.val == 0:
                    self.error("cannot divide by zero!", expr.rhs.pos)

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

    def _get_array_index(self, ind: ArrayIndex) -> tuple[int, int | None]:
        index_v = self.visit_expr(ind.idx_outer)  # type: ignore
        if index_v.kind != BCPrimitiveType.INTEGER:
            self.error(f"type of array index is {index_v.kind}, not an INTEGER!")

        v = self.visit_expr(ind.expr)
        a: BCArray = v.val  # type: ignore

        if a.typ.is_matrix():
            if ind.idx_inner is None:
                self.error("expected 2 indices for matrix indexing", ind.pos)

            inner_index_v = self.visit_expr(ind.idx_inner)  # type: ignore
            if inner_index_v.kind != BCPrimitiveType.INTEGER:
                self.error(
                    f"type of inner array index is {index_v.kind}, not an INTEGER!"
                )

            return (index_v.val, inner_index_v.val)  # type: ignore
        else:
            if ind.idx_inner is not None:
                self.error("expected only 1 index for array indexing", ind.pos)
            return (index_v.val, None)  # type: ignore

    def visit_array_index(self, ind: ArrayIndex) -> BCValue:  # type: ignore
        v = self.visit_expr(ind.expr)

        if v.is_array:
            a = v.get_array()
            tup = self._get_array_index(ind)
            if a.typ.is_matrix():
                outer, inner = tup
                bounds = a.get_matrix_bounds()
                if inner is None:
                    self.error(
                        "second index not present for matrix index", ind.expr.pos
                    )

                bounds = a.get_matrix_bounds()
                if outer not in range(bounds[0], bounds[1] + 1):  # type: ignore
                    self.error(
                        f'cannot access out of bounds array element "{tup[0]}"',
                        ind.idx_outer.pos,
                    )

                if inner not in range(bounds[2], bounds[3] + 1):  # type: ignore
                    self.error(
                        f'cannot access out of bounds array element "{tup[1]}"',
                        ind.idx_inner.pos,  # type: ignore
                    )

                idx1 = outer - bounds[0]
                idx2 = inner - bounds[2]
                res = a.get_matrix()[idx1][idx2]
                return res
            else:
                if tup[0] not in range(
                    a.get_flat_bounds()[0], a.get_flat_bounds()[1] + 1
                ):
                    if tup[0] == 0:
                        self.error(
                            "cannot access the 0th array element, which is disallowed in pseudocode",
                            ind.idx_outer.pos,
                        )
                    else:
                        self.error(
                            f"cannot access out of bounds array element {tup[0]}",
                            ind.idx_outer.pos,
                        )

                res = a.get_flat()[tup[0] - a.get_flat_bounds()[0]]
                return res
        elif v.kind == BCPrimitiveType.STRING:
            self.error(
                # FIXME: nicer diagnostics with the AST printer
                f"cannot index a string!\nplease use SUBSTRING(YourString, YourIndex, 1) instead.",
                ind.expr.pos,
            )
        else:
            self.error(f"cannot index {v.kind}", ind.expr.pos)

    def _eval_libroutine_args(
        self,
        args: list[Expr],
        lr: Libroutine,
        name: str,
        pos: Pos | None,
    ) -> list[BCValue]:
        if lr and len(args) < len(lr):
            self.error(
                f"expected {len(lr)} args, but got {len(args)} in call to library routine {name.upper()}",
                pos,
            )

        evargs: list[BCValue] = []
        if lr:
            for idx, (arg, arg_type) in enumerate(zip(args, lr)):
                new = self.visit_expr(arg)

                mismatch = False
                if isinstance(arg_type, tuple):
                    if new.kind not in arg_type:
                        mismatch = True
                elif not arg_type:
                    pass
                elif arg_type != new.kind:
                    mismatch = True

                if mismatch and new.is_null():
                    self.error(
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
                    self.error(err_base, pos)

                evargs.append(new)
        else:
            evargs = [self.visit_expr(e) for e in args]

        return evargs

    def visit_libroutine(self, stmt: FunctionCall) -> BCValue:  # type: ignore
        name = stmt.ident
        lr = LIBROUTINES[name]
        evargs = self._eval_libroutine_args(stmt.args, lr, name, stmt.pos)

        try:
            match name:
                case "initarray":
                    bean_initarray(stmt.pos, evargs)
                    return BCValue.new_null()
                case "format":
                    return bean_format(stmt.pos, evargs)
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
                    return bean_getchar(stmt.pos)
                case "random":
                    return bean_random(stmt.pos)
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
                    [cmd, *_] = evargs

                    if sys.platform in {"wasi", "emscripten"}:
                        raise BCError("EXECUTE is not supported in the browser!", stmt.pos)

                    try:
                        out = subprocess.check_output(cmd.get_string(), shell=True)
                        out = out.decode('utf-8')
                    except Exception as e:
                        raise BCError(f"Error: {e}", stmt.pos)

                    return BCValue.new_string(out)
                case "putchar":
                    [ch, *_] = evargs
                    bean_putchar(stmt.pos, ch.get_char())
                    return BCValue.new_null()
                case "exit":
                    [code, *_] = evargs
                    bean_exit(stmt.pos, code.get_integer())
                case "sleep":
                    [duration, *_] = evargs
                    bean_sleep(stmt.pos, duration.get_real())
                    return BCValue.new_null()
                case "flush":
                    sys.stdout.flush()
                    return BCValue.new_null()
                case "clear":
                    print("\x1b[2J\x1b[H", end='', flush=True)
                    return BCValue.new_null()
        except BCError as e:
            e.pos = stmt.pos
            raise e

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
                f"FFI function {func.name} returns {str(func.returns).upper()} but returned a null/uninitialized value.",
                stmt.pos,
            )

        return retval

    def visit_fncall(self, stmt: FunctionCall, tracer: Tracer | None = None) -> BCValue:
        if stmt.libroutine:
            return self.visit_libroutine(stmt)

        if stmt.ident in self.variables:
            self.error(f'"{stmt.ident}" is a variable, not a function!', stmt.pos)

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

        intp = self.new(func.block, func=True, tracer=tracer)
        intp.files = self.files.copy()
        intp.calls = self.calls.copy()
        intp.calls.append(CallStackEntry(func.name, func.returns, func=True))
        if self.tracer is not None and tracer is None:
            intp.tracer = self.tracer

        if len(func.args) != len(stmt.args):
            self.error(
                f"function {func.name} declares {len(func.args)} variables but only found {len(stmt.args)} in procedure call",
                stmt.pos,
            )

        intp.variables = self.variables.copy()
        for argdef, argval in zip(func.args, stmt.args):
            val = self.visit_expr(argval)
            intp.variables[argdef.name] = Variable(val=val, const=False, export=False)

        intp.functions = self.functions.copy()
        intp.visit_block(func.block)
        intp.calls.pop()
        if intp._returned is False:
            self.error(
                f"function with return type {func.returns} did not return a value!",
                stmt.pos,
            )

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

    def visit_call(self, stmt: CallStatement, tracer: Tracer | None = None):
        if stmt.ident in self.variables:
            self.error(f'"{stmt.ident}" is a variable, not a procedure!', stmt.pos)

        if stmt.libroutine:
            self.error(
                f"{stmt.ident} is a library routine\nplease remove the CALL!",
                stmt.pos,
            )

        try:
            proc = self.functions[stmt.ident]
        except KeyError:
            self.error(f"no procedure named {stmt.ident} exists", stmt.pos)

        if isinstance(proc, FunctionStatement):
            self.error(
                "cannot run CALL on a function!\nPlease call the function without the CALL keyword instead.",
                stmt.pos,
            )

        if isinstance(proc, BCFunction):
            self.error(
                "cannot run CALL on an FFI function!\nPlease call the function without the CALL keyword instaed.",
                stmt.pos,
            )

        if isinstance(proc, BCProcedure):
            return self.visit_ffi_call(proc, stmt)

        intp = self.new(proc.block, proc=True, tracer=tracer)
        intp.calls = self.calls.copy()
        intp.calls.append(CallStackEntry(proc.name, None, proc=True))

        if self.tracer is not None and tracer is None:
            intp.tracer = self.tracer

        if len(proc.args) != len(stmt.args):
            self.error(
                f"procedure {proc.name} declares {len(proc.args)} variables but only found {len(stmt.args)} in procedure call",
                stmt.pos,
            )

        intp.functions = self.functions.copy()
        intp.variables = self.variables.copy()
        intp.files = self.files.copy()
        for argdef, argval in zip(proc.args, stmt.args):
            val = self.visit_expr(argval)
            intp.variables[argdef.name] = Variable(val=val, const=False, export=False)

        intp.visit_block(proc.block)
        intp.calls.pop()

    def _typecast_string(self, inner: BCValue, pos: Pos) -> BCValue:
        _ = pos  # shut up the type checker
        s = ""

        if inner.is_array:
            arr = inner.get_array()
            s = self._display_array(arr)
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

    def _typecast_integer(self, inner: BCValue, pos: Pos) -> BCValue:
        i = 0
        match inner.kind:
            case BCPrimitiveType.STRING:
                s = inner.get_string()
                try:
                    i = int(s.strip())
                except ValueError:
                    self.error(f'impossible to convert "{s}" to an INTEGER!', pos)
            case BCPrimitiveType.INTEGER:
                return inner
            case BCPrimitiveType.REAL:
                i = int(inner.get_real())
            case BCPrimitiveType.CHAR:
                i = ord(inner.get_char()[0])
            case BCPrimitiveType.BOOLEAN:
                i = 1 if inner.get_boolean() else 0

        return BCValue.new_integer(i)

    def _typecast_real(self, inner: BCValue, pos: Pos) -> BCValue:
        r = 0.0

        match inner.kind:
            case BCPrimitiveType.STRING:
                s = inner.get_string()
                try:
                    r = float(s.strip())
                except ValueError:
                    self.error(f'impossible to convert "{s}" to a REAL!', pos)
            case BCPrimitiveType.INTEGER:
                r = float(inner.get_integer())
            case BCPrimitiveType.REAL:
                return inner
            case BCPrimitiveType.CHAR:
                self.error(f"impossible to convert a REAL to a CHAR!", pos)
            case BCPrimitiveType.BOOLEAN:
                r = 1.0 if inner.get_boolean() else 0.0

        return BCValue.new_real(r)

    def _typecast_char(self, inner: BCValue, pos: Pos) -> BCValue:
        c = ""

        match inner.kind:
            case BCPrimitiveType.STRING:
                self.error(
                    f"cannot convert a STRING to a CHAR! use SUBSTRING(str, begin, 1) to get a character.",
                    pos,
                )
            case BCPrimitiveType.INTEGER:
                c = chr(inner.get_integer())
            case BCPrimitiveType.REAL:
                self.error(f"impossible to convert a CHAR to a REAL!", pos)
            case BCPrimitiveType.CHAR:
                return inner
            case BCPrimitiveType.BOOLEAN:
                self.error(f"impossible to convert a BOOLEAN to a CHAR!", pos)

        return BCValue.new_char(c)

    def _typecast_boolean(self, inner: BCValue) -> BCValue:
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

    def visit_typecast(self, tc: Typecast) -> BCValue:  # type: ignore
        inner = self.visit_expr(tc.expr)

        if inner.kind == BCPrimitiveType.NULL:
            self.error("cannot cast NULL to anything!", tc.pos)

        if inner.is_array and tc.typ != BCPrimitiveType.STRING:
            self.error(f"cannot cast an array to a {tc.typ}", tc.pos)

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

    def visit_matrix_literal(self, expr: ArrayLiteral) -> BCValue:
        first_matrix_elem: Expr = expr.items[0].items[0]  # type: ignore
        matrix: list[list[BCValue]] = list()

        # since we checked earlier, we know this is always a primitive
        typ: BCPrimitiveType = self.visit_expr(first_matrix_elem).kind  # type: ignore
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
        arrtyp = BCArrayType.new_matrix(typ, bounds)
        return BCValue.new_array(BCArray.new_matrix(arrtyp, matrix))

    def visit_array_literal(self, expr: ArrayLiteral) -> BCValue:
        if isinstance(expr.items[0], ArrayLiteral):
            return self.visit_matrix_literal(expr)

        # we know there is at least one item always
        vals = [self.visit_expr(expr.items[0])]
        # we know that this is a primitive
        typ: BCPrimitiveType = vals[0].kind  # type: ignore

        for val in expr.items[1:]:
            newval = self.visit_expr(val)
            if newval.kind != typ:
                self.error(
                    "array literal may not contain items of multiple types!", val.pos
                )
            vals.append(newval)

        bounds = (1, len(vals))
        t = BCArrayType.new_flat(typ, bounds)
        return BCValue.new_array(BCArray.new_flat(t, vals))

    def visit_identifier(self, expr: Identifier) -> BCValue:
        if expr.libroutine:
            self.error(
                f'"{expr.ident}" is a library routine!\nplease call it with an argument list: {expr.ident}(args...)',
                expr.pos,
            )

        try:
            var = self.variables[expr.ident]
        except KeyError:
            if expr.ident in self.functions:
                f = self.functions[expr.ident]
                if isinstance(f, BCFunction) or isinstance(f, FunctionStatement):
                    self.error(
                        f'undeclared variable "{expr.ident}" is a function!\nplease call it with an argument list: {expr.ident}(args...)',
                        expr.pos,
                    )
                else:
                    self.error(
                        f'undeclared variable "{expr.ident}" is a procedure!\nplease call it with the CALL keyword: CALL {expr.ident}(args...)',
                        expr.pos,
                    )
            self.error(f'cannot access undeclared variable "{expr.ident}"', expr.pos)

        return var.val

    def visit_expr(self, expr: Expr) -> BCValue:  # type: ignore
        match expr:
            case Typecast():
                return self.visit_typecast(expr)
            case Grouping():
                return self.visit_expr(expr.inner)
            case Negation():
                inner = self.visit_expr(expr.inner)
                if inner.kind not in [BCPrimitiveType.INTEGER, BCPrimitiveType.REAL]:
                    self.error(
                        f"cannot negate a value of type {inner.kind}", expr.inner.pos
                    )

                if inner.kind == BCPrimitiveType.INTEGER:
                    return BCValue.new_integer(-inner.get_integer())  # type: ignore
                elif inner.kind == BCPrimitiveType.REAL:
                    return BCValue.new_real(-inner.get_real())  # type: ignore
            case Not():
                inner = self.visit_expr(expr.inner)
                if inner.kind != BCPrimitiveType.BOOLEAN:
                    self.error(
                        f"cannot perform NOT on {inner.kind}",
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
            case Sqrt():
                # Only the optimizer can generate this node, so we know the type is checked.
                return BCValue.new_real(math.sqrt(self.visit_expr(expr.inner).val))  # type: ignore
        self.error(
            "whoops something is very wrong. this is a rare error, please report it to the developers."
        )

    def _display_array(self, arr: BCArray) -> str:
        if arr.typ.is_flat():
            res = list()
            res.append("[")
            flat = arr.get_flat()
            for idx, item in enumerate(flat):
                if item.is_uninitialized():
                    res.append("(null)")
                else:
                    res.append(str(item))

                if idx != len(flat) - 1:
                    res.append(", ")
            res.append("]")

            return "".join(res)
        else:
            matrix = arr.get_matrix()
            outer_res = list()
            outer_res.append("[")
            res = list()
            for oidx, a in enumerate(matrix):
                res.append("[")
                for iidx, item in enumerate(a):
                    if item.is_uninitialized():
                        res.append("(null)")
                    else:
                        res.append(str(item))

                    if iidx != len(a) - 1:
                        res.append(", ")
                res.append("]")

                outer_res.append("".join(res))
                res.clear()
                if oidx != len(matrix) - 1:
                    outer_res.append(", ")
            outer_res.append("]")

            return "".join(outer_res)

    def visit_output_stmt(self, stmt: OutputStatement):
        res = (
            str(self.visit_expr(stmt.items[0]))
            if len(stmt.items) == 1
            else "".join(
                [
                    (
                        self._display_array(evaled.val)  # type: ignore
                        if evaled.is_array
                        else str(evaled)
                    )
                    for evaled in map(self.visit_expr, stmt.items)
                ]
            )
        )

        if self.tracer_outputs is not None:
            if not self.loop and self.tracer:
                self.tracer.collect_new({}, stmt.pos.row, outputs=[res])
            else:
                self.tracer_outputs.append(res)  # type: ignore

        if self.tracer and self.tracer.config.show_outputs:
            print("(tracer output): " + res, flush=True)
        elif not self.tracer:
            print(res, end=("\n" if stmt.newline else ""), flush=True)

    def _guess_input_type(self, inp: str) -> BCValue:
        if is_real(inp):
            return BCValue.empty(BCPrimitiveType.REAL)
        elif is_integer(inp):
            return BCValue.empty(BCPrimitiveType.INTEGER)

        if inp.strip().lower() in {"true", "false", "no", "yes"}:
            return BCValue.empty(BCPrimitiveType.BOOLEAN)

        if len(inp.strip()) == 1:
            return BCValue.empty(BCPrimitiveType.CHAR)
        else:
            return BCValue.empty(BCPrimitiveType.STRING)

    def visit_input_stmt(self, s: InputStatement):
        prompt = str()
        if self.tracer and self.tracer.config.prompt_on_inputs:
            prompt = "(tracer input): "

        inp = input(prompt)
        target: BCValue

        if self.tracer_inputs is not None:
            self.tracer_inputs.append(inp)  # type: ignore

        if isinstance(s.ident, ArrayIndex):
            target = self.visit_array_index(s.ident)
        else:
            id = s.ident.ident

            data: Variable | None = self.variables.get(id)
            if data is None:
                val = self._guess_input_type(inp)
                data = Variable(val, False, export=False)
                self.variables[id] = data
            target = data.val  # type: ignore

            if data.const:
                self.error(f'cannot call "INPUT" into constant {id}', s.ident.pos)

            if data.val.is_array:
                self.error(f'cannot call "INPUT" on an array', s.ident.pos)

        match target.kind:
            case BCPrimitiveType.STRING:
                target.kind = BCPrimitiveType.STRING
                target.val = inp
            case BCPrimitiveType.CHAR:
                if len(inp) > 1:
                    self.error(
                        f'expected single character but got "{inp}" for CHAR', s.pos
                    )

                target.kind = BCPrimitiveType.CHAR
                target.val = inp[0]
            case BCPrimitiveType.BOOLEAN:
                if inp.lower() not in {"true", "false", "yes", "no"}:
                    self.error(
                        f'expected TRUE, FALSE, YES or NO including lowercase for BOOLEAN but got "{inp}"',
                        s.pos,
                    )

                inp = inp.lower()
                target.kind = BCPrimitiveType.BOOLEAN
                if inp in {"true", "yes"}:
                    target.val = True
                elif inp in {"false", "no"}:
                    target.val = False
            case BCPrimitiveType.INTEGER:
                inp = inp.lower().strip()
                if is_integer(inp):
                    try:
                        res = int(inp)
                        target.kind = BCPrimitiveType.INTEGER
                        target.val = res
                    except ValueError:
                        self.error("expected INTEGER for INPUT", s.ident.pos)
                else:
                    self.error("expected INTEGER for INPUT", s.ident.pos)
            case BCPrimitiveType.REAL:
                inp = inp.lower().strip()
                if is_real(inp) or is_integer(inp):
                    try:
                        res = float(inp)
                        target.kind = BCPrimitiveType.REAL
                        target.val = res
                    except ValueError:
                        self.error("expected REAL for INPUT", s.ident.pos)
                else:
                    self.error("expected REAL for INPUT", s.ident.pos)

        self.trace(s.pos.row)

    def visit_return_stmt(self, stmt: ReturnStatement):
        proc, func = self.can_return()

        if not proc and not func:
            self.error(
                f"did not find function or procedure to return from!",
                stmt.pos,
            )

        if func:
            if stmt.expr is None:
                self.error("you must return something from a function!", stmt.pos)

            res = self.visit_expr(stmt.expr)
            rtype = self.get_return_type()

            if rtype is None:
                self.error(
                    "return type for function not set! this is an internal error, please report it.",
                    stmt.pos,
                )

            actual_type = self.visit_type(rtype)

            if res.kind != actual_type:
                self.error(
                    f"return type {actual_type} does not match return value's type {res.kind}!",
                    stmt.pos,
                )

            self.retval = res
            self._returned = True
        elif proc:
            if stmt.expr is not None:
                self.error("you cannot return a value from a procedure!", stmt.pos)

            self._returned = True

    def visit_include_ffi_stmt(self, stmt: IncludeStatement):
        if sys.platform in {"emscripten", "wasi"}:
            self.error("INCLUDE_FFI is not supported on the browser!", stmt.pos)

        # XXX: this is probably the most scuffed code in existence.
        try:
            mod: Exports = importlib.import_module(
                f"beancode.modules.{stmt.file}"
            ).EXPORTS
        except Exception as e:
            self.error(
                f"could not include module {stmt.file}!\nreason: {e.args[0]}", stmt.pos
            )

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
            self.error(f"file {filename} in include does not exist!", stmt.pos)

        try:
            with open(filename, "r+") as f:
                file_content = f.read()
        except IsADirectoryError:
            self.error(f'"{stmt.file}" is a directory', stmt.pos)

        lexer = Lexer(file_content)
        toks = lexer.tokenize()
        parser = Parser(toks)
        try:
            program = parser.program()
        except BCError as err:
            err.print(filename, file_content)
            print(file=sys.stderr)
            self.error(f'error in included file "{stmt.file}".', stmt.pos)

        intp = self.new(program.stmts)
        try:
            intp.visit_block(None)
        except BCError as err:
            err.print(filename, file_content)
            print(file=sys.stderr)
            self.error(f'error in included file "{stmt.file}".', stmt.pos)

        for name, var in intp.variables.items():
            if var.export:
                self.variables[name] = var

        for name, fn in intp.functions.items():
            if isinstance(fn, BCFunction) or isinstance(fn, BCProcedure):
                continue

            if fn.export:  # type: ignore
                self.functions[name] = fn

    def visit_if_stmt(self, stmt: IfStatement):
        cond: BCValue = self.visit_expr(stmt.cond)

        if cond.kind != BCPrimitiveType.BOOLEAN:
            self.error("condition of while loop must be a boolean!", stmt.cond.pos)

        saved_cur = self.cur_stmt
        if cond.get_boolean():
            self.visit_block(stmt.if_block)
        else:
            self.visit_block(stmt.else_block)
        self.cur_stmt = saved_cur

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        value: BCValue = self.visit_expr(stmt.expr)

        for branch in stmt.branches:
            if not isinstance(branch, CaseofBranch):
                continue

            rhs = self.visit_expr(branch.expr)
            if value == rhs:
                self.visit_stmt(branch.stmt)
                return

        if stmt.otherwise is not None:
            self.visit_stmt(stmt.otherwise)

    def visit_while_stmt(self, stmt: WhileStatement):
        cond: Expr = stmt.cond  # type: ignore

        block: list[Statement] = stmt.block  # type: ignore

        intp = self._make_new_interpreter(block)
        intp.loop = True

        while True:
            evcond = self.visit_expr(cond)
            if evcond.kind != BCPrimitiveType.BOOLEAN:
                self.error("condition of while loop must be a boolean!", stmt.cond.pos)
            if not evcond.get_boolean():
                break

            intp.visit_block(block)

            # trace all I/O that happened
            intp.trace(stmt.end_pos.row, loop_trace=True)

            # FIXME: barbaric aah
            # reset all declares
            intp.variables = self.variables.copy()
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    self.error(
                        f"did not find function or procedure to return from!",
                        stmt.pos,
                    )

                self._returned = True
                self.retval = intp.retval
                return

    def visit_for_stmt(self, stmt: ForStatement):
        begin = self.visit_expr(stmt.begin)

        if begin.kind != BCPrimitiveType.INTEGER:
            self.error("non-integer expression used for for loop begin", stmt.begin.pos)

        end = self.visit_expr(stmt.end)

        if end.kind != BCPrimitiveType.INTEGER:
            self.error("non-integer expression used for for loop end", stmt.end.pos)

        step = 1
        if stmt.step is None:
            if begin.val > end.val:  # type: ignore
                step = -1
        else:
            step_val = self.visit_expr(stmt.step)  # type: ignore
            if step_val.kind != BCPrimitiveType.INTEGER:
                self.error("non-integer expression used for loop step", stmt.step.pos)
            step: int = step.val  # type: ignore
            if step == 0:
                self.error("step for for loop cannot be 0!", stmt.step.pos)

        intp = self._make_new_interpreter(stmt.block)
        intp.loop = True

        var_existed = stmt.counter.ident in intp.variables
        if var_existed:
            var_prev_value = intp.variables[stmt.counter.ident]

        counter = Variable(begin.copy(), const=False)
        intp.variables[stmt.counter.ident] = counter

        if step > 0:
            cond = (
                lambda *_: counter.val.val  # type: ignore
                <= self.visit_expr(stmt.end).val
            )
        else:
            cond = (
                lambda *_: counter.val.val  # type: ignore
                >= self.visit_expr(stmt.end).val
            )

        while cond():
            intp.visit_block(None)
            intp.trace(stmt.end_pos.row, loop_trace=True)

            #  FIXME: barbaric
            # clear declared variables
            c = intp.variables[stmt.counter.ident]
            intp.variables = self.variables.copy()
            intp.variables[stmt.counter.ident] = c

            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    self.error(
                        f"did not find function or procedure to return from!",
                        stmt.pos,
                    )

                self._returned = True
                self.retval = intp.retval
                return

            counter.val.val += step  # type: ignore

        if not var_existed:
            intp.variables.pop(stmt.counter.ident)
        else:
            intp.variables[stmt.counter.ident] = var_prev_value  # type: ignore

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        cond: Expr = stmt.cond  # type: ignore

        intp = self._make_new_interpreter(stmt.block)
        intp.loop = True

        while True:
            intp.visit_block(None)

            intp.trace(stmt.end_pos.row, loop_trace=True)

            intp.variables = dict(self.variables)
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    self.error(
                        f"did not find function or procedure to return from!",
                        stmt.pos,
                    )

                self._returned = True
                self.retval = intp.retval
                return

            evcond = self.visit_expr(cond)
            if evcond.kind != BCPrimitiveType.BOOLEAN:
                self.error(
                    "condition of repeat-until loop must be a boolean!", stmt.cond.pos
                )
            if evcond.get_boolean():
                break

    def visit_scope_stmt(self, stmt: ScopeStatement):
        intp = self._make_new_interpreter(stmt.block)
        intp.visit_block(None)

        for name, var in intp.variables.items():
            if var.export:
                self.variables[name] = var

        for name, fn in intp.functions.items():
            if fn.export:  # type: ignore
                self.functions[name] = fn

    def visit_procedure(self, stmt: ProcedureStatement):
        if stmt.name in LIBROUTINES:
            self.error(
                f"cannot redefine library routine {stmt.name.upper()}!", stmt.pos
            )

        if stmt.name in self.variables:
            self.error(
                f'cannot redefine variable "{stmt.name}" as a procedure', stmt.pos
            )

        self.functions[stmt.name] = stmt

    def visit_function(self, stmt: FunctionStatement):
        if stmt.name in LIBROUTINES:
            self.error(
                f"cannot redefine library routine {stmt.name.upper()}!", stmt.pos
            )

        if stmt.name in self.variables:
            self.error(
                f'cannot redefine variable "{stmt.name}" as a function', stmt.pos
            )

        self.functions[stmt.name] = stmt

    def visit_assign_stmt(self, s: AssignStatement):
        val = self.visit_expr(s.value)

        if s.is_ident:  # isinstance(s.ident, Identifier)
            key: str = s.ident.ident  # type: ignore

            if s.ident.libroutine:  # type: ignore
                self.error(f'cannot shadow library routine named "{key}"', s.pos)

            target = self.variables.get(key)

            if target is None:
                if key in self.functions:
                    self.error(
                        f'cannot shadow existing function or procedure named "{key}"',
                        s.pos,
                    )

                target = Variable(val.copy(), False, export=False)
                self.variables[key] = target

            if self.variables[key].const:
                self.error(f"cannot assign constant {key}", s.ident.pos)

            target = target.val

            if val.is_array:
                a: BCArray = val.val  # type: ignore
                t: BCArray = target.val  # type: ignore
                if a.typ.is_matrix() and a.typ.bounds != t.typ.bounds:  # type: ignore
                    self.error(f"mismatched matrix sizes in matrix assignment", s.pos)
                elif a.typ.is_flat() and a.typ.bounds != t.typ.bounds:  # type: ignore
                    self.error(f"mismatched array sizes in array assignment", s.pos)
        else:  # elif isinstance(s.ident, ArrayIndex)
            target = self.visit_expr(s.ident)

        should_promote_real = (
            target.kind == BCPrimitiveType.REAL and val.kind == BCPrimitiveType.INTEGER
        )
        if target.kind != val.kind:
            if not should_promote_real:
                self.error(
                    f"cannot assign {val.kind} to {target.kind}",
                    s.ident.pos,
                )
            else:
                val = BCValue(
                    BCPrimitiveType.REAL, value=float(val.val), is_array=False  # type: ignore
                )

        target.replace_inner(val.copy())

        self.trace(s.pos.row)

    def visit_constant_stmt(self, s: ConstantStatement):
        key = s.ident.ident

        is_prev_func = len(self.calls) > 0 and (
            self.calls[-1].proc or self.calls[-1].func
        )
        if key in self.variables and not is_prev_func:
            existing_var = self.variables[key]
            if not existing_var.const:
                self.error(
                    f'cannot shadow variable declaration for constant "{key}"', s.pos
                )
            else:
                self.error(f"variable or constant {key} declared!", s.pos)

        if s.ident.libroutine:
            self.error(f'cannot shadow library routine named "{key}"!', s.pos)

        if key in self.functions:
            self.error(
                f'cannot shadow existing function or procedure named "{key}"', s.pos
            )

        val = self.visit_expr(s.value)
        self.variables[key] = Variable(val, True, export=s.export)
        self.trace(s.pos.row)

    def _declare_array(self, d: DeclareStatement, key: str):
        at: ArrayType = d.typ  # type: ignore
        inner_type = at.inner
        t = self.visit_array_type(at)
        if t.is_matrix():
            bounds = t.get_matrix_bounds()
            ob, oe, ib, ie = bounds

            if ob < 0:
                self.error(
                    "outer beginning value for array bound declaration cannot be <0",
                    at.get_matrix_bounds()[0].pos,
                )  # type: ignore

            if oe < 0:
                self.error(
                    "outer ending value for array bound declaration cannot be <0",
                    at.get_matrix_bounds()[1].pos,
                )  # type: ignore

            if ib < 0:
                self.error(
                    "inner beginning value for array bound declaration cannot be <0",
                    at.get_matrix_bounds()[2].pos,
                )  # type: ignore

            if ie < 0:
                self.error(
                    "inner ending value for array bound declaration cannot be <0",
                    at.get_matrix_bounds()[3].pos,
                )  # type: ignore

            if ob > oe:
                self.error(
                    "invalid outer range for 2D array bound declaration",
                    at.get_matrix_bounds()[0].pos,
                )

            if ib > ie:
                self.error(
                    "invalid inner range for 2D array bound declaration",
                    at.get_matrix_bounds()[2].pos,
                )

            # Directly setting the result of the comprehension results in multiple pointers pointing to the same list
            in_size = ie - ib
            out_size = oe - ob
            # array bound declarations are inclusive
            outer_arr = [
                [BCValue(inner_type) for _ in range(in_size + 1)]
                for _ in range(out_size + 1)
            ]

            atype = BCArrayType.new_matrix(inner_type, bounds)
            res = BCArray.new_matrix(atype, outer_arr)
        else:
            bounds = t.get_flat_bounds()
            begin, end = bounds

            if begin < 0:
                self.error(
                    "beginning value for array bound declaration cannot be <0",
                    at.get_flat_bounds()[0].pos,
                )  # type: ignore

            if end < 0:
                self.error(
                    "ending value for array bound declaration cannot be <0",
                    at.get_flat_bounds()[1].pos,
                )  # type: ignore

            if begin > end:
                self.error(
                    "invalid range for array bound declaration",
                    at.get_flat_bounds()[0].pos,
                )  # type: ignore

            size = end - begin
            arr = [BCValue(t.inner) for _ in range(size + 1)]

            atype = BCArrayType.new_flat(inner_type, bounds)
            res = BCArray.new_flat(atype, arr)

        self.variables[key] = Variable(BCValue.new_array(res), False, export=d.export)

    def visit_declare_stmt(self, s: DeclareStatement):
        for ident in s.ident:
            key: str = ident.ident
            is_prev_func = len(self.calls) > 0 and (
                self.calls[-1].proc or self.calls[-1].func
            )
            if key in self.variables and not is_prev_func:
                existing_var = self.variables[key]
                actual_type = self.visit_type(s.typ)
                if existing_var.val.kind != actual_type:
                    self.error(
                        f'variable "{key}" declared with a different type!', s.pos
                    )
                elif existing_var.const:
                    self.error(
                        f'cannot shadow variable declaration for constant "{key}"',
                        s.pos,
                    )
                else:
                    self.error(f"variable or constant {key} declared!", s.pos)

            if ident.libroutine:
                self.error(
                    f'cannot shadow existing library routine "{key}" with a variable of the same name',
                    s.pos,
                )

            if key in self.functions:
                self.error(
                    f'cannot shadow existing function or procedure named "{key}" with a variable of the same name',
                    s.pos,
                )

            if isinstance(s.typ, ArrayType):
                self._declare_array(s, key)
            else:
                self.variables[key] = Variable(
                    BCValue(kind=s.typ), False, export=s.export
                )
        self.trace(s.pos.row)

    def visit_trace_stmt(self, stmt: TraceStatement):
        vars = stmt.vars
        tracer = Tracer(vars)

        tracer.config.write_to_default_location()
        tracer.load_config()

        intp = self.new(stmt.block, loop=False, tracer=tracer)
        intp.variables = dict(self.variables)
        intp.functions = dict(self.functions)
        intp.visit_block(None)

        written_path = tracer.write_out(stmt.file_name)
        if self.tracer_open:
            tracer.open(written_path)

    def _get_file_name(self, id: Expr | str, pos: Pos):
        name = str()
        if isinstance(id, Expr):
            if isinstance(id, Identifier):
                name = id.ident
            else:
                exp = self.visit_expr(id)
                if exp.kind != BCPrimitiveType.STRING:
                    self.error("file name must be a string!", pos)
                name = exp.get_string()
        else:
            name = id

        if len(name) == 0:
            self.error("empty string given as file name!", pos)

        return name

    def visit_openfile_stmt(self, stmt: OpenfileStatement):
        name = self._get_file_name(stmt.file_ident, stmt.pos)

        mode = _get_file_mode(*stmt.mode)
        if not mode:
            self.error(
                "Bogus Amogus file mode!\n"
                + "This error was not anticipated. Please report this to the developers.",
                stmt.pos,
            )

        try:
            self.files[name] = File(stream=open(name, mode), mode=stmt.mode)
        except PermissionError as e:
            self.error(
                f'not enough permissions to open "{name}"\n'
                + f"Do you have access to this file? [Error Code: {e.errno}]",
                stmt.pos,
            )
        except FileNotFoundError as e:
            self.error(
                f'file "{name}" was not found\n' + f"Error Code: {e.errno}", stmt.pos
            )
        except IsADirectoryError as e:
            self.error(
                f'"{name}" is a folder/directory\n' + f"Error Code: {e.errno}", stmt.pos
            )
        except Exception as e:
            self.error(
                f'could not open file "{name}"\n' + f"Python Exception: {str(e)}",
                stmt.pos,
            )

        self.files[name] = File(open(name, mode), stmt.mode)

    def _get_file_obj(self, fileid: Any, pos: Pos) -> tuple[str, File]:
        name = self._get_file_name(fileid, pos)
        file = self.files.get(name)
        if not file:
            self.error(f"file does not exist!\n" + "did you forget to open it?", pos)
        return (name, file)

    def visit_readfile_stmt(self, stmt: ReadfileStatement):
        _, file = self._get_file_obj(stmt.file_ident, stmt.pos)

        if not file.mode[0]:
            self.error("file not open for reading!", stmt.pos)

        target: BCValue
        if isinstance(stmt.target, ArrayIndex):
            target = self.visit_array_index(stmt.target)
        else:
            target = self.visit_identifier(stmt.target)

        target.val = str(file.stream.read())
        file.stream.seek(0)

    def visit_writefile_stmt(self, stmt: WritefileStatement):
        _, file = self._get_file_obj(stmt.file_ident, stmt.pos)

        if not file.mode[1]:
            self.error("file not open for writing!", stmt.pos)

        contents = str(self.visit_expr(stmt.src))
        file.stream.write(contents)

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        name, file = self._get_file_obj(stmt.file_ident, stmt.pos)

        file.stream.close()
        self.files.pop(name)

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
            case ClosefileStatement():
                self.visit_closefile_stmt(stmt)
            case ExprStatement():
                self.visit_expr(stmt.inner)
            # ignore newline statement if ever given

    def visit_block(self, block: list[Statement] | None):
        blk = block if block is not None else self.block
        cur = 0
        while cur < len(blk):
            stmt = blk[cur]
            self.cur_stmt = cur
            self.visit_stmt(stmt)
            if self._returned:
                return
            cur += 1

    def visit_program(self, program: Program):
        if program is not None:
            self.visit_block(program.stmts)
