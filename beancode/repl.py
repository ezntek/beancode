import sys
import os

from typing import Any

from .error import *
from .lexer import *
from .parser import *
from .interpreter import *
from .bean_ast import *
from .bean_ffi import *
from .runner import run_file, trace
from . import __version__

from enum import IntEnum

BANNER = f"""\033[1m=== welcome to beancode \033[0m{__version__}\033[1m ===\033[0m
\033[2mUsing Python {sys.version}\033[0m
type ".help" for a list of REPL commands, ".exit" to exit, or start typing some code."""

PROMPT = "\001\033[1m\002>> \001\033[0m\002"

HELP = """\033[1mAVAILABLE COMMANDS:\033[0m
 .var [names]          get info regarding a declared variable/constant
 .vars                 get info regarding all declared variables/constants
 .func [names]         get info regarding a declared procedure/function
 .funcs                get info regarding all declared procedures/functions
 .delete [names]       delete a variable/constant/procedure/function
 .runfile (name)       run a beancode file. not specifying a name will open a
                       file picker dialog.
 .trace (name) [vars]  trace a beancode file. you must specify a path and all
                       variables to record. the configuration file will be
                       loaded from the default paths.
 .reset      reset the interpreter
 .help       show this help message
 .clear      clear the screen
 .version    print the version
 .license    print a license notice
 .exit       exit the interpreter (.quit also works)
"""

LICENSE = """This software is copyright (c) Eason Qin, <eason@ezntek.com>.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


def setup_readline():
    try:
        import readline
        import atexit

        histfile = os.path.join(os.path.expanduser("~"), ".beancode_history")
        try:
            readline.read_history_file(histfile)
            # default history len is -1 (infinite), which may grow unruly
            readline.set_history_length(10000)
        except FileNotFoundError:
            open(histfile, "wb").close()
        atexit.register(readline.write_history_file, histfile)
    except ImportError:
        if sys.platform not in {"emscripten", "wasi"}:
            warn("could not import readline, continuing without shell history")


class DotCommandResult(IntEnum):
    NO_OP = 1
    BREAK = 2
    UNKNOWN_COMMAND = 3
    RESET = 4


class ContinuationResult(IntEnum):
    BREAK = 1
    ERROR = 2
    SUCCESS = 3


class Repl:
    lx: Lexer
    p: Parser
    i: Interpreter
    buf: list[str]
    func_src: dict[str, str]
    func_src: dict[str, str]
    debug: bool
    no_run: bool

    def __init__(self, debug=False, no_run=False):
        self.lx = Lexer(str())
        self.p = Parser(list())
        self.i = Interpreter(list())
        self.buf = list()
        self.func_src = dict()
        self.func_src = dict()
        self.debug = debug
        self.no_run = no_run

    def print_var(self, var: Variable):
        val = var.val
        rep: str
        typ: str
        if isinstance(val.kind, BCArrayType):
            a = val.get_array()
            rep = self.i._display_array(a)
            typ = str(a.typ)
        else:
            rep = repr(val)
            typ = str(val.kind)

        if isinstance(val.kind, BCArrayType):
            print(f"'{typ}' {rep}")
        else:
            print(f"'{typ.upper()}' ({rep})")

    def _args_list_to_string(self, args: list[tuple[str, Any]]) -> str:
        # Any: either an BCType or Type
        buf = list()
        buf.append("(")
        for i, (name, typ) in enumerate(args):
            buf.append(f"{name}: ")
            buf.append(str(typ).upper())

            if i != len(args) - 1:
                buf.append(", ")
        buf.append(")")
        return "".join(buf)

    def print_proc(self, proc: ProcedureStatement | BCProcedure):
        buf = list()
        buf.append("PROCEDURE ")
        ffi = False

        if isinstance(proc, ProcedureStatement):
            buf.append(proc.name)
            if len(proc.args) != 0:
                args = [(arg.name, arg.typ) for arg in proc.args]
                buf.append(self._args_list_to_string(args))
        else:
            ffi = True
            buf.append(proc.name)
            if len(proc.params) != 0:
                args = list(proc.params.items())
                buf.append(self._args_list_to_string(args))

        if ffi:
            buf.append("\033[2m <FFI>\033[0m")

        print("".join(buf))

    def print_func(self, func: FunctionStatement | BCFunction):
        sio = list()
        sio.append("FUNCTION ")
        ffi = False

        if isinstance(func, FunctionStatement):
            sio.append(func.name)
            if len(func.args) != 0:
                args = [(arg.name, arg.typ) for arg in func.args]
                sio.append(self._args_list_to_string(args))
        else:
            ffi = True
            sio.append(func.name)
            if len(func.params) != 0:
                args = list(func.params.items())
                sio.append(self._args_list_to_string(args))

        sio.append(" RETURNS ")
        sio.append(str(func.returns).upper())

        if ffi:
            sio.append("\033[2m <FFI>\033[0m")

        print("".join(sio))

    def _var(self, args: list[str]) -> DotCommandResult:
        if len(args) < 2:
            error("not enough args for var")
            return DotCommandResult.NO_OP

        for arg in args[1:]:
            var = self.i.variables.get(arg)
            if var is None:
                error(f'variable "{arg}" does not exist!')
                continue

            print(f"{arg}: ", end="")
            self.print_var(var)
        return DotCommandResult.NO_OP

    def _vars(self, args: list[str]) -> DotCommandResult:
        _ = args

        if len(self.i.variables) == 0:  # null, NULL
            info("no variables")

        for name, var in self.i.variables.items():
            print(f"{name}: ", end="")
            self.print_var(var)

        return DotCommandResult.NO_OP

    def _func(self, args: list[str]) -> DotCommandResult:
        if len(args) < 2:
            error("not enough args for func")
            return DotCommandResult.NO_OP

        for func_name in args[1:]:
            func = self.i.functions.get(func_name)
            if func is None:
                error(f"no procedure or function named {func} found")
                continue

            if isinstance(func, ProcedureStatement) or isinstance(func, BCProcedure):
                self.print_proc(func)
            else:
                self.print_func(func)

        return DotCommandResult.NO_OP

    def _funcs(self, args: list[str]) -> DotCommandResult:
        _ = args

        if len(self.i.functions) == 0:
            info("no functions or procedures")

        for func in self.i.functions.values():
            if isinstance(func, ProcedureStatement) or isinstance(func, BCProcedure):
                self.print_proc(func)
            else:
                self.print_func(func)

        return DotCommandResult.NO_OP

    def _delete(self, args: list[str]) -> DotCommandResult:
        if len(args) == 1:
            error("not enough args for delete")
            return DotCommandResult.NO_OP

        for arg in args[1:]:
            if arg in self.i.variables:
                self.i.variables.__delitem__(arg)
                info(f'deleted variable "{arg}"')
            elif arg in self.i.functions:
                self.i.functions.__delitem__(arg)
                info(f'deleted function/procedure "{arg}"')
            else:
                error(f'no name "{arg}" found')
        return DotCommandResult.NO_OP

    def _runfile(self, args: list[str]):
        if len(args) > 2:
            error("you may only specify one or no arguments to .runfile!")
            return DotCommandResult.NO_OP

        if len(args) == 1:
            run_file()
        else:
            run_file(args[1])

        return DotCommandResult.NO_OP

    def _trace(self, args: list[str]):
        if len(args) < 2:
            error("you must at least specify the path of the script to trace!")
            return DotCommandResult.NO_OP

        path = args[1]
        vars = args[2:]

        trace(path, vars=vars)
        return DotCommandResult.NO_OP

    def handle_dot_command(self, s: str) -> DotCommandResult:
        args = s.strip().split(" ")
        base = args[0]

        match base:
            case "exit" | "quit":
                print("\033[1mbye\033[0m")
                return DotCommandResult.BREAK
            case "clear":
                sys.stdout.write("\033[2J\033[H")
                return DotCommandResult.NO_OP
            case "reset":
                info("reset interpreter")
                return DotCommandResult.RESET
            case "help":
                print(HELP)
                return DotCommandResult.NO_OP
            case "version":
                print(f"beancode version \033[1m{__version__}\033[0m")
                return DotCommandResult.NO_OP
            case "license":
                print(LICENSE)
                return DotCommandResult.NO_OP
            case "runfile":
                return self._runfile(args)
            case "trace":
                return self._trace(args)
            case "var":
                return self._var(args)
            case "vars":
                return self._vars(args)
            case "func":
                return self._func(args)
            case "funcs":
                return self._funcs(args)
            case "delete":
                return self._delete(args)

        return DotCommandResult.UNKNOWN_COMMAND

    def get_continuation(self) -> tuple[Program | None, ContinuationResult]:
        while True:
            oldrow = self.lx.row
            self.lx.reset()
            self.lx.row = oldrow + 1

            try:
                inp = input("\001\033[0m\033[1m\002..\001\033[0m\002 ")
            except KeyboardInterrupt:
                print()
                return (None, ContinuationResult.ERROR)

            self.buf.append(inp + "\n")

            if len(inp) == 0:
                continue

            if inp[0] == ".":
                match self.handle_dot_command(inp[1:]):
                    case DotCommandResult.NO_OP:
                        continue
                    case DotCommandResult.BREAK:
                        return (None, ContinuationResult.BREAK)
                    case DotCommandResult.UNKNOWN_COMMAND:
                        error("invalid dot command")
                        print(HELP, file=sys.stderr)
                        continue
                    case DotCommandResult.RESET:
                        self.i.reset_all()
                        continue

            self.lx.src = inp

            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", "".join(self.buf))
                print()
                return (None, ContinuationResult.ERROR)

            self.p.reset()
            self.p.tokens += toks

            try:
                prog = self.p.program()
            except BCError as err:
                if err.eof:
                    continue
                else:
                    err.print("(repl)", "".join(self.buf))
                    print()
                    return (None, ContinuationResult.ERROR)

            return (prog, ContinuationResult.SUCCESS)

    def handle_error(self, err: BCError):
        src: str
        repl_txt = "(repl)"
        if err.proc is not None:
            res = self.func_src.get(err.proc)  # type: ignore
            if res is None:
                warn(
                    f'could not find source code for procedure "{err.proc}":\n    ({err.msg.strip()})'
                )
                return
            src = res  # type: ignore
            repl_txt = f'(repl "{err.proc}")'
        elif err.func is not None:
            res = self.func_src.get(err.func)  # type: ignore
            if res is None:
                warn(
                    f'could not find source code for function "{err.func}":\n    ({err.msg.strip()})'
                )
                return
            src = res  # type: ignore
            repl_txt = f"(repl {err.func})"
        else:
            src = "".join(self.buf)
        err.print(repl_txt, src)
        print()

    def repl(self):
        setup_readline()
        print(BANNER, end="\n")

        inp = ""
        while True:
            self.lx.reset()
            self.p.reset()
            self.i.reset()

            self.buf.clear()

            try:
                inp = input(PROMPT)
            except KeyboardInterrupt:
                print()
                warn('type ".exit" or ".quit" to exit the REPL.')
                continue
            self.buf.append(inp + "\n")

            if len(inp) == 0:
                continue

            if inp[0] == ".":
                match self.handle_dot_command(inp[1:]):
                    case DotCommandResult.NO_OP:
                        continue
                    case DotCommandResult.BREAK:
                        break
                    case DotCommandResult.UNKNOWN_COMMAND:
                        error("invalid dot command")
                        print(HELP, file=sys.stderr)
                        continue
                    case DotCommandResult.RESET:
                        self.i.reset_all()
                        continue

            self.lx.src = inp
            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", "".join(self.buf))
                print()
                continue

            program: Program
            self.p.tokens = toks

            if self.debug:
                print("\033[2m=== TOKENS ===", file=sys.stderr)
                for tok in toks:
                    tok.print(file=sys.stderr)
                print("==============\033[0m\n", file=sys.stderr)

            try:
                program = self.p.program()
            except BCError as err:
                if err.eof:
                    cont = self.get_continuation()
                    match cont[1]:
                        case ContinuationResult.SUCCESS:
                            program = cont[0]  # type: ignore
                        case ContinuationResult.BREAK:
                            break
                        case ContinuationResult.ERROR:
                            continue
                else:
                    self.handle_error(err)
                    continue

            if len(program.stmts) < 1:
                continue

            if isinstance(program.stmts[0], ProcedureStatement):
                proc = program.stmts[0]
                self.func_src[proc.name] = "".join(self.buf)
            elif isinstance(program.stmts[0], FunctionStatement):
                func = program.stmts[0]
                self.func_src[func.name] = "".join(self.buf)

            if isinstance(program.stmts[-1], ExprStatement):
                exp = program.stmts[-1].inner
                program.stmts[-1] = OutputStatement(pos=Pos(1, 1, 0), items=[exp])

            if self.debug:
                print("\033[2m=== AST ===", file=sys.stderr)
                for stmt in program.stmts:
                    print(stmt, file=sys.stderr)
                print("===========\033[0m", file=sys.stderr)

            if self.no_run:
                continue

            self.i.block = program.stmts
            self.i.toplevel = True
            try:
                self.i.visit_block(None)
            except BCError as err:
                self.handle_error(err)
                continue
            except KeyboardInterrupt:
                warn("caught keyboard interrupt during REPL code execution")
                continue

        return

    def repl_and_exit(self):
        self.repl()
        exit(0)
