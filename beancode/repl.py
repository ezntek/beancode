from . import bean_ast as ast
from . import lexer
from . import parser
from . import interpreter as intp
from . import bean_ffi as ffi
from . import __version__
from .error import *

from io import StringIO

import sys

def _info(msg: str):
    print(
        f"\033[34;1minfo:\033[0m {msg}",
        file=sys.stderr,
    )

def _warn(msg: str):
    print(
        f"\033[33;1mwarn:\033[0m {msg}",
        file=sys.stderr,
    )

def _error(msg: str):
    print(
        f"\033[31;1merror:\033[0m {msg}",
        file=sys.stderr,
    )

try:
    import readline
except ImportError:
    _warn("could not import readline, continuing without shell history")

from enum import Enum

BANNER = f"""\033[1m=== welcome to beancode \033[0m{__version__}\033[1m ===\033[0m
\033[2mUsing Python {sys.version}\033[0m
type ".help" for a list of REPL commands, ".exit" to exit, or start typing some code.
"""

HELP = """\033[1mAVAILABLE COMMANDS:\033[0m
 .var [names]   get information regarding a variable/constant
 .vars          get information regarding all variables/constants
 .func [names]  get information regarding a procedure/function
 .funcs         get information regarding all procedures/functions
 .help      show this help message
 .clear     clear the screen
 .reset     reset the interpreter
 .version   print the version
 .exit      exit the interpreter (.quit also works)
"""


class DotCommandResult(Enum):
    NO_OP = (0,)
    BREAK = (1,)
    UNKNOWN_COMMAND = (2,)
    RESET = (3,)


class ContinuationResult(Enum):
    BREAK = (0,)
    ERROR = (1,)
    SUCCESS = (2,)

class Repl:
    lx: lexer.Lexer
    p: parser.Parser
    i: intp.Interpreter
    buf: StringIO
    proc_src: dict[str, str]
    func_src: dict[str, str]

    def __init__(self):
        self.lx = lexer.Lexer(str())
        self.p = parser.Parser(list())
        self.i = intp.Interpreter(list())
        self.buf = StringIO()
        self.proc_src = dict()
        self.func_src = dict()

    def print_var(self, var: intp.Variable):
        val = var.val
        rep: str
        typ: str
        if isinstance(val.kind, ast.BCArrayType):
            a = val.get_array()
            rep = self.i._display_array(a)
            typ = a.get_type_str()
        else:
            rep = repr(val)
            typ = val.kind

        if isinstance(val.kind, ast.BCArrayType):
            print(f"'{typ}' {rep}")
        else:
            print(f"'{typ.upper()}' ({rep})")

    def _get_args_list(self, args: list[tuple[str, ast.BCType]]) -> str:
        sio = StringIO()
        sio.write("(")
        for i, (name, typ) in enumerate(args):
            sio.write(f"{name}: ")
            if isinstance(typ, ast.BCArrayType):
                sio.write(f"ARRAY OF {typ.inner.upper()}")
            else:
                sio.write(typ.upper())

            if i != len(args) - 1:
                sio.write(", ")
        sio.write(")")
        return sio.getvalue()


    def print_proc(self, proc: ast.ProcedureStatement | ffi.BCProcedure):
        sio = StringIO()
        sio.write("PROCEDURE ")
        ffi = False

        if isinstance(proc, ast.ProcedureStatement):
            sio.write(proc.name)
            if len(proc.args) != 0:
                args: list[tuple[str, ast.BCType]] = [(arg.name, arg.typ) for arg in proc.args]
                sio.write(self._get_args_list(args))
        else:
            ffi = True
            sio.write(proc.name)
            if len(proc.params) != 0:
                args = list(proc.params.items())
                sio.write(self._get_args_list(args))
        
        if ffi:
            sio.write("\033[2m <FFI>\033[0m")

        print(sio.getvalue())


    def print_func(self, func: ast.FunctionStatement | ffi.BCFunction):
        sio = StringIO()
        sio.write("FUNCTION ")
        ffi = False

        if isinstance(func, ast.FunctionStatement):
            sio.write(func.name)
            if len(func.args) != 0:
                args: list[tuple[str, ast.BCType]] = [(arg.name, arg.typ) for arg in func.args]
                sio.write(self._get_args_list(args))
        else:
            ffi = True
            sio.write(func.name)
            if len(func.params) != 0:
                args = list(func.params.items())
                sio.write(self._get_args_list(args))
        
        sio.write(" RETURNS ")
        if isinstance(func.returns, ast.BCArrayType):
            sio.write(f"ARRAY OF {func.returns.inner.upper()}")
        else:
            sio.write(func.returns.upper())
        
        if ffi:
            sio.write("\033[2m <FFI>\033[0m")

        print(sio.getvalue())

    def _var(self, args: list[str]) -> DotCommandResult:
        if len(args) < 2:
            _error("not enough args for var")
            return DotCommandResult.NO_OP

        for arg in args[1:]:
            var = self.i.variables.get(arg)
            if var is None:
                _error(f"variable \"{arg}\" does not exist!")
                continue

            print(f"{arg}: ", end='')
            self.print_var(var)
        return DotCommandResult.NO_OP

    def _vars(self, args: list[str]) -> DotCommandResult:
        _ = args

        if len(self.i.variables) <= 2: # null, NULL
            _info("no variables")

        for name, var in self.i.variables.items():
            if name.lower() == "null":
                continue

            print(f"{name}: ", end='')
            self.print_var(var)

        return DotCommandResult.NO_OP

    def _func(self, args: list[str]) -> DotCommandResult:
        if len(args) < 2:
            _error("not enough args for func")
            return DotCommandResult.NO_OP

        for func_name in args[1:]:
            func = self.i.functions.get(func_name)
            if func is None:
                _error(f"no procedure or function named {func} found")
                continue
            
            if isinstance(func, ast.ProcedureStatement) or isinstance(func, ffi.BCProcedure): 
                self.print_proc(func)
            else:
                self.print_func(func)

        return DotCommandResult.NO_OP

    def _funcs(self, args: list[str]) -> DotCommandResult:
        _ = args
        
        if len(self.i.functions) == 0: # null, NULL
            _info("no functions or procedures")

        for func in self.i.functions.values():
            if isinstance(func, ast.ProcedureStatement) or isinstance(func, ffi.BCProcedure): 
                self.print_proc(func)
            else:
                self.print_func(func)
            
        return DotCommandResult.NO_OP

    def handle_dot_command(self, s: str) -> DotCommandResult:
        args = s.split(" ")
        base = args[0]

        match base:
            case "exit" | "quit":
                print("\033[1mbye\033[0m")
                return DotCommandResult.BREAK
            case "clear":
                sys.stdout.write("\033[2J\033[H")
                return DotCommandResult.NO_OP
            case "reset":
                _info("reset interpreter")
                return DotCommandResult.RESET
            case "help":
                print(HELP)
                return DotCommandResult.NO_OP
            case "version":
                print(f"beancode version \033[1m{__version__}\033[0m")
                return DotCommandResult.NO_OP
            case "var":
                return self._var(args)
            case "vars":
                return self._vars(args)
            case "func":
                return self._func(args)
            case "funcs":
                return self._funcs(args)
        
        return DotCommandResult.UNKNOWN_COMMAND


    def get_continuation(self) -> tuple[ast.Program | None, ContinuationResult]:
        while True:
            oldrow = self.lx.row
            self.lx.reset()
            self.lx.row = oldrow + 1

            inp = input("\033[0m.. ")
            self.buf.write(inp + "\n")

            if len(inp) == 0:
                continue

            if inp[0] == ".":
                match self.handle_dot_command(inp[1:]):
                    case DotCommandResult.NO_OP:
                        continue
                    case DotCommandResult.BREAK:
                        return (None, ContinuationResult.BREAK)
                    case DotCommandResult.UNKNOWN_COMMAND:
                        _error("invalid dot command")
                        print(HELP, file=sys.stderr)
                        continue
                    case DotCommandResult.RESET:
                        self.i.reset_all()
                        continue

            self.lx.file = inp

            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", self.buf.getvalue())
                print()
                continue

            self.p.reset()
            self.p.tokens += toks

            try:
                prog = self.p.program()
            except BCError as err:
                if err.eof:
                    continue
                else:
                    err.print("(repl)", self.buf.getvalue())
                    print()
                    return (None, ContinuationResult.ERROR)
            except BCWarning as w:
                w.print("(repl)", inp)
                print()
                return (None, ContinuationResult.ERROR)

            return (prog, ContinuationResult.SUCCESS)

    def repl(self) -> int:
        print(BANNER, end=str())

        inp = str()
        while True:
            self.lx.reset()
            self.p.reset()
            self.i.reset()

            self.buf.truncate(0)
            self.buf.seek(0)
            
            try:
                inp = input("\033[0;1m>> \033[0m")
            except KeyboardInterrupt:
                print()
                _warn("type \".exit\" or \".quit\" to exit the REPL.")
                continue
            self.buf.write(inp + "\n")

            if len(inp) == 0:
                continue

            if inp[0] == ".":
                match self.handle_dot_command(inp[1:]):
                    case DotCommandResult.NO_OP:
                        continue
                    case DotCommandResult.BREAK:
                        break
                    case DotCommandResult.UNKNOWN_COMMAND:
                        _error("invalid dot command")
                        print(HELP, file=sys.stderr)
                        continue
                    case DotCommandResult.RESET:
                        self.i.reset_all()
                        continue

            self.lx.file = inp
            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", self.buf.getvalue())
                print()
                continue

            program: ast.Program
            self.p.tokens = toks
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
                    src: str
                    if err.proc is not None:
                        res = self.proc_src.get(err.proc) # type: ignore
                        if res is None:
                            _warn(f"could not find source code for procedure \"{err.proc}\"")
                            continue
                        src = res # type: ignore
                    elif err.func is not None:
                        res = self.func_src.get(err.func) # type: ignore
                        if res is None:
                            _warn(f"could not find source code for function \"{err.func}\"")
                            continue
                        src = res # type: ignore
                    else:
                        src = self.buf.getvalue()

                    err.print("(repl)", src)
                    print()
                    continue
            except BCWarning as w:
                if isinstance(w.data, ast.Expr):
                    exp: ast.Expr = w.data  # type: ignore
                    output_stmt = ast.OutputStatement(pos=(0, 0, 0), items=[exp])
                    s = ast.Statement(kind="output", output=output_stmt)
                    program = ast.Program([s])
                else:
                    w.print("(repl)", inp)
                    continue

            if program.stmts[0].kind == "procedure":
                proc: ast.ProcedureStatement = program.stmts[0].procedure # type: ignore
                self.proc_src[proc.name] = self.buf.getvalue()
            elif program.stmts[0].kind == "function":
                func: ast.FunctionStatement = program.stmts[0].function # type: ignore
                self.func_src[func.name] = self.buf.getvalue()

            if program.stmts[0].kind == "fncall":
                fncall: ast.FunctionCall = program.stmts[0].fncall # type: ignore
                program.stmts[0] = ast.Statement("output", output=ast.OutputStatement(pos=(0, 0, 0), items=[fncall]))

            self.i.block = program.stmts
            self.i.toplevel = True
            try:
                self.i.visit_block(None)
            except BCError as err:
                src: str
                repl_txt = "(repl)"
                if err.proc is not None:
                    res = self.proc_src.get(err.proc) # type: ignore
                    if res is None:
                        _warn(f"could not find source code for procedure \"{err.proc}\"")
                        continue
                    src = res # type: ignore
                    repl_txt = f"(repl \"{err.proc}\")"
                elif err.func is not None:
                    res = self.func_src.get(err.func) # type: ignore
                    if res is None:
                        _warn(f"could not find source code for function \"{err.func}\"")
                        continue
                    src = res # type: ignore
                    repl_txt = f"(repl {err.func})"
                else:
                    src = self.buf.getvalue()

                err.print(repl_txt, src)
                print()
                continue

        return 0
