from . import BCError, BCWarning, bean_ast as ast
from . import lexer
from . import parser
from . import interpreter as intp
from . import __version__

import sys
import readline
from enum import Enum

BANNER = f"""\033[1m=== welcome to beancode \033[0m{__version__}\033[1m ===\033[0m
\033[2mUsing Python {sys.version}\033[0m
type ".help" for a list of REPL commands, or start typing some code.
"""

HELP = """\033[1mAVAILABLE COMMANDS:\033[0m
 .help:    show this help message
 .clear:   clear the screen
 .reset:   reset the interpreter
 .version: print the version
 .exit:    exit the interpreter (same as .quit)
"""

class DotCommandResult(Enum):
    NO_OP = 0,
    BREAK = 1,
    UNKNOWN_COMMAND = 2,
    RESET = 3,

def handle_dot_command(s: str) -> DotCommandResult:
    match s:
        case "exit" | "quit":
            print("\033[1mbye\033[0m")
            return DotCommandResult.BREAK
        case "clear":
            sys.stdout.write("\033[2J\033[H")
            return DotCommandResult.NO_OP
        case "reset":
            print("\033[1mreset interpreter\033[0m")
            return DotCommandResult.RESET
        case "help":
            print(HELP)
            return DotCommandResult.NO_OP
        case "version":
            print(f"beancode version \033[1m{__version__}\033[0m")
            return DotCommandResult.NO_OP
    return DotCommandResult.UNKNOWN_COMMAND

def repl() -> int:
    print(BANNER, end=str())

    lx = lexer.Lexer(str())
    p = parser.Parser(list())
    i = intp.Interpreter(list())

    inp = str()
    while True:
        lx.reset()
        p.reset()
        i.reset()

        inp = input(">> ")

        if len(inp) == 0:
            continue
        
        if inp[0] == ".":
            match handle_dot_command(inp[1:]):
                case DotCommandResult.NO_OP:
                    continue
                case DotCommandResult.BREAK:
                    break
                case DotCommandResult.UNKNOWN_COMMAND:
                    print("\033[1minvalid dot command\033[0m")
                    print(HELP)
                    continue
                case DotCommandResult.RESET:
                    i.reset_all()
                    continue

        lx.file = inp
        try:
            toks = lx.tokenize()
        except BCError as err:
            err.print("(repl)", inp)
            print()
            continue

        program: ast.Program
        p.tokens = toks
        try:
            program = p.program()
        except BCError as err:
            err.print("(repl)", inp)
            print()
            continue
        except BCWarning as w:
            if isinstance(w.data, ast.Expr):
                exp: ast.Expr = w.data # type: ignore
                s = ast.Statement(kind="output", output=ast.OutputStatement(pos=(0,0,0), items=[exp]))
                program = ast.Program([s])
            else:
                w.print("(repl)", inp)
                continue

        i.block = program.stmts
        i.toplevel = True
        try:
            i.visit_block(None)
        except BCError as err:
            err.print("(repl)", inp)
            print()
            continue

    return 0
