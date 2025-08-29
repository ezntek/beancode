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
type ".help" for a list of REPL commands, ".exit" to exit, or start typing some code.
"""

HELP = """\033[1mAVAILABLE COMMANDS:\033[0m
 .help:    show this help message
 .clear:   clear the screen
 .reset:   reset the interpreter
 .version: print the version
 .exit:    exit the interpreter (.quit also works)
"""

class DotCommandResult(Enum):
    NO_OP = 0,
    BREAK = 1,
    UNKNOWN_COMMAND = 2,
    RESET = 3,

class ContinuationResult(Enum):
    BREAK = 0,
    ERROR = 1,
    SUCCESS = 2,

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


class Repl:
    lx: lexer.Lexer
    p: parser.Parser
    i: intp.Interpreter

    def __init__(self):
        self.lx = lexer.Lexer(str())
        self.p = parser.Parser(list())
        self.i = intp.Interpreter(list())

    def get_continuation(self) -> tuple[ast.Program | None, ContinuationResult]:
        while True:
            oldrow = self.lx.row
            self.lx.reset()
            self.lx.row = oldrow + 1

            inp = input("\033[0m.. ")

            if len(inp) == 0:
                continue
            
            if inp[0] == ".":
                match handle_dot_command(inp[1:]):
                    case DotCommandResult.NO_OP:
                        continue
                    case DotCommandResult.BREAK:
                        return (None, ContinuationResult.BREAK)
                    case DotCommandResult.UNKNOWN_COMMAND:
                        print("\033[1minvalid dot command\033[0m")
                        print(HELP)
                        continue
                    case DotCommandResult.RESET:
                        self.i.reset_all()
                        continue
            
            self.lx.file = inp

            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", inp)
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
                    err.print("(repl)", inp)
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

            inp = input("\033[0;1m>> \033[0m")

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
                        self.i.reset_all()
                        continue

            self.lx.file = inp
            try:
                toks = self.lx.tokenize()
            except BCError as err:
                err.print("(repl)", inp)
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
                            program = cont[0] # type: ignore
                        case ContinuationResult.BREAK:
                            break
                        case ContinuationResult.ERROR:
                            continue
                else:
                    err.print("(repl)", inp)
                    print()
                    continue
            except BCWarning as w:
                if isinstance(w.data, ast.Expr):
                    exp: ast.Expr = w.data # type: ignore
                    output_stmt = ast.OutputStatement(pos=(0,0,0), items=[exp])
                    s = ast.Statement(kind="output", output=output_stmt)
                    program = ast.Program([s])
                else:
                    w.print("(repl)", inp)
                    continue

            self.i.block = program.stmts
            self.i.toplevel = True
            try:
                self.i.visit_block(None)
            except BCError as err:
                err.print("(repl)", inp)
                print()
                continue

        return 0
