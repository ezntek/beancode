from . import BCError, BCWarning, bean_ast as ast
from . import lexer
from . import parser
from . import interpreter as intp

import sys
import readline

HELP = """\033[1mAVAILABLE COMMANDS:\033[0m
 .help:  show this help message
 .clear: clear the screen
 .exit:  exit the interpreter (same as .quit)
"""

def handle_dot_command(s: str) -> int:
    match s:
        case "exit" | "quit":
            print("\033[1mbye\033[0m")
            return 1
        case "clear":
            sys.stdout.write("\033[2J\033[H")
            return 0
        case "help":
            print(HELP)
            return 0
    return 2

def repl() -> int:
    lx = lexer.Lexer(str())
    p = parser.Parser(list())
    i = intp.Interpreter(list())

    inp = str()
    while True:
        lx.reset()
        p.reset()
        i.reset()

        inp = input("> ")

        if len(inp) == 0:
            continue
        
        if inp[0] == ".":
            match handle_dot_command(inp[1:]):
                case 0:
                    continue
                case 1:
                    break
                case 2:
                    print("\033[1minvalid dot command\033[0m", end=str())
                    print(HELP)
                    continue

        lx.file = inp
        try:
            toks = lx.tokenize()
        except BCError as err:
            err.print("(repl)", inp)
            continue

        program: ast.Program
        p.tokens = toks
        try:
            program = p.program()
        except BCError as err:
            err.print("(repl)", inp)
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
            continue

    return 0
