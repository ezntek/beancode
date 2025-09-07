import os
import sys
import argparse

from .repl import Repl

from .interpreter import Interpreter
from .lexer import *
from .parser import Parser
from .error import BCError, BCWarning
from . import error, __version__

def main():
    if len(sys.argv) == 1:
        try:
            sys.exit(Repl().repl())
        except KeyboardInterrupt:
            sys.exit(1)
        except EOFError:
            sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument(
        "-d", "--debug", action="store_true", help="show debugging information"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="only print the program's AST"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"beancode version {__version__}"
    )
    args = parser.parse_args()

    if args.no_run:
        args.debug = True

    if not os.path.exists(args.file):
        error(f"file {args.file} does not exist!")

    with open(args.file, "r+") as f:
        file_content = f.read()

    lexer = Lexer(file_content)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)

    if args.debug:
        for tok in toks:
            tok.print()

    parser = Parser(toks)

    try:
        program = parser.program()
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)
    except BCWarning as w:
        w.print(args.file, file_content)
        exit(1)

    if args.debug:
        print("\033[1m----- BEGINNING OF AST -----\033[0m", file=sys.stderr)
        for stmt in program.stmts:
            print(stmt)
            print()
        print("\033[0m\033[1m----- END OF AST -----\033[0m", file=sys.stderr)

    if args.no_run:
        return

    try:
        i = Interpreter(program.stmts)
        i.toplevel = True
        i.visit_block(None)
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)


if __name__ == "__main__":
    main()
