import os
import sys
import argparse

from .repl import Repl

from .interpreter import Interpreter
from .lexer import *
from .parser import Parser
from .error import *
from . import __version__


def _error(s: str) -> NoReturn:
    error(s)
    exit(1)


def real_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="show debugging information"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="only print the program's AST"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"beancode version {__version__}"
    )
    parser.add_argument(
        "-i",
        "--stdin",
        action="store_true",
        help="read source from stdin",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--command",
        help="pass a program in as a string",
    )
    group.add_argument("file", nargs="?", type=str)
    args = parser.parse_args()

    if args.no_run:
        args.debug = True

    if args.command is not None:
        file_content = args.command
    elif args.stdin:
        file_content = sys.stdin.read()
    else:
        if not os.path.exists(args.file):
            _error(f"file {args.file} does not exist!")

        try:
            with open(args.file, "r+") as f:
                file_content = f.read()
        except IsADirectoryError:
            _error(f"{args.file} is not a file, but a directory!")
        except Exception as e:
            _error(f"failed to open file {args.file}: {e}")

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


def main():
    try:
        if len(sys.argv) == 1:
            sys.exit(Repl().repl())

        real_main()
    except KeyboardInterrupt:
        warn("caught keyboard interrupt")
        exit(1)
    except EOFError:
        warn("caught EOF")
        exit(1)


if __name__ == "__main__":
    main()
