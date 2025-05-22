import os
import typing
from codegen import LuaCodegen
from interpreter import Interpreter
from lexer import *
from parser import Parser
from util import BCError, BCWarning, panic

from tap import Tap

Backend = typing.Literal["interpreter", "lua", "none"]


class Args(Tap):
    file: str  # filename
    out_file: str = ""
    backend: Backend = "interpreter"
    print: bool = True
    debug: bool = False


def main():
    args = Args().parse_args()

    if not os.path.exists(args.file):
        panic(f"file {args.file} does not exist!")

    with open(args.file, "r+") as f:
        file_content = f.read()

    lexer = Lexer(file_content)
    toks = lexer.tokenize()

    if args.debug:
        for tok in toks:
            print(tok)

    warnings: list[BCWarning]

    parser = Parser(toks)

    try:
        program, warnings = parser.program()
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)

    if warnings:
        for warning in warnings:
            warning.print(args.file, file_content)
        exit(1)

    match args.backend:
        case "interpreter":
            if args.debug:
                print("\033[1m----- BEGINNING OF AST -----\033[0m")
                for stmt in program.stmts:
                    print(stmt)
                    print()

            print("\033[1m----- BEGINNING OF INTERPRETER OUTPUT -----\033[0m")
            try:
                i = Interpreter(program.stmts)
                i.toplevel = True
                i.visit_block(None)
            except BCError as err:
                err.print(args.file, file_content)
                exit(1)
        case "lua":
            cg = LuaCodegen(program.stmts)
            code = cg.generate()
            if args.out_file == "":
                print("\033[1m----- BEGINNING OF GENERATED CODE -----\033[0m")
                print(code)
                print("\033[1m----- END OF GENERATED CODE -----\033[0m")
            else:
                with open(args.out_file) as f:
                    f.write(code)

        case "none":         
            if args.print:
                print("\033[0m\033[1m----- BEGINNING OF AST -----\033[0m\033[2m")
                for stmt in program.stmts:
                    print(stmt)
                    print()
                print("\033[0m\033[1m----- END OF AST -----\033[0m")
            print("\033[0m")
                    

if __name__ == "__main__":
    main()
