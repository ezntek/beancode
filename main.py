from sys import argv
from interpreter import Interpreter
from lexer import *
from parser import Parser
from util import BCError, BCWarning

PRINT = False
INTERPRET = True 

def main():
    fn = argv[1]
    print("\033[2m", end='')

    with open(fn, "r+") as f:
        file_content = f.read()
        
    lexer = Lexer(file_content)
    toks = lexer.tokenize()

    if PRINT:
        for tok in toks:
            print(tok)

    warnings: list[BCWarning]

    parser = Parser(toks)

    try:
        program, warnings = parser.program()
    except BCError as err:
        err.print(fn, file_content)
        exit(1)

    if warnings:
        for warning in warnings:
            warning.print(fn, file_content)

    if PRINT:
        print("\033[0m\033[1m----- BEGINNING OF AST -----\033[0m\033[2m")
        for stmt in program.stmts:
            print(stmt)
            print()
        print("\033[0m\033[1m----- END OF AST -----\033[0m")
    print("\033[0m")

    if INTERPRET:
        print("\033[1m----- BEGINNING OF INTERPRETER OUTPUT -----\033[0m")
        i = Interpreter(program.stmts)
        i.toplevel = True
        i.visit_block(None)

if __name__ == "__main__":
    main()

