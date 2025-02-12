from sys import argv
from interpreter import Intepreter
from lexer import *
from parser import Parser

PRINT = True
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

        parser = Parser(toks)
        program = parser.program()

        if PRINT:
            for stmt in program.stmts:
                print(stmt)
                print()
        print("\033[0m")
        if INTERPRET:
            print("\033[1m----- BEGINNING OF INTERPRETER OUTPUT -----\033[0m")
            i = Intepreter(program.stmts)
            i.visit_block(None)

if __name__ == "__main__":
    main()

