from sys import argv
from interpreter import Intepreter
from lexer import *
from parser import Parser

PRINT = True
INTERPRET = True

def main():
    fn = "a.bean"
    print("\033[2m")
    with open(fn, "r+") as f:
        file_content = f.read()

        lexer = Lexer(file_content)
        toks = lexer.tokenize()

        if False:
            for tok in toks:
                print(tok)

        parser = Parser(toks)
        program = parser.program()

        if PRINT:
            print(program)
        
        print("\033[0m")
        if INTERPRET:
            print("\n\033[1m----- BEGINNING OF INTERPRETER OUTPUT -----\033[0m")
            i = Intepreter(program)
            i.visit_program()

if __name__ == "__main__":
    main()

