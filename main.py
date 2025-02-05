from sys import argv
from lexer import *
from parser import Parser

def main():
    fn = "a.bean"
    with open(fn, "r+") as f:
        s = f.read()

        l = Lexer(s)
        toks = l.tokenize()
        for tok in toks:
            print(tok)

        #p = Parser(toks)
        #e = p.program()
        #print(e)

if __name__ == "__main__":
    main()

