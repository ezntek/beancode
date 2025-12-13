import argparse
import os
import sys

from beancode.error import *
from beancode.formatter import *
from beancode.lexer import Lexer
from beancode.parser import Parser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, help="output path of file"
    )
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    out = sys.stdout
    has_output = False
    if args.output:
        has_output = True
        out = open(args.output, "w")

    src = ""
    if not os.path.exists(args.file):
        error(f"file {args.file} does not exist!")

    with open(args.file, "r") as f:
        src = f.read()

    try:
        blk = Parser(Lexer(src).tokenize()).block(True)
        f = Formatter(blk)
        res = "".join(f.visit_block())
    except BCError as e:
        e.print(args.file, src)
        sys.exit(1)

    out.write(res + "\n")

    if has_output:
        out.close()

if __name__ == "__main__":
    main()
