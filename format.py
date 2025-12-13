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
    parser.add_argument(
        "--debug", action="store_true", help="print debugging information"
    )
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    out = sys.stdout
    has_output = False
    if args.output:
        has_output = True
        out = open(args.output, "w")

    file_content = ""
    if not os.path.exists(args.file):
        error(f"file {args.file} does not exist!")

    with open(args.file, "r") as f:
        file_content = f.read()

    lexer = Lexer(file_content, preserve_comments=True)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)

    if args.debug:
        print("\033[2m=== TOKENS ===\033[0m", file=sys.stderr)
        for tok in toks:
            tok.print(file=sys.stderr)
        print("\033[2m==============\033[0m", file=sys.stderr)
        sys.stderr.flush()

    parser = Parser(toks, preserve_trivia=True)

    try:
        blk = parser.block()
    except BCError as err:
        err.print(args.file, file_content)
        exit(1)

    if args.debug:
        print("\033[2m=== CST ===\033[0m", file=sys.stderr)
        for stmt in blk:
            print(stmt, file=sys.stderr)
            print(file=sys.stderr)
        print("\033[0m\033[2m===========\033[0m", file=sys.stderr)
        sys.stderr.flush()

    try:
        f = Formatter(blk)
        res = "".join(f.visit_block())
    except BCError as e:
        e.print(args.file, file_content)
        sys.exit(1)

    out.write(res + "\n")

    if has_output:
        out.close()

if __name__ == "__main__":
    main()
