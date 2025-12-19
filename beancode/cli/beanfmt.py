#!/usr/bin/env python3

import argparse
import sys

from io import StringIO
from pathlib import Path
from typing import NoReturn

from beancode.error import *
from beancode.formatter import *
from beancode.lexer import Lexer
from beancode.parser import Parser
from beancode.optimizer import Optimizer

optimize = False

def _error(s: str) -> NoReturn:
    error(s)
    sys.exit(1)


def _format_file(args, name: str, file_content: str, file=sys.stdout):
    global optimize
    lexer = Lexer(file_content, preserve_comments=True)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(name, file_content)
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
        if optimize:
            opt = Optimizer(blk)
            blk = opt.visit_block(None)
    except BCError as err:
        err.print(name, file_content)
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
        e.print(name, file_content)
        sys.exit(1)

    file.write(res)


def _format_in_place(args, src: str, path: str, f):
    out = StringIO()
    _format_file(args, path, src, file=out)
    f.truncate(0)
    f.write(out.getvalue())
    print(f"Formatted {path}", file=sys.stderr)


def _format_one(args, in_path: Path, out_path: Path | None, stdout=False):
    src = in_path.read_text()
    if out_path:
        with out_path.open(mode="w") as f:
            _format_file(args, str(in_path), src, file=f)
            print(f"Formatted {in_path} to {out_path}", file=sys.stderr)
    elif stdout:
        _format_file(args, str(in_path), src)
    else:
        with in_path.open(mode="r+") as f:
            _format_in_place(args, src, str(in_path), f)


def _format_many(args, in_path: Path):
    for file in in_path.iterdir():
        with file.open(mode="r+") as f:
            _format_in_place(args, file.read_text(), str(file), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="output path of file")
    parser.add_argument("--stdout", action="store_true", help="print output to stdout")
    parser.add_argument("-O", "--optimize", action="store_true", help="format and optimize code at the same time")
    parser.add_argument(
        "--debug", action="store_true", help="print debugging information"
    )
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    global optimize
    optimize = args.optimize

    in_path = Path(args.file)
    if not in_path.exists():
        _error(f"file or directory {in_path} does not exist!")

    if args.stdout and in_path.is_dir():
        _error("you must only pass --in-place to format a directory!")

    if in_path.is_dir():
        _format_many(args, in_path)
    elif not (args.output or args.stdout):
        _format_one(args, in_path, None)
    else:
        stdout = False
        out_path = None
        if args.output == "stdout" or args.stdout:
            stdout = True
        else:
            out_path = Path(args.output)
            if out_path.is_dir():
                _error("you must pass --in-place to format a directory!")

        _format_one(args, in_path, out_path, stdout)


if __name__ == "__main__":
    main()
