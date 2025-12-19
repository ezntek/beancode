#!/usr/bin/env python3

import os
from beancode.runner import *
from argparse import ArgumentParser


def log(s: str):
    print(f"\033[33;1m===> {s}\033[0m")

p = ArgumentParser()
p.add_argument("-O", "--optimize", action="store_true", help="run the optimizer before execution")
args = p.parse_args()

src = ""
for file in sorted(os.listdir("examples")):
    if "raylib" in file:
        continue

    try:
        p = os.path.join("examples", file)
        if not os.path.isfile(p):
            continue
        with open(p, "r") as f:
            src = f.read()
        log(f"running example {file}")
        if not execute(src, filename=file, save_interpreter=True, optimize=args.optimize):
            exit(1)
    except KeyboardInterrupt:
        log("continuing")
    except EOFError:
        exit(1)
    except Exception as e:
        log(f"exception caught:")
        print(e)
        exit(1)
