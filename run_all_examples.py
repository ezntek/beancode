#!/usr/bin/env python3

import os
from beancode.runner import *

def log(s: str):
    print(f"\033[33;1m===> {s}\033[0m")

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
        if not execute(src, filename=file, save_interpreter=True):
            exit(1)
    except KeyboardInterrupt:
        log("continuing")
    except Exception as e:
        log(f"exception caught:")
        print(e)
        exit(1)
