from beancode.bean_ffi import *

import time
import math
import sys


def _write(args: BCArgsList):
    s = args["s"].get_string()
    sys.stdout.write(s)
    sys.stdout.flush()


def _write_err(args: BCArgsList):
    s = args["s"].get_string()
    sys.stderr.write(s)
    sys.stderr.flush()


def _flush(_: BCArgsList):
    sys.stdout.flush()


def _flush_err(_: BCArgsList):
    sys.stderr.flush()


def _writeln(args: BCArgsList):
    s = args["s"].get_string()
    sys.stdout.write(s + "\n")


def _writeln_err(args: BCArgsList):
    s = args["s"].get_string()
    sys.stderr.write(s + "\n")


consts = []
vars = []
procs = [
    BCProcedure("Write", {"s": "string"}, _write),
    BCProcedure("WriteErr", {"s": "string"}, _write_err),
    BCProcedure("Flush", {}, _flush),
    BCProcedure("FlushErr", {}, _flush_err),
    BCProcedure("WriteLn", {"s": "string"}, _writeln),
    BCProcedure("WriteLnErr", {"s": "string"}, _writeln_err),
]
funcs = []

EXPORTS: Exports = {
    "constants": consts,
    "variables": vars,
    "procs": procs,
    "funcs": funcs,
}
