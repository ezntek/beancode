import os
from beancode.bean_ffi import *

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


def _get_env(args: BCArgsList) -> BCValue:
    s = args["s"].get_string()
    res = os.environ.get(s)
    res = res if res else ""
    return BCValue.new_string(res)


consts = []
vars = []
procs = [
    BCProcedure("Write", {"s": BCPrimitiveType.STRING}, _write),
    BCProcedure("WriteErr", {"s": BCPrimitiveType.STRING}, _write_err),
    BCProcedure("Flush", {}, _flush),
    BCProcedure("FlushErr", {}, _flush_err),
    BCProcedure("WriteLn", {"s": BCPrimitiveType.STRING}, _writeln),
    BCProcedure("WriteLnErr", {"s": BCPrimitiveType.STRING}, _writeln_err),
]
funcs = [
    BCFunction(
        "GetEnv", {"s": BCPrimitiveType.STRING}, BCPrimitiveType.STRING, _get_env
    ),
]

EXPORTS: Exports = {
    "constants": consts,
    "variables": vars,
    "procs": procs,
    "funcs": funcs,
}
