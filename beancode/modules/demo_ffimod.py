# this source code is just to test dynamically loading python files and doing ffi.

from beancode.bean_ffi import *


def say_hello(_: BCArgsList):
    print("hello, world!")


def one_more(args: BCArgsList):
    print(args["n"].get_integer() + 1)


def gimme_five(_: BCArgsList) -> BCValue:
    return BCValue.new_integer(5)


consts = [BCConstant("Name", BCValue.new_string("Charles"))]

vars = [BCDeclare("Age", "integer", BCValue.new_integer(69))]

procs = [
    BCProcedure("SayHello", {}, say_hello),
    BCProcedure("OneMore", {"n": "integer"}, one_more),
]

funcs = [BCFunction("GimmeFive", {}, "integer", gimme_five)]

EXPORTS: Exports = {
    "constants": consts,
    "variables": vars,
    "procs": procs,
    "funcs": funcs,
}
