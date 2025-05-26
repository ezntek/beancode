# this source code is just to test dynamically loading python files and doing ffi.

from bean_ffi import *

def say_hello(_: BCArgsList):
    print("hello, world!")

def one_more(args: BCArgsList): 
    print(args["n"].get_integer() + 1)

def gimme_five(_: BCArgsList) -> BCValue:
    return BCValue.new_integer(5)

procs = [
    BCProcedure("SayHello", {}, say_hello),
    BCProcedure("OneMore", {"n": "integer"}, one_more)
]

funcs = [
    BCFunction("GimmeFive", "integer", {}, gimme_five)
]

EXPORTS: Exports = {
    "constants": [BCConstant("Name", BCValue.new_string("Charles"))],
    "variables": [BCDeclare("Age", "integer", BCValue.new_integer(69))],
    "procs": procs,
    "funcs": funcs,
}

