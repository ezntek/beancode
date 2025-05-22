# this source code is just to test dynamically loading python files and doing ffi.

from bean_ffi import *

EXPORTS: Exports = {
    "constants": [BCConstant("Name", BCValue.new_string("Charles"))],
    "variables": [],
    "procs": [],
    "funcs": [],
}

# include_ffi "demo_ffimod"
