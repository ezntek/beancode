import beancode
import ast

from beancode.bean_ast import BCValue

S = """
val = BCValue.new_string("hello, world!")
print(val)
"""
locals = {}
globals = {
    "beancode": beancode,
    "BCValue": BCValue
}

tree = ast.parse(S)
ast.fix_missing_locations(tree)
code = compile(tree, filename="<generated code>", mode="exec")
exec(code, locals, globals)
