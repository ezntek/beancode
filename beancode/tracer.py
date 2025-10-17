from typing import Callable

from .bean_ast import *

class Tracer:
    table: dict[str, list[BCValue]]
    
    def __init__(self, wanted_vars: list[str]) -> None:
        self.table = dict()
        for var in wanted_vars:
            self.table[var] = list()

    def collect_new(self, vars: dict[str, Variable]) -> None:
        for k, v in self.table.items():
            if k not in vars:
                v.append(BCValue.new_null())
            else:
                v.append(vars[k].val)

    def print_items(self) -> None:
        for key, items in self.table.items():
            print(f"{key}: {items}")
