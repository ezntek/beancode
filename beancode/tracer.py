import copy

from .bean_ast import *
from typing import Callable

TABLE_STYLE = """
table {
    border-collapse: collapse;
}
tr, td, th {
    border: 1px solid;
    padding-left: 20px;
    padding-right: 20px;
    text-align: center;
}
"""

class Tracer:
    table: dict[str, list[BCValue]]
    outputs: dict[int, list[str]]
    inputs: dict[int, list[str]]

    def __init__(self, wanted_vars: list[str]) -> None:
        self.table = dict()
        self.outputs = dict()
        self.inputs = dict()

        for var in wanted_vars:
            self.table[var] = list()

    def collect_new(
        self,
        vars: dict[str, Variable],
        outputs: list[str] | None = None,
        inputs: list[str] | None = None,
    ) -> None:
        lastidx = int() 
        for k, v in self.table.items():
            if k not in vars:
                v.append(BCValue.new_null())
            else:
                v.append(vars[k].val)
            lastidx = len(v) 

        if outputs is not None:
            self.outputs[lastidx] = copy.copy(outputs)

        if inputs is not None:
            self.inputs[lastidx] = copy.copy(inputs)

    def print_raw(self) -> None:
        for key, items in self.table.items():
            print(f"{key}: {items}")

        print(f"Inputs: {self.outputs}")
        print(f"Outputs: {self.inputs}")

    def _gen_html_table(self) -> str:
        res = StringIO()
        res.write("<table>\n")

        # generate header
        res.write("<tr>\n")
        for k in self.table:
            res.write(f"<th>{k}</th>\n")
        res.write(f"<th>Inputs</th>\n")
        res.write(f"<th>Outputs</th>\n")
        res.write("</tr>\n")

        # cursed python
        for i, row in enumerate(zip(*self.table.values())): 
            res.write("<tr>\n")
            
            for itm in row:
                itm: BCValue
                res.write(f"<td><pre>{str(itm)}</pre></td>\n")
           
            s = str()
            if i in self.inputs:
                l = self.inputs[i]
                s = '\n'.join(l)
            res.write(f"<td><pre>{s}</pre></td>\n")

            s = str()
            if i in self.outputs:
                l = self.outputs[i]
                s = '\n'.join(l)
            res.write(f"<td><pre>{s}</pre></td>\n")
            
            res.write("</tr>\n")

        res.write("</table>\n")        

        return res.getvalue()

    def gen_html(self, filename: str | None = None) -> str:
        res = StringIO()
        res.write("<!DOCTYPE html>\n")
        res.write("<!-- Generated HTML by beancode's TRACE statement -->\n")
        res.write("<html>\n")
        res.write(f"<head>\n")
        title_s = ""
        if filename is not None:
            title_s = " for " + filename
        title = f"Generated Trace Table{title_s}"
       
        res.write(f"<title>{title}</title>\n")
        res.write(f"<style>\n{TABLE_STYLE}\n</style>\n")
        res.write("</head>\n")
        res.write("<body>\n")

        res.write(f"<h1>{title}</h1>\n")
        res.write(self._gen_html_table() + '\n')
        
        res.write("</body>\n")
        res.write("</html>\n")
        return res.getvalue() 
