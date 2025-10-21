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
.fls {
    color: red; 
}
.tru {
    color: green;
}
.int {
    color: yellow; 
}
.dim {
    color: gray;
}
caption {
    caption-side: bottom;
}
"""


class Tracer:
    vars: dict[str, list[BCValue | None]]
    line_numbers: dict[int, int]
    outputs: dict[int, list[str]]
    inputs: dict[int, list[str]]

    def __init__(self, wanted_vars: list[str]) -> None:
        self.vars = dict()
        self.outputs = dict()
        self.inputs = dict()
        self.line_numbers = dict()

        for var in wanted_vars:
            self.vars[var] = list()

    def collect_new(
        self,
        vars: dict[str, Variable],
        line_num: int,
        outputs: list[str] | None = None,
        inputs: list[str] | None = None,
    ) -> None:
        last_idx = int()
        for k, v in self.vars.items():
            if k not in vars:
                v.append(BCValue.new_null())
            else:
                v.append(copy.copy(vars[k].val))
            last_idx = len(v)
        last_idx -= 1

        if outputs is not None and len(outputs) > 0:
            self.outputs[last_idx] = copy.copy(outputs)

        if inputs is not None and len(inputs) > 0:
            self.inputs[last_idx] = copy.copy(inputs)

        self.line_numbers[last_idx] = line_num

    def print_raw(self) -> None:
        for key, items in self.vars.items():
            print(f"{key}: {items}")

        print(f"Lines: {self.line_numbers}")
        print(f"Inputs: {self.outputs}")
        print(f"Outputs: {self.inputs}")

    def _should_print_line_numbers(self) -> bool:
        first = tuple(self.line_numbers.keys())[0]
        print_lines = False
        for idx in self.line_numbers:
            if idx not in self.inputs and idx not in self.outputs:
                line_empty = True
                for v in self.vars.keys():
                    var = self.vars[v][idx]
                    if var is None:
                        break
                    if not var.is_uninitialized():
                        line_empty = False
                        break
                if line_empty:
                    continue
            if self.line_numbers[idx] != first:
                print_lines = True
                break
        return print_lines

    def _gen_html_table(self) -> str:
        res = StringIO()
        res.write("<table>\n")

        should_print_line_nums = self._should_print_line_numbers()

        if not should_print_line_nums:
            res.write("<caption>")
            res.write(f"All values are captured at line {self.line_numbers}")
            res.write("</caption>\n")

        # generate header
        res.write("<thead>\n")
        res.write("<tr>\n")

        if should_print_line_nums:
            res.write("<th style=padding:0>Line</th>")

        for row_num in self.vars:
            res.write(f"<th>{row_num}</th>\n")

        if len(self.inputs) > 0:
            res.write("<th>Inputs</th>\n")

        if len(self.outputs) > 0:
            res.write("<th>Outputs</th>\n")

        res.write("</tr>\n")
        res.write("</thead>\n")

        res.write("<tbody>\n")

        # cursed python
        rows = tuple(enumerate(zip(*self.vars.values())))
        nskipped = 0
        for row_num, row in rows:
            # skip empty lines where nothing happens
            if row_num not in self.inputs and row_num not in self.outputs:
                line_empty = True  # assume empty line due to no I/O first
                for var in row:
                    if not var.is_uninitialized():
                        line_empty = False
                        break
                if line_empty:
                    nskipped += 1
                    continue

            res.write("<tr>\n")

            if should_print_line_nums:
                if row_num in self.line_numbers:
                    res.write(f"<td>{self.line_numbers[row_num]}</td>\n")

            for col, var in enumerate(row):
                var: BCValue

                if row_num - nskipped != 0:
                    prev = rows[row_num - 1][1][col]
                    if prev == var:
                        res.write(f"<td></td>\n")
                        continue

                if var.is_uninitialized():
                    res.write(f"<td><pre class=dim>(uninitialized)</pre></td>\n")
                    continue

                match var.kind:
                    case "boolean":
                        klass = "tru" if var.boolean == True else "fls"
                        res.write(f"<td><pre class={klass}>{str(var)}</pre></td>\n")
                    case "integer" | "real":
                        res.write(f"<td><pre class=int>{str(var)}</pre></td>\n")
                    case _:
                        res.write(f"<td><pre>{str(var)}</pre></td>\n")

            if len(self.inputs) > 0:
                s = str()
                if row_num in self.inputs:
                    l = self.inputs[row_num]
                    s = "\n".join(l)
                res.write(f"<td><pre>{s}</pre></td>\n")

            if len(self.outputs) > 0:
                s = str()
                if row_num in self.outputs:
                    l = self.outputs[row_num]
                    s = "\n".join(l)
                res.write(f"<td><pre>{s}</pre></td>\n")

            res.write("</tr>\n")

        res.write("</tbody\n")
        res.write("</table>\n")

        return res.getvalue()

    def gen_html(self, filename: str | None = None) -> str:
        res = StringIO()
        res.write("<!DOCTYPE html>\n")
        res.write("<!-- Generated HTML by beancode's TRACE statement -->\n")
        res.write("<html>\n")
        res.write(f"<head>\n")
        res.write("<meta charset=UTF-8>\n")
        res.write('<meta name=color-scheme content="dark light">\n')

        title_s = ""
        if filename is not None:
            title_s = " for " + filename
        title = f"Generated Trace Table{title_s}"

        res.write(f"<title>{title}</title>\n")
        res.write(f"<style>\n{TABLE_STYLE}\n</style>\n")
        res.write("</head>\n")
        res.write("<body><center>\n")

        res.write(f"<h1>{title}</h1>\n")
        res.write(self._gen_html_table() + "\n")

        res.write("</center></body>\n")
        res.write("</html>\n")
        return res.getvalue()
