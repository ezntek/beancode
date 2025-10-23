import copy

from .bean_ast import *

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
pre {
    font-size: 1.2em;
}
"""


class Tracer:
    vars: dict[str, list[BCValue | None]]
    var_types: dict[str, BCType]
    last_updated_vals: dict[str, BCValue | None] # None only initially
    line_numbers: dict[int, int]
    outputs: dict[int, list[str]]
    inputs: dict[int, list[str]]

    def __init__(self, wanted_vars: list[str]) -> None:
        self.vars = dict()
        self.outputs = dict()
        self.inputs = dict()
        self.line_numbers = dict()
        self.last_updated_vals = dict()
        self.var_types = dict()

        for var in wanted_vars:
            self.vars[var] = list()
            self.var_types[var] = "null"
            self.last_updated_vals[var] = None

    def collect_new(
        self,
        vars: dict[str, Variable],
        line_num: int,
        outputs: list[str] | None = None,
        inputs: list[str] | None = None,
    ) -> None:
        should_collect = False
        for k in self.vars:
            if k not in vars:
                continue
            if not vars[k].is_uninitialized():
                should_collect = True

        if not should_collect:
            return
                
        last_idx = int()
        for k, v in self.vars.items():
            if k not in vars:
                v.append(BCValue.new_null())
            else:
                if len(v) > 0 and vars[k].val == self.last_updated_vals[k]:
                    v.append(None)
                else:
                    new_obj = copy.deepcopy(vars[k].val)
                    v.append(new_obj)
                    self.last_updated_vals[k] = new_obj
                    self.var_types[k] = new_obj.kind
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

    def _highlight_var(self, var: BCValue) -> str:
        if var.is_uninitialized():
            return f"<td><pre class=dim>(null)</pre></td>"

        match var.kind:
            case "boolean":
                klass = "tru" if var.boolean == True else "fls"
                return f"<td><pre class={klass}>{str(var)}</pre></td>\n"
            case "integer" | "real":
                return f"<td><pre class=int>{str(var)}</pre></td>\n"
            case _:
                return f"<td><pre>{str(var)}</pre></td>\n"

    def _gen_html_table_header(self, should_print_line_nums: bool) -> str:
        res = StringIO()

        res.write("<thead>\n")
        res.write("<tr>\n")

        if should_print_line_nums:
            res.write("<th style=padding:0>Line</th>")

        for name, var in self.vars.items():
            if var[0] is not None and isinstance(var[0].kind, BCArrayType):
                arrtyp = var[0].kind 
                if not arrtyp.is_matrix:
                    bounds: tuple[int, int] = arrtyp.flat_bounds # type: ignore
                    for num in range(bounds[0], bounds[1]+1): # never None
                        res.write(f"<th>{name}[{num}]</th>") 
            else:
                res.write(f"<th>{name}</th>\n")

        if len(self.inputs) > 0:
            res.write("<th>Inputs</th>\n")

        if len(self.outputs) > 0:
            res.write("<th>Outputs</th>\n")

        res.write("</tr>\n")
        res.write("</thead>\n")

        res.write("<tbody>\n")

        return res.getvalue() 

    def _gen_html_table_row(self, rows: list[tuple[int, tuple[BCValue, ...]]], row_num: int, row: tuple[BCValue, ...]) -> str:
        res = StringIO()

        for col, (var_name, var) in enumerate(zip(self.vars, row)):
            if isinstance(self.var_types[var_name], BCArrayType):
                if not var:
                    # blank the region out
                    bounds = self.var_types[var_name].get_flat_bounds() # type: ignore
                    for _ in range(bounds[0], bounds[1]+1):
                        res.write(f"<td></td>")
                else:
                    # rows[row_num] is enumerated, col+1 compensates for the index at the front
                    arr: BCArray = var.get_array()
                    if not arr.typ.is_matrix:
                        prev: list[BCValue] | None = None
                        if row_num != 0:
                            prev_var = rows[row_num-1][1][col]
                            if prev_var:
                                prev = prev_var.get_array().get_flat()
                        
                        for idx, itm in enumerate(arr.get_flat()):
                            if prev and prev[idx] == itm: # if it is a repeated entry
                                res.write("<td></td>\n")
                            else:
                                res.write(self._highlight_var(itm))
            elif not var:
                res.write(f"<td></td>\n")
                continue
            else: 
                res.write(self._highlight_var(var))

        return res.getvalue()

    def _gen_html_table_row_io(self, row_num: int) -> str:
        res = StringIO()

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

        return res.getvalue()

    def _gen_html_table(self) -> str:
        res = StringIO()
        res.write("<table>\n")

        # generate header
        should_print_line_nums = self._should_print_line_numbers()

        if not should_print_line_nums:
            res.write("<caption>")
            res.write(f"All values are captured at line {self.line_numbers}")
            res.write("</caption>\n")

        res.write(self._gen_html_table_header(should_print_line_nums))

        # cursed python
        rows = list(enumerate(zip(*self.vars.values())))
        for row_num, row in rows:
            res.write("<tr>\n")

            if should_print_line_nums:
                if row_num in self.line_numbers:
                    res.write(f"<td>{self.line_numbers[row_num]}</td>\n")

            res.write(self._gen_html_table_row(rows, row_num, row))
            res.write(self._gen_html_table_row_io(row_num))
            
            res.write("</tr>\n")

        res.write("</tbody>\n")
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
