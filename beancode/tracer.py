import copy

from .bean_ast import *

TABLE_STYLE = """
table {
    border-collapse: collapse;
}
tr, td, th {
    border: 1px solid;
    padding-left: 0.7em;
    padding-right: 0.7em;
    text-align: center;
}
caption {
    color: rgb(150, 150, 150);
}
.fls {
    color: rgb(230, 41, 55); 
}
.tru {
    color: rgb(0, 158, 47);
}
.int {
    color: rgb(245, 193, 0); 
}
.dim {
    color: rgb(130, 130, 130);
}
caption {
    caption-side: bottom;
}
pre {
    font-size: 1.2em;
}
"""


@dataclass
class TracerConfig:
    trace_every_line = False
    hide_repeating_entries = True
    condense_arrays = False
    syntax_highlighting = True
    # handled by the interpreter
    show_outputs = False
    prompt_on_inputs = True
    debug = False
    i_will_not_cheat = False

    @classmethod
    def from_config(cls, cfg: dict[str, BCValue]) -> "TracerConfig":
        res = cls()

        if "TraceEveryLine" in cfg:
            data = cfg["TraceEveryLine"]
            if data.kind == "boolean":
                res.trace_every_line = data.get_boolean()

        if "HideRepeatingEntries" in cfg:
            data = cfg["HideRepeatingEntries"]
            if data.kind == "boolean":
                res.hide_repeating_entries = data.get_boolean()

        if "CondenseArrays" in cfg:
            data = cfg["CondenseArrays"]
            if data.kind == "boolean":
                res.condense_arrays = data.get_boolean()

        if "SyntaxHighlighting" in cfg:
            data = cfg["SyntaxHighlighting"]
            if data.kind == "boolean":
                res.syntax_highlighting = data.get_boolean()

        if "ShowOutputs" in cfg:
            data = cfg["ShowOutputs"]
            if data.kind == "boolean":
                res.show_outputs = data.get_boolean()

        if "PromptOnInputs" in cfg:
            data = cfg["PromptOnInputs"]
            if data.kind == "boolean":
                res.prompt_on_inputs = data.get_boolean()

        if "Debug" in cfg:
            data = cfg["Debug"]
            if data.kind == "boolean":
                res.debug = data.get_boolean()

        if "IWillNotCheat" in cfg:
            data = cfg["IWillNotCheat"]
            if data.kind == "boolean":
                res.i_will_not_cheat = data.get_boolean()

        return res


class Tracer:
    vars: dict[str, list[BCValue | None]]
    var_types: dict[str, BCType]
    last_updated_vals: dict[str, BCValue | None]  # None only initially
    line_numbers: dict[int, int]
    outputs: dict[int, list[str]]
    inputs: dict[int, list[str]]
    last_idx: int

    def __init__(
        self, wanted_vars: list[str], config: TracerConfig | None = None
    ) -> None:
        self.vars = dict()
        self.outputs = dict()
        self.inputs = dict()
        self.line_numbers = dict()
        self.last_updated_vals = dict()
        self.var_types = dict()
        self.last_idx = -1

        # weird python object copy/move semantics
        if config:
            self.config = config
        else:
            self.config = TracerConfig()

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
        should_collect = self.config.trace_every_line

        for k in self.vars:
            if k not in vars:
                continue
            if not vars[k].is_uninitialized():
                should_collect = True
                break

        if outputs and len(outputs) > 0:
            should_collect = True

        if inputs and len(inputs) > 0:
            should_collect = True

        if not should_collect:
            return

        last_idx = 0
        for k, v in self.vars.items():
            if k not in vars:
                # NOTE: No value.
                # BCValue.new_null() will result in (null) being printed, but uninitialized
                # variables look the same.
                v.append(None)
            else:
                # XXX: The algorithm to figure out repeated values breaks when you use arrays.
                # Therefore, we just copy the pointers for repeated values to make sure that at
                # the data level, there are no blank rows (unlike before). We use magic iterator
                # trickery to figure it out at generation-time.
                #
                if len(v) > 0 and vars[k].val == self.last_updated_vals[k]:
                    # copy the pointer of the last owned value if it is repeated
                    v.append(self.last_updated_vals[k])  # type: ignore
                else:
                    # copy the object and create a new owned value
                    new_obj = copy.deepcopy(vars[k].val)
                    v.append(new_obj)
                    self.last_updated_vals[k] = new_obj
                    self.var_types[k] = new_obj.kind
            last_idx = len(v) - 1

        if len(self.vars) == 0:
            self.last_idx += 1
        else:
            self.last_idx = last_idx

        if outputs is not None and len(outputs) > 0:
            self.outputs[self.last_idx] = copy.copy(outputs)

        if inputs is not None and len(inputs) > 0:
            self.inputs[self.last_idx] = copy.copy(inputs)

        self.line_numbers[self.last_idx] = line_num

    def print_raw(self) -> None:
        for key, items in self.vars.items():
            print(f"{key}: {items}")

        print(f"Lines: {self.line_numbers}")
        print(f"Outputs: {self.outputs}")
        print(f"Inputs: {self.inputs}")

    def _should_print_line_numbers(self) -> bool:
        if self.config.trace_every_line:
            return True

        if len(self.line_numbers) == 0:
            return False

        first = tuple(self.line_numbers.values())[0]
        print_lines = False
        for idx in self.line_numbers:
            if self.line_numbers[idx] != first:
                print_lines = True
                break

        return print_lines

    def _has_array(self) -> bool:
        has_array = False
        for typ in self.var_types.values():
            if (
                not self.config.condense_arrays
                and isinstance(typ, BCArrayType)
                and not typ.is_matrix
            ):
                has_array = True
                break
        return has_array

    def _highlight_var(self, var: BCValue) -> str:
        if var.is_uninitialized():
            if self.config.syntax_highlighting:
                return f"<td><pre class=dim>null</pre></td>"
            else:
                return "<td><pre>(null)</pre></td>"

        if not self.config.syntax_highlighting:
            return f"<td><pre>{str(var)}</pre></td>"

        match var.kind:
            case "boolean":
                klass = "tru" if var.boolean == True else "fls"
                return f"<td><pre class={klass}>{str(var)}</pre></td>"
            case "integer" | "real":
                return f"<td><pre class=int>{str(var)}</pre></td>"
            case _:
                return f"<td><pre>{str(var)}</pre></td>"

    def _gen_html_table_header(self, should_print_line_nums: bool) -> str:
        res = StringIO()

        res.write("<thead>\n")
        res.write("<tr>\n")

        has_array = self._has_array()
        rs = " rowspan=2" if has_array else ""

        if should_print_line_nums:
            res.write(f"<th style=padding:0.2em{rs}>Line</th>\n")

        # first pass
        for name, typ in self.var_types.items():
            if not self.config.condense_arrays and isinstance(typ, BCArrayType):
                if not typ.is_matrix:
                    width = typ.flat_bounds[1] - typ.flat_bounds[0] + 1  # type: ignore
                    res.write(f"<th colspan={width}>{name}</th>")
            else:
                res.write(f"<th{rs}>{name}</th>")

        if len(self.inputs) > 0:
            res.write(f"<th{rs}>Inputs</th>")

        if len(self.outputs) > 0:
            res.write(f"<th{rs}>Outputs</th>")

        # second pass
        if has_array:
            res.write("</tr><tr>")
            for name, typ in self.var_types.items():
                if isinstance(typ, BCArrayType) and not typ.is_matrix:
                    bounds: tuple[int, int] = typ.flat_bounds  # type: ignore
                    for num in range(bounds[0], bounds[1] + 1):  # never None
                        res.write(f"<th>[{num}]</th>")

        res.write("</tr>\n")
        res.write("</thead>\n")

        return res.getvalue()

    def _gen_html_table_line_num(self, row_num: int) -> str:
        if self._should_print_line_numbers():
            if row_num in self.line_numbers:
                return f"<td>{self.line_numbers[row_num]}</td>\n"
        return str()

    def _gen_html_table_row(
        self,
        rows: list[tuple[int, tuple[BCValue | None, ...]]],
        row_num: int,
        row: tuple[BCValue | None, ...],
        printed_first: bool = True,
    ) -> str:
        res = StringIO()

        for col, (var_name, var) in enumerate(zip(self.vars, row)):
            if not self.config.condense_arrays and isinstance(
                self.var_types[var_name], BCArrayType
            ):
                if not var:
                    # blank the region out
                    bounds = self.var_types[var_name].get_flat_bounds()  # type: ignore
                    for _ in range(bounds[0], bounds[1] + 1):
                        res.write(f"<td></td>")
                else:
                    # rows[row_num] is enumerated, col+1 compensates for the index at the front
                    arr: BCArray = var.get_array()
                    if not arr.typ.is_matrix:
                        prev_arr: list[BCValue] | None = None
                        if row_num != 0:
                            prev_var = rows[row_num - 1][1][col]
                            if prev_var:
                                prev_arr = prev_var.get_array().get_flat()

                        for idx, itm in enumerate(arr.get_flat()):
                            repeated = self.config.hide_repeating_entries and (
                                prev_arr and prev_arr[idx] == itm
                            )
                            if repeated or not prev_arr and printed_first:
                                res.write("<td></td>")
                            else:
                                res.write(self._highlight_var(itm))
            else:
                prev: BCValue | None = None
                if row_num != 0:
                    prev = rows[row_num - 1][1][col]

                repeated = self.config.hide_repeating_entries and var == prev
                if not var or repeated and printed_first:
                    res.write("<td></td>")
                else:
                    res.write(self._highlight_var(var))

        return res.getvalue()

    def _gen_html_table_row_io(self, row_num: int) -> str:
        res = StringIO()

        if len(self.inputs) > 0:
            s = str()
            if row_num in self.inputs:
                l = self.inputs[row_num]
                s = "<br></br>".join(l)
            res.write(f"<td><pre>{s}</pre></td>\n")

        if len(self.outputs) > 0:
            s = str()
            if row_num in self.outputs:
                l = self.outputs[row_num]
                s = "<br></br>".join(l)
            res.write(f"<td><pre>{s}</pre></td>\n")

        return res.getvalue()

    def _gen_html_table_body(self):
        res = StringIO()

        res.write("<tbody>\n")

        if len(self.vars) == 0:
            keys = set()
            for k in self.outputs:
                keys.add(k)
            for k in self.inputs:
                keys.add(k)

            for k in keys:
                res.write("<tr>")
                res.write(self._gen_html_table_line_num(k))
                res.write(self._gen_html_table_row_io(k))
                res.write("</tr>\n")

            res.write("</tbody>\n")
            return res.getvalue()

        rows: list[tuple[int, tuple[BCValue | None, ...]]] = list(
            enumerate(zip(*self.vars.values()))
        )
        printed_first = False
        for row_num, row in rows:
            # skip empty rows
            if not self.config.trace_every_line and (
                row_num not in self.inputs and row_num not in self.outputs
            ):
                # no I/O
                empty = True
                for col, var in enumerate(row):
                    if var is None:
                        continue

                    if isinstance(var.kind, BCArrayType) and not var.kind.is_matrix:
                        prev_arr: list[BCValue] | None = None
                        if row_num != 0:
                            prev_var = rows[row_num - 1][1][col]
                            if prev_var:
                                prev_arr = prev_var.get_array().get_flat()
                        arr = var.get_array().get_flat()

                        for idx, itm in enumerate(arr):
                            if prev_arr:
                                if prev_arr[idx] == itm:
                                    continue

                            empty = False
                            break
                    else:
                        prev = rows[row_num - 1][1][col]

                        if prev and prev == var:
                            continue

                        empty = False
                        break

                if empty:
                    continue

            res.write("<tr>")

            res.write(self._gen_html_table_line_num(row_num))
            res.write(self._gen_html_table_row(rows, row_num, row, printed_first))
            res.write(self._gen_html_table_row_io(row_num))

            printed_first = True

            res.write("</tr>\n")

        res.write("</tbody>\n")
        return res.getvalue()

    def _gen_html_table(self) -> str:
        res = StringIO()
        res.write("<table>\n")

        # generate header
        should_print_line_nums = self._should_print_line_numbers()

        if not should_print_line_nums:
            res.write("<caption>")
            if len(self.line_numbers) == 0:
                res.write("No values were captured.")
            else:
                res.write(f"All values are captured at line {self.line_numbers[0]}")
            res.write("</caption>\n")

        res.write(self._gen_html_table_header(should_print_line_nums))
        res.write(self._gen_html_table_body())

        res.write("</table>\n")
        return res.getvalue()

    def gen_html(self, filename: str | None = None) -> str:
        res = StringIO()
        res.write("<!DOCTYPE html>\n")
        res.write("<!-- Generated HTML by beancode's trace table generator -->\n")
        res.write("<html>\n")
        res.write(f"<head>\n")
        res.write("<meta charset=UTF-8>\n")
        res.write('<meta name=color-scheme content="dark light">\n')

        title_s = ""
        if filename is not None:
            title_s = " for " + filename
        title = f"Generated Trace Table{title_s}"

        res.write(f"<title>{title}</title>\n")
       
        noselect = str()
        if not self.config.i_will_not_cheat:
            noselect = """
body {
  user-select: none;
}
            """

        res.write(f"<style>\n{TABLE_STYLE}\n{noselect}</style>\n")
        res.write("</head>\n")

        res.write(f"<body><center>\n")

        res.write(f"<h1>{title}</h1>\n")
        res.write(self._gen_html_table() + "\n")

        res.write("</center></body>\n")
        res.write("</html>\n")
        return res.getvalue()

    def write_out(self, file_name: str | None = None):
        """write out tracer output with console output."""

        real_name = "tracer_output.html" if not file_name else file_name
        if os.path.splitext(real_name)[1] != ".html":
            warn(f"provided file path does not have the .html file extension!")
            real_name += ".html"

        if os.path.exists(real_name):
            warn(f'"{real_name}" already exists on disk! overwriting...')
        else:
            info(f'writing output to "{real_name}"...')

        try:
            with open(real_name, "w") as f:
                f.write(self.gen_html())
        except IsADirectoryError:
            error(f"cannot write the tracer's output to a directory!")
        except PermissionError:
            error(f"no permission to write tracer's output")
