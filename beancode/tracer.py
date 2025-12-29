import subprocess

from .cfgparser import parse_config_from_file

from . import __version__
from .bean_ast import *

TABLE_STYLE = ".bean-table{border-collapse:collapse}.bean-table td,.bean-table th,.bean-table tr{border:1px solid;padding:0 0.7em;text-align:center}.bean-table pre{font-size:1.3em}.bean-table .io{font-weight:normal}@media (prefers-color-scheme: light){.bean-table pre{font-weight:bold}.bean-table caption{color:rgb(95, 95, 95);caption-side:bottom}.bean-table .F{color:rgb(230, 41, 55)}.bean-table .T{color:rgb(0, 158, 47)}.bean-table .I{color:rgb(230, 156, 29)}.bean-table .D{font-weight:normal;color:rgb(95, 95, 95)}}@media (prefers-color-scheme: dark){.bean-table caption{color:rgb(150, 150, 150);caption-side:bottom}.bean-table .F{color:rgb(230, 41, 55)}.bean-table .T{color:rgb(0, 158, 47)}.bean-table .I{color:rgb(245, 193, 0)}.bean-table .D{color:rgb(130, 130, 130)}}"
NOSELECT_STYLE = "body{-webkit-user-drag:none;-webkit-touch-callout:none;pointer-events:none;user-select:none !important;-ms-user-select:none}"


def _pascal_case_to_snake(name: str) -> str:
    if len(name) == 0:
        return name

    res = [name[0].lower()]
    for c in name[1:]:
        if c.isupper():
            res.append("_")
            res.append(c.lower())
        else:
            res.append(c)

    return "".join(res)


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
        KEYS = {
            "TraceEveryLine",
            "HideRepeatingEntries",
            "CondenseArrays",
            "SyntaxHighlighting",
            "ShowOutputs",
            "PromptOnInputs",
            "Debug",
            "IWillNotCheat",
        }
        for key in KEYS:
            if key in cfg:
                data = cfg[key]
                if data.kind == BCPrimitiveType.BOOLEAN:
                    setattr(res, _pascal_case_to_snake(key), data.get_boolean())

        return res

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> "TracerConfig":
        res = cls()
        for key, val in cfg.items():
            if key in res.__dict__:
                setattr(res, key, val)
        return res

    def write_out(self, path: str):
        s = f"""
TraceEveryLine <- {str(self.trace_every_line).upper()}
HideRepeatingEntries <- {str(self.hide_repeating_entries).upper()}
CondenseArrays <- {str(self.condense_arrays).upper()}
SyntaxHighlighting <- {str(self.syntax_highlighting).upper()}
ShowOutputs <- {str(self.show_outputs).upper()}
PromptOnInputs <- {str(self.prompt_on_inputs).upper()}
Debug <- {str(self.debug).upper()}
IWillNotCheat <- {str(self.i_will_not_cheat).upper()}
"""
        with open(path, "w") as f:
            f.write(s)

    def write_to_default_location(self, force=False):
        cfgpath = str()
        if sys.platform != "win32":
            cfgpath = os.environ.get("XDG_CONFIG_HOME")
            if not cfgpath:
                cfgpath = os.path.join(os.environ["HOME"], ".config")

            # darwin
            if not os.path.exists(cfgpath):
                os.mkdir(cfgpath)

            cfgpath = os.path.join(cfgpath, "beancode", "tracerconfig.bean")
        else:
            cfgpath = f"{os.environ['APPDATA']}\\beancode\\tracerconfig.bean"

        dir = os.path.dirname(cfgpath)
        if not os.path.exists(dir):
            os.mkdir(dir)

        if force or not os.path.exists(cfgpath):
            self.write_out(cfgpath)


class Tracer:
    vars: dict[str, list[BCValue | None]]
    var_types: dict[str, BCType]
    last_updated_vals: dict[str, BCValue | None]  # None only initially
    line_numbers: dict[int, int]
    outputs: dict[int, list[str]]
    inputs: dict[int, list[str]]
    last_idx: int
    # internal use only !
    cols: int

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
        self.cols = 0

        # weird python object copy/move semantics
        if config:
            self.config = config
        else:
            self.config = TracerConfig()

        for var in wanted_vars:
            self.vars[var] = list()
            self.var_types[var] = BCPrimitiveType.NULL
            self.last_updated_vals[var] = None

    def load_config(self, search_paths: list[str] | None = None):
        if not search_paths:
            config_paths = list()

            if sys.platform != "win32":
                cfgpath = os.environ.get("XDG_CONFIG_HOME")
                if not cfgpath:
                    cfgpath = os.path.join(os.environ["HOME"], ".config")
                cfgpath = os.path.join(cfgpath, "beancode", "tracerconfig.bean")

                config_paths = [
                    cfgpath,
                    "./tracerconfig.bean",
                ]
            else:
                config_paths = [
                    f"{os.getenv('APPDATA')}\\beancode\\tracerconfig.bean",
                    ".\\tracerconfig.bean",
                ]
        else:
            config_paths = search_paths

        for path in config_paths:
            if os.path.exists(path):
                cfg = parse_config_from_file(path)
                self.config = TracerConfig.from_config(cfg)
                break

    def open(self, path: str):
        match sys.platform:
            case "darwin":
                subprocess.run(["open", path])
            case "linux" | "freebsd":
                subprocess.run(["xdg-open", path])
            case "win32":
                subprocess.run(["cmd", "/c", "start", "", path])

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
                    new_obj = vars[k].val.copy()
                    v.append(new_obj)
                    self.last_updated_vals[k] = new_obj
                    self.var_types[k] = new_obj.kind
            last_idx = len(v) - 1

        if len(self.vars) == 0:
            self.last_idx += 1
        else:
            self.last_idx = last_idx

        if outputs is not None and len(outputs) > 0:
            self.outputs[self.last_idx] = list(outputs)

        if inputs is not None and len(inputs) > 0:
            self.inputs[self.last_idx] = list(inputs)

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
                and typ.is_flat()
            ):
                has_array = True
                break
        return has_array

    def _highlight_var(self, var: BCValue) -> str:
        if var.is_uninitialized():
            if self.config.syntax_highlighting:
                return f"<td><pre class=D>null</pre></td>"
            else:
                return "<td><pre>(null)</pre></td>"

        if not self.config.syntax_highlighting:
            return f"<td><pre>{str(var)}</pre></td>"

        match var.kind:
            case BCPrimitiveType.BOOLEAN:
                klass = "T" if var.val == True else "F"
                return f"<td><pre class={klass}>{str(var)}</pre></td>"
            case BCPrimitiveType.INTEGER | BCPrimitiveType.REAL:
                return f"<td><pre class=I>{str(var)}</pre></td>"
            case _:
                return f"<td><pre>{str(var)}</pre></td>"

    def _gen_html_table_header(self, should_print_line_nums: bool) -> str:
        res = list()

        res.append("<thead>")
        res.append("<tr>")

        has_array = self._has_array()
        rs = " rowspan=2" if has_array else ""

        if should_print_line_nums:
            res.append(f"<th style=padding:0.23em{rs}>Line</th>")

        # first pass
        for name, typ in self.var_types.items():
            if (
                not self.config.condense_arrays
                and isinstance(typ, BCArrayType)
                and typ.is_flat()
            ):
                width = typ.get_flat_bounds()[1] - typ.get_flat_bounds()[0] + 1  # type: ignore
                res.append(f"<th colspan={width}>{name}</th>")
            else:
                res.append(f"<th{rs}>{name}</th>")
            self.cols += 1

        if len(self.inputs) > 0:
            res.append(f"<th{rs}>Inputs</th>")
            self.cols += 1

        if len(self.outputs) > 0:
            res.append(f"<th{rs}>Outputs</th>")
            self.cols += 1

        # second pass
        if has_array:
            res.append("</tr><tr>")
            for name, typ in self.var_types.items():
                if isinstance(typ, BCArrayType) and typ.is_flat():
                    bounds = typ.get_flat_bounds()
                    for num in range(bounds[0], bounds[1] + 1):
                        res.append(f"<th>[{num}]</th>")
                        self.cols += 1

        res.append("</tr>")
        res.append("</thead>")

        return "".join(res)

    def _gen_html_table_line_num(self, row_num: int) -> str:
        if self._should_print_line_numbers():
            if row_num in self.line_numbers:
                return f"<td>{self.line_numbers[row_num]}</td>"
        return str()

    def _gen_html_table_row(
        self,
        rows: list[tuple[int, tuple[BCValue | None, ...]]],
        row_num: int,
        row: tuple[BCValue | None, ...],
        printed_first: bool = True,
    ) -> str:
        res = list()

        for col, (var_name, var) in enumerate(zip(self.vars, row)):
            if not self.config.condense_arrays and isinstance(
                self.var_types[var_name], BCArrayType
            ):
                if not var:
                    # blank the region out
                    bounds = self.var_types[var_name].get_flat_bounds()  # type: ignore
                    for _ in range(bounds[0], bounds[1] + 1):
                        res.append(f"<td/>")
                else:
                    # rows[row_num] is enumerated, col+1 compensates for the index at the front
                    arr: BCArray = var.get_array()
                    if arr.typ.is_flat():
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
                                res.append("<td/>")
                            else:
                                res.append(self._highlight_var(itm))
            else:
                prev: BCValue | None = None
                if row_num != 0:
                    prev = rows[row_num - 1][1][col]

                repeated = self.config.hide_repeating_entries and var == prev
                if not var or repeated and printed_first:
                    res.append("<td/>")
                else:
                    res.append(self._highlight_var(var))

        return "".join(res)

    def _gen_html_table_row_io(self, row_num: int) -> str:
        res = list()

        if len(self.inputs) > 0:
            s = str()
            if row_num in self.inputs:
                l = self.inputs[row_num]
                s = "<br/>".join(l)
            res.append(f"<td><pre class=io>{s}</pre></td>")

        if len(self.outputs) > 0:
            s = str()
            if row_num in self.outputs:
                l = self.outputs[row_num]
                s = "<br/>".join(l)
            res.append(f"<td><pre class=io>{s}</pre></td>")

        return "".join(res)

    def _gen_html_table_body(self):
        res = list()

        res.append("<tbody>")

        if len(self.vars) == 0:
            keys = set()
            for k in self.outputs:
                keys.add(k)
            for k in self.inputs:
                keys.add(k)

            for k in keys:
                res.append("<tr>")
                res.append(self._gen_html_table_line_num(k))
                res.append(self._gen_html_table_row_io(k))
                res.append("</tr>")

            res.append("</tbody>")
            return "".join(res)

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

                    if var.is_array and var.kind.is_flat():  # type: ignore
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

            res.append("<tr>")

            res.append(self._gen_html_table_line_num(row_num))
            res.append(self._gen_html_table_row(rows, row_num, row, printed_first))
            res.append(self._gen_html_table_row_io(row_num))

            printed_first = True

            res.append("</tr>")

        if self._should_print_line_numbers():
            self.cols += 1

        res.append(
            f'<tr><td style="padding: 0.1em" colspan={self.cols}><small class=D>'
            + f"Generated by beancode version <strong>{__version__}</strong>"
            + "</small></td></tr>"
        )
        res.append("</tbody>")
        return "".join(res)

    def _gen_html_table(self) -> str:
        res = list()
        res.append("<table class=bean-table>")

        # generate header
        should_print_line_nums = self._should_print_line_numbers()

        if not should_print_line_nums:
            res.append("<caption>")
            if len(self.line_numbers) == 0:
                res.append("No values were captured.")
            else:
                res.append(f"All values are captured at line {self.line_numbers[0]}")
            res.append("</caption>")

        res.append(self._gen_html_table_header(should_print_line_nums))
        res.append(self._gen_html_table_body())

        res.append("</table>")
        return "".join(res)

    def gen_html(self) -> str:
        res = list()
        res.append("<!DOCTYPE html>\n")
        res.append(
            f"<!-- Generated HTML by beancode's trace table generator, version {__version__} -->\n"
        )
        # HTML tag is optional, so is head
        res.append("<meta charset=UTF-8>")
        res.append('<meta name=color-scheme content="dark light">')

        title = f"Generated Trace Table"

        res.append(f"<title>{title}</title>")

        noselect = "" if self.config.i_will_not_cheat else NOSELECT_STYLE
        res.append(f"<style>{TABLE_STYLE}{noselect}</style>")

        res.append(f"<center>")

        res.append(f"<h1>Generated Trace Table</h1>")
        res.append(self._gen_html_table())

        res.append("</center>")
        return "".join(res)

    def write_out(self, file_name: str | None = None) -> str:
        """write out tracer output with console output."""

        real_name = "tracer_output.html" if not file_name else file_name
        if os.path.splitext(real_name)[1] != ".html":
            warn(f"provided file path does not have the .html file extension!")
            real_name += ".html"

        full_path = os.path.abspath(os.path.join("./", real_name))

        if os.path.exists(real_name):
            warn(f'"{full_path}" already exists on disk! overwriting...')
        else:
            info(f'writing output to "{full_path}"...')

        try:
            with open(real_name, "w") as f:
                f.write(self.gen_html())
        except IsADirectoryError:
            error(f"cannot write the tracer's output to a directory!")
        except PermissionError:
            error(f"no permission to write tracer's output")

        return full_path
