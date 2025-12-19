from .error import *
from .lexer import Lexer
from .parser import Parser
from .optimizer import Optimizer
from .interpreter import Interpreter
from .cfgparser import parse_config_from_source
from .tracer import Tracer, TracerConfig

_tk_root = None


def get_file_path_with_dialog() -> str:
    try:
        # inspired by https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/
        import tkinter as tk
        from tkinter import filedialog as fd

        pass  # XXX: I made DeepSeek R1 write a bundler, this has to be here for that to work!
    except ImportError:
        warn("could not import tkinter to show a file picker!")
        return input("\033[1mEnter a file to run: \033[0m")

    global _tk_root
    if not _tk_root:
        _tk_root = tk.Tk()
        _tk_root.withdraw()

    filetypes = (
        ("Pseudocode/beancode scripts", "*.bean"),
        ("Pseudocode scripts", "*.pseudo"),
        ("Pseudocode scripts", "*.psc"),
        ("Pseudocode scripts", "*.pseudocode"),
        ("All files", "*.*"),
    )
    res = fd.askopenfilename(
        title="Select file to run", initialdir=".", filetypes=filetypes
    )
    res = os.path.expanduser(res)
    _tk_root.update()
    return res


def _read_whole_file_nicely(path: str) -> str | None:
    """reads a whole file, running expanduser and catching exceptions."""

    real_path = os.path.expanduser(path)
    file_content = str()
    try:
        with open(real_path, "r") as f:
            file_content = f.read()
    except IsADirectoryError:
        error("expected a file but got a directory!")
        return None
    except FileNotFoundError:
        error(f'file "{path}" was not found!')
        return None
    except PermissionError:
        error(f'no permissions to access "{path}"!')
        return None
    except Exception as e:
        error(f"a Python exception was caught: {e}")
        return None

    return file_content


def run_file(filename: str | None = None):
    if not filename:
        info("Opening tkinter file picker")
        real_path = get_file_path_with_dialog()
    else:
        real_path = filename

    file_content = _read_whole_file_nicely(real_path)
    if not file_content:
        return

    execute(file_content, filename=real_path)


def trace(
    filename: str | None = None,
    vars: list[str] | None = None,
    target_file: str | None = None,
    config_path: str | None = None,
):
    if not filename:
        info("Opening tkinter file picker")
        real_path = get_file_path_with_dialog()
    else:
        real_path = filename
    src = _read_whole_file_nicely(real_path)
    if not src:
        return

    if vars:
        vars = [val.strip() for val in vars]
    else:
        warn("not tracing script with any vars!")
        vars = list()

    tracer = Tracer(vars)
    tracer.config.write_to_default_location()

    actual_path = config_path
    if config_path and config_path in ("", "default"):
        tracer.load_config()

    if actual_path is not None:
        file_content = _read_whole_file_nicely(actual_path)
        if not file_content:
            return

        try:
            cfg = parse_config_from_source(file_content)
        except BCError as e:
            e.print(actual_path, file_content)
            exit(1)

        tracer.config = TracerConfig.from_config(cfg)

    execute(
        src,
        filename=real_path,
        save_interpreter=False,
        tracer=tracer,
        notify_when_done=True,
    )
    output_path = tracer.write_out(target_file)
    tracer.open(output_path)


def execute(
    src: str,
    filename="(execute)",
    save_interpreter=False,
    tracer: "Tracer | None" = None,
    notify_when_done=False,
    optimize=False,
) -> "Interpreter | BCError":  # type: ignore
    lexer = Lexer(src)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            exit(1)
        else:
            return err

    parser = Parser(toks)

    try:
        program = parser.program()
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            return err
        else:
            exit(1)

    if optimize:
        try:
            opt = Optimizer(program.stmts)
            program.stmts = opt.visit_block(None)
        except BCError as err:
            err.print(filename, src)
            if save_interpreter:
                return err
            else:
                exit(1)

    if tracer is None:
        i = Interpreter(program.stmts)
    else:
        i = Interpreter(program.stmts, tracer=tracer, tracer_open=True)

    i.toplevel = True
    try:
        i.visit_block(None)
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            return err
        else:
            exit(1)

    if notify_when_done:
        info("execution of script/code complete!")

    if save_interpreter:
        return i
