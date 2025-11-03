from beancode.error import *

_tk_root = None


def get_file_path_with_dialog() -> str:
    try:
        # inspired by https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/
        import tkinter as tk
        from tkinter import filedialog as fd
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
    _tk_root.update()
    return res


def run_repl():
    from .repl import Repl

    Repl(debug=False).repl()


def run_file(filename: str | None = None):
    if not filename:
        info("Opening tkinter file picker")
        real_path = get_file_path_with_dialog()
    else:
        real_path = filename

    real_path = os.path.expanduser(real_path)
    file_content = str()
    try:
        with open(real_path, "r") as f:
            file_content = f.read()
    except IsADirectoryError:
        error("cannot run a directory!")
    except FileNotFoundError:
        error("file to run was not found!")
    except PermissionError:
        error("no permissions to run script!")
    except Exception as e:
        error(f"a Python exception was caught: {e}")

    execute(file_content, filename=real_path)


def trace(filename: str | None = None):
    pass


def execute(src: str, filename="(execute)", save_interpreter=False) -> "Interpreter | None":  # type: ignore
    from .error import BCError
    from .lexer import Lexer
    from .parser import Parser
    from .interpreter import Interpreter

    lexer = Lexer(src)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            exit(1)
        else:
            return

    parser = Parser(toks)

    try:
        program = parser.program()
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            exit(1)
        else:
            return

    i = Interpreter(program.stmts)
    i.toplevel = True
    try:
        i.visit_block(None)
    except BCError as err:
        err.print(filename, src)
        if save_interpreter:
            exit(1)
        else:
            return

    if save_interpreter:
        return i
