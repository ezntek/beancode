from beancode.error import warn, info

def _get_file_path_with_dialog() -> str:
    try:
        # inspired by https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/
        from tkinter import filedialog as fd
    except ImportError:
        warn("could not import tkinter to show a file picker!")
        return input("\033[1mEnter a file to run: \033[0m")
   
    filetypes = (
        ('Pseudocode/beancode scripts', '*.bean'),
        ('Pseudocode scripts', '*.pseudo'),
        ('Pseudocode scripts', '*.psc'),
        ('Pseudocode scripts', '*.pseudocode'),
        ('All files', '*.*'),
    )
    return fd.askopenfilename(title='Select file to run', initialdir='.', filetypes=filetypes)

def run_file(filename: str | None = None):
    if not filename:
        info("opening tkinter file picker")
        real_path = _get_file_path_with_dialog()
    else:
        real_path = filename

    file_content = str()
    with open(real_path, "r") as f:
        file_content = f.read()
    execute(file_content)


def execute(src: str, filename="(execute)", save_interpreter=False) -> "Interpreter | None": # type: ignore
    from .error import BCError
    from .lexer import Lexer
    from .parser import Parser
    from .interpreter import Interpreter

    lexer = Lexer(src)

    try:
        toks = lexer.tokenize()
    except BCError as err:
        err.print(filename, src)
        exit(1)

    parser = Parser(toks)

    try:
        program = parser.program()
    except BCError as err:
        err.print(filename, src)
        exit(1)

    i = Interpreter(program.stmts)
    i.toplevel = True
    try:
        i.visit_block(None)
    except BCError as err:
        err.print(filename, src)
        exit(1)
    
    if save_interpreter:
        return i
