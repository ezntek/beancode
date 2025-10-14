__version__ = "0.4.0-beta1"


def prefix_string_with_article(s: str) -> str:
    if s[0].lower() in "aeiou":
        return "an " + s
    else:
        return "a " + s


def humanize_index(idx: int) -> str:
    s = str(idx)
    last = s[-1]

    if len(s) == 1 or (len(s) > 1 and s[-2] != "1"):
        match last:
            case "1":
                return s + "st"
            case "2":
                return s + "nd"
            case "3":
                return s + "rd"
            case _:
                return s + "th"

    return s + "th"


def is_case_consistent(s: str) -> bool:
    return s.isupper() or s.islower()


def error(msg: str):
    print(f"\033[31;1merror: \033[0m{msg}")
    exit(1)


def panic(msg: str):
    print(f"\033[31;1mpanic! \033[0m{msg}")
    print(
        "\033[31mplease report this error to the developers. A traceback is provided:\033[0m"
    )
    raise Exception("panicked")

def run(filename: str):
    """Execute code from a file path."""

    file_content = str()
    with open(filename, "r") as f:
        file_content = f.read()
    execute(file_content)

def execute(src: str, filename="(execute)"):
    """Execute code from a Python string."""

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
