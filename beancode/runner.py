
def run(filename: str):
    file_content = str()
    with open(filename, "r") as f:
        file_content = f.read()
    execute(file_content)


def execute(src: str, filename="(execute)"):
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
