from .bean_ast import BCValue, Literal
from .lexer import Lexer, TokenKind
from .parser import Parser
from .error import BCError


# parser manhandling sure is fun!
def parse_config_from_source(src: str) -> dict[str, BCValue]:
    lx = Lexer(src)
    tokens = lx.tokenize()

    res = dict()
    ps = Parser(tokens)
    while ps.cur < len(ps.tokens):
        ps.consume_newlines()
        # Ident Assign Literal
        ident = ps.ident("for configuration file key")
        ps.consume_and_expect(TokenKind.ASSIGN, "after each configuration file key")
        lit: Literal | None = ps.literal()  # type: ignore
        if lit is None:
            raise BCError(
                "invalid or no literal after configuration file key!", ps.pos()
            )

        if ps.cur != len(ps.tokens) - 1:
            ps.consume_and_expect(TokenKind.NEWLINE, "after configuration entry")

        ps.consume_newlines()
        res[ident.ident] = lit.val

    return res


def parse_config_from_file(file_name: str) -> dict[str, BCValue]:
    """parses a config from a file, and exits on errors."""

    src = str()
    with open(file_name, "r") as f:
        src = f.read()

    try:
        return parse_config_from_source(src)
    except BCError as e:
        e.print(file_name, src)
        exit(1)
