from dataclasses import dataclass

from .bean_ast import TokenKind, BCPrimitiveType, Comment
from .error import *
from . import Pos, __version__, is_case_consistent, panic


@dataclass
class Token:
    kind: TokenKind
    pos: Pos
    data: str | BCPrimitiveType | Comment | None = None

    def print(self, file=sys.stdout):
        match self.kind:
            case TokenKind.LITERAL_STRING:
                s = f'"{self.data}"'
            case TokenKind.LITERAL_CHAR:
                s = f"'{self.data}'"
            case TokenKind.LITERAL_NUMBER:
                s = self.data
            case TokenKind.TYPE:
                s = f"<{str(self.data).upper()}>"
            case TokenKind.IDENT:
                s = f"{{{str(self.data)}}}"
            case _:
                s = f"<{self.kind}>"

        print(f"token[{self.pos}]: {s}", file=file)

    def to_humanized_string(self) -> str:
        match self.kind:
            case TokenKind.TYPE:
                return f"{str(self.data).upper()}"
            case _:
                return self.kind.humanize()

    def __repr__(self) -> str:
        return f"token({self.kind})"


TYPES = {"integer", "boolean", "real", "char", "string", "array"}
KEYWORDS = {
    "declare",
    "constant",
    "output",
    "input",
    "and",
    "or",
    "not",
    "if",
    "then",
    "else",
    "endif",
    "case",
    "of",
    "otherwise",
    "endcase",
    "while",
    "do",
    "endwhile",
    "repeat",
    "until",
    "for",
    "to",
    "step",
    "next",
    "procedure",
    "endprocedure",
    "call",
    "function",
    "returns",
    "return",
    "endfunction",
    "openfile",
    "readfile",
    "writefile",
    "closefile",
    "read",
    "write",
    "append",
    "trace",
    "endtrace",
    "scope",
    "endscope",
    "include",
    "include_ffi",
    "export",
    "print",
}


class Lexer:
    src: str
    row: int
    bol: int
    cur: int
    found_shebang: bool
    preserve_comments: bool
    cur_comment: Comment | None
    cur_comment_pos: Pos | None

    def __init__(self, src: str, preserve_comments=False) -> None:
        self.src = src
        self.found_shebang = False
        self.preserve_comments = preserve_comments
        self.cur_comment = None
        self.cur_comment_pos = None
        self.reset()

    def reset(self):
        self.row = 1
        self.bol = 0
        self.cur = 0

    def get_cur(self):
        return self.src[self.cur]

    def peek(self):
        return self.src[self.cur + 1]

    def bump_newline(self):
        self.row += 1
        self.cur += 1
        self.bol = self.cur

    def in_bounds(self) -> bool:
        return self.cur < len(self.src)

    def pos(self, span: int) -> Pos:
        return Pos(row=self.row, col=self.cur - self.bol + 1 - span, span=span)

    def pos_here(self, span: int) -> Pos:
        return Pos(row=self.row, col=self.cur - self.bol + 1, span=span)

    def is_separator(self, ch: str) -> bool:
        return ch in "{}[]();:,"

    def is_operator_start(self, ch: str) -> bool:
        # catch % for error
        return ch in "%+-*/<>=^\uf0ac←"

    def trim_spaces(self) -> Comment | None:
        if not self.in_bounds():
            return

        cur = str()
        while self.in_bounds() and (cur := self.get_cur()).isspace() and cur != "\n":
            self.cur += 1

        self.trim_comments()

    def trim_comments(self) -> Comment | None:
        # if there are not 2 chars more in the stream (// and /*)
        if self.cur + 2 > len(self.src):
            return

        pair = self.src[self.cur : self.cur + 2]
        if pair not in {"//", "/*", "#!"}:
            return

        multiline = False
        if pair == "/*":
            self.cur_comment_pos = self.pos_here(2)
            multiline = True
            self.cur += 2

            buf = []
            cur_begin = self.cur
            while self.in_bounds() and self.src[self.cur : self.cur + 2] != "*/":
                if self.preserve_comments:
                    if self.get_cur() == "\n":
                        s = self.src[cur_begin : self.cur]
                        buf.append(s)
                        self.cur += 1
                        cur_begin = self.cur
                    else:
                        self.cur += 1
                    continue

                if self.get_cur() == "\n":
                    self.bump_newline()
                else:
                    self.cur += 1

            # found */
            self.cur += 2
            self.cur_comment = Comment(buf, multiline, shebang=False)
        else:
            self.cur_comment_pos = self.pos_here(2)
            shebang = False
            if pair == "#!":
                if self.found_shebang:
                    raise BCError(
                        "cannot have more than one shebang in one source file!",
                        self.pos_here(2),
                    )
                self.found_shebang = True
                shebang = True

            self.cur += 2  # skip past comment marker
            begin = self.cur
            while self.in_bounds() and self.get_cur() != "\n":
                self.cur += 1
            data = [self.src[begin : self.cur]]
            self.cur_comment = Comment(data, multiline=False, shebang=shebang)

        self.trim_spaces()

    def next_double_symbol(self) -> Token | None:
        if not self.is_operator_start(self.get_cur()):
            return None

        if self.cur + 2 > len(self.src):
            return None

        TABLE: dict[str, TokenKind] = {
            "<>": TokenKind.NOT_EQUAL,
            ">=": TokenKind.GREATER_THAN_OR_EQUAL,
            "<=": TokenKind.LESS_THAN_OR_EQUAL,
            "<-": TokenKind.ASSIGN,
        }

        pair = self.src[self.cur : self.cur + 2]
        kind = TABLE.get(pair)

        if kind is not None:
            self.cur += 2
            return Token(kind, self.pos(2))

    def next_single_symbol(self) -> Token | None:
        cur = self.get_cur()
        if not self.is_separator(cur) and not self.is_operator_start(cur):
            return None

        if cur == "%":
            raise BCError(
                "% as an operator is not supported!\nPlease use the MOD(a, b) library routine instead of a % b!",
                self.pos(1),
            )

        TABLE: dict[str, TokenKind] = {
            "{": TokenKind.LEFT_CURLY,
            "}": TokenKind.RIGHT_CURLY,
            "[": TokenKind.LEFT_BRACKET,
            "]": TokenKind.RIGHT_BRACKET,
            "(": TokenKind.LEFT_PAREN,
            ")": TokenKind.RIGHT_PAREN,
            ":": TokenKind.COLON,
            ";": TokenKind.NEWLINE,
            ",": TokenKind.COMMA,
            "=": TokenKind.EQUAL,
            "<": TokenKind.LESS_THAN,
            ">": TokenKind.GREATER_THAN,
            "*": TokenKind.MUL,
            "/": TokenKind.DIV,
            "+": TokenKind.ADD,
            "-": TokenKind.SUB,
            "^": TokenKind.POW,
            "←": TokenKind.ASSIGN,
            "\uf0ac": TokenKind.ASSIGN,
        }

        kind = TABLE.get(cur)
        if kind is not None:
            self.cur += 1
            return Token(kind, self.pos(1))

    def next_word(self) -> str:
        begin = self.cur
        len = 0
        DELIMS = "\"'"
        is_delimited_literal = self.src[begin] in DELIMS
        delim = str()
        if is_delimited_literal:
            delim = self.get_cur()
            self.cur += 1
            len += 1

        while True:
            stop = False
            if not self.in_bounds():
                break

            cur = self.get_cur()
            if is_delimited_literal:
                stop = cur == delim
            else:
                stop = (
                    self.is_operator_start(cur)
                    or self.is_separator(cur)
                    or cur.isspace()
                    or cur in DELIMS
                )

            if cur == "\\":
                len += 1
                self.cur += 1

            if stop:
                break

            len += 1
            self.cur += 1

        if is_delimited_literal:
            if not self.in_bounds():
                # we don't set eof to true, becuase we do not allow for multile string literals, and this
                # would break the REPL.
                raise BCError(
                    "could not find ending delimiter in literal\n"
                    + "did you forget to insert an ending quotation mark?",
                    self.pos(len),
                )
            else:
                self.cur += 1
                len += 1

        res = self.src[begin : begin + len]
        return res

    def next_keyword(self, word: str) -> Token | None:
        if is_case_consistent(word):
            if word.lower() not in KEYWORDS:
                return None
        else:
            return None

        kind = TokenKind.from_str(word.lower())
        return Token(kind, self.pos(len(word)))

    def next_type(self, word: str) -> Token | None:
        if is_case_consistent(word):
            if word.lower() not in TYPES:
                return None
        else:
            return None

        return Token(TokenKind.TYPE, self.pos(len(word)), data=word.lower())

    def _is_number(self, word: str) -> bool:
        found_decimal = False

        for ch in word:
            if ch.isdigit():
                continue

            if ch == ".":
                if found_decimal:
                    return False
                else:
                    found_decimal = True
                continue

            return False

        if found_decimal and len(word) == 1:
            return False

        return True

    def next_literal(self, word: str) -> Token | None:
        if word[0] in "\"'":
            if len(word) == 1:
                panic("reached unreachable code")

            res = word[1:-1]
            kind: TokenKind = (
                TokenKind.LITERAL_STRING if word[0] == '"' else TokenKind.LITERAL_CHAR
            )

            return Token(kind, self.pos(len(word)), data=res)

        if self._is_number(word):
            return Token(TokenKind.LITERAL_NUMBER, self.pos(len(word)), data=word)
        elif word[0].isdigit():
            raise BCError("invalid number literal", self.pos(len(word)))

        if is_case_consistent(word):
            if word.lower() == "true":
                return Token(TokenKind.TRUE, self.pos(len(word)))
            elif word.lower() == "false":
                return Token(TokenKind.FALSE, self.pos(len(word)))
            elif word.lower() == "null":
                return Token(TokenKind.NULL, self.pos(len(word)))

    def _is_ident(self, word: str) -> bool:
        if not word[0].isalpha() and word[0] not in "_":
            return False

        for ch in word:
            if not ch.isalnum() and ch not in "_.":
                return False

        return True

    def next_ident(self, word: str) -> Token:
        p = self.pos(len(word))
        if self._is_ident(word):
            if is_case_consistent(word):
                p = self.pos(len(word))
                match word.lower():
                    case "endfor":
                        raise BCError(
                            "ENDFOR is not a valid keyword!\nPlease use NEXT <your counter> to end a for loop instead.",
                            p,
                        )
                    case "open":
                        raise BCError(
                            "OPEN is not a valid keyword!\nPlease use OPENFILE instead. If you are copying from the textbook\n"
                            + "(ISBN 9781398318281), their File I/O examples are incorrect to\n"
                            + "Cambridge's official pseudocode.",
                            p,
                        )
                    case "close":
                        raise BCError(
                            "CLOSE is not a valid keyword!\nPlease used CLOSEFILE instead.",
                            p,
                        )
            return Token(TokenKind.IDENT, p, data=word)
        else:
            raise BCError("invalid identifier or symbol", p)

    def next_token(self) -> Token | None:
        self.trim_spaces()

        if self.cur_comment and self.preserve_comments:
            t = Token(TokenKind.COMMENT, self.cur_comment_pos, self.cur_comment)  # type: ignore
            self.cur_comment = None
            self.cur_comment_pos = None
            return t

        if not self.in_bounds():
            return

        if self.get_cur() == "\n":
            t = Token(TokenKind.NEWLINE, self.pos_here(1))
            self.bump_newline()
            return t

        res: Token | None
        if res := self.next_double_symbol():
            return res

        if res := self.next_single_symbol():
            return res

        word = self.next_word()

        if res := self.next_keyword(word):
            return res

        if res := self.next_type(word):
            return res

        if res := self.next_literal(word):
            return res

        return self.next_ident(word)

    def tokenize(self) -> list[Token]:
        res = list()
        while self.in_bounds():
            t = self.next_token()
            if not t:
                break
            res.append(t)
        res.append(Token(TokenKind.NEWLINE, self.pos_here(1)))
        return res
